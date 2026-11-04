"""Free-only failover, bounded use, grounding guards, and secret-safe failures."""

from dataclasses import replace
import json

import pytest

from src.llm_client import (ExplanationService, LLMConfig, Reply, RouterRuntime,
                            ServiceFailure, load_config, validate_explanation)
from src.llm_results import demo_snapshot, template_text


PRIMARY = "dots-studio/dots-3-note-preview:free"
BACKUP = "nvidia/nemotron-3.5-super-vl:free"
CONFIG = LLMConfig("test-secret-must-never-appear", PRIMARY, BACKUP)


def valid_content(snapshot):
    return {"snapshot_id": snapshot.identity,
            "observations": [{"evidence_id": "E001", "text": "This is a recorded estimate for an eligible observation."}],
            "interpretation": "Compare the recorded estimate with its stated quality and evidence scope.",
            "limitations": snapshot.payload["limitations"],
            "review_checks": snapshot.payload["review_checks"][:1]}


def success(snapshot, model=PRIMARY, content=None):
    return Reply(200, {"model": model, "choices": [{"finish_reason": "stop", "message": {
        "content": json.dumps(valid_content(snapshot) if content is None else content)}}], "usage": {"cost": 0}})


class FakeTransport:
    def __init__(self, snapshot, primary=None, backup=None, quota=50):
        self.snapshot, self.primary, self.backup = snapshot, primary, backup
        self.quotas = [quota] if isinstance(quota, int) else list(quota)
        self.calls = []
        self.catalog = {model: {"id": model, "pricing": {"prompt": "0", "completion": "0"},
            "supported_parameters": ["structured_outputs", "response_format", "reasoning"]} for model in (PRIMARY, BACKUP)}

    def send(self, method, path, api_key, payload, deadline):
        self.calls.append((method, path, payload))
        if path == "/models":
            assert api_key == ""
            return Reply(200, {"data": list(self.catalog.values())})
        assert api_key == CONFIG.api_key
        if path == "/key":
            quota = self.quotas.pop(0) if len(self.quotas) > 1 else self.quotas[0]
            return Reply(200, {"data": {"free_model_daily_requests": {"remaining": quota}}})
        action = self.primary if payload["model"] == PRIMARY else self.backup
        assert payload["provider"]["max_price"] == {"prompt": 0, "completion": 0}
        assert "models" not in payload and "tools" not in payload and "plugins" not in payload
        assert CONFIG.api_key not in json.dumps(payload)
        if isinstance(action, Exception):
            raise action
        return action if action is not None else success(self.snapshot, payload["model"])

    @property
    def posts(self):
        return [payload["model"] for method, path, payload in self.calls if method == "POST"]


def run(primary=None, backup=None, *, config=CONFIG, quota=50, runtime=None):
    snapshot = demo_snapshot()
    transport = FakeTransport(snapshot, primary, backup, quota)
    service = ExplanationService(runtime, transport)
    return service.generate(snapshot, config, session_id="test-visitor"), transport, service


def test_primary_has_priority_and_only_allowlisted_snapshot_is_sent():
    result, transport, _ = run()
    assert result["source"] == "primary" and transport.posts == [PRIMARY]
    assert result["reported_cost"] == 0
    assert CONFIG.api_key not in json.dumps(result) and CONFIG.api_key not in repr(CONFIG)


@pytest.mark.parametrize("failure", [Reply(503, {}), Reply(404, {}), Reply(400, {}),
    Reply(429, {"error": {"code": 429}}), ServiceFailure("timeout"), ServiceFailure("connection_error"),
    Reply(200, {"error": {"code": 503}}), Reply(200, {"choices": []})])
def test_model_failure_tries_backup_once(failure):
    result, transport, _ = run(failure)
    assert result["source"] == "backup" and result["model"] == BACKUP
    assert transport.posts == [PRIMARY, BACKUP]


def test_both_models_fail_into_a_fact_based_local_template():
    result, transport, _ = run(ServiceFailure("timeout"), Reply(503, {}))
    assert result["source"] == "template" and result["model"] is None
    assert transport.posts == [PRIMARY, BACKUP]
    text = template_text(demo_snapshot())
    assert "3.2 g/L" in text and "0.45 g/L" in text
    assert "Unavailable / not assessed" in text and "not provided" in text


@pytest.mark.parametrize("config", [replace(CONFIG, enabled=False), replace(CONFIG, api_key=""),
    replace(CONFIG, primary="paid/model", backup=""), replace(CONFIG, primary="", backup="")])
def test_disabled_missing_key_or_no_free_models_never_connect(config):
    result, transport, _ = run(config=config)
    assert result["source"] == "template" and transport.calls == []


def test_paid_primary_is_skipped_and_duplicate_model_is_not_retried():
    result, transport, _ = run(config=replace(CONFIG, primary="paid/model"))
    assert result["source"] == "backup" and transport.posts == [BACKUP]
    result, transport, _ = run(Reply(503, {}), config=replace(CONFIG, backup=PRIMARY))
    assert result["source"] == "template" and transport.posts == [PRIMARY]


@pytest.mark.parametrize("status", [401, 402, 403])
def test_account_restrictions_skip_backup(status):
    result, transport, _ = run(Reply(status, {}))
    assert result["source"] == "template" and result["reason"] == "account_restricted"
    assert transport.posts == [PRIMARY]


def test_known_shared_daily_quota_skips_both_models_and_blocks_until_reset():
    runtime = RouterRuntime()
    result, transport, service = run(quota=0, runtime=runtime)
    assert result["reason"] == "shared_quota_exhausted" and transport.posts == []
    before = len(transport.calls)
    again = service.generate(demo_snapshot(), CONFIG)
    assert again["reason"] == "account_cooldown" and len(transport.calls) == before


@pytest.mark.parametrize("embedded", [False, True])
def test_daily_quota_exhaustion_discovered_on_error_does_not_try_backup(embedded):
    response = Reply(200 if embedded else 429, {"error": {"code": 429}})
    result, transport, _ = run(response, quota=[1, 0])
    assert result["source"] == "template" and transport.posts == [PRIMARY]
    assert result["reason"] == "shared_rate_limit"


def test_shared_minute_limit_does_not_try_another_model():
    result, transport, _ = run(Reply(429, {}, {"x-ratelimit-limit": "20", "retry-after": "60"}))
    assert result["source"] == "template" and transport.posts == [PRIMARY]


@pytest.mark.parametrize("damage", ["numeric", "claim", "unknown_evidence", "snapshot", "missing_limit",
    "url", "invented_gap", "not_performed", "wrong_rmse", "duplicate", "bad_json", "extra_field", "process_acceptability"])
def test_bad_explanations_are_rejected_and_backup_is_used(damage):
    snapshot = demo_snapshot()
    content = valid_content(snapshot)
    if damage == "numeric": content["interpretation"] = "The predicted value is 42 g/L."
    elif damage == "claim": content["interpretation"] = "This model is manufacturing validated and accurate."
    elif damage == "unknown_evidence": content["observations"][0]["evidence_id"] = "E999"
    elif damage == "snapshot": content["snapshot_id"] = "wrong"
    elif damage == "missing_limit": content["limitations"] = ["educational"]
    elif damage == "url": content["interpretation"] = "Read this guidance at https://untrusted.example."
    elif damage == "invented_gap": content["interpretation"] = "The missing prediction is 2.0 g/L."
    elif damage == "not_performed": content["interpretation"] = "No independent test was performed."
    elif damage == "wrong_rmse": content["interpretation"] = "RMSE is the average absolute error."
    elif damage == "duplicate": content["observations"] *= 2
    elif damage == "extra_field": content["new_metric"] = 10
    elif damage == "process_acceptability": content["interpretation"] = "This limit defines an acceptable boundary for safe operating conditions."
    response = success(snapshot, content=content)
    if damage == "bad_json": response.body["choices"][0]["message"]["content"] = "not JSON"
    result, transport, _ = run(response)
    assert result["source"] == "backup" and transport.posts == [PRIMARY, BACKUP]
    assert result["attempts"][0]["reason"] == "invalid_explanation"


def test_model_disappears_price_changes_or_lacks_schema_support():
    for change in ("missing", "price", "schema"):
        snapshot = demo_snapshot()
        transport = FakeTransport(snapshot)
        if change == "missing": del transport.catalog[PRIMARY]
        elif change == "price": transport.catalog[PRIMARY]["pricing"]["prompt"] = "0.1"
        else: transport.catalog[PRIMARY]["supported_parameters"] = []
        result = ExplanationService(transport=transport).generate(snapshot, CONFIG)
        assert result["source"] == "backup" and transport.posts == [BACKUP]


def test_failure_cooldown_skips_primary_on_the_next_action():
    _, transport, service = run(ServiceFailure("timeout"))
    result = service.generate(demo_snapshot(), CONFIG, session_id="another-visitor")
    assert result["source"] == "backup"
    assert transport.posts == [PRIMARY, BACKUP, BACKUP]


def test_visitor_and_app_usage_limits_are_shared_across_page_requests():
    _, transport, service = run()
    service.generate(demo_snapshot(), CONFIG, session_id="test-visitor")
    service.generate(demo_snapshot(), CONFIG, session_id="test-visitor")
    result = service.generate(demo_snapshot(), CONFIG, session_id="test-visitor")
    assert result["reason"] == "local_usage_limit" and len(transport.posts) == 3
    for index in range(27):
        service.generate(demo_snapshot(), CONFIG, session_id=f"visitor-{index}")
    result = service.generate(demo_snapshot(), CONFIG, session_id="last-visitor")
    assert result["reason"] == "local_usage_limit" and len(transport.posts) == 30


def test_request_schema_and_value_guards_do_not_claim_semantic_certainty():
    snapshot = demo_snapshot()
    assert validate_explanation(json.dumps(valid_content(snapshot)), snapshot)["observations"]
    for value in (None, "[]", "null", '{"unrecognized":true}'):
        with pytest.raises(ServiceFailure):
            validate_explanation(value, snapshot)


def test_refusal_goes_to_template_and_does_not_evade_it_using_backup():
    response = success(demo_snapshot())
    response.body["choices"][0]["finish_reason"] = "content_filter"
    result, transport, _ = run(response)
    assert result["reason"] == "response_refused" and transport.posts == [PRIMARY]


def test_prepared_source_requires_consent_before_any_network_access():
    from src.llm_results import ResultSnapshot
    payload = demo_snapshot().payload
    payload["source_kind"] = "prepared_sources"
    snapshot = ResultSnapshot(json.dumps(payload))
    transport = FakeTransport(snapshot)
    result = ExplanationService(transport=transport).generate(snapshot, CONFIG)
    assert result["reason"] == "consent_required" and transport.calls == []
    result = ExplanationService(transport=transport).generate(snapshot, CONFIG, consent=True)
    assert result["source"] == "primary"


def test_configuration_precedence_aliases_public_opt_in_and_no_environment_mutation(tmp_path):
    path = tmp_path / ".env"
    path.write_text(f"OPEN_ROUTER_API=local-secret\nOPENROUTER_MODEL={PRIMARY}\nOPENROUTER_MODEL_BACKUP={BACKUP}\n")
    env = {"OPEN_ROUTER_API": "environment-secret"}
    config = load_config(environ=env, secrets={"api_key": "secret-store-value"}, env_path=path)
    assert config.api_key == "environment-secret" and config.backup == BACKUP
    assert env == {"OPEN_ROUTER_API": "environment-secret"}
    assert load_config(environ={}, secrets={"api_key": "secret-store-value"}, env_path=path).api_key == "secret-store-value"
    assert not load_config(environ={}, env_path=path, public=True).enabled
    assert load_config(environ={}, secrets={"public_enabled": True}, env_path=path, public=True).enabled
    assert not load_config(environ={"OPENROUTER_ENABLED": "false"}, env_path=path).enabled
    assert load_config(environ={}, env_path=tmp_path / "missing").api_key == ""
