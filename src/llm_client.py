"""Free-only OpenRouter explanation chain. No network activity on import/render.

Only aggregate snapshots enter prompts. All errors leaving this module use
fixed reason codes, never remote messages, credentials, or request objects.
"""

from collections import OrderedDict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from email.utils import parsedate_to_datetime
from hashlib import sha256
import json
import os
from pathlib import Path
import re
from threading import Lock
import time

from dotenv import dotenv_values
import requests

from src.llm_results import PROMPT_VERSION, template_sections


ROOT = Path(__file__).resolve().parents[1]
BASE_URL = "https://openrouter.ai/api/v1"
FREE_MODEL = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.:-]+:free\Z")
STYLES = ("Learning explanation", "Technical explanation")


@dataclass(frozen=True)
class LLMConfig:
    api_key: str = field(default="", repr=False)
    primary: str = "dots-studio/dots-3-note-preview:free"
    backup: str = ""
    enabled: bool = True

    @property
    def identity(self):
        # Private session/runtime key only; never included in downloads or UI.
        return sha256(json.dumps([self.api_key, self.primary, self.backup, self.enabled]).encode()).hexdigest()


def load_config(*, environ=None, secrets=None, env_path=ROOT / ".env", public=False):
    """Environment > [llm] secrets > local dotenv; no os.environ mutation."""
    env = os.environ if environ is None else environ
    try:
        local = dotenv_values(env_path, interpolate=False) if Path(env_path).is_file() else {}
    except OSError:
        local = {}
    layers = [env, secrets or {}, local]

    def setting(names, default=""):
        for layer in layers:
            for name in names:
                if name in layer:
                    return layer[name]
        return default

    def text(names, default=""):
        value = setting(names, default)
        return value.strip() if isinstance(value, str) else ""

    def flag(names, default):
        value = setting(names, default)
        return value is True or isinstance(value, str) and value.lower() in ("true", "1", "yes")

    key = text(("OPENROUTER_API_KEY", "OPEN_ROUTER_API", "api_key"))
    if "\n" in key or "\r" in key:
        key = ""
    enabled = flag(("OPENROUTER_ENABLED", "enabled"), True)
    if public:
        enabled = enabled and flag(("OPENROUTER_PUBLIC_ENABLED", "public_enabled"), False)
    return LLMConfig(key, text(("OPENROUTER_MODEL", "model"), LLMConfig.primary),
                     text(("OPENROUTER_MODEL_BACKUP", "model_backup")), enabled)


class ServiceFailure(Exception):
    def __init__(self, reason):
        self.reason = reason
        super().__init__(reason)


@dataclass
class Reply:
    status: int
    body: dict
    headers: dict = field(default_factory=dict)


class HTTPTransport:
    def send(self, method, path, api_key, payload, deadline):
        remaining = deadline - time.monotonic()
        if remaining <= 0.2:
            raise ServiceFailure("time_budget")
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = "Bearer " + api_key
        response = None
        try:
            response = requests.request(method, BASE_URL + path, headers=headers, json=payload,
                timeout=(min(2.0, remaining / 4), min(8.0, remaining / 2)),
                allow_redirects=False, stream=True)
            # Do not follow redirects with a credential. Bound both bytes and time.
            chunks = bytearray()
            byte_limit = 2_000_000 if path == "/models" else 80_000
            for part in response.iter_content(chunk_size=4096):
                if time.monotonic() > deadline:
                    raise ServiceFailure("time_budget")
                chunks.extend(part)
                if len(chunks) > byte_limit:
                    raise ServiceFailure("response_too_large")
            try:
                body = json.loads(chunks, parse_constant=lambda value: (_ for _ in ()).throw(ValueError()))
            except (ValueError, UnicodeError):
                raise ServiceFailure("invalid_response") from None
            if not isinstance(body, dict):
                raise ServiceFailure("invalid_response")
            return Reply(response.status_code, body, {key.lower(): value for key, value in response.headers.items()})
        except requests.Timeout:
            raise ServiceFailure("timeout") from None
        except requests.RequestException:
            raise ServiceFailure("connection_error") from None
        finally:
            if response is not None:
                response.close()


class RouterRuntime:
    """Process-local bounded counters/cooldowns. Stores no prompts or responses.

    Quota state is deliberately advisory: restarts clear these local limits;
    OpenRouter still enforces its account-wide quotas. No paid route is used.
    """
    def __init__(self):
        self.lock = Lock()
        self.accounts = OrderedDict()
        self.catalog = None
        self.catalog_until = 0.0

    def _account(self, key):
        if key not in self.accounts:
            if len(self.accounts) >= 128:
                self.accounts.popitem(last=False)
            self.accounts[key] = {"quota_until": 0.0, "remaining": None, "blocked_until": 0.0,
                                  "cooldowns": {}, "daily": {}, "sessions": OrderedDict()}
        self.accounts.move_to_end(key)
        return self.accounts[key]

    def blocked(self, key):
        with self.lock:
            return self._account(key)["blocked_until"] > time.time()

    def block(self, key, seconds):
        with self.lock:
            account = self._account(key)
            account["blocked_until"] = max(account["blocked_until"], time.time() + seconds)

    def cooling(self, key, model):
        with self.lock:
            return self._account(key)["cooldowns"].get(model, 0) > time.time()

    def cooldown(self, key, model, seconds):
        with self.lock:
            cooldowns = self._account(key)["cooldowns"]
            if len(cooldowns) >= 16:
                cooldowns.clear()
            cooldowns[model] = time.time() + min(max(seconds, 5), 3600)

    def get_quota(self, key):
        with self.lock:
            account = self._account(key)
            return account["quota_until"] > time.time(), account["remaining"]

    def set_quota(self, key, remaining):
        with self.lock:
            account = self._account(key)
            account.update(remaining=remaining, quota_until=time.time() + 30)

    def reserve(self, key, session_id):
        with self.lock:
            account = self._account(key)
            day = datetime.now(timezone.utc).date().isoformat()
            account["daily"] = {day: account["daily"].get(day, 0)}
            sessions = account["sessions"]
            if session_id not in sessions:
                if len(sessions) >= 128:
                    sessions.popitem(last=False)
                sessions[session_id] = {"day": day, "count": 0, "recent": deque()}
            visitor = sessions[session_id]
            if visitor["day"] != day:
                visitor.update(day=day, count=0)
            now = time.time()
            while visitor["recent"] and visitor["recent"][0] <= now - 60:
                visitor["recent"].popleft()
            if account["daily"][day] >= 30 or visitor["count"] >= 10 or len(visitor["recent"]) >= 3:
                return False
            if account["remaining"] is not None and account["remaining"] <= 0:
                return False
            account["daily"][day] += 1
            visitor["count"] += 1
            visitor["recent"].append(now)
            if account["remaining"] is not None:
                account["remaining"] -= 1
            return True


def _retry_after(headers, default=60):
    value = headers.get("retry-after", "")
    try:
        return min(max(float(value), 5), 3600)
    except (ValueError, TypeError):
        try:
            return min(max(parsedate_to_datetime(value).timestamp() - time.time(), 5), 3600)
        except (ValueError, TypeError, OverflowError):
            return default


def response_schema(snapshot):
    data = snapshot.payload
    return {"type": "object", "properties": {
        "snapshot_id": {"type": "string", "enum": [snapshot.identity]},
        "observations": {"type": "array", "minItems": 1, "maxItems": 4, "items": {
            "type": "object", "properties": {
                "evidence_id": {"type": "string", "enum": [fact["id"] for fact in data["facts"]]},
                "text": {"type": "string"}}, "required": ["evidence_id", "text"], "additionalProperties": False}},
        "interpretation": {"type": "string"},
        "limitations": {"type": "array", "items": {"type": "string", "enum": data["limitations"]}},
        "review_checks": {"type": "array", "items": {"type": "string", "enum": data["review_checks"]}},
    }, "required": ["snapshot_id", "observations", "interpretation", "limitations", "review_checks"], "additionalProperties": False}


def _text_valid(text):
    if not isinstance(text, str) or not 8 <= len(text.strip()) <= 1000:
        return False
    # Facts/units are rendered by Python. Generated prose cannot invent numbers,
    # URLs, code, regulatory claims or operational instructions.
    if any(char.isnumeric() for char in text) or re.search(r"https?://|www\.|[<>{}`]|!\[|\]\(", text, re.I):
        return False
    forbidden = (r"\b(manufactur\w*|gmp|potency|purity|certif\w*|guarantee\w*|proves?|conclusively|"
                 r"accurate|reliable|retrain\w*|setpoints?|untested)\b|safe operating|acceptable boundary|"
                 r"(?:increase|decrease|adjust|change)\s+(?:the\s+)?(?:feed|temperature|agitation|ph|dose)|"
                 r"(?:no|not|never).{0,35}(?:test.{0,15}(?:performed|conducted)|data.{0,10}exists)|"
                 r"independently validated|validated accuracy|rmse.{0,40}(?:mean|average) absolute")
    return not re.search(forbidden, text, re.I)


def validate_explanation(content, snapshot):
    try:
        data = json.loads(content)
    except (ValueError, TypeError):
        raise ServiceFailure("invalid_explanation") from None
    expected = {"snapshot_id", "observations", "interpretation", "limitations", "review_checks"}
    if not isinstance(data, dict) or set(data) != expected or data["snapshot_id"] != snapshot.identity:
        raise ServiceFailure("invalid_explanation")
    observations = data["observations"]
    ids = {fact["id"] for fact in snapshot.payload["facts"]}
    if not isinstance(observations, list) or not 1 <= len(observations) <= 4:
        raise ServiceFailure("invalid_explanation")
    seen = set()
    for item in observations:
        if not isinstance(item, dict) or set(item) != {"evidence_id", "text"} or not isinstance(item["evidence_id"], str):
            raise ServiceFailure("invalid_explanation")
        if item["evidence_id"] not in ids or item["evidence_id"] in seen or not _text_valid(item["text"]):
            raise ServiceFailure("invalid_explanation")
        seen.add(item["evidence_id"])
    if not _text_valid(data["interpretation"]):
        raise ServiceFailure("invalid_explanation")
    for field_name, allowed, complete in (("limitations", snapshot.payload["limitations"], True),
                                          ("review_checks", snapshot.payload["review_checks"], False)):
        values = data[field_name]
        if not isinstance(values, list) or not values or any(not isinstance(value, str) for value in values):
            raise ServiceFailure("invalid_explanation")
        if len(set(values)) != len(values) or not set(values).issubset(allowed) or complete and set(values) != set(allowed):
            raise ServiceFailure("invalid_explanation")
    return {name: data[name] for name in ("observations", "interpretation", "limitations", "review_checks")}


def request_payload(model, info, snapshot, style):
    glossary = ("RMSE is root mean squared prediction error, not mean absolute error. MAE is mean absolute error. "
        "Bias is mean prediction minus reference. R-squared can be negative or unavailable. "
        "CV was used for selection, so it can be optimistic. Missing metrics mean not provided, not that testing never occurred. "
        "A distribution shift is descriptive and does not establish a root cause or model failure.")
    page = snapshot.payload["page"]
    if page == "Spectral Analytics":
        glossary = ("Explain spectral exploration only. SNV/MSC can remove concentration-related amplitude. "
            "Derivatives suppress smooth backgrounds but can amplify noise. Raw-minus-SG means raw spectrum minus its SG-smoothed spectrum, "
            "not minus the AsLS baseline; that residual may include real peaks. "
            "FFT frequencies are along Raman shift, not temporal Hz. Wavelets describe localized scales; thresholding can remove peaks. "
            "Exploratory PCA reference variance and reconstruction error do not measure concentration prediction accuracy or calibrated alarms. "
            "Do not assign molecules or describe a predictive model or validation results.")
    elif page == "Process Monitoring":
        glossary = ("Explain the captured historical monitoring view only, never claim it is current live process status. "
            "Advisory bounds and input plausibility/domain/freshness checks are different. "
            "An unavailable or ineligible estimate stays unavailable even if a historical estimate exists. "
            "Assay results are references available at capture, not model inputs. Never infer a process fault or a safe operating state.")
    elif page == "Multivariate Monitoring":
        glossary = ("Explain reference-fitted process PCA monitoring. T-squared is distance in retained standardized PCA coordinates. "
            "SPE/Q is the squared residual outside the retained subspace. Use the words T-squared in prose. "
            "Discuss both statistics together when provided; an in-limit T-squared never overrules an SPE exceedance. "
            "Fitting, limit-calibration and evaluation batches have separate roles. Empirical per-chart quantiles do not guarantee "
            "joint false-alarm probabilities. Unscorable observations are unknown, not normal. Residual contributions are descriptive, not causal. "
            "The inspected observation is selected separately from the chart batch scope. Do not describe concentration predictions or predictive validation.")
    instruction = (f"You explain educational PAT results. Style: {style}. Use only the supplied allowlisted facts. {glossary} "
        "Choose up to four useful observations and attach each to its evidence_id. Explain what the fact means, not its numeric value. "
        "IMPORTANT: All exact values and units are rendered separately by Python. Use NO numerals, numeric words, percentages, "
        "model IDs, evidence IDs, URLs, markdown, code, or equations in text/interpretation. Say R-squared, not R2. "
        "Never fill a missing prediction, invent a metric, claim accuracy or root cause, discuss manufacturing suitability, "
        "or recommend process adjustments/retraining. A chart limit is an empirical review threshold, not process acceptability. "
        "Explain only selected facts; do not recite the glossary or restate limitations. Avoid the words prove, guarantee, "
        "accurate, reliable and manufacturing even in negations; fixed limitations are displayed by the application. "
        "Use at most a short sentence per observation, under two hundred characters. "
        "Keep interpretation to at most three short sentences and under six hundred characters. "
        "Return the exact snapshot_id, ALL supplied limitation IDs, and one to three supplied review_check IDs. "
        "The application renders those statements; do not invent extra limitations or checks.")
    payload = {"model": model, "messages": [{"role": "system", "content": instruction},
        {"role": "user", "content": snapshot.serialized}], "temperature": 0.1, "max_tokens": 1200,
        "provider": {"require_parameters": True, "max_price": {"prompt": 0, "completion": 0}},
        "response_format": {"type": "json_schema", "json_schema": {"name": "pat_explanation", "strict": True,
            "schema": response_schema(snapshot)}}}
    if "reasoning" in info.get("supported_parameters", []) and not (info.get("reasoning") or {}).get("mandatory", False):
        payload["reasoning"] = {"enabled": False}
    return payload


def local_result(snapshot, reason="offline_requested", attempts=None):
    return {"source": "template", "model": None, "snapshot_id": snapshot.identity,
            "prompt_version": PROMPT_VERSION, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "reason": reason, "attempts": attempts or [], "reported_cost": None,
            "sections": template_sections(snapshot)}


class ExplanationService:
    def __init__(self, runtime=None, transport=None):
        self.runtime = runtime or RouterRuntime()
        self.transport = transport or HTTPTransport()

    def _quota(self, config, key, deadline, *, force=False):
        fresh, remaining = self.runtime.get_quota(key)
        if fresh and not force:
            return remaining
        reply = self.transport.send("GET", "/key", config.api_key, None, deadline)
        if reply.status != 200 or "error" in reply.body:
            self.runtime.block(key, _retry_after(reply.headers) if reply.status == 429 else 60)
            raise ServiceFailure("account_unavailable")
        data = reply.body.get("data")
        if not isinstance(data, dict):
            raise ServiceFailure("invalid_response")
        quota = data.get("free_model_daily_requests") or {}
        if not isinstance(quota, dict):
            raise ServiceFailure("invalid_response")
        remaining = quota.get("remaining")
        if remaining is not None and (isinstance(remaining, bool) or not isinstance(remaining, int) or remaining < 0):
            raise ServiceFailure("invalid_response")
        self.runtime.set_quota(key, remaining)
        if remaining == 0:
            now = datetime.now(timezone.utc)
            tomorrow = (now.replace(hour=0, minute=0, second=0, microsecond=0).timestamp() + 86400)
            self.runtime.block(key, max(1, tomorrow - now.timestamp()))
        return remaining

    def _catalog(self, deadline):
        with self.runtime.lock:
            if self.runtime.catalog is not None and self.runtime.catalog_until > time.time():
                return self.runtime.catalog
        reply = self.transport.send("GET", "/models", "", None, deadline)
        if reply.status != 200 or not isinstance(reply.body.get("data"), list):
            raise ServiceFailure("catalog_unavailable")
        catalog = {item["id"]: item for item in reply.body["data"] if isinstance(item, dict) and isinstance(item.get("id"), str)}
        with self.runtime.lock:
            self.runtime.catalog, self.runtime.catalog_until = catalog, time.time() + 300
        return catalog

    def generate(self, snapshot, config, *, style=STYLES[0], session_id="local", consent=False):
        if style not in STYLES:
            raise ValueError("Unsupported explanation style.")
        if snapshot.requires_consent and not consent:
            return local_result(snapshot, "consent_required")
        if not config.enabled or not config.api_key:
            return local_result(snapshot, "disabled" if not config.enabled else "missing_key")
        key = sha256(config.api_key.encode()).hexdigest()
        if self.runtime.blocked(key):
            return local_result(snapshot, "account_cooldown")
        candidates, seen = [], set()
        for stage, model in (("primary", config.primary), ("backup", config.backup)):
            if isinstance(model, str) and FREE_MODEL.fullmatch(model) and model not in seen:
                candidates.append((stage, model))
                seen.add(model)
        if not candidates:
            return local_result(snapshot, "no_free_model")
        deadline = time.monotonic() + 25
        attempts = []
        try:
            if self._quota(config, key, deadline) == 0:
                return local_result(snapshot, "shared_quota_exhausted")
            catalog = self._catalog(deadline)
        except ServiceFailure as error:
            return local_result(snapshot, error.reason)
        for stage, model in candidates:
            reason = "model_unavailable"
            try:
                if self.runtime.cooling(key, model):
                    raise ServiceFailure("model_cooldown")
                info = catalog.get(model)
                if not info:
                    raise ServiceFailure("model_unavailable")
                pricing = info.get("pricing") or {}
                if any(Decimal(str(pricing.get(name, "-1"))) != 0 for name in ("prompt", "completion")) or Decimal(str(pricing.get("request", "0"))) != 0:
                    raise ServiceFailure("model_not_free")
                if not {"structured_outputs", "response_format"}.issubset(info.get("supported_parameters", [])):
                    raise ServiceFailure("structured_output_unavailable")
                if not self.runtime.reserve(key, session_id):
                    return local_result(snapshot, "local_usage_limit", attempts)
                reply = self.transport.send("POST", "/chat/completions", config.api_key,
                    request_payload(model, info, snapshot, style), deadline)
                embedded = reply.body.get("error")
                if isinstance(embedded, dict) and str(embedded.get("code")) in ("400", "401", "402", "403", "429", "500", "502", "503"):
                    reply = Reply(int(embedded["code"]), reply.body, reply.headers)
                if reply.status in (401, 402, 403):
                    self.runtime.block(key, 60)
                    attempts.append({"stage": stage, "model": model, "reason": "account_restricted"})
                    return local_result(snapshot, "account_restricted", attempts)
                if reply.status == 429:
                    # Refresh the shared counter before spending an attempt on a
                    # second model. Platform headers indicate a shared rate cap.
                    remaining = self._quota(config, key, deadline, force=True)
                    if remaining == 0 or "x-ratelimit-limit" in reply.headers:
                        self.runtime.block(key, _retry_after(reply.headers))
                        attempts.append({"stage": stage, "model": model, "reason": "shared_rate_limit"})
                        return local_result(snapshot, "shared_rate_limit", attempts)
                    self.runtime.cooldown(key, model, _retry_after(reply.headers))
                    raise ServiceFailure("model_rate_limit")
                if reply.status != 200 or "error" in reply.body:
                    raise ServiceFailure("model_unavailable")
                if reply.body.get("model") != model:
                    raise ServiceFailure("unexpected_model")
                choice = reply.body.get("choices", [{}])[0]
                if choice.get("finish_reason") == "content_filter" or choice.get("message", {}).get("refusal"):
                    return local_result(snapshot, "response_refused", attempts)
                if choice.get("finish_reason") != "stop":
                    raise ServiceFailure("incomplete_response")
                sections = validate_explanation(choice.get("message", {}).get("content"), snapshot)
                cost = reply.body.get("usage", {}).get("cost")
                if cost is not None and Decimal(str(cost)) != 0:
                    raise ServiceFailure("unexpected_cost")
                attempts.append({"stage": stage, "model": model, "reason": "success"})
                return {"source": stage, "model": model, "snapshot_id": snapshot.identity,
                        "prompt_version": PROMPT_VERSION, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                        "reason": "success", "attempts": attempts, "reported_cost": cost, "sections": sections}
            except (ServiceFailure, InvalidOperation, ValueError, TypeError, KeyError, IndexError, AttributeError) as error:
                reason = error.reason if isinstance(error, ServiceFailure) else "invalid_response"
                attempts.append({"stage": stage, "model": model, "reason": reason})
                if reason not in ("model_cooldown", "model_rate_limit"):
                    self.runtime.cooldown(key, model, 60)
                if self.runtime.blocked(key) or time.monotonic() >= deadline:
                    break
        return local_result(snapshot, attempts[-1]["reason"] if attempts else "models_unavailable", attempts)
