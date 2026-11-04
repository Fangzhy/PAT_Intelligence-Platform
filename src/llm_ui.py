"""Manual, session-private explanations on already computed page results."""

from hashlib import sha256
import json
from uuid import uuid4

import pandas as pd
import streamlit as st

from src.llm_client import ExplanationService, RouterRuntime, STYLES, load_config, local_result
from src.llm_results import CHECKS, LIMITATIONS, PROMPT_VERSION, template_text
from src.runtime import public_demo


@st.cache_resource(show_spinner=False)
def shared_router_runtime():
    # Only counters and model availability are shared. Text/results stay in
    # this visitor's session_state, never a global Streamlit data cache.
    return RouterRuntime()


def _configuration():
    try:
        secrets = dict(st.secrets.get("llm", {}))
    except FileNotFoundError:
        secrets = {}
    return load_config(secrets=secrets, public=public_demo())


def render_explanation(snapshot, *, key, empty_message="Prepare a saved study first. Explanations use recorded results and never train or evaluate a model.", provenance=None):
    with st.expander("Explain these results · AI or local summary", expanded=False):
        if snapshot is None:
            st.caption(empty_message)
            return
        config = _configuration()
        context_key = f"ai_{key}_context"
        consent_key = f"ai_{key}_consent"
        if st.session_state.get(context_key) != snapshot.identity:
            st.session_state.pop(consent_key, None)
            st.session_state[context_key] = snapshot.identity
        st.caption("The exact facts below come from Python. AI provides optional educational interpretation; check it against those facts. It does not change predictions, alarms, model selection, or frozen evidence.")
        with st.expander("Preview the aggregate data sent to OpenRouter"):
            st.json(snapshot.payload)
        style = st.radio("Explanation style", STYLES, key=f"ai_{key}_style", horizontal=True)
        consent = not snapshot.requires_consent
        if snapshot.requires_consent:
            consent = st.checkbox("I approve sending this prepared-data summary to OpenRouter and its model provider.", key=consent_key)
        else:
            st.caption("Clicking Explain sends this synthetic-results summary to OpenRouter and its provider. Raw rows, batch/sample IDs, and credentials are excluded from the prompt.")
        if not config.enabled:
            st.info("External AI is disabled for this deployment. A local template summary is available.")
        elif not config.api_key:
            st.info("OpenRouter is not configured. A local template summary is available.")
        signature = sha256((snapshot.identity + style + config.identity + PROMPT_VERSION + str(consent)).encode()).hexdigest()
        cache_key = f"ai_{key}_cache"
        cache = st.session_state.setdefault(cache_key, {})
        requested = st.button("Explain these results", key=f"ai_{key}_explain", disabled=not consent)
        offline = st.button("Use local template summary", key=f"ai_{key}_template")
        if offline:
            cache[signature] = local_result(snapshot)
        elif requested and (signature not in cache or cache[signature]["source"] == "template"):
            session_id = st.session_state.setdefault("ai_visitor_id", uuid4().hex)
            with st.spinner("Trying the configured free models; a local summary is used if unavailable…"):
                cache[signature] = ExplanationService(shared_router_runtime()).generate(
                    snapshot, config, style=style, session_id=session_id, consent=consent)
        while len(cache) > 4:
            del cache[next(iter(cache))]
        if signature not in cache:
            if cache:
                st.caption("Results or explanation settings changed. Generate a summary for the current snapshot.")
            return
        result = cache[signature]
        source = {"primary": "Primary LLM", "backup": "Backup LLM", "template": "Template summary (offline)"}[result["source"]]
        st.caption(f"{source} · {result['model'] or 'Local Python template'} · {result['generated_at_utc']}")
        if result["source"] == "template":
            st.caption(f"Summary reason: {result['reason']}. No AI interpretation is claimed.")
        for fact in snapshot.payload["facts"]:
            value = "Unavailable / not assessed" if fact["value"] is None else f"{fact['value']:.6g}"
            st.write(f"{fact['id']} · {fact['label']}: {value} {fact['unit']}")
        sections = result["sections"]
        for item in sections["observations"]:
            st.caption(f"Explanation of {item['evidence_id']}")
            st.text(item["text"])
        st.text(sections["interpretation"])
        for name in sections["limitations"]:
            st.caption(LIMITATIONS[name])
        for name in sections["review_checks"]:
            st.write(CHECKS[name])
        report = {"snapshot": snapshot.payload, "explanation": result}
        if provenance is not None:
            report["capture"] = provenance
        if result["source"] == "template":
            report["template_text"] = template_text(snapshot)
        st.download_button("Download explanation and provenance (JSON)", json.dumps(report, indent=2, allow_nan=False),
                           file_name="pat_result_explanation.json", mime="application/json", key=f"ai_{key}_download")


def render_captured_explanation(snapshot, *, key, context_id, cutoff):
    """Pin a manual monitoring snapshot; never generate from timer reruns.

    Forward progress may keep an explicitly historical capture. Context changes,
    reset/empty history and backward review clear it before it can be reused.
    """
    state_key = f"ai_{key}_capture"
    saved = st.session_state.get(state_key)
    cutoff = pd.Timestamp(cutoff)
    if saved is not None and (snapshot is None or saved["context"] != context_id or cutoff < pd.Timestamp(saved["cutoff"])):
        st.session_state.pop(state_key, None)
        saved = None
    st.caption("Optional explanation: capture this monitoring view, then explain that fixed historical snapshot. Polling never requests an explanation automatically.")
    if st.button("Capture visible view for explanation", key=f"ai_{key}_capture_button", disabled=snapshot is None):
        saved = {"snapshot": snapshot, "context": context_id, "cutoff": cutoff.isoformat(),
                 "captured_at_utc": pd.Timestamp.now(tz="UTC").isoformat()}
        st.session_state[state_key] = saved
    if saved is not None:
        st.caption(f"Captured historical snapshot · UTC {saved['captured_at_utc']} · Displayed data cutoff {saved['cutoff']}")
        comparable = lambda item: {name: value for name, value in item.payload.items() if name != "snapshot_id"}
        if comparable(saved["snapshot"]) != comparable(snapshot):
            st.warning("Monitoring has changed since capture. This explanation concerns the captured historical snapshot only. Capture again to update it.")
    render_explanation(saved["snapshot"] if saved is not None else None, key=key,
                       empty_message="Reveal/receive observations and capture the visible view to enable an explanation. Polling never automatically calls an LLM.",
                       provenance={"captured_at_utc": saved["captured_at_utc"], "data_cutoff": saved["cutoff"],
                                   "scope": "captured historical monitoring snapshot"} if saved else None)
