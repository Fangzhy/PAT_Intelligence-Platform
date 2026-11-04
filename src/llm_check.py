"""Explicit synthetic-only live check: python -m src.llm_check --live.

Never invoked by application rendering or CI. Does not modify .env or studies.
"""

import argparse
from dataclasses import replace
import json
from time import perf_counter

from src.llm_client import ExplanationService, load_config
from src.llm_results import demo_snapshot


def main():
    parser = argparse.ArgumentParser(description="Check optional OpenRouter configuration without exposing credentials.")
    parser.add_argument("--live", action="store_true", help="Send a small synthetic-only request to each configured free model.")
    parser.add_argument("--page", choices=("reliability", "spectral", "process", "multivariate"), default="reliability",
                        help="Choose the built-in synthetic explanation scope; no model fitting is performed.")
    args = parser.parse_args()
    config = load_config()
    if not args.live:
        print(json.dumps({"key_configured": bool(config.api_key), "enabled": config.enabled,
                          "live_requests_sent": False}))
        return
    passed = True
    page = {"reliability": "Model Reliability", "spectral": "Spectral Analytics",
            "process": "Process Monitoring", "multivariate": "Multivariate Monitoring"}[args.page]
    for role, model in (("primary", config.primary), ("backup", config.backup)):
        started = perf_counter()
        result = ExplanationService().generate(demo_snapshot(page), replace(config, primary=model, backup=""), session_id="explicit-live-check")
        ok = result["source"] == "primary"
        passed = passed and ok
        print(json.dumps({"page": page, "configured_role": role, "status": "passed" if ok else "failed",
            "resolved_model": result["model"], "reason": result["reason"], "reported_cost": result["reported_cost"],
            "latency_seconds": round(perf_counter() - started, 2)}))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
