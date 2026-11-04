"""Recover old LLM module exports retained by a long-running Streamlit server.

Only already-loaded explanation modules are refreshed, in dependency order.
Ordinary reruns do nothing. Scientific/modeling modules, fitted artifacts, and
session state are not reloaded or cleared. This guard performs no API calls.
"""

import importlib
import sys
from threading import RLock


_LOCK = RLock()
_EXPORTS = {
    "src.llm_results": ("PROMPT_VERSION", "ResultSnapshot", "spectral_snapshot", "process_snapshot", "multivariate_snapshot"),
    "src.llm_client": ("PROMPT_VERSION", "ExplanationService", "RouterRuntime", "load_config"),
    "src.llm_ui": ("PROMPT_VERSION", "render_explanation", "render_captured_explanation"),
}


def _outdated():
    versions = set()
    for name, exports in _EXPORTS.items():
        module = sys.modules.get(name)
        if module is None:
            continue
        if any(not hasattr(module, export) for export in exports):
            return True
        versions.add(module.PROMPT_VERSION)
    return len(versions) > 1


def refresh_legacy_llm_modules():
    """Return True only when cached exports/protocols needed an upgrade.

    Streamlit can rerun a changed page while keeping an earlier imported module
    in sys.modules. Refresh dependents as well: their from-imports otherwise
    retain old prompt versions, limitation maps, and snapshot classes.
    """
    with _LOCK:
        if not _outdated():
            return False
        importlib.invalidate_caches()
        for name in _EXPORTS:
            module = sys.modules.get(name)
            if module is not None:
                importlib.reload(module)
        if _outdated():
            raise RuntimeError("The explanation modules are incomplete. Synchronize the application files and restart Streamlit.")
        return True
