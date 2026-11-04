"""Read-only deployment preflight: python -m src.deployment_check [--json]."""

import argparse
from importlib import metadata
import json
from pathlib import Path
import platform

from packaging.requirements import Requirement


ROOT = Path(__file__).resolve().parents[1]


def deployment_preflight(root=ROOT):
    root = Path(root)
    python_ok = platform.python_version_tuple()[:2] == ("3", "11")
    dependencies = []
    for line in (root / "requirements.txt").read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        requirement = Requirement(text)
        if requirement.marker is not None and not requirement.marker.evaluate():
            continue
        pins = [specifier.version for specifier in requirement.specifier if specifier.operator == "=="]
        try:
            actual = metadata.version(requirement.name)
        except metadata.PackageNotFoundError:
            actual = None
        dependencies.append({"package": requirement.name, "expected": pins[0] if len(pins) == 1 else "exact pin required",
            "installed": actual, "ok": len(pins) == 1 and actual == pins[0]})
    required = ["app.py", ".python-version", ".streamlit/config.toml", "LICENSE", "deployment/README.md",
        *[f"pages/{name}" for name in ("1_Overview.py", "2_Data_Integration.py", "3_Spectral_Analytics.py", "4_Predictive_Modeling.py",
        "5_Process_Monitoring.py", "6_Multivariate_Monitoring.py", "7_Model_Reliability.py", "8_Model_Deployment.py", "9_Test_Acquisition.py")]]
    files = {name: (root / name).is_file() for name in required}
    return {"ready": python_ok and bool(dependencies) and all(row["ok"] for row in dependencies) and all(files.values()),
        "python": {"expected": "3.11", "actual": platform.python_version(), "ok": python_ok},
        "system": platform.system(), "dependencies": dependencies, "files": files,
        "scope": "Local runtime/files only; does not certify a hosted URL, Linux execution, or model manufacturing validation."}


def main():
    parser = argparse.ArgumentParser(description="Check pinned runtime and deployment files without training or publishing.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = deployment_preflight()
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"Deployment preflight: {'PASS' if result['ready'] else 'FAIL'}")
        print(f"Python {result['python']['actual']} · {result['system']} · {len(result['dependencies'])} applicable pins")
        for row in result["dependencies"]:
            if not row["ok"]:
                print(f"Dependency mismatch: {row['package']} expected {row['expected']}, installed {row['installed']}")
        for name, exists in result["files"].items():
            if not exists:
                print(f"Missing deployment file: {name}")
    raise SystemExit(0 if result["ready"] else 1)


if __name__ == "__main__":
    main()
