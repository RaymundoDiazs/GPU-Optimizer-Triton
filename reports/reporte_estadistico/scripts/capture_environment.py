from __future__ import annotations

import json
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "reports" / "reporte_estadistico" / "results" / "environment_snapshot.json"


PACKAGES = [
    "torch",
    "triton",
    "transformers",
    "xgrammar",
    "pandas",
    "matplotlib",
    "numpy",
    "pytest",
]


def _package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not_installed"


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def _gpu_info() -> dict:
    try:
        import torch
    except ImportError:
        return {"torch_available": False}

    info = {
        "torch_available": True,
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_cuda_version": getattr(torch.version, "cuda", None),
    }
    if torch.cuda.is_available():
        info.update(
            {
                "gpu_count": torch.cuda.device_count(),
                "gpu_name_0": torch.cuda.get_device_name(0),
            }
        )
    return info


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    snapshot = {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "git_commit": _git_commit(),
        "packages": {package: _package_version(package) for package in PACKAGES},
        "gpu": _gpu_info(),
    }
    OUTPUT.write_text(json.dumps(snapshot, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Snapshot escrito en {OUTPUT}")


if __name__ == "__main__":
    main()
