from __future__ import annotations

import importlib.util
from typing import Iterable


REQUIRED_PACKAGES = ("transformers", "xgrammar")
BACKEND_PACKAGES = ("torch", "accelerate")


def _is_module_available(name: str) -> bool:
    spec = importlib.util.find_spec(name)
    return spec is not None


def verify_xgrammar_environment() -> None:
    """Verify the local environment required for Qwen + XGrammar constrained decoding."""
    missing = [pkg for pkg in REQUIRED_PACKAGES if not _is_module_available(pkg)]
    if missing:
        raise ImportError(
            "Missing required packages for XGrammar constrained decoding: "
            + ", ".join(missing)
            + ". Install them with `pip install -r requirements.txt` or `pip install transformers xgrammar`."
        )

    backend_missing = [pkg for pkg in BACKEND_PACKAGES if not _is_module_available(pkg)]
    if len(backend_missing) == len(BACKEND_PACKAGES):
        raise ImportError(
            "XGrammar constrained decoding requires a local PyTorch or Accelerate backend. "
            "Install `torch` or `accelerate` to run the Qwen HF model locally."
        )


def get_missing_xgrammar_dependencies() -> list[str]:
    """Return the list of missing packages required for XGrammar constrained decoding."""
    missing = [pkg for pkg in REQUIRED_PACKAGES if not _is_module_available(pkg)]
    if not any(_is_module_available(pkg) for pkg in BACKEND_PACKAGES):
        missing.append("torch or accelerate")
    return missing
