from pathlib import Path
from typing import Any

import yaml


DEFAULT_SETTINGS_PATH = Path(__file__).resolve().parents[1] / "config" / "settings.yaml"


def _load_settings(settings_path: Path = DEFAULT_SETTINGS_PATH) -> dict[str, Any]:
    if not settings_path.exists():
        return {}

    with settings_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def decide_strategy(code: str, ast_repr: dict, grammar: str) -> dict[str, Any]:
    """Choose a deterministic optimization strategy for the current input."""
    settings = _load_settings()
    optimization_level = settings.get("generation", {}).get("optimization_level", "basic")
    normalized_code = code.lower()

    if "sum" in normalized_code or "reduce" in normalized_code:
        strategy = "reduce_specialized"
        confidence = 0.9
        explanation = "Detected a reduction-like operation in the input code."
    elif optimization_level == "advanced" and any(token in normalized_code for token in ("+", "-", "*", "/")):
        strategy = "fusion"
        confidence = 0.75
        explanation = "Advanced optimization mode enables element-wise fusion heuristics."
    else:
        strategy = "simple"
        confidence = 0.8
        explanation = f"Using deterministic fallback for optimization level '{optimization_level}'."

    # TODO: replace with a real LLM call while keeping this public interface stable.
    return {
        "strategy": strategy,
        "confidence": confidence,
        "explanation": explanation,
    }
