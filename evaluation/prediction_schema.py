from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = 1
ROOT = Path(__file__).resolve().parents[1]
TRITONBENCH_DATASET = ROOT / "data" / "tritonbench_t_simp_subset166.json"

PROVIDER_MODELS = {
    "qwen": {
        "id": "small_qwen25_coder_1_5b",
        "display_name": "Qwen2.5-Coder-1.5B (Ollama local)",
        "tier": "small",
        "provider": "ollama",
    },
    "gpt4o": {
        "id": "frontier_openai",
        "display_name": "GPT-4o (OpenAI)",
        "tier": "frontier",
        "provider": "openai",
    },
    "claude": {
        "id": "frontier_anthropic",
        "display_name": "Claude Haiku 4.5 (Anthropic)",
        "tier": "frontier",
        "provider": "anthropic",
    },
}


def load_tritonbench_dataset() -> list[dict[str, Any]]:
    with TRITONBENCH_DATASET.open("r", encoding="utf-8") as file:
        return json.load(file)


def infer_provider_key(path: Path | None) -> str:
    name = path.name.lower() if path is not None else ""
    for key in PROVIDER_MODELS:
        if key in name:
            return key
    return "unknown"


def _record_id(model_id: str, task_id: str, mode: str, sample_index: int) -> str:
    raw = f"{model_id}|{task_id}|{mode}|{sample_index}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:20]


def make_prediction_record(
    *,
    model: dict[str, Any],
    task: dict[str, Any],
    output: str,
    mode: str,
    sample_index: int = 1,
    prompt: str = "",
    latency_seconds: float = 0.0,
    constrained_decoding_backend: str = "",
    source_index: int | None = None,
) -> dict[str, Any]:
    model_id = str(model.get("id", "unknown"))
    task_id = str(task.get("id", "unknown"))
    record = {
        "schema_version": SCHEMA_VERSION,
        "record_id": _record_id(model_id, task_id, mode, sample_index),
        "model": {
            "id": model_id,
            "display_name": str(model.get("display_name", model_id)),
            "tier": str(model.get("tier", "unknown")),
            "provider": str(model.get("provider", "unknown")),
        },
        "task": dict(task),
        "mode": mode,
        "sample_index": int(sample_index),
        "prompt": prompt,
        "output": output,
        "latency_seconds": round(float(latency_seconds), 6),
        "constrained_decoding_backend": constrained_decoding_backend,
    }
    if source_index is not None:
        record["source_index"] = int(source_index)
    return record


def normalize_prediction_record(
    record: dict[str, Any],
    *,
    source_path: Path | None = None,
    source_index: int | None = None,
    dataset: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if "model" in record and "task" in record and ("output" in record or "extracted_code" in record):
        output = str(record.get("output", record.get("extracted_code", "")))
        normalized = make_prediction_record(
            model=record["model"],
            task=record["task"],
            output=output,
            mode=str(record.get("mode", "baseline")),
            sample_index=int(record.get("sample_index", 1)),
            prompt=str(record.get("prompt", "")),
            latency_seconds=float(record.get("latency_seconds", 0.0) or 0.0),
            constrained_decoding_backend=str(record.get("constrained_decoding_backend", "")),
            source_index=record.get("source_index", source_index),
        )
        normalized.update(
            {
                key: value
                for key, value in record.items()
                if key not in normalized and key not in {"predict", "instruction"}
            }
        )
        return normalized

    if "instruction" not in record or "predict" not in record:
        raise ValueError("prediction record must use the canonical schema or contain instruction/predict")

    if source_index is None:
        raise ValueError("source_index is required for legacy instruction/predict records")

    rows = dataset if dataset is not None else load_tritonbench_dataset()
    if source_index < 1 or source_index > len(rows):
        raise ValueError(f"source_index {source_index} is outside the TritonBench dataset")

    dataset_row = rows[source_index - 1]
    provider_key = infer_provider_key(source_path)
    model = PROVIDER_MODELS.get(
        provider_key,
        {
            "id": provider_key,
            "display_name": provider_key,
            "tier": "unknown",
            "provider": provider_key,
        },
    )
    task = {
        "id": f"tritonbench_t_{source_index:03d}",
        "benchmark": "tritonbench_t",
        "instruction": str(record["instruction"]),
        "expected_output": dataset_row.get("output", ""),
    }
    return make_prediction_record(
        model=model,
        task=task,
        output=str(record["predict"]),
        mode=str(record.get("mode", "baseline")),
        sample_index=int(record.get("sample_index", 1)),
        prompt=str(record["instruction"]),
        latency_seconds=float(record.get("latency_seconds", 0.0) or 0.0),
        constrained_decoding_backend=str(record.get("constrained_decoding_backend", "")),
        source_index=source_index,
    )


def load_prediction_jsonl(path: Path) -> list[dict[str, Any]]:
    dataset = load_tritonbench_dataset()
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for source_index, line in enumerate(file, start=1):
            if not line.strip():
                continue
            rows.append(
                normalize_prediction_record(
                    json.loads(line),
                    source_path=path,
                    source_index=source_index,
                    dataset=dataset,
                )
            )
    return rows


def write_prediction_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
