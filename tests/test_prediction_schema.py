import json

from evaluation.prediction_schema import (
    SCHEMA_VERSION,
    load_prediction_jsonl,
    make_prediction_record,
)


def test_legacy_prediction_is_normalized(tmp_path):
    path = tmp_path / "predictions_qwen.jsonl"
    path.write_text(
        json.dumps(
            {
                "instruction": "Generate a Triton wrapper.",
                "predict": "import triton\n@triton.jit\ndef kernel():\n    pass\n",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    record = load_prediction_jsonl(path)[0]

    assert record["schema_version"] == SCHEMA_VERSION
    assert record["model"]["id"] == "small_qwen25_coder_1_5b"
    assert record["task"]["id"] == "tritonbench_t_001"
    assert record["source_index"] == 1
    assert record["output"].startswith("import triton")


def test_canonical_prediction_round_trips(tmp_path):
    path = tmp_path / "predictions.jsonl"
    original = make_prediction_record(
        model={"id": "model", "provider": "local"},
        task={"id": "vector_add"},
        output="x = 1",
        mode="baseline",
    )
    path.write_text(json.dumps(original) + "\n", encoding="utf-8")

    loaded = load_prediction_jsonl(path)[0]

    assert loaded["record_id"] == original["record_id"]
    assert loaded["task"]["id"] == "vector_add"
    assert loaded["output"] == "x = 1"
