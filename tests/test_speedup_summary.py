import json

from benchmarks.compare_launch_policies import _summarize


def test_speedup_summary_only_uses_correct_kernels(tmp_path):
    path = tmp_path / "results.jsonl"
    rows = [
        {
            "triton_compiles": True,
            "triton_numerically_correct": True,
            "triton_speedup": 2.0,
        },
        {
            "triton_compiles": True,
            "triton_numerically_correct": False,
            "triton_speedup": 100.0,
        },
    ]
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    summary = _summarize(path, "shape-aware")

    assert summary["speedup_mean"] == 2.0
    assert summary["speedups_recorded"] == 1
