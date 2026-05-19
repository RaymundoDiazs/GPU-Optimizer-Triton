import csv

from evaluation.model_evaluation import run_model_evaluation


def test_model_evaluation_writes_artifacts(tmp_path):
    rows = run_model_evaluation(output_dir=tmp_path, samples_per_model=1)

    assert rows
    results_path = tmp_path / "model_eval_results.csv"
    generated_path = tmp_path / "generated_kernels.jsonl"
    summary_path = tmp_path / "model_eval_summary.md"

    assert results_path.exists()
    assert generated_path.exists()
    assert summary_path.exists()

    with results_path.open("r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        fieldnames = set(reader.fieldnames or [])
        saved_rows = list(reader)

    assert {"model_id", "mode", "syntax_valid", "correctness_proxy"}.issubset(fieldnames)
    assert len(saved_rows) == len(rows)
