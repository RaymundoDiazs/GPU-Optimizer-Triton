import csv
import json

from benchmarks.run_benchmarks import FIELDNAMES, run_benchmarks


def test_run_benchmarks_writes_csv(tmp_path):
    problems_path = tmp_path / "problems.json"
    output_path = tmp_path / "results.csv"
    problems_path.write_text(
        json.dumps([{"id": "vector_add", "code": "C = A + B"}]),
        encoding="utf-8",
    )

    rows = run_benchmarks(runs=2, problems_path=problems_path, output_path=output_path)

    assert len(rows) == 2
    with output_path.open("r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        assert reader.fieldnames == FIELDNAMES
        saved_rows = list(reader)

    assert len(saved_rows) == 2
    assert saved_rows[0]["id"] == "vector_add"
    assert saved_rows[0]["problem_type"] == "element_wise"
