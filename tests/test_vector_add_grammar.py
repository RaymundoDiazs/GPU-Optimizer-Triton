from pathlib import Path

from parsing.triton_grammar_rules import validate_vector_add_kernel


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "grammars" / "examples"


def _read_example(name: str) -> str:
    return (EXAMPLES / name).read_text(encoding="utf-8")


def test_valid_vector_add_example_passes():
    result = validate_vector_add_kernel(_read_example("valid_vector_add.py"))

    assert result.valid is True
    assert result.errors == []


def test_missing_store_example_fails():
    result = validate_vector_add_kernel(_read_example("invalid_missing_store.py"))

    assert result.valid is False
    assert any("store" in error for error in result.errors)


def test_wrong_operation_example_fails():
    result = validate_vector_add_kernel(_read_example("invalid_wrong_operation.py"))

    assert result.valid is False
    assert any("store_add" in error for error in result.errors)


def test_no_mask_example_fails():
    result = validate_vector_add_kernel(_read_example("invalid_no_mask.py"))

    assert result.valid is False
    assert any("mask" in error for error in result.errors)

