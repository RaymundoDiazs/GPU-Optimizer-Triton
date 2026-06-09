from pathlib import Path
import importlib.util

import pytest


ROOT = Path(__file__).resolve().parents[1]
GRAMMAR_PATH = ROOT / "grammars" / "tritonbench_t" / "general_kernel_family.ebnf"
MODULE_GRAMMAR_PATH = ROOT / "grammars" / "triton_module.gbnf"
UNIVERSAL_GRAMMAR_PATH = ROOT / "grammars" / "tritonbench_t" / "universal_triton_kernel.ebnf"


def test_general_kernel_family_grammar_exists():
    assert GRAMMAR_PATH.exists(), f"Grammar file not found: {GRAMMAR_PATH}"
    grammar = GRAMMAR_PATH.read_text(encoding="utf-8")
    assert "@triton.jit" in grammar
    assert "tl.load" in grammar
    assert "tl.store" in grammar
    assert "tl.program_id" in grammar
    assert "tl.arange" in grammar
    assert "mask = offsets < n_elements" in grammar
    assert "generic-statement" not in grammar
    assert "line_rest" not in grammar
    assert "torch_fallback" not in grammar


def test_single_module_grammar_is_not_free_form():
    assert MODULE_GRAMMAR_PATH.exists(), f"Grammar file not found: {MODULE_GRAMMAR_PATH}"
    grammar = MODULE_GRAMMAR_PATH.read_text(encoding="utf-8")

    assert "def generated_kernel" in grammar
    assert "tl.load(input_ptr + offsets, mask=mask" in grammar
    assert "tl.store(output_ptr + offsets, result, mask=mask)" in grammar
    assert "torch.from_buffer" not in grammar
    assert "line_rest" not in grammar
    assert "line_char" not in grammar


def test_universal_triton_kernel_grammar_exists_and_is_strict():
    assert UNIVERSAL_GRAMMAR_PATH.exists(), f"Grammar file not found: {UNIVERSAL_GRAMMAR_PATH}"
    grammar = UNIVERSAL_GRAMMAR_PATH.read_text(encoding="utf-8")
    assert "required_parallelism" in grammar
    assert "required_memory_io" in grammar
    assert "top_level_block+" in grammar
    assert "triton_kernel_block" in grammar
    assert "wrapper_function" in grammar
    assert "grid_launch" in grammar
    assert "elementwise_statement" in grammar
    assert "reduction_statement" in grammar
    assert "dot_statement" in grammar
    assert "triton_library_statement" in grammar
    assert "@triton.jit" in grammar
    assert "tl.program_id" in grammar
    assert "tl.load" in grammar
    assert "tl.store" in grammar
    assert "torch_fallback" not in grammar
    assert "raw_statement" not in grammar
    assert "loop_statement" not in grammar


def _compile_with_xgrammar(grammar_path: Path):
    if importlib.util.find_spec("xgrammar") is None:
        pytest.skip("xgrammar is not installed")
    if importlib.util.find_spec("transformers") is None:
        pytest.skip("transformers is not installed")

    import xgrammar as xgr
    from transformers import AutoTokenizer

    grammar = grammar_path.read_text(encoding="utf-8")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-1.5B-Instruct", use_fast=True)
    tokenizer_info = xgr.TokenizerInfo.from_huggingface(
        tokenizer,
        vocab_size=getattr(tokenizer, "vocab_size", None),
    )
    compiler = xgr.GrammarCompiler(tokenizer_info)
    return compiler.compile_grammar(grammar)


def test_general_kernel_family_grammar_compiles_if_xgrammar_installed():
    assert _compile_with_xgrammar(GRAMMAR_PATH) is not None


def test_universal_triton_kernel_grammar_compiles_if_xgrammar_installed():
    assert _compile_with_xgrammar(UNIVERSAL_GRAMMAR_PATH) is not None
