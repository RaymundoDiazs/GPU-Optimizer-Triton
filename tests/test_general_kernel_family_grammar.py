from pathlib import Path
import importlib.util

import pytest


GRAMMAR_PATH = Path(__file__).resolve().parents[1] / "grammars" / "tritonbench_t" / "general_kernel_family.ebnf"
MODULE_GRAMMAR_PATH = Path(__file__).resolve().parents[1] / "grammars" / "triton_module.gbnf"


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


def test_general_kernel_family_grammar_compiles_if_xgrammar_installed():
    if importlib.util.find_spec("xgrammar") is None:
        pytest.skip("xgrammar is not installed")
    if importlib.util.find_spec("transformers") is None:
        pytest.skip("transformers is not installed")

    import xgrammar as xgr
    from transformers import AutoTokenizer

    grammar = GRAMMAR_PATH.read_text(encoding="utf-8")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-1.5B-Instruct", use_fast=True)
    tokenizer_info = xgr.TokenizerInfo.from_huggingface(
        tokenizer,
        vocab_size=getattr(tokenizer, "vocab_size", None),
    )
    compiler = xgr.GrammarCompiler(tokenizer_info)
    compiled = compiler.compile_grammar(grammar)
    assert compiled is not None
