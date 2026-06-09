import json
import sys
import types
from pathlib import Path

from generation.xgrammar_llm_decoder import (
    DEFAULT_TRITON_EBNF,
    XGrammarLLMDecoder,
    build_triton_prompt,
)
from evaluation.collect_real_outputs import build_tritonbench_constrained_prompt


class DummyTensor:
    def __init__(self):
        self.device = "cpu"

    def to(self, device):
        return self


class DummyTokenizer:
    vocab_size = 100

    def __call__(
        self,
        text,
        return_tensors=None,
    ):
        class FakeTensor:
            def __init__(self):
                self.device = "cpu"
                self.shape = (-1, 3)

            def to(self, device):
                return self

        return {
            "input_ids": FakeTensor(),
        }

    def decode(self, token_ids, skip_special_tokens=True):
        return "import triton\nimport triton.language as tl\n\n@triton.jit\ndef vector_add_kernel(x_ptr, y_ptr, z_ptr, N, BLOCK_SIZE: tl.constexpr):\n    pass\n"


class DummyModel:
    config = type(
        "Config",
        (),
        {"vocab_size": 100},
    )()

    def parameters(self):
        yield DummyTensor()


class DummyXGrammarModel(DummyModel):
    def __init__(self):
        super().__init__()
        self.generate_kwargs = None

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        return [[0, 1, 2, 3]]


def test_default_triton_grammar():
    assert "root ::= imports" in DEFAULT_TRITON_EBNF
    assert "@triton.jit" in DEFAULT_TRITON_EBNF
    assert "tl.store" in DEFAULT_TRITON_EBNF
    assert "torch" not in DEFAULT_TRITON_EBNF


def test_decoder_creation():
    model = DummyModel()
    tokenizer = DummyTokenizer()

    decoder = XGrammarLLMDecoder(
        model,
        tokenizer,
    )

    assert decoder.model is model
    assert decoder.tokenizer is tokenizer


def test_prompt_builder():
    prompt = build_triton_prompt(
        "Generate vector addition kernel"
    )

    assert "Triton" in prompt
    assert "@triton.jit" in prompt


def test_compile_grammar_uses_xgrammar_tokenizer_and_compiler(monkeypatch):
    class TokenizerInfo:
        @staticmethod
        def from_huggingface(tokenizer, vocab_size=None):
            return {"tokenizer": tokenizer, "vocab_size": vocab_size}

    class GrammarCompiler:
        def __init__(self, tokenizer_info):
            self.tokenizer_info = tokenizer_info

        def compile_grammar(self, grammar_text):
            return {"compiled_grammar": grammar_text}

    class DummyLogitsProcessor:
        def __init__(self, compiled_grammar):
            self.compiled_grammar = compiled_grammar

    fake_xgrammar = types.SimpleNamespace(
        TokenizerInfo=TokenizerInfo,
        GrammarCompiler=GrammarCompiler,
        contrib=types.SimpleNamespace(hf=types.SimpleNamespace(LogitsProcessor=DummyLogitsProcessor)),
    )
    monkeypatch.setitem(sys.modules, "xgrammar", fake_xgrammar)

    model = DummyModel()
    tokenizer = DummyTokenizer()
    decoder = XGrammarLLMDecoder(model, tokenizer, grammar_text="root ::= kernel")

    compiled = decoder.compile_grammar()

    assert compiled == {"compiled_grammar": "root ::= kernel"}
    assert decoder.compiled_grammar == compiled


def test_generate_uses_logits_processor(monkeypatch):
    class TokenizerInfo:
        @staticmethod
        def from_huggingface(tokenizer, vocab_size=None):
            return {"tokenizer": tokenizer, "vocab_size": vocab_size}

    class GrammarCompiler:
        def __init__(self, tokenizer_info):
            self.tokenizer_info = tokenizer_info

        def compile_grammar(self, grammar_text):
            return {"compiled_grammar": grammar_text}

    class DummyLogitsProcessor:
        def __init__(self, compiled_grammar):
            self.compiled_grammar = compiled_grammar

    fake_xgrammar = types.SimpleNamespace(
        TokenizerInfo=TokenizerInfo,
        GrammarCompiler=GrammarCompiler,
        contrib=types.SimpleNamespace(hf=types.SimpleNamespace(LogitsProcessor=DummyLogitsProcessor)),
    )
    monkeypatch.setitem(sys.modules, "xgrammar", fake_xgrammar)

    module = __import__("generation.xgrammar_llm_decoder", fromlist=["validate_triton_kernel"])
    monkeypatch.setattr(module, "validate_triton_kernel", lambda code: types.SimpleNamespace(valid=True, errors=[], warnings=[]))

    model = DummyXGrammarModel()
    tokenizer = DummyTokenizer()
    decoder = XGrammarLLMDecoder(model, tokenizer, grammar_text="root ::= kernel")
    result = decoder.generate(prompt="Generate a kernel", max_new_tokens=5)

    assert isinstance(result.generated_code, str)
    assert model.generate_kwargs is not None
    assert "logits_processor" in model.generate_kwargs
    assert isinstance(model.generate_kwargs["logits_processor"][0], DummyLogitsProcessor)


def test_constrained_prompt_does_not_include_other_tritonbench_examples():
    """Verify no data contamination: TritonBench examples don't leak into prompts."""
    repo_root = Path(__file__).resolve().parents[1]
    dataset_path = repo_root / "data" / "tritonbench_t_simp_subset166.json"
    with dataset_path.open("r", encoding="utf-8") as f:
        dataset = json.load(f)

    prompt, _ = build_tritonbench_constrained_prompt("tritonbench_t_001", mode="family")
    selected_instruction = dataset[0]["instruction"].strip()

    assert selected_instruction in prompt
    for other in [item["instruction"].strip() for item in dataset[1:10]]:
        assert other not in prompt


def test_full_qwen_xgrammar_pipeline_writes_and_evaluates(tmp_path, monkeypatch):
    """End-to-end: collection + generation + evaluation with XGrammar backend."""
    from evaluation.collect_real_outputs import collect_for_model_generation

    def fake_build_prompt(task_id, mode="family"):
        return "Generate vector add kernel", "root ::= kernel"

    def fake_call_hf_xgrammar(prompt, grammar_text=None, max_new_tokens=512):
        return (
            "import triton\nimport triton.language as tl\n\n@triton.jit\ndef vector_add_kernel(x_ptr, y_ptr, z_ptr, N, BLOCK_SIZE: tl.constexpr):\n    pass\n",
            0.12,
        )

    monkeypatch.setattr(sys.modules["evaluation.collect_real_outputs"], "build_tritonbench_constrained_prompt", fake_build_prompt)
    monkeypatch.setattr(sys.modules["evaluation.collect_real_outputs"], "call_hf_xgrammar", fake_call_hf_xgrammar)

    model = {
        "id": "small_qwen25_coder_1_5b",
        "display_name": "Qwen2.5-Coder-1.5B (Ollama local)",
        "tier": "small",
        "provider": "ollama",
        "model_name": "qwen2.5-coder:1.5b",
    }
    task = {
        "id": "tritonbench_t_001",
        "prompt": "vector add task prompt",
        "expected_terms": ["@triton.jit", "tl.load", "tl.store"],
    }

    qwen_path = tmp_path / "predictions_qwen_constrained.jsonl"
    main_path = tmp_path / "real_outputs.jsonl"
    with qwen_path.open("w", encoding="utf-8") as qwen_out_file:
        with main_path.open("w", encoding="utf-8") as main_out_file:
            count = collect_for_model_generation(
                model,
                [task],
                modes=["constrained"],
                samples=1,
                out_file=main_out_file,
                qwen_out_file=qwen_out_file,
            )

    assert count == 1
    assert qwen_path.exists()

    from evaluation.model_evaluation import run_model_evaluation

    results_dir = tmp_path / "results"
    rows = run_model_evaluation(output_dir=results_dir, manual_outputs=qwen_path)

    assert len(rows) == 1
    assert rows[0]["mode"] == "constrained"
    assert rows[0]["constrained_decoding_backend"] == "xgrammar_hf"
    generated = json.loads(qwen_path.read_text(encoding="utf-8").strip().splitlines()[0])
    assert generated["constrained_decoding_backend"] == "xgrammar_hf"
