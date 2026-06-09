from dataclasses import dataclass
from pathlib import Path
from typing import Any

from parsing.triton_grammar_rules import validate_triton_kernel


DEFAULT_TRITON_EBNF = r"""
root ::= imports "\n" kernel
imports ::= "import triton\n" "import triton.language as tl\n"
kernel ::= "@triton.jit\n" "def generated_kernel(input_ptr, other_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):\n" body
body ::= "    pid = tl.program_id(0)\n"
         "    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)\n"
         "    mask = offsets < n_elements\n"
         "    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)\n"
         "    y = tl.load(other_ptr + offsets, mask=mask, other=0.0)\n"
         "    result = " expression "\n"
         "    tl.store(output_ptr + offsets, result, mask=mask)\n"
expression ::= "x" | "x + y" | "x - y" | "x * y" | "x / y" | "tl.tanh(x)" | "tl.sqrt(x)" | "tl.exp(x)" | "tl.maximum(x, 0.0)"
"""

DEFAULT_GRAMMAR_PATH = Path(__file__).resolve().parents[1] / "grammars" / "tritonbench_t" / "general_kernel_family.ebnf"


@dataclass
class XGrammarGenerationResult:
    prompt: str
    generated_code: str
    accepted: bool
    errors: list[str]
    warnings: list[str]
    used_xgrammar: bool


class XGrammarLLMDecoder:

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        grammar_text: str | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.grammar_text = grammar_text or self._load_default_grammar()
        self.compiled_grammar = None

    @staticmethod
    def _load_default_grammar() -> str:
        if DEFAULT_GRAMMAR_PATH.exists():
            return DEFAULT_GRAMMAR_PATH.read_text(encoding="utf-8")
        return DEFAULT_TRITON_EBNF

    def compile_grammar(self):

        import xgrammar as xgr

        vocab_size = getattr(self.tokenizer, "vocab_size", None)

        if vocab_size is None and hasattr(self.model, "config"):
            vocab_size = getattr(self.model.config, "vocab_size", None)

        tokenizer_info = xgr.TokenizerInfo.from_huggingface(
            self.tokenizer,
            vocab_size=vocab_size,
        )

        compiler = xgr.GrammarCompiler(tokenizer_info)

        self.compiled_grammar = compiler.compile_grammar(
            self.grammar_text
        )

        return self.compiled_grammar

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
    ):

        import xgrammar as xgr

        if self.compiled_grammar is None:
            self.compile_grammar()

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
        )

        # Move inputs to the same device as the model if possible.
        try:
            device = next(self.model.parameters()).device
            inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
        except (StopIteration, AttributeError):
            pass

        logits_processor = xgr.contrib.hf.LogitsProcessor(
            self.compiled_grammar
        )

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            logits_processor=[logits_processor],
        )

        prompt_length = inputs["input_ids"].shape[-1]

        generated_ids = output_ids[0][prompt_length:]

        generated_code = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        )

        validation = validate_triton_kernel(
            generated_code
        )

        return XGrammarGenerationResult(
            prompt=prompt,
            generated_code=generated_code,
            accepted=validation.valid,
            errors=validation.errors,
            warnings=validation.warnings,
            used_xgrammar=True,
        )


def build_triton_prompt(
    problem_description: str,
):

    return f"""
You are generating Triton GPU kernels.

Rules:
- Output only code.
- Do not use markdown fences.
- Use @triton.jit.
- Inside @triton.jit use only Triton primitives: tl.program_id, tl.arange, tl.load, tl.store, tl.* math, and vector expressions.
- Do not use torch, numpy, math, isinstance, range, for loops, or triton.jit.get_* inside @triton.jit.
- Put torch allocation and launch logic only in a plain Python wrapper.
- Do not explain anything.

Task:
{problem_description}
""".strip()
