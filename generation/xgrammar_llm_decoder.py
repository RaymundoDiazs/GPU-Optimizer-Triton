from dataclasses import dataclass
from typing import Any

from parsing.triton_grammar_rules import validate_triton_kernel


DEFAULT_TRITON_EBNF = r"""
root ::= kernel
kernel ::= decorator ws function_def ws body
decorator ::= "@triton.jit"
function_def ::= "def" ws identifier "(" params ")" ":"
params ::= param ("," ws param)*
param ::= identifier | identifier ":" ws "tl.constexpr"
body ::= line+
line ::= ws statement
statement ::= program_id | arange | load | store | assignment
program_id ::= identifier ws "=" ws "tl.program_id(" integer ")"
arange ::= identifier ws "=" ws "tl.arange(" integer "," ws identifier ")"
load ::= identifier ws "=" ws "tl.load(" text ")"
store ::= "tl.store(" text ")"
assignment ::= identifier ws "=" ws text
identifier ::= [a-zA-Z_][a-zA-Z0-9_]*
integer ::= [0-9]+
text ::= [^\n]+
ws ::= [ \t\n]*
"""


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
        grammar_text: str = DEFAULT_TRITON_EBNF,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.grammar_text = grammar_text
        self.compiled_grammar = None

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
- Use @triton.jit.
- Use tl.program_id.
- Use tl.load.
- Use tl.store.
- Do not explain anything.

Task:
{problem_description}
""".strip()