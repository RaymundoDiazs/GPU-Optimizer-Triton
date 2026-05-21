from dataclasses import dataclass

from parsing.triton_grammar_rules import (
    GrammarCheckResult,
    get_valid_next_tokens,
    validate_triton_kernel,
)


@dataclass
class ConstrainedGenerationResult:
    accepted: bool
    code: str
    grammar_result: GrammarCheckResult
    valid_next_tokens: list[str]


class ConstrainedDecoder:
    """
    First prototype for connecting grammar rules with model output.

    This does not replace the model yet.
    It works as a control layer that validates generated Triton code
    before accepting it.
    """

    def __init__(self, initial_state: str = "START"):
        self.current_state = initial_state

    def allowed_tokens(self) -> list[str]:
        return get_valid_next_tokens(self.current_state)

    def update_state(self, token: str) -> None:
        self.current_state = token

    def validate_model_output(self, generated_code: str) -> ConstrainedGenerationResult:
        grammar_result = validate_triton_kernel(generated_code)

        return ConstrainedGenerationResult(
            accepted=grammar_result.valid,
            code=generated_code,
            grammar_result=grammar_result,
            valid_next_tokens=self.allowed_tokens(),
        )


def constrain_model_output(generated_code: str) -> dict:
    """
    Simple public function that can be used by the pipeline or tests.
    """

    decoder = ConstrainedDecoder()
    result = decoder.validate_model_output(generated_code)

    return {
        "accepted": result.accepted,
        "code": result.code,
        "errors": result.grammar_result.errors,
        "warnings": result.grammar_result.warnings,
        "valid_next_tokens": result.valid_next_tokens,
    }