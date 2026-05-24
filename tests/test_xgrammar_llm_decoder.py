from generation.xgrammar_llm_decoder import (
    DEFAULT_TRITON_EBNF,
    XGrammarLLMDecoder,
    build_triton_prompt,
)


class DummyTokenizer:
    vocab_size = 100

    def __call__(
        self,
        text,
        return_tensors=None,
    ):
        return {
            "input_ids": [[1, 2, 3]]
        }


class DummyModel:
    config = type(
        "Config",
        (),
        {"vocab_size": 100},
    )()


def test_default_triton_grammar():
    assert "root ::= kernel" in DEFAULT_TRITON_EBNF
    assert "@triton.jit" in DEFAULT_TRITON_EBNF
    assert "tl.store" in DEFAULT_TRITON_EBNF


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