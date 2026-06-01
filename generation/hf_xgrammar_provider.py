import time
from functools import lru_cache

from generation.xgrammar_llm_decoder import XGrammarLLMDecoder


DEFAULT_HF_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"


@lru_cache(maxsize=1)
def load_hf_model(model_name: str = DEFAULT_HF_MODEL):
    """
    Load a local HuggingFace model for real XGrammar constrained decoding.

    This is required because XGrammar needs token-level access through
    logits processors, which commercial APIs do not expose.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
    )

    return model, tokenizer


def call_hf_xgrammar(
    prompt: str,
    model_name: str = DEFAULT_HF_MODEL,
    grammar_text: str | None = None,
    max_new_tokens: int = 512,
) -> tuple[str, float]:
    """
    Real XGrammar constrained decoding provider.

    This routes constrained generation through HuggingFace + XGrammar,
    instead of using only prompt-based constraints.
    """
    model, tokenizer = load_hf_model(model_name)

    decoder = XGrammarLLMDecoder(
        model=model,
        tokenizer=tokenizer,
        **({"grammar_text": grammar_text} if grammar_text is not None else {}),
    )

    start = time.perf_counter()

    result = decoder.generate(
        prompt=prompt,
        max_new_tokens=max_new_tokens,
    )

    latency = time.perf_counter() - start

    return result.generated_code, latency
