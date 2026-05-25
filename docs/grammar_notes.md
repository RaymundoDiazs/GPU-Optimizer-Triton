# Grammar Notes

## Responsibility

This document describes the formal grammar part of the project.

The grammar designer defines the valid shape of a kernel. The constrained-decoding integrator later connects those rules to the model so invalid tokens are blocked during generation.

## Difference From Constrained Decoding

Grammar design answers:

```text
What code is valid?
```

Constrained decoding answers:

```text
How do we force the model to generate only valid code?
```

For this project, the grammar is the rulebook. Constrained decoding is the mechanism that applies the rulebook while the LLM is writing.

## Current Grammar Scope

The first grammar supports only vector addition:

```text
Z = X + Y
```

This is enough for the first research milestone because the course requirement allows a grammar for one specific operation.

## Valid Kernel Shape

A valid kernel must look structurally like this:

```python
import triton
import triton.language as tl

@triton.jit
def vector_add_kernel(X, Y, Z, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(X + offsets, mask=mask)
    y = tl.load(Y + offsets, mask=mask)
    tl.store(Z + offsets, x + y, mask=mask)
```

## Why These Rules Matter

- `@triton.jit` tells Triton to compile the function as a kernel.
- `tl.program_id(0)` gives each program instance an id.
- `tl.arange(0, BLOCK_SIZE)` creates per-block offsets.
- `mask = offsets < N` avoids out-of-bounds memory access.
- `tl.load` reads from GPU memory.
- `tl.store` writes the result.
- `x + y` enforces the target operation.

## Files

- `grammars/vector_add.ebnf`: formal EBNF-style grammar.
- `grammars/vector_add_contract.md`: human-readable rule contract.
- `grammars/examples/valid_vector_add.py`: valid example.
- `grammars/examples/invalid_missing_store.py`: invalid example.
- `grammars/examples/invalid_wrong_operation.py`: invalid example.
- `grammars/examples/invalid_no_mask.py`: invalid example.
- `parsing/triton_grammar_rules.py`: practical Python validator that approximates these rules.

## Current Limitation

The grammar currently validates code after generation. It does not yet constrain the model token by token.

That means this is not the final XGrammar integration yet. It is the grammar design and validation layer that the constrained-decoding work should use next.

