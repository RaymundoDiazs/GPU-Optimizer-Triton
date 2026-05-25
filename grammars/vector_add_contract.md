# Vector Add Grammar Contract

## Scope

This grammar accepts a narrow Triton kernel for vector addition:

```text
Z = X + Y
```

The grammar is intentionally small so the team can first prove the constrained-generation idea on one operation.

## Required Structures

A valid kernel must include:

- `import triton`
- `import triton.language as tl`
- `@triton.jit`
- a kernel function named `vector_add_kernel`
- arguments `X`, `Y`, `Z`, `N`, and `BLOCK_SIZE`
- `pid = tl.program_id(0)`
- block offsets using `tl.arange(0, BLOCK_SIZE)`
- a bounds mask: `mask = offsets < N`
- one load from `X`
- one load from `Y`
- one store into `Z`
- the operation `x + y`
- masked memory access using `mask=mask`

## Rejected Errors

The grammar should reject or flag kernels that:

- omit `@triton.jit`
- omit `tl.program_id`
- omit `tl.arange`
- omit `tl.load`
- omit `tl.store`
- do not use a mask
- write the wrong operation, such as `x - y`
- never write to the output pointer
- return explanations instead of code
- have unbalanced parentheses or brackets

## Current Implementation Status

The EBNF file is the formal design artifact.

The current Python validator in `parsing/triton_grammar_rules.py` implements practical checks that approximate this contract. It is still post-generation validation. The next integration step is to use this grammar with XGrammar or an equivalent constrained-decoding engine so invalid tokens are blocked during generation.

