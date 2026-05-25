# Grammars

This folder contains the formal grammar assets for the project.

The current scoped grammar targets one operation:

```text
vector_add: Z = X + Y
```

It is intentionally narrow. The goal is not to describe every possible Triton program, but to define a controlled subset that can be used for constrained decoding and evaluation.

Files:

- `vector_add.ebnf`: human-readable EBNF grammar for a valid vector-add Triton kernel.
- `vector_add_contract.md`: design contract explaining required structures and rejected errors.
- `examples/valid_vector_add.py`: accepted example.
- `examples/invalid_missing_store.py`: rejected because it never writes output.
- `examples/invalid_wrong_operation.py`: rejected because it computes subtraction instead of addition.
- `examples/invalid_no_mask.py`: rejected because memory loads/stores are unmasked.

