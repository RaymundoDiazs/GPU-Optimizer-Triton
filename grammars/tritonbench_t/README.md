# TritonBench-T Grammar Contracts

This folder contains generated grammar contracts for all 166 examples in:

```text
external/TritonBench/data/TritonBench_T_simp_alpac_v1.json
```

Those examples are copied into:

```text
data/tritonbench_t_simp_subset166.json
```

The individual contracts are not universal Triton grammars. They define the
minimum accepted module structure for each TritonBench-T task so generated LLM
candidates can be filtered before GPU execution.

`general_kernel_family.ebnf` is the reusable family-level grammar.
`universal_triton_kernel.ebnf` is the task-agnostic Triton structural grammar:
it checks for a generic Triton module shape without hard-coding a task family or
wrapper name. The individual contracts are used to compare whether task-specific
checks, family checks, and universal checks make the same accept/reject decision
before GPU execution.

Current grammar contracts:

- `tritonbench_t_001_*.ebnf` through `tritonbench_t_166_*.ebnf`
- `general_kernel_family.ebnf`
- `universal_triton_kernel.ebnf`
- `scripts/generate_tritonbench_contracts.py` regenerates the dataset subset,
  manifest, and any missing per-task EBNF files from the upstream dataset.

The Python validator in `parsing/tritonbench_grammar_rules.py` implements
practical post-generation checks for the JSON-backed subset.
