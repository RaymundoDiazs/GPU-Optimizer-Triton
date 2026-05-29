# TritonBench-T Grammar Contracts

This folder contains narrow grammar contracts for the first ten examples in:

```text
external/TritonBench/data/TritonBench_T_simp_alpac_v1.json
```

Those ten examples are copied into:

```text
data/tritonbench_t_simp_subset10.json
```

The contracts are not universal Triton grammars. They define the minimum accepted
module structure for each TritonBench-T task so generated LLM candidates can be
filtered before GPU execution.

`general_kernel_family.ebnf` is the reusable family-level grammar. The individual
contracts are used to compare whether task-specific checks and family checks make
the same accept/reject decision on the first ten examples.

Current subset:

- `tritonbench_t_001_fused_bmm_rmsnorm_gelu_dropout_sub.ebnf`
- `tritonbench_t_002_div.ebnf`
- `tritonbench_t_003_sigmoid_conv2d.ebnf`
- `tritonbench_t_004_solve_multiple_lu.ebnf`
- `tritonbench_t_005_tanh.ebnf`
- `tritonbench_t_006_relu_sqrt.ebnf`
- `tritonbench_t_007_sqrt.ebnf`
- `tritonbench_t_008_sigmoid_argmax.ebnf`
- `tritonbench_t_009_sub.ebnf`
- `tritonbench_t_010_grid_sample.ebnf`
- `general_kernel_family.ebnf`

The Python validator in `parsing/tritonbench_grammar_rules.py` implements
practical post-generation checks for these contracts.
