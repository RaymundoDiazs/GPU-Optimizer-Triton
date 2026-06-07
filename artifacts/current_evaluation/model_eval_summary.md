# PyTorch to Triton Translation — Model Evaluation Summary

| Model | Mode | Samples | Syntax valid | Safe | Contract valid | Correctness proxy |
|---|---:|---:|---:|---:|---:|---:|
| frontier_anthropic | baseline | 166 | 99% | 98% | 90% | 89% |
| frontier_openai | baseline | 166 | 100% | 100% | 65% | 65% |
| small_qwen25_coder_1_5b | baseline | 166 | 94% | 93% | 2% | 2% |

Notes:
- Static checks cover Python syntax, a restricted execution policy, and task-specific contracts.
- GPU compilation and numerical correctness are validated downstream by TritonBench.
