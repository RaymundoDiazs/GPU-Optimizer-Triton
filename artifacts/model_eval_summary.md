# Kernel Generation Model Evaluation Summary

| Model | Mode | Samples | Syntax valid | Kernel shape valid | Correctness proxy |
|---|---:|---:|---:|---:|---:|
| frontier_anthropic | baseline | 150 | 100% | 100% | 75% |
| frontier_anthropic | constrained | 150 | 100% | 100% | 100% |
| frontier_openai | baseline | 150 | 100% | 100% | 75% |
| frontier_openai | constrained | 150 | 100% | 100% | 100% |
| small_qwen25_coder_1_5b | baseline | 150 | 67% | 33% | 33% |
| small_qwen25_coder_1_5b | constrained | 150 | 100% | 100% | 100% |

Notes:
- These are lightweight checks intended for the second progress video.
- Replace mock rows with manually collected or API-generated outputs before claiming real model results.
- GPU compilation and functional correctness should be validated in the final experiment harness.
