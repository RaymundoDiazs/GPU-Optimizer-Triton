# Second Progress Video Plan

Deadline: Saturday, May 23, 2026.

Goal: explain the current state of the LLM-based kernel generation method through decisions, errors, and lessons learned. This is not a pipeline walkthrough.

## Model Comparison Scope

Use one small model and two frontier baselines.

Current repo defaults are placeholders in `config/model_eval.yaml`:

- Small: `Qwen2.5-Coder-1.5B` candidate for local/on-prem feasibility.
- Frontier A: high-quality API model placeholder.
- Frontier B: second high-quality API model placeholder from another provider.

Before recording, replace `Frontier Model A/B` with the actual model versions the team tested.

## What To Run

Generate demo artifacts with deterministic mock outputs:

```bash
python evaluation/model_evaluation.py --samples 3
```

For real video evidence, collect outputs from the chosen models using the same prompt in:

```text
prompts/kernel_generation_prompt.txt
```

Then replace the mock labels/results with the real model names and outputs. The expected artifacts are:

- `evaluation/artifacts/generated_kernels.jsonl`
- `evaluation/artifacts/model_eval_results.csv`
- `evaluation/artifacts/model_eval_summary.md`

## What Each Person Can Say

Ray:
- We scoped the first task to vector addition because it is small enough to constrain with a grammar and still representative of Triton kernel structure.
- The main design decision was not to create a full Triton grammar. We define a narrow grammar for valid kernel shape first.

Ocampo:
- We compared a small coder model against two frontier models using the same kernel generation prompt.
- The trade-off we focused on was local feasibility and cost versus reliability and instruction following.

Cfsl:
- The hard part is not writing a grammar on paper. The hard part is applying it during decoding so invalid tokens are filtered before the model emits them.
- Early constrained runs should focus on fewer valid outputs rather than broad expressiveness.

German:
- A generated kernel is not useful until it can be compiled and tested.
- We separate lightweight static checks for fast iteration from GPU execution checks for final validation.

## Narrative Beats

1. Model selection was driven by cost, latency, reproducibility, and code quality.
2. The small model is important because it represents local/on-prem deployment, but it is expected to make more syntax and API mistakes.
3. The frontier models are stronger baselines, but they are not automatically correct and may still produce plausible wrong kernels.
4. The team learned that prompt quality helps, but it does not provide guarantees.
5. The next step is to make the grammar active during decoding and move from heuristic checks to GPU compile/functionality checks.

## Honest Caveat

Do not claim mock outputs are real model results. They are for rehearsing the video structure and validating the data pipeline. Replace them with real runs before the final recording if possible.
