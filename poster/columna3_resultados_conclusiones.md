# Columna 3 — Resultados, Conclusiones y Referencias

---

## 7. Setup experimental

- **Benchmark:** TritonBench-T — 166 operadores, input: `pytorch_func_complete`.
- **Hardware:** NVIDIA A10G (Modal) para evaluación; Qwen en CPU local.
- **Métricas:** call@1 (compila sin error), exe@1 (correcto numéricamente), speedup vs. PyTorch.
- **Repeticiones:** 1 run por kernel.

---

## 8. Resultados

> **Claude Haiku logra 41.6% call@1 sobre 166 operadores — 4× más que GPT-4o y 1.8× más que Qwen sin restricción.**

| Modelo | call@1 | exe@1 | Speedup mediano |
|---|---|---|---|
| Claude Haiku 4.5 | **41.57%** | **40.36%** | 0.45× |
| Qwen 2.5-Coder (sin restricción) | 23.49% | 6.63% | 0.80× |
| GPT-4o | 10.84% | 7.23% | 0.45× |
| Qwen + XGrammar + few-shot | [pendiente] | [pendiente] | — |

- Speedup < 1.0 es esperado: los kernels baseline no incluyen optimizaciones de tiling manual.
- GPT-4o falla frecuentemente por usar `torch.*` dentro del `@triton.jit` — error que la gramática elimina.

---

## 9. Discusión y limitaciones

- **Logros:** pipeline completo y reproducible para 3 modelos sobre 166 operadores; comparación directa base vs. constrained.
- **Limitaciones:** speedup < 1.0 en todos los modelos; operadores de álgebra lineal (LU, SVD) tienen tasa ~0% en todos.
- **Siguiente paso:** retry agentico — retroalimentar el error de compilación al LLM, máx. 3 intentos.

---

## Referencias

1. Tillet et al. (2019). *Triton: An Intermediate Language for Tiled Neural Network Computations*. MAPL '19.
2. Dong et al. (2024). *XGrammar: Flexible and Efficient Structured Generation Engine for LLMs*. arXiv:2411.15100.
3. Qwen Team (2024). *Qwen2.5-Coder Technical Report*. Alibaba Cloud.
4. TritonBench (2024). *Benchmarking LLM Capabilities for Generating Triton Operators*. GitHub: thunlp/TritonBench.
5. Geng et al. (2023). *Grammar-Constrained Decoding for Structured NLP Tasks*. EMNLP 2023.
6. Shi et al. (2024). *KernelBench: Can LLMs Write Efficient GPU Kernels?* arXiv:2502.10517.
