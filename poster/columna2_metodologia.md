# Columna 2 — Metodología

---

## 4. Arquitectura del sistema

```
[PyTorch spec] → [Clasificador de categoría]
                          ↓
               [Prompt Builder + ejemplo few-shot]
                          ↓
               [LLM / XGrammar (Qwen)]
                          ↓
                   [Triton JIT]
                          ↓
          [TritonBench Evaluator]
          call@1 · exe@1 · speedup
```

- Claude / GPT-4o: API remota en Modal (GPU A10G).
- Qwen 2.5-Coder 1.5B: local vía HuggingFace + XGrammar `LogitsProcessor`.

---

## 5. Categorización de operadores

Los 166 operadores se clasifican en familias funcionales. A cada categoría le corresponde un ejemplo de kernel Triton externo al benchmark:

| Categoría | Ejemplos de operadores |
|---|---|
| Elementwise | `relu`, `sigmoid`, `erf`, `log`, `exp` |
| Reducción | `sum`, `max`, `mean`, `std` |
| Álgebra lineal | `matmul`, `lu`, `svd`, `cholesky` |
| Fusión | `relu_batch_norm`, `fused_transformer_block` |
| Especiales | `fft`, `bessel`, `airy_ai` |

El ejemplo insertado en el prompt es de la misma categoría que el operador objetivo, pero **nunca** un operador de la lista de evaluación.

---

## 6. Pipeline de generación

1. **Clasificación:** el operador se asigna a su categoría funcional.
2. **Prompt:** descripción funcional + código PyTorch de referencia + ejemplo Triton de la misma categoría.
3. **Decodificación (Qwen):** gramática EBNF compilada con `xgr.GrammarCompiler` → `LogitsProcessor` → greedy decoding.
4. **Evaluación:** compilación Triton JIT → ejecución → comparación numérica vs. PyTorch.

| Modelo | Constrained decoding | Few-shot categorizado |
|---|---|---|
| Claude Haiku 4.5 | No | No (baseline) |
| GPT-4o | No | No (baseline) |
| Qwen 2.5-Coder 1.5B | **Sí** | **Sí** |
