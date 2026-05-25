# HumanEval — Contexto de capacidad de los modelos

HumanEval es el benchmark estándar de generación de código Python.
Mide pass@1: qué porcentaje de 164 problemas el modelo resuelve correctamente
en el primer intento.

## Resultados públicos

| Modelo | HumanEval pass@1 | Fuente |
|---|---|---|
| Qwen2.5-Coder-1.5B (local) | ~43.9% | Alibaba Qwen2.5 Tech Report 2024 |
| GPT-4o (OpenAI) | ~90.2% | OpenAI evals 2024 |
| Claude Haiku 4.5 (Anthropic) | ~88.0% | Anthropic model card 2025 |

## Interpretación para este proyecto

Qwen2.5-Coder-1.5B resuelve correctamente ~44 de cada 100 problemas
estándar de Python. GPT-4o y Claude Haiku resuelven ~90.

Esto explica directamente los resultados de TritonBench:
- Si Qwen ya tiene dificultades con Python general (44% pass@1),
  los kernels Triton son aún más difíciles porque requieren conocimiento
  especializado de paralelismo GPU que no está en la mayoría del código
  de entrenamiento.
- Los modelos frontier pasan ~90% en Python general, y eso se refleja
  en que generan kernels Triton correctos el 100% de las veces.

## Conclusión para el video

No es que Qwen sea un modelo malo. Es que la tarea de generación de
kernels GPU es una tarea muy especializada para un modelo de 1.5B parámetros.
La brecha entre 44% y 90% en HumanEval predice exactamente la brecha
que observamos en nuestros resultados de Triton.
