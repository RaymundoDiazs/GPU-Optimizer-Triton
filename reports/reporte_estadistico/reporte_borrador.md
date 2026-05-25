# Parte II - Resultados y Analisis

Este documento contiene un borrador del reporte estadistico. Los resultados actuales son preliminares y deben interpretarse como validacion del pipeline, no como evidencia concluyente.

## 2.1 Resultados descriptivos

Fuente de datos actual:

```text
evaluation/artifacts/model_eval_results.csv
```

Fuente planeada para la evaluacion final:

```text
TritonBench / TritonBench4Modal
```

Las metricas actuales son una capa preliminar de validacion estructural. Las metricas finales deben venir de ejecucion real: call accuracy, execution accuracy y speedup vs PyTorch.

Tamano actual:

```text
18 outputs = 3 modelos x 2 modos x 3 muestras
```

Resumen observado:

| Modelo | Modo | Muestras | Syntax valid | Kernel shape valid | Correctness proxy |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-Coder-1.5B | baseline | 3 | 100% | 33% | 0% |
| Qwen2.5-Coder-1.5B | constrained | 3 | 100% | 100% | 0% |
| GPT-4o | baseline | 3 | 100% | 100% | 100% |
| GPT-4o | constrained | 3 | 100% | 100% | 100% |
| Claude Haiku 4.5 | baseline | 3 | 100% | 100% | 100% |
| Claude Haiku 4.5 | constrained | 3 | 100% | 100% | 100% |

Observacion preliminar:

El modo constrained mejora la estructura del kernel generado por Qwen, pasando de 33% a 100% en `kernel_shape_valid`. Sin embargo, `correctness_proxy` permanece en 0%, lo que indica que la estructura no garantiza comportamiento correcto.

## 2.2 Visualizacion de datos

Las graficas se generan con:

```bash
python reports/reporte_estadistico/scripts/analyze_results.py
```

Figuras generadas:

- `figures/kernel_shape_valid_rate.png`
- `figures/correctness_proxy_rate.png`
- `figures/latency_by_model_mode.png`

Interpretacion esperada:

- La grafica de `kernel_shape_valid` muestra mejora clara de Qwen en modo constrained.
- La grafica de `correctness_proxy` muestra que Qwen sigue fallando la metrica proxy.
- Las graficas de latencia deben interpretarse con cautela porque dependen del provider y del entorno.

## 2.3 Inferencia estadistica

Con n=3 por grupo, no se deben hacer conclusiones inferenciales fuertes.

El script incluye una prueba aproximada de diferencia de proporciones, intervalos Wilson al 95%, tamano del efecto para proporciones mediante Cohen's h y correccion Holm-Bonferroni. Estos resultados son exploratorios; el reporte final debera repetir el analisis con al menos 30 muestras por grupo.

Comparacion preliminar principal:

```text
Qwen baseline vs Qwen constrained en kernel_shape_valid
```

Resultado observado:

```text
baseline = 1/3
constrained = 3/3
diferencia = +0.667
```

Interpretacion:

La diferencia observada es grande, pero la muestra es demasiado pequena para sostener una conclusion estadistica robusta.

Los resultados generados por `pairwise_proportion_tests.csv` deben interpretarse de forma conservadora. Si una comparacion muestra p-value bajo con n pequeno, no debe reportarse como evidencia concluyente sin repetir el experimento con mayor muestra.

## 2.4 Comparacion critica

El hallazgo mas importante no es que los modelos frontier obtengan 100% en esta tarea simple, sino que el modelo pequeno muestra una separacion entre:

- sintaxis valida,
- estructura valida,
- correctness proxy.

Esto sugiere que medir solo sintaxis puede ser enganoso. Un modelo puede generar Python valido y aun asi fallar como kernel Triton.

La comparacion tambien sugiere que restricciones estructurales pueden ayudar a modelos pequenos, pero no necesariamente resuelven correccion funcional.

Esto es relevante porque muestra una diferencia entre forma y comportamiento:

```text
estructura valida != resultado correcto
```

Por eso el siguiente paso metodologico es reemplazar o complementar `correctness_proxy` con equivalencia PyTorch-vs-Triton.

## 2.4.1 Conexion con TritonBench

El material de clase sobre TritonBench define una evaluacion mas cercana al reto final. En lugar de revisar solo texto generado, TritonBench ejecuta el codigo y separa tres niveles:

- call accuracy: el codigo generado puede ser llamado con la firma esperada,
- execution accuracy: la salida coincide con PyTorch,
- efficiency: se mide speedup contra la implementacion PyTorch.

Esta separacion es importante porque un output puede tener buena estructura pero fallar en ejecucion, o puede ser correcto pero no competitivo en rendimiento.

Para el reporte final, los resultados preliminares de `syntax_valid`, `kernel_shape_valid` y `correctness_proxy` deben reportarse como diagnostico inicial. Las conclusiones principales deben basarse en las metricas de TritonBench cuando esten disponibles.

## 2.4.2 Analisis de robustez preliminar

El analisis de robustez todavia no puede ejecutarse completamente con los datos actuales.

Robustez pendiente:

- repetir con mas muestras,
- repetir con otra seed,
- repetir con otra temperatura baja,
- probar mas shapes de `vector_add`,
- comparar proxy vs equivalencia real.
- repetir con subconjuntos distintos de operadores de TritonBench.
- separar resultados por tipo de operador: elementwise, reduccion y operaciones con mayor complejidad de memoria.

Conclusion de robustez preliminar:

La tendencia observada en Qwen es interesante, pero no robusta todavia por el tamano de muestra y por usar una sola tarea simple.

## 2.5 Amenazas observadas a la validez

### Validez interna

- Muestra actual demasiado pequena.
- Providers externos pueden variar en latencia y comportamiento.
- El modo constrained actual usa contrato de prompt y validacion posterior; aun no es XGrammar end-to-end.

### Validez externa

- La tarea `vector_add` es simple.
- Los resultados pueden no generalizar a matmul, reducciones o kernels con patrones de memoria complejos.

### Validez de constructo

- `correctness_proxy` no es equivalencia funcional real.
- `kernel_shape_valid` mide forma del kernel, no rendimiento.
- La latencia de generacion no mide velocidad del kernel generado.

## 2.6 Discusion

La evidencia preliminar apoya la idea de que las restricciones pueden mejorar la estructura generada por un modelo pequeno. Sin embargo, la evidencia no es suficiente para afirmar mejora general en calidad de kernels.

Para sostener una conclusion fuerte se necesita:

- mas muestras,
- ejecucion real de kernels,
- comparacion PyTorch vs Triton,
- medicion de runtime,
- analisis estadistico con intervalos de confianza y tamano del efecto.

Adicionalmente, se debe analizar si las diferencias observadas se explican por la escala del modelo. Es esperable que modelos frontier superen al modelo pequeno; por eso una pregunta mas interesante es si las restricciones reducen parcialmente esa brecha.

En el estado actual, la evidencia sugiere que el modo constrained mejora estructura en Qwen, pero no cierra la brecha de correctness proxy frente a GPT-4o o Claude.

## 2.7 Conclusion preliminar

Con la evidencia actual no se puede rechazar formalmente H0. La muestra es insuficiente para inferencia estadistica fuerte.

No obstante, los resultados preliminares muestran una tendencia relevante:

```text
Qwen constrained mejora kernel_shape_valid respecto a Qwen baseline.
```

La conclusion defendible por ahora es:

> El pipeline experimental ya permite recolectar outputs, evaluar estructura y comparar modelos. Los resultados iniciales sugieren que las restricciones ayudan al modelo pequeno en estructura, pero falta validacion funcional y estadistica para sostener conclusiones finales.

## Trabajo faltante

- Aumentar a 30-50 muestras por grupo.
- Ejecutar validacion PyTorch-vs-Triton.
- Integrar XGrammar end-to-end.
- Medir runtime y speedup.
- Repetir analisis estadistico con mayor muestra.
