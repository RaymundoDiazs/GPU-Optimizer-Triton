# Parte I - Pre-registro Experimental

## 1.1 Objetivo experimental

El objetivo del experimento es evaluar un metodo de generacion de kernels Triton basado en un modelo pequeno y restricciones gramaticales, comparandolo contra modelos comerciales frontier usados como baseline.

El problema evaluado es la generacion de codigo Triton para una tarea inicial de suma de vectores:

```text
Z = X + Y
```

El experimento busca medir si las restricciones de estructura, mediante prompt constrained y posteriormente gramatica formal/XGrammar, mejoran la calidad de los kernels generados.

Metodos comparados:

- Qwen2.5-Coder-1.5B via Ollama local como modelo pequeno.
- GPT-4o via OpenAI como baseline frontier.
- Claude Haiku 4.5 via Anthropic como segundo baseline frontier.

Modos comparados:

- `baseline`: generacion con prompt base.
- `constrained`: generacion con contrato de restricciones estructurales.

## 1.2 Hipotesis

### Hipotesis principal

Metrica principal: `kernel_shape_valid`.

H0: No existe diferencia en la tasa de kernels con estructura valida entre el modo baseline y el modo constrained para el modelo pequeno Qwen2.5-Coder-1.5B.

H1: El modo constrained obtiene mayor tasa de kernels con estructura valida que el modo baseline para Qwen2.5-Coder-1.5B.

### Hipotesis secundaria

Metrica secundaria: `correctness_proxy`.

H0: No existe diferencia en la tasa de correctness proxy entre modelos frontier y el modelo pequeno.

H1: Los modelos frontier obtienen mayor tasa de correctness proxy que el modelo pequeno.

### Hipotesis futura para validacion GPU

Metrica futura: equivalencia PyTorch vs Triton.

H0: No existe diferencia significativa en la tasa de equivalencia funcional entre el metodo propuesto y los baselines.

H1: El metodo propuesto con restricciones gramaticales mejora la tasa de equivalencia funcional respecto al baseline sin restricciones.

## 1.3 Variables experimentales

### Variables independientes

- Modelo usado:
  - Qwen2.5-Coder-1.5B.
  - GPT-4o.
  - Claude Haiku 4.5.
- Modo de generacion:
  - baseline.
  - constrained.
- Provider:
  - Ollama local.
  - OpenAI API.
  - Anthropic API.

### Variables dependientes

- `syntax_valid`: indica si la salida es codigo Python sintacticamente valido.
- `kernel_shape_valid`: indica si el codigo contiene estructura minima esperada de kernel Triton.
- `correctness_proxy`: aproximacion heuristica de si el kernel corresponde a la operacion esperada.
- `latency_seconds`: tiempo de generacion registrado para cada salida.

Metricas planeadas para la version final:

- call accuracy de TritonBench,
- execution accuracy de TritonBench,
- equivalencia numerica PyTorch vs Triton,
- runtime,
- speedup contra PyTorch.

### Variables controladas

- Misma tarea inicial: `vector_add`.
- Mismo prompt base.
- Misma temperatura: 0.2.
- Misma semilla configurada: 7 cuando el provider lo permite.
- Mismo numero de muestras por modelo/modo en cada corrida experimental.
- Mismo evaluador para todos los modelos.

## 1.4 Diseno experimental

El diseno es within-task y between-methods.

Todos los modelos resuelven la misma tarea `vector_add`, pero se comparan diferentes metodos/modelos y modos de generacion.

Grupos:

```text
3 modelos x 2 modos = 6 grupos
```

Comparaciones planeadas:

- Qwen baseline vs Qwen constrained.
- Qwen constrained vs GPT-4o constrained.
- Qwen constrained vs Claude constrained.
- GPT-4o baseline vs GPT-4o constrained.
- Claude baseline vs Claude constrained.

Justificacion:

Usar la misma tarea y el mismo prompt base reduce variabilidad por diferencias de entrada. Comparar baseline contra constrained permite aislar el efecto de las restricciones estructurales.

## 1.5 Metricas

### Syntax valid

Definicion: porcentaje de outputs que pueden parsearse como Python valido.

Interpretacion: mide errores sintacticos basicos.

Limitacion: un codigo Python valido puede no ser un kernel Triton correcto.

### Kernel shape valid

Definicion: porcentaje de outputs que contienen estructuras requeridas de Triton, como `@triton.jit`, `tl.program_id`, `tl.load` y `tl.store`.

Interpretacion: mide cumplimiento estructural de kernel.

Limitacion: no garantiza correccion numerica ni rendimiento.

### Correctness proxy

Definicion: heuristica que revisa si el codigo contiene terminos esperados para la tarea, por ejemplo `x + y`.

Interpretacion: aproximacion temprana a correccion funcional.

Limitacion: no sustituye ejecucion real ni comparacion numerica.

### Call accuracy

Definicion: porcentaje de outputs generados que pueden ser cargados por el harness de evaluacion y llamados con la firma esperada.

Interpretacion: mide si la salida del modelo es usable dentro del benchmark.

Limitacion: no garantiza que el resultado numerico sea correcto.

### Execution accuracy

Definicion: porcentaje de outputs que producen el mismo resultado que la implementacion PyTorch de referencia en los casos de prueba del benchmark.

Interpretacion: mide correccion funcional.

Limitacion: depende de la cobertura de inputs y shapes usados por el benchmark.

### Speedup vs PyTorch

Definicion: razon entre el tiempo de ejecucion de la referencia PyTorch y el tiempo de ejecucion del kernel Triton generado.

Interpretacion: valores mayores a 1 indican mejora de rendimiento.

Limitacion: solo debe interpretarse para kernels que pasaron correccion funcional.

### Latency seconds

Definicion: tiempo de generacion registrado por output.

Interpretacion: aproxima costo temporal de generacion.

Limitacion: puede depender del provider, red, carga del servidor y hardware local.

## 1.6 Tamano del efecto esperado

Para la hipotesis principal se espera un efecto grande en `kernel_shape_valid` para el modelo pequeno, porque las restricciones deberian ayudar especialmente a modelos con menor capacidad.

Efecto esperado:

```text
diferencia de proporciones >= 0.30
```

Ejemplo esperado:

```text
Qwen baseline: 40% - 60%
Qwen constrained: 70% - 90%
```

## 1.7 Analisis de poder

Parametros planeados:

- nivel de significancia: alfa = 0.05,
- poder estadistico deseado: 0.80,
- efecto esperado: mediano a grande.

Debido a limitaciones de costo y tiempo, se propone un minimo operativo de:

```text
30 muestras por modelo/modo
```

Idealmente:

```text
50 muestras por modelo/modo
```

Los resultados actuales con n=3 por grupo no tienen poder suficiente para inferencia estadistica fuerte.

## 1.8 Plan estadistico

### Comparaciones planeadas

Comparaciones primarias:

- Qwen baseline vs Qwen constrained en `kernel_shape_valid`.
- Qwen constrained vs GPT-4o constrained en `correctness_proxy`.
- Qwen constrained vs Claude constrained en `correctness_proxy`.

Comparaciones secundarias:

- GPT-4o baseline vs GPT-4o constrained.
- Claude baseline vs Claude constrained.
- Comparacion de latencia entre modelos.

### Pruebas estadisticas

Para proporciones binarias:

- prueba z de dos proporciones cuando n sea suficiente,
- prueba exacta de Fisher si hay celdas pequenas.

Para latencia:

- prueba t si la distribucion es aproximadamente normal,
- prueba Mann-Whitney U si no se cumple normalidad.

Para comparaciones multiples:

- correccion Holm-Bonferroni.

### Regla de decision

Se rechazara H0 si:

```text
p < 0.05
```

despues de correccion por comparaciones multiples cuando aplique.

Tambien se reportaran intervalos de confianza del 95% y tamano del efecto.

## 1.9 Reproducibilidad experimental

Informacion a reportar:

- Repositorio: `https://github.com/RaymundoDiazs/GPU-Optimizer-Triton`
- Commit: debe fijarse antes de la corrida final.
- Python: registrar version exacta.
- Librerias: `torch`, `triton`, `transformers`, `xgrammar`, `pandas`, `matplotlib`.
- Modelos:
  - Qwen2.5-Coder-1.5B via Ollama.
  - GPT-4o via OpenAI.
  - Claude Haiku 4.5 via Anthropic.
- Prompt: `prompts/kernel_generation_prompt.txt`.
- Tarea: `data/kernel_generation_tasks.json`.
- Configuracion: `config/model_eval.yaml`.
- Semilla: 7 cuando el provider lo permite.
- Temperatura: 0.2.
- Benchmark final: TritonBench-T o TritonBench4Modal, documentando dataset, limite de operadores y GPU usada.

Para validacion GPU futura:

- GPU,
- driver CUDA,
- version de PyTorch CUDA,
- version de Triton,
- sistema operativo.

## 1.10 Amenazas a la validez

### Validez interna

- Diferencias entre providers pueden introducir variabilidad no controlada.
- APIs comerciales pueden cambiar comportamiento aunque el nombre del modelo sea el mismo.
- Latencias dependen de red y carga del provider.
- `correctness_proxy` puede no reflejar correccion real.

### Validez externa

- `vector_add` es una tarea simple y puede no representar kernels complejos.
- Los resultados pueden no generalizar a reducciones, matmul o kernels con memoria compartida.
- El numero actual de muestras es pequeno.

### Validez de constructo

- `kernel_shape_valid` mide estructura, no rendimiento.
- `syntax_valid` puede sobreestimar calidad.
- `correctness_proxy` no sustituye PyTorch-vs-Triton equivalence.

## 1.11 Plan de robustez

El analisis de robustez busca verificar si las conclusiones se mantienen bajo cambios razonables de configuracion.

Pruebas planeadas:

1. Repeticion con mas semillas cuando el provider lo permita.
2. Repeticion con otra temperatura baja, por ejemplo `temperature = 0.0`.
3. Aumento del numero de muestras por grupo.
4. Evaluacion de al menos dos shapes para `vector_add`:
   - vector pequeno: `N = 128`.
   - vector mediano: `N = 4096`.
   - vector grande: `N = 1_000_000`.
5. Comparacion entre metrica proxy y validacion real PyTorch-vs-Triton cuando haya GPU disponible.

Criterio de robustez:

Una conclusion se considerara robusta solo si la direccion del efecto se mantiene al cambiar semillas, tamano de muestra o shape de entrada.

Ejemplo:

Si el modo constrained mejora `kernel_shape_valid` en Qwen con n=50 y el efecto se mantiene en diferentes shapes, entonces la conclusion es mas confiable.

Si el efecto solo aparece en `vector_add` pequeno o con una sola seed, se reportara como evidencia limitada.

## 1.12 Criterios de inclusion y exclusion

Se incluiran outputs que:

- provengan de uno de los modelos registrados,
- usen el prompt configurado,
- correspondan a la tarea definida,
- tengan modo `baseline` o `constrained`,
- incluyan metadata de modelo, provider, modo, tarea y muestra.

Se excluiran outputs que:

- no correspondan a la tarea evaluada,
- hayan sido generados con temperatura o prompt distinto sin documentarlo,
- no tengan metadata suficiente,
- sean duplicados exactos por error de recoleccion,
- fallen por error de API no relacionado con el modelo.

Toda exclusion posterior a la recoleccion debera registrarse explicitamente.

## 1.13 Criterios para conclusiones

No se aceptaran conclusiones basadas solo en el mejor caso observado.

Para afirmar una mejora, se requiere:

- diferencia consistente en la metrica primaria,
- intervalo de confianza reportado,
- tamano del efecto reportado,
- prueba estadistica segun el plan,
- discusion de amenazas a la validez.

Si n es menor a 30 por grupo, las conclusiones deberan presentarse como preliminares.
