Fine tuning the model using a larger SOTA LLM, and also integrating it into a ready-for-use code review tool in github so that developers can get optimization suggestions right from their pull requests

We use progressive optimization combined with hardware-in-the-loop feedback to systematically find optimal kernels. Our approach optimizes in five intelligent phases: memory hierarchy, parallelism, memory access, fine-tuning, and micro-optimizations, where each phase builds on the previous one and is guided by real GPU profiling metrics (throughput, occupancy, memory bandwidth). After each phase, we execute the code on actual hardware, measure concrete performance data, and feed these metrics back to the LLM with phase-specific constraints enforced by formal grammars.

"Genera kernels GPU en tres niveles de optimización garantizados por gramática
Si el nivel más alto falla, el sistema baja automáticamente al siguiente sin romper el pipeline
El usuario siempre recibe el kernel más optimizado posible dado su contexto"

Ademas de generar codigo con Triton con una gramatica, generaremos codigo en CUDA con el LLM para comparar la efectividad de generacion de codigo de manera cruda y en el lenguaje mas complejo de GPUs VS un lenguaje mas simplificado y apoyado con el uso de LLMS.

Implementación de un ciclo de retroalimentación donde los errores del compilador de Triton corrigen automáticamente el código vía LLM y Sistema de votación entre dos LLMs especializados para pre-validar kernels, reduciendo el riesgo de errores de memoria en la GPU antes de la ejecución.

We go beyond basic grammar constraints by designing a domain-specific, performance-aware grammar for stencil operations that enforces both correctness and GPU optimization patterns during code generation.

Agent loop for resolving compiler errors, code optimization agent, property-based (a test with x cases using random variables), test-driven, agent generating specs and tests (chain of thought before starting)

What differentiates TRIED is that it goes beyond one-shot code generation by integrating a continuous feedback loop. Instead of simply generating Triton code, it compiles, tests, and validates each kernel against real outputs, learning from both successes and failures. Over time, this creates a self-improving system that gets better through execution, not just prompting, making it more reliable and adaptive than traditional approaches.

---

## Propuestas de diferenciadores para nuestro equipo

### Opción A — Few-shot dinámico por tipo de operador ⭐ MUY FÁCIL
**Qué es:** en vez de usar siempre los mismos 2 ejemplos (vector_add y softmax) en el prompt, seleccionar el ejemplo más parecido al operador que se va a generar.

**Cómo:** clasificar los 166 operadores en categorías (elementwise, reduction, matmul, fused) y tener un ejemplo few-shot por categoría. Cuando se va a generar un operador de tipo reduction, el prompt incluye un ejemplo de reduction en vez de vector_add.

**Por qué ayuda:** el modelo tiene un ejemplo más cercano a lo que necesita generar, lo que aumenta la probabilidad de que el kernel tenga la estructura correcta.

**Esfuerzo:** bajo. Solo requiere clasificar los operadores y agregar 2-3 ejemplos más al prompt. No toca ningún otro componente.

---

### Opción B — Self-repair: reintentar con el error del compilador ⭐ FÁCIL
**Qué es:** si un kernel falla call@1 (no compila), tomar el mensaje de error y mandárselo al modelo con el contexto original para que intente corregirlo.

**Cómo:**
```
Intento 1: generar kernel → falla call@1 con "NameError: tl.program_id not defined"
Intento 2: prompt = original + "Tu intento anterior falló con este error: <error>. Corrígelo."
```

**Por qué ayuda:** los errores de compilación de Triton son muy específicos y el modelo puede corregirlos si se le dice exactamente qué falló. Especialmente útil para Qwen que tiene call@1 bajo.

**Esfuerzo:** bajo-medio. Requiere un loop de reintento en el script de generación, pero la lógica es simple. Se puede limitar a 1-2 reintentos para no disparar el costo de API.

---

### Opción C — Chain-of-thought antes del código
**Qué es:** agregar al prompt una instrucción para que el modelo piense en voz alta antes de escribir el kernel: qué tipo de paralelismo necesita, cuántas dimensiones tiene el grid, si requiere reducción.

**Cómo:** modificar el prompt para incluir:
```
Before writing the kernel, briefly answer:
1. What parallelism pattern does this operation need? (elementwise / reduction / matmul)
2. How many program_id axes are needed?
3. Does it require atomic operations?
Then write the kernel.
```

**Por qué ayuda:** fuerza al modelo a planear antes de codificar, lo que reduce errores estructurales. Especialmente útil para operadores complejos (fused ops, reductions 2D).

**Esfuerzo:** muy bajo. Solo es un cambio de prompt. El evaluador ignora el texto previo al bloque de código.

---

### Opción D — Prompt con restricciones negativas explícitas
**Qué es:** el prompt actual dice qué hay que hacer. Esta opción agrega explícitamente qué NO hacer, basado en los errores más frecuentes que ya observamos en los kernels de Qwen.

**Cómo:** agregar una sección al prompt:
```
COMMON MISTAKES TO AVOID:
- Never use Python for loops to iterate over elements — use tl.program_id
- Never access tensors with direct indexing (x[i]) — use tl.load / tl.store
- Never define BLOCK_SIZE as a variable — it must be tl.constexpr
- Never omit the boundary mask — every tl.load and tl.store needs mask=
```

**Por qué ayuda:** los errores de Qwen en el baseline son predecibles (loops Python, sin mask, sin program_id). Nombrarlos explícitamente reduce su frecuencia.

**Esfuerzo:** mínimo. Es una línea de texto en el prompt. Ya tenemos datos del baseline de Qwen para saber cuáles errores son los más frecuentes.

---

### Recomendación

Las opciones **B + D** juntas son las más fáciles de implementar y tienen el mayor impacto esperado en call@1 de Qwen, que es donde más margen de mejora hay. La opción **A** es la más interesante para el análisis porque permite ver si la calidad del few-shot predice la calidad del kernel generado. La opción **C** es bajo riesgo y se puede probar en 10 minutos.