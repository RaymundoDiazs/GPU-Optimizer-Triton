El objetivo del reto es evaluar qué tan bien distintos modelos de lenguaje pueden traducir código de PyTorch a Triton, midiendo no solo si el código generado funciona, sino también si es eficiente, seguro y competitivo frente a la versión original.

Para esto se utilizará TritonBench, que permite ejecutar los kernels generados y medir:

Cuántos ejemplos se ejecutan correctamente.
Cuántos fallan por errores de compilación o ejecución.
Qué tan correctos son los resultados numéricos.
Qué tan rápido es el código generado frente a la versión en PyTorch.

La primera parte consiste en usar modelos comerciales, como OpenAI o Anthropic, y también modelos pequeños o abiertos, para generar código Triton a partir de implementaciones en PyTorch. Después, cada resultado se evalúa automáticamente con TritonBench.

La segunda parte consiste en comparar modelos. Además de TritonBench, se puede usar HumanEval para medir la capacidad general de programación en Python, y luego comparar si esa habilidad realmente se traduce en mejores kernels Triton.

La tercera parte del reto es explorar generación restringida, usando herramientas como constrained decoding o xgrammar. La idea es limitar lo que el modelo puede generar para que el resultado tenga una estructura válida, reduciendo errores de sintaxis y mejorando la probabilidad de que el código compile.

Finalmente, los resultados deben analizarse estadísticamente, comparando métricas como tasa de compilación, tasa de ejecución correcta, speedup promedio, errores frecuentes y desempeño por modelo.

Pregunta central del reto

El reto no solo busca responder qué modelo genera mejor código Triton, sino también:

¿Qué técnicas adicionales pueden mejorar el proceso de traducción de PyTorch a Triton para obtener código más correcto, más seguro y más eficiente?