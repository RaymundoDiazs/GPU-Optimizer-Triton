Hola profesor, buenos días. Esperamos se encuentre bien.

Le escribimos con algo de pena porque sentimos que como equipo no logramos coordinarnos bien desde el inicio, cada quien tenía una idea distinta de hacia dónde iba el proyecto y eso nos generó bastante confusión. Después de la asesoría de ayer nos juntamos para alinear nuestros pensamientos, entender bien el reto y definir un approach en el que todos estuviéramos de acuerdo.

Queremos compartirle el enfoque que estamos tomando, mencionarle algunas cosas que ya tenemos avanzadas y otras que aún están en proceso, y de paso resolver unas dudas puntuales. También tenemos una propuesta de diferenciador que nos gustaría que nos dijera si va por buen camino antes de invertir tiempo en implementarlo. Sobre todo nos gustaría que nos confirmara si el approach general y la implementación que tenemos pensada están correctos y alineados con lo que se pide, para estar seguros de que vamos por el camino correcto y que todo el equipo está en la misma página.


Nuestro enfoque

1. Generación baseline con 3 modelos

Se generan los kernels Triton para los 166 operadores de TritonBench-T usando tres modelos. Para GPT y Claude usamos modal_app.py directamente, que genera y evalúa en una sola corrida en GPU. Para Qwen, al ser un modelo local, generamos las predicciones por separado y las pasamos a evaluate_only para la evaluación en GPU. Los tres se evalúan con el mismo pipeline de TritonBench4Modal midiendo call@1, exe@1 y speedup.

Si, en este caso se espera que le hagan constrained decoding a qwen. Asi tendrían al menos 4 experimentos. 

2. Comparación con HumanEval

Mediremos pass@1 en HumanEval para los 3 modelos. El proceso sería: correr el benchmark HumanEval estándar sobre los 3 modelos con temperatura 0.8 para obtener diversidad, calcular pass@1 y cruzar ese número contra los resultados de call@1 en TritonBench para ver si hay correlación.

No tiene caso correr human eval, son tareas diferentes. Sospecho que aventaron todas las slides del curso a una ia y les sugiere eso.

3. Constrained decoding con XGrammar

Para la parte de constrained decoding entendemos que aplica únicamente a Qwen, por que es el único modelo que corre localmente y da acceso directo a los logits. 
Si.

El proceso sería: definir una gramática EBNF general que describa la estructura válida de cualquier kernel Triton, compilarla con xgr.GrammarCompiler, y conectarla a Qwen durante la generación mediante xgr.contrib.hf.LogitsProcessor. Qwen se carga con HuggingFace Transformers para tener acceso directo al proceso de decodificación. De ahi obtenemos el archivo predictions_qwen_constrained.jsonl que se evalúa con el mismo pipeline de TritonBench4Modal.

Si. 

4. Diferenciador

Como diferenciador, planeamos clasificar los operadores en categorías y seleccionar el ejemplo más cercano al tipo de operador que se va a generar. POr ejemplo, cuando el modelo va a generar un operador de tipo reduction, el prompt incluye ejemplos de reduction. Pensamos que los ejemplos estructuralmente similares aumentan la probabilidad de que el kernel tenga la arquitectura correcta desde el primer intento.

Está bien. Solo tengan cuidado de no contaminar el experimento usando los mismos ejemplos del tritón bench. Sería como hacer un examen viendo las respuestas. 

5. Análisis

Con los archivos evaluados compararemos los 3 modelos baseline entre sí, Qwen baseline vs Qwen constrained y HumanEval vs desempeño en Triton.

HumanEval es para python, no para PyTorch ni triton. No usen human eval para nada, solo fue un ejemplo en una clase.

Dudas que queremos confirmar

1. Confirmamos que el dataset correcto es TritonBench_T_simp_alpac_v1.json, el input es una descripción funcional y firma de la función no código PyTorch fuente. ¿Es así, o debemos usar la versión que incluye el código fuente?

Deben usar la versión que tiene el código en PyTorch. La tarea es traducir de PyTorch a tritón 

2. Para la gramática, estamos tomando como referencia los datasets del studen_package: curated_100 y adversarial_100 para identificar patrones de qué hace válido o inválido un kernel. Entendemos que la evaluación final es con TritonBench. ¿La gramática debe ser una sola gramática general que cubra cualquier kernel Triton válido, o debe ser específica para cada uno de los 166 operadores?

Una solo gramática. No tiene sentido 166. 

3. ¿Se espera que el constrained decoding aplique solo a Qwen, o también se espera alguna forma de restricción estructurada para los otros modelos?
Solo qwen 

4. Para el diferenciador, el plan es clasificar los 166 operadores en categorías y usar un ejemplo por categoría. ¿Este es un diferenciador válido para el reto?es válido solo usando ejemplos que no estén en el benchmark. Si usas los mismos estás haciendo trampa. 

5. ¿Usar fine-tuning es parte del scope esperado para todos los equipos?
No es requisito. Algunos lo hacen como diferenciador., 