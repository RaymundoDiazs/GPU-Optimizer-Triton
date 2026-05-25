# Plan de Integracion con TritonBench

Este documento conecta el reporte estadistico con el material visto en clase sobre TritonBench y TritonBench4Modal.

## Por que usar TritonBench

La validacion actual del proyecto revisa principalmente:

- si la salida es Python valido,
- si contiene estructura de kernel Triton,
- si cumple un proxy simple de correccion.

Eso sirve como primera capa, pero no basta para el reto final. TritonBench permite evaluar lo que realmente importa en generacion de kernels:

- si el codigo se puede llamar correctamente,
- si ejecuta sin fallar,
- si produce el mismo resultado que PyTorch,
- si mejora o empeora el tiempo de ejecucion.

## Metricas que se deben reportar

### Call accuracy

Pregunta: el codigo generado respeta la firma esperada y se puede ejecutar dentro del harness?

Interpretacion: mide si el modelo genero una solucion usable por el benchmark.

Limitacion: pasar call accuracy no significa que el resultado numerico sea correcto.

### Execution accuracy

Pregunta: el resultado generado por Triton coincide con la referencia PyTorch?

Interpretacion: mide correccion funcional.

Limitacion: depende de los casos de prueba y shapes usados por el benchmark.

### Speedup vs PyTorch

Pregunta: el kernel Triton generado es mas rapido que la implementacion PyTorch?

Interpretacion: mide utilidad practica del kernel.

Limitacion: un kernel correcto puede ser mas lento que PyTorch, especialmente para shapes pequenos o kernels mal parametrizados.

## Pipeline propuesto para el reporte final

```text
1. Tomar una funcion PyTorch del benchmark.
2. Pedir a cada modelo que genere una version Triton.
3. Aplicar validacion estructural propia.
4. Ejecutar TritonBench sobre los outputs generados.
5. Guardar call accuracy, execution accuracy y speedup.
6. Analizar resultados por modelo y por modo de generacion.
```

La validacion estructural propia no reemplaza TritonBench. Funciona como filtro previo para entender que tipo de errores produce cada modelo antes de llegar a ejecucion real.

## Relacion con el notebook de clase

El notebook didactico muestra tres ideas importantes que deben aparecer en el reporte:

- traducir PyTorch a Triton no es solo generar texto; el resultado debe mantener la firma esperada,
- un kernel puede pasar la llamada inicial y aun asi fallar la equivalencia numerica,
- la metrica de rendimiento solo tiene sentido despues de pasar correccion funcional.

Por eso el orden correcto de evaluacion es:

```text
syntax_valid -> kernel_shape_valid -> call accuracy -> execution accuracy -> speedup
```

## Relacion con TritonBench4Modal

TritonBench4Modal permite ejecutar TritonBench-T en una GPU remota de Modal. Su flujo tiene tres fases:

```text
Phase 1: call accuracy
Phase 2: execution accuracy
Phase 3: efficiency / speedup vs PyTorch
```

Esto encaja directamente con la rubrica del reporte porque produce metricas cuantitativas, reproducibles y comparables entre modelos.

## Como se debe usar en nuestro proyecto

Para el video actual:

- mencionar TritonBench como el siguiente paso de evaluacion real,
- aclarar que los resultados actuales son preliminares y estructurales,
- explicar que el reporte final usara call accuracy, execution accuracy y speedup.

Para el reporte final:

- correr un smoke test con pocos operadores primero,
- guardar los JSON/CSV de resultados,
- aumentar gradualmente el numero de operadores,
- comparar el metodo propuesto contra GPT-4o y Claude usando las mismas tareas.

## Recomendacion practica

Empezar con un subconjunto pequeno:

```text
5 a 10 operadores
```

Despues escalar a:

```text
30 a 50 operadores
```

Si el tiempo y costo lo permiten, correr el benchmark completo.

## Como reportarlo

En el reporte estadistico, TritonBench debe aparecer como la fuente principal de las metricas finales:

- tasa de compilacion / llamada exitosa,
- tasa de ejecucion correcta,
- speedup promedio o geometrico,
- errores frecuentes por modelo,
- variabilidad por operador o shape.

La conclusion no debe decir solo "modelo A fue mejor". Debe explicar en que etapa fue mejor:

```text
modelo A genero codigo mas estructurado,
modelo B compilo mas veces,
modelo C tuvo mejor equivalencia numerica,
modelo D fue mas rapido solo despues de pasar correctness.
```

Esa separacion hace que el analisis sea mas fuerte y mas defendible.
