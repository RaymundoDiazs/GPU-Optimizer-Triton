# Columna 1 — Problema, Background y Diferenciador

---

## Header

**Título:** Grammar-Constrained GPU Kernel Generation via LLMs

**Subtítulo:** Category-Aware Few-Shot Prompting for Triton Code Synthesis

**Autores:** [Nombre 1] · [Nombre 2] · [Nombre 3] · [Nombre 4]
**Afiliación:** Tecnológico de Monterrey — TC3002B (ACA-2026)

---

## Abstract

Generamos kernels Triton a partir de código PyTorch usando LLMs con decodificación restringida por gramática EBNF (XGrammar). Evaluamos Claude Haiku, GPT-4o y Qwen 2.5-Coder sobre 166 operadores de TritonBench-T. Claude Haiku logra 41.6% call@1 y 40.4% exe@1; Qwen sin restricción alcanza 23.5% call@1 pero solo 6.6% exe@1.

---

## 1. Problema

- Escribir kernels GPU en Triton manualmente requiere experiencia experta.
- Los LLMs generan código estructuralmente inválido: usan APIs de PyTorch dentro de `@triton.jit`, omiten máscaras, firmas incorrectas.
- **¿Puede una gramática EBNF + few-shot por categoría de operador aumentar la tasa de kernels compilables generados por un LLM local?**

---

## 2. Background

- **Triton DSL:** lenguaje sobre Python para kernels GPU; compila a PTX via JIT. Estructura semi-regular: `@triton.jit`, `tl.load`, `tl.store`, `tl.program_id`.
- **Gramáticas formales (EBNF):** definen sintaxis válida con reglas de producción; permiten restringir el espacio de tokens del LLM en cada paso.
- **XGrammar:** compila gramáticas EBNF en bitmasks y expone un `LogitsProcessor` para HuggingFace, sin modificar el modelo.

---

## 3. Nuestro diferenciador

> **Clasificamos cada operador en una categoría funcional e insertamos en el prompt un ejemplo de kernel Triton válido de esa categoría — sin usar operadores del benchmark.**

Cada operador de TritonBench-T se clasifica automáticamente en una familia (elementwise, reducción, álgebra lineal, fusión, etc.). El prompt que recibe el LLM incluye un ejemplo de código Triton real y correcto de esa misma familia, guiando la estructura del kernel generado sin revelar la respuesta.

- **C1.** El ejemplo en el prompt es siempre de la misma familia funcional que el operador objetivo, aumentando la probabilidad de que el kernel tenga la arquitectura correcta desde el primer intento.
- **C2.** Los ejemplos provienen de fuentes externas al benchmark (no de los 166 operadores de TritonBench-T), evitando contaminación de datos.
