# Plantilla de Reproducibilidad Experimental

Este archivo debe completarse antes de entregar el reporte final.

## Identificacion del experimento

- Nombre del experimento:
- Fecha de ejecucion:
- Responsable:
- Repositorio:
- Commit exacto:
- Rama:

## Hardware

- Sistema operativo:
- CPU:
- RAM:
- GPU:
- VRAM:
- Driver NVIDIA:
- CUDA:

## Software

- Python:
- PyTorch:
- Triton:
- XGrammar:
- Transformers:
- Pandas:
- Matplotlib:
- Ollama:

Comando recomendado:

```bash
python reports/reporte_estadistico/scripts/capture_environment.py
```

## Modelos

| Modelo | Provider | Version/checkpoint | Tamano | Acceso |
|---|---|---|---:|---|
| Qwen2.5-Coder-1.5B | Ollama | qwen2.5-coder:1.5b | 1.5B | local |
| GPT-4o | OpenAI | gpt-4o | no reportado | API |
| Claude Haiku 4.5 | Anthropic | claude-haiku-4-5-20251001 | no reportado | API |

## Configuracion de generacion

- Prompt:
- Temperatura:
- Seed:
- Muestras por modelo/modo:
- Max tokens:
- Modo baseline:
- Modo constrained:
- Uso de gramatica:

## Benchmark / evaluador

- Evaluador usado:
- Dataset/tarea:
- Shapes:
- Metricas:
- Script de analisis:

## Pasos para reproducir

```bash
pip install -r requirements.txt
python evaluation/collect_real_outputs.py --provider ollama
python evaluation/collect_real_outputs.py --provider openai
python evaluation/collect_real_outputs.py --provider anthropic
python evaluation/model_evaluation.py --manual-outputs evaluation/real_outputs.jsonl
python reports/reporte_estadistico/scripts/analyze_results.py
```

## Limitaciones de reproducibilidad

- APIs comerciales pueden cambiar internamente.
- Latencia depende de red y carga del provider.
- Resultados de modelos comerciales pueden no ser perfectamente deterministas.
- Validacion GPU depende de hardware y version CUDA.

