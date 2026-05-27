"""
Tests para evaluation/model_evaluation.py y evaluation/collect_real_outputs.py

Scope: traducción PyTorch → Triton con LLM (parte de Dilan Ocampo).
Estos tests no llaman a ninguna API ni a Ollama — todo corre localmente
usando datos de prueba sintéticos que simulan el formato real de los outputs.
"""

import csv
import json
import sys
import pytest
from pathlib import Path

from evaluation.model_evaluation import (
    extract_code,
    evaluate_generated_code,
    load_manual_outputs,
    run_model_evaluation,
    _build_summary,
    _rate,
)
from evaluation.collect_real_outputs import (
    build_translation_prompt,
    load_pytorch_examples,
)

# ---------------------------------------------------------------------------
# Fixtures — datos sintéticos que simulan outputs reales de traducción
# ---------------------------------------------------------------------------

VALID_KERNEL = """\
import triton
import triton.language as tl

@triton.jit
def vector_add_kernel(x_ptr, y_ptr, z_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(z_ptr + offsets, x + y, mask=mask)
"""

INVALID_SYNTAX_KERNEL = "def broken_kernel(x_ptr\n    pass"

INCOMPLETE_KERNEL = """\
import triton
import triton.language as tl

def not_a_jit_kernel(x, y):
    return x + y
"""

VECTOR_ADD_TASK = {
    "id": "vector_add",
    "pytorch_code": "z = x + y",
    "operation_description": "element-wise addition of two vectors Z = X + Y",
    "expected_kernel_name": "vector_add_kernel",
    "expected_terms": ["@triton.jit", "tl.load", "tl.store", "tl.program_id"],
}

MODEL_SMALL = {
    "id": "small_qwen25_coder_1_5b",
    "display_name": "Qwen2.5-Coder-1.5B (Ollama local)",
    "tier": "small",
    "provider": "ollama",
}

MODEL_FRONTIER = {
    "id": "frontier_openai",
    "display_name": "GPT-4o (OpenAI)",
    "tier": "frontier",
    "provider": "openai",
}


def make_record(model, task, output, sample_index=1, latency=1.5):
    """Construye un registro JSONL sintético con el formato real de translation_outputs.jsonl."""
    return {
        "model": model,
        "task": task,
        "mode": "translation",
        "sample_index": sample_index,
        "prompt": "fake prompt",
        "output": output,
        "latency_seconds": latency,
    }


def write_jsonl(path: Path, records: list) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


# ---------------------------------------------------------------------------
# Tests: extract_code
# ---------------------------------------------------------------------------

class TestExtractCode:
    def test_extrae_bloque_python(self):
        raw = "Aquí está:\n```python\nx = 1\n```\nFin."
        assert extract_code(raw) == "x = 1"

    def test_extrae_bloque_sin_lenguaje(self):
        raw = "```\nx = 1\n```"
        assert extract_code(raw) == "x = 1"

    def test_sin_markdown_devuelve_texto_limpio(self):
        raw = "  x = 1  "
        assert extract_code(raw) == "x = 1"

    def test_kernel_completo_preservado(self):
        raw = f"```python\n{VALID_KERNEL}\n```"
        result = extract_code(raw)
        assert "@triton.jit" in result
        assert "tl.load" in result
        assert "tl.store" in result


# ---------------------------------------------------------------------------
# Tests: evaluate_generated_code
# ---------------------------------------------------------------------------

class TestEvaluateGeneratedCode:
    def test_kernel_valido_pasa_todo(self):
        result = evaluate_generated_code(VALID_KERNEL, VECTOR_ADD_TASK)
        assert result["syntax_valid"] is True
        assert result["kernel_shape_valid"] is True
        assert result["correctness_proxy"] is True

    def test_sintaxis_invalida_falla_todo(self):
        result = evaluate_generated_code(INVALID_SYNTAX_KERNEL, VECTOR_ADD_TASK)
        assert result["syntax_valid"] is False
        assert result["kernel_shape_valid"] is False
        assert result["correctness_proxy"] is False
        assert "syntax_error" in result["notes"]

    def test_sin_decorador_jit_falla_shape(self):
        result = evaluate_generated_code(INCOMPLETE_KERNEL, VECTOR_ADD_TASK)
        assert result["syntax_valid"] is True
        assert result["kernel_shape_valid"] is False

    def test_expected_terms_faltantes_falla_correctness(self):
        # Tiene estructura válida pero le falta tl.program_id
        kernel_sin_program_id = """\
import triton
import triton.language as tl

@triton.jit
def vector_add_kernel(x_ptr, y_ptr, z_ptr, N, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(z_ptr + offsets, x + y, mask=mask)
"""
        result = evaluate_generated_code(kernel_sin_program_id, VECTOR_ADD_TASK)
        assert result["syntax_valid"] is True
        assert result["kernel_shape_valid"] is True
        assert result["correctness_proxy"] is False  # falta tl.program_id

    def test_sin_expected_terms_definidos_pasa_correctness(self):
        task_sin_terms = {**VECTOR_ADD_TASK, "expected_terms": []}
        result = evaluate_generated_code(VALID_KERNEL, task_sin_terms)
        assert result["correctness_proxy"] is True


# ---------------------------------------------------------------------------
# Tests: load_manual_outputs
# ---------------------------------------------------------------------------

class TestLoadManualOutputs:
    def test_carga_registros_jsonl(self, tmp_path):
        records = [
            make_record(MODEL_SMALL, VECTOR_ADD_TASK, VALID_KERNEL, sample_index=1),
            make_record(MODEL_SMALL, VECTOR_ADD_TASK, VALID_KERNEL, sample_index=2),
        ]
        jsonl = tmp_path / "test.jsonl"
        write_jsonl(jsonl, records)

        loaded = load_manual_outputs(jsonl)
        assert len(loaded) == 2
        assert loaded[0]["model"]["id"] == "small_qwen25_coder_1_5b"
        assert loaded[1]["sample_index"] == 2

    def test_ignora_lineas_vacias(self, tmp_path):
        jsonl = tmp_path / "test.jsonl"
        jsonl.write_text(
            json.dumps(make_record(MODEL_SMALL, VECTOR_ADD_TASK, VALID_KERNEL)) + "\n\n\n",
            encoding="utf-8",
        )
        loaded = load_manual_outputs(jsonl)
        assert len(loaded) == 1


# ---------------------------------------------------------------------------
# Tests: run_model_evaluation — genera artefactos correctamente
# ---------------------------------------------------------------------------

class TestRunModelEvaluation:
    def _make_jsonl(self, tmp_path, records):
        jsonl = tmp_path / "translation_outputs.jsonl"
        write_jsonl(jsonl, records)
        return jsonl

    def test_falla_sin_manual_outputs(self, tmp_path):
        """Sin --manual-outputs el script debe salir con error."""
        with pytest.raises(SystemExit) as exc:
            run_model_evaluation(output_dir=tmp_path, manual_outputs=None)
        assert exc.value.code == 1

    def test_genera_tres_artefactos(self, tmp_path):
        records = [make_record(MODEL_FRONTIER, VECTOR_ADD_TASK, VALID_KERNEL)]
        jsonl = self._make_jsonl(tmp_path, records)

        run_model_evaluation(output_dir=tmp_path, manual_outputs=jsonl)

        assert (tmp_path / "model_eval_results.csv").exists()
        assert (tmp_path / "generated_kernels.jsonl").exists()
        assert (tmp_path / "model_eval_summary.md").exists()

    def test_csv_tiene_columnas_correctas(self, tmp_path):
        records = [make_record(MODEL_FRONTIER, VECTOR_ADD_TASK, VALID_KERNEL)]
        jsonl = self._make_jsonl(tmp_path, records)

        run_model_evaluation(output_dir=tmp_path, manual_outputs=jsonl)

        with (tmp_path / "model_eval_results.csv").open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            cols = set(reader.fieldnames or [])

        expected = {"model_id", "mode", "task_id", "syntax_valid", "kernel_shape_valid", "correctness_proxy"}
        assert expected.issubset(cols)

    def test_kernel_valido_pasa_checks_en_csv(self, tmp_path):
        records = [make_record(MODEL_FRONTIER, VECTOR_ADD_TASK, f"```python\n{VALID_KERNEL}\n```")]
        jsonl = self._make_jsonl(tmp_path, records)

        rows = run_model_evaluation(output_dir=tmp_path, manual_outputs=jsonl)

        assert rows[0]["syntax_valid"] is True
        assert rows[0]["kernel_shape_valid"] is True
        assert rows[0]["correctness_proxy"] is True

    def test_kernel_invalido_falla_checks_en_csv(self, tmp_path):
        records = [make_record(MODEL_SMALL, VECTOR_ADD_TASK, INVALID_SYNTAX_KERNEL)]
        jsonl = self._make_jsonl(tmp_path, records)

        rows = run_model_evaluation(output_dir=tmp_path, manual_outputs=jsonl)

        assert rows[0]["syntax_valid"] is False
        assert rows[0]["kernel_shape_valid"] is False

    def test_generated_kernels_tiene_extracted_code(self, tmp_path):
        raw_output = f"```python\n{VALID_KERNEL}\n```"
        records = [make_record(MODEL_FRONTIER, VECTOR_ADD_TASK, raw_output)]
        jsonl = self._make_jsonl(tmp_path, records)

        run_model_evaluation(output_dir=tmp_path, manual_outputs=jsonl)

        with (tmp_path / "generated_kernels.jsonl").open(encoding="utf-8") as f:
            kernel_record = json.loads(f.readline())

        assert "extracted_code" in kernel_record
        assert "@triton.jit" in kernel_record["extracted_code"]
        # El markdown fue eliminado
        assert "```" not in kernel_record["extracted_code"]

    def test_multiples_modelos_y_ejemplos(self, tmp_path):
        relu_task = {
            "id": "vector_relu",
            "pytorch_code": "z = torch.relu(x)",
            "operation_description": "ReLU activation",
            "expected_kernel_name": "relu_kernel",
            "expected_terms": ["@triton.jit", "tl.load", "tl.store", "tl.program_id"],
        }
        records = [
            make_record(MODEL_SMALL,    VECTOR_ADD_TASK, VALID_KERNEL,       sample_index=1),
            make_record(MODEL_SMALL,    VECTOR_ADD_TASK, VALID_KERNEL,       sample_index=2),
            make_record(MODEL_FRONTIER, VECTOR_ADD_TASK, VALID_KERNEL,       sample_index=1),
            make_record(MODEL_FRONTIER, relu_task,       INVALID_SYNTAX_KERNEL, sample_index=1),
        ]
        jsonl = self._make_jsonl(tmp_path, records)

        rows = run_model_evaluation(output_dir=tmp_path, manual_outputs=jsonl)

        assert len(rows) == 4

        with (tmp_path / "generated_kernels.jsonl").open(encoding="utf-8") as f:
            kernels = [json.loads(line) for line in f if line.strip()]
        assert len(kernels) == 4

    def test_summary_menciona_modelos(self, tmp_path):
        records = [
            make_record(MODEL_SMALL,    VECTOR_ADD_TASK, VALID_KERNEL),
            make_record(MODEL_FRONTIER, VECTOR_ADD_TASK, VALID_KERNEL),
        ]
        jsonl = self._make_jsonl(tmp_path, records)

        run_model_evaluation(output_dir=tmp_path, manual_outputs=jsonl)

        summary = (tmp_path / "model_eval_summary.md").read_text(encoding="utf-8")
        assert "small_qwen25_coder_1_5b" in summary
        assert "frontier_openai" in summary
        assert "PyTorch to Triton Translation" in summary

    def test_latencia_guardada_en_csv(self, tmp_path):
        records = [make_record(MODEL_FRONTIER, VECTOR_ADD_TASK, VALID_KERNEL, latency=3.7)]
        jsonl = self._make_jsonl(tmp_path, records)

        run_model_evaluation(output_dir=tmp_path, manual_outputs=jsonl)

        with (tmp_path / "model_eval_results.csv").open(encoding="utf-8") as f:
            row = list(csv.DictReader(f))[0]
        assert float(row["latency_seconds"]) == pytest.approx(3.7, abs=0.01)


# ---------------------------------------------------------------------------
# Tests: collect_real_outputs — build_translation_prompt y load_pytorch_examples
# ---------------------------------------------------------------------------

class TestCollectRealOutputs:
    def test_load_pytorch_examples_devuelve_lista(self):
        examples = load_pytorch_examples()
        assert isinstance(examples, list)
        assert len(examples) == 3

    def test_ejemplos_tienen_campos_requeridos(self):
        examples = load_pytorch_examples()
        for ex in examples:
            assert "id" in ex
            assert "pytorch_code" in ex
            assert "operation_description" in ex
            assert "expected_terms" in ex

    def test_ids_esperados_presentes(self):
        examples = load_pytorch_examples()
        ids = {ex["id"] for ex in examples}
        assert ids == {"vector_add", "vector_relu", "vector_scale"}

    def test_build_translation_prompt_rellena_pytorch_code(self):
        example = {
            "pytorch_code": "z = x + y",
            "operation_description": "element-wise addition",
        }
        prompt = build_translation_prompt(example)
        assert "z = x + y" in prompt
        assert "{pytorch_code}" not in prompt  # el placeholder fue reemplazado

    def test_build_translation_prompt_rellena_operation_description(self):
        example = {
            "pytorch_code": "z = torch.relu(x)",
            "operation_description": "ReLU activation",
        }
        prompt = build_translation_prompt(example)
        assert "ReLU activation" in prompt
        assert "{operation_description}" not in prompt

    def test_build_translation_prompt_contiene_instrucciones_triton(self):
        example = {
            "pytorch_code": "z = x + y",
            "operation_description": "element-wise addition",
        }
        prompt = build_translation_prompt(example)
        assert "@triton.jit" in prompt
        assert "tl.load" in prompt
        assert "tl.store" in prompt
        assert "tl.program_id" in prompt

    def test_build_translation_prompt_diferentes_ejemplos(self):
        examples = load_pytorch_examples()
        prompts = [build_translation_prompt(ex) for ex in examples]
        # Cada prompt es diferente porque el código PyTorch es diferente
        assert len(set(prompts)) == 3


# ---------------------------------------------------------------------------
# Tests: _rate helper
# ---------------------------------------------------------------------------

class TestRate:
    def test_todos_true(self):
        rows = [{"ok": True}, {"ok": True}, {"ok": True}]
        assert _rate(rows, "ok") == 1.0

    def test_todos_false(self):
        rows = [{"ok": False}, {"ok": False}]
        assert _rate(rows, "ok") == 0.0

    def test_mitad(self):
        rows = [{"ok": True}, {"ok": False}]
        assert _rate(rows, "ok") == 0.5

    def test_lista_vacia(self):
        assert _rate([], "ok") == 0.0

    def test_acepta_string_true(self):
        # El CSV guarda booleans como strings "True"/"False"
        rows = [{"ok": "True"}, {"ok": "False"}]
        assert _rate(rows, "ok") == 0.5
