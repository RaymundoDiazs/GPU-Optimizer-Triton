from main import run_pipeline
from validation import gpu_validator


def test_validate_generated_kernel_skips_without_gpu_stack(monkeypatch):
    monkeypatch.setattr(gpu_validator, "_load_gpu_stack", lambda: (None, "CUDA GPU is not available"))

    result = gpu_validator.validate_generated_kernel("element_wise", "irrelevant")

    assert result["status"] == "not_run"
    assert result["gpu_available"] is False
    assert result["reason"] == "CUDA GPU is not available"


def test_run_pipeline_can_include_gpu_validation(monkeypatch):
    def fake_validate(problem_type, kernel_code):
        assert problem_type == "element_wise"
        assert "def launch_elementwise_add" in kernel_code
        return {
            "gpu_available": False,
            "status": "not_run",
            "problem_type": problem_type,
            "reason": "CUDA GPU is not available",
        }

    monkeypatch.setattr(gpu_validator, "validate_generated_kernel", fake_validate)

    output = run_pipeline("C = A + B", validate_gpu=True)

    assert "GPU validation:" in output
    assert '"status": "not_run"' in output
    assert '"problem_type": "element_wise"' in output
