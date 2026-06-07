from evaluation.code_safety import validate_generated_code_safety


def test_allows_normal_triton_module():
    code = """
import torch
import triton
import triton.language as tl

@triton.jit
def kernel(X):
    value = tl.load(X)
"""

    result = validate_generated_code_safety(code)

    assert result.safe is True
    assert result.errors == []


def test_rejects_filesystem_and_process_access():
    code = """
import os
import subprocess

open("/tmp/output", "w")
subprocess.run(["echo", "unsafe"])
"""

    result = validate_generated_code_safety(code)

    assert result.safe is False
    assert any("blocked import: os" in error for error in result.errors)
    assert any("blocked call: open" in error for error in result.errors)


def test_worker_environment_does_not_forward_api_keys(monkeypatch):
    from benchmarks.run_generated_kernels import _worker_environment

    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    monkeypatch.setenv("PATH", "/usr/bin")

    environment = _worker_environment()

    assert environment["PATH"] == "/usr/bin"
    assert "OPENAI_API_KEY" not in environment
