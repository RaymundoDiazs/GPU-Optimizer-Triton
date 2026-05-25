from __future__ import annotations

import linecache
import time
from typing import Any, Callable


Validator = Callable[[dict[str, Any], Any, int], dict[str, Any]]


def _load_gpu_stack() -> tuple[dict[str, Any] | None, str]:
    try:
        import torch
    except ImportError:
        return None, "torch is not installed"

    try:
        import triton  # noqa: F401
        import triton.language as tl  # noqa: F401
    except ImportError:
        return None, "triton is not installed"

    if not torch.cuda.is_available():
        return None, "CUDA GPU is not available"

    return {"torch": torch}, ""


def _exec_generated_code(kernel_code: str) -> dict[str, Any]:
    filename = "<generated_triton_kernel>"
    linecache.cache[filename] = (
        len(kernel_code),
        None,
        [line + "\n" for line in kernel_code.splitlines()],
        filename,
    )
    namespace: dict[str, Any] = {"__name__": "generated_triton_kernel"}
    exec(compile(kernel_code, filename, "exec"), namespace)
    return namespace


def _get_wrapper(namespace: dict[str, Any], *names: str) -> tuple[str, Callable[..., Any]]:
    for name in names:
        candidate = namespace.get(name)
        if callable(candidate):
            return name, candidate
    raise ValueError(f"missing launch wrapper; expected one of: {', '.join(names)}")


def _compare_outputs(torch: Any, actual: Any, expected: Any, *, rtol: float, atol: float) -> dict[str, Any]:
    if actual.shape != expected.shape:
        return {
            "correct": False,
            "reason": f"shape mismatch: actual={tuple(actual.shape)} expected={tuple(expected.shape)}",
            "max_abs_error": None,
        }

    diff = (actual - expected).abs()
    max_abs_error = 0.0 if diff.numel() == 0 else float(diff.max().detach().cpu())
    close = torch.isclose(actual, expected, rtol=rtol, atol=atol)
    correct = bool(close.all().detach().cpu())
    return {
        "correct": correct,
        "reason": "outputs match" if correct else "outputs differ from PyTorch reference",
        "max_abs_error": max_abs_error,
    }


def _validate_elementwise(namespace: dict[str, Any], torch: Any, size: int) -> dict[str, Any]:
    wrapper_name, wrapper = _get_wrapper(namespace, "launch_elementwise_add")
    x = torch.randn(size, device="cuda", dtype=torch.float32)
    y = torch.randn(size, device="cuda", dtype=torch.float32)
    actual = wrapper(x, y)
    expected = x + y
    comparison = _compare_outputs(torch, actual, expected, rtol=1e-5, atol=1e-6)
    return {
        **comparison,
        "wrapper": wrapper_name,
        "input_shape": [size],
        "reference": "torch add",
    }


def _validate_generic(namespace: dict[str, Any], torch: Any, size: int) -> dict[str, Any]:
    wrapper_name, wrapper = _get_wrapper(namespace, "launch_generic")
    x = torch.randn(size, device="cuda", dtype=torch.float32)
    y = torch.randn(size, device="cuda", dtype=torch.float32)
    actual = wrapper(x, y)
    expected = x + y
    comparison = _compare_outputs(torch, actual, expected, rtol=1e-5, atol=1e-6)
    return {
        **comparison,
        "wrapper": wrapper_name,
        "input_shape": [size],
        "reference": "torch add",
    }


def _validate_reduction(namespace: dict[str, Any], torch: Any, size: int) -> dict[str, Any]:
    wrapper_name, wrapper = _get_wrapper(namespace, "launch_reduction_sum")
    x = torch.randn(size, device="cuda", dtype=torch.float32)
    actual = wrapper(x)
    expected = torch.sum(x)
    comparison = _compare_outputs(torch, actual.reshape(()), expected.reshape(()), rtol=1e-4, atol=1e-3)
    return {
        **comparison,
        "wrapper": wrapper_name,
        "input_shape": [size],
        "reference": "torch.sum",
    }


def _validate_matmul(namespace: dict[str, Any], torch: Any, size: int) -> dict[str, Any]:
    del size
    wrapper_name, wrapper = _get_wrapper(namespace, "launch_matmul")
    m, k, n = 31, 37, 29
    a = torch.randn((m, k), device="cuda", dtype=torch.float16)
    b = torch.randn((k, n), device="cuda", dtype=torch.float16)
    actual = wrapper(a, b)
    expected = a @ b
    comparison = _compare_outputs(torch, actual, expected, rtol=1e-2, atol=1e-1)
    return {
        **comparison,
        "wrapper": wrapper_name,
        "input_shape": [m, k, n],
        "reference": "torch matmul",
    }


VALIDATORS: dict[str, Validator] = {
    "element_wise": _validate_elementwise,
    "reduction": _validate_reduction,
    "matrix_operation": _validate_matmul,
    "generic": _validate_generic,
}


def validate_generated_kernel(
    problem_type: str,
    kernel_code: str,
    *,
    size: int = 4097,
    seed: int = 0,
) -> dict[str, Any]:
    """Compile and run a generated Triton wrapper on CUDA, comparing with PyTorch."""
    stack, unavailable_reason = _load_gpu_stack()
    if stack is None:
        return {
            "gpu_available": False,
            "status": "not_run",
            "problem_type": problem_type,
            "reason": unavailable_reason,
        }

    torch = stack["torch"]
    validator = VALIDATORS.get(problem_type, _validate_generic)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    try:
        namespace = _exec_generated_code(kernel_code)
        torch.cuda.synchronize()
        start = time.perf_counter()
        details = validator(namespace, torch, size)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
    except Exception as exc:
        return {
            "gpu_available": True,
            "status": "failed",
            "problem_type": problem_type,
            "device_name": torch.cuda.get_device_name(0),
            "reason": f"{type(exc).__name__}: {exc}",
        }

    return {
        "gpu_available": True,
        "status": "ok" if details["correct"] else "failed",
        "problem_type": problem_type,
        "device_name": torch.cuda.get_device_name(0),
        "elapsed_time": elapsed,
        **details,
    }
