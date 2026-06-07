from __future__ import annotations

import argparse
import inspect
import json
import linecache
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generation.shape_aware_selector import select_vector_launch_config
from evaluation.code_safety import validate_generated_code_safety


EXTRA_COLUMNS = ("triton_compiles", "triton_numerically_correct", "triton_speedup")
DEFAULT_NUM_ELEMENTS = 1_048_576
DEFAULT_WARMUP = 25
DEFAULT_REPEATS = 100
DEFAULT_FIXED_BLOCK_SIZE = 1024
DEFAULT_WORKER_TIMEOUT = 120
SCALE_ALPHA = 2.5


class KernelEvaluationError(RuntimeError):
    pass


def _jsonl_load(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def _jsonl_write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def _json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
        file.write("\n")


def _load_stack() -> tuple[dict[str, Any] | None, str]:
    try:
        import torch
    except Exception as exc:
        return None, f"torch import failed: {type(exc).__name__}: {exc}"

    try:
        import triton
        import triton.language as tl
    except Exception as exc:
        return None, f"triton import failed: {type(exc).__name__}: {exc}"

    if not torch.cuda.is_available():
        return None, "CUDA GPU is not available"

    torch.cuda.set_device(0)
    return {"torch": torch, "triton": triton, "tl": tl}, ""


def _probe_import_version(module_name: str) -> str | None:
    try:
        module = __import__(module_name)
    except Exception:
        return None
    return getattr(module, "__version__", None)


def _probe_host_gpus() -> list[str]:
    probes = [
        (
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            lambda output: [line.strip() for line in output.splitlines() if line.strip()],
        ),
        (
            ["system_profiler", "SPDisplaysDataType"],
            lambda output: [
                line.split(":", 1)[1].strip()
                for line in output.splitlines()
                if "Chipset Model:" in line
            ],
        ),
    ]
    for command, parser in probes:
        try:
            completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=10)
        except (OSError, subprocess.SubprocessError):
            continue
        if completed.returncode == 0:
            devices = parser(completed.stdout)
            if devices:
                return devices
    return []


def _stack_metadata(stack: dict[str, Any] | None, unavailable_reason: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "status": "ok" if stack is not None else "not_run",
        "unavailable_reason": unavailable_reason or None,
        "torch_version": _probe_import_version("torch"),
        "triton_version": _probe_import_version("triton"),
        "gpu_used": None,
        "host_gpus_detected": _probe_host_gpus(),
    }
    if stack is None:
        return payload

    torch = stack["torch"]
    triton = stack["triton"]
    payload.update(
        {
            "torch_version": getattr(torch, "__version__", None),
            "triton_version": getattr(triton, "__version__", None),
            "cuda_version": getattr(torch.version, "cuda", None),
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_used": torch.cuda.get_device_name(0),
            "gpu_capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
            "gpu_index": 0,
        }
    )
    return payload


def _exec_extracted_code(code: str, stack: dict[str, Any], row_number: int) -> dict[str, Any]:
    filename = f"<generated_kernel_{row_number}>"
    linecache.cache[filename] = (
        len(code),
        None,
        [line + "\n" for line in code.splitlines()],
        filename,
    )
    namespace = {
        "__name__": f"generated_kernel_{row_number}",
        "torch": stack["torch"],
        "triton": stack["triton"],
        "tl": stack["tl"],
    }
    exec(compile(code, filename, "exec"), namespace)
    return namespace


def _is_triton_jit_function(candidate: Any) -> bool:
    return callable(candidate) and hasattr(candidate, "run") and hasattr(candidate, "src")


def _arity(fn: Callable[..., Any]) -> int:
    try:
        return len(inspect.signature(fn).parameters)
    except (TypeError, ValueError):
        return 0


def _call_wrapper(fn: Callable[..., Any], *patterns: tuple[Any, ...]) -> Any:
    last_error: Exception | None = None
    for args in patterns:
        try:
            return fn(*args)
        except TypeError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    raise KernelEvaluationError("no call patterns provided")


def _launch_add(
    namespace: dict[str, Any],
    x: Any,
    y: Any,
    stack: dict[str, Any],
    *,
    launch_policy: str,
    fixed_block_size: int,
) -> Any:
    torch = stack["torch"]
    triton = stack["triton"]
    n = x.numel()
    launch_config = select_vector_launch_config(n, launch_policy, fixed_block_size)
    block_size = launch_config.block_size

    add = namespace.get("add")
    if callable(add) and not _is_triton_jit_function(add):
        if _arity(add) <= 2:
            return add(x, y)
        z = torch.empty_like(x)
        result = _call_wrapper(add, (x, y, z, n, block_size), (x, y, z, n))
        return z if result is None else result

    kernel = namespace.get("add_kernel")
    if _is_triton_jit_function(kernel):
        z = torch.empty_like(x)
        grid = launch_config.grid
        kernel[grid](x, y, z, n, BLOCK_SIZE=block_size)
        return z

    add_vectors = namespace.get("add_vectors")
    if _is_triton_jit_function(add_vectors):
        z = torch.empty_like(x)
        grid = (triton.cdiv(n, 256),)
        add_vectors[grid](x, y, z, n)
        return z

    raise KernelEvaluationError("no supported vector_add wrapper or kernel found")


def _launch_relu(
    namespace: dict[str, Any],
    x: Any,
    stack: dict[str, Any],
    *,
    launch_policy: str,
    fixed_block_size: int,
) -> Any:
    torch = stack["torch"]
    triton = stack["triton"]
    n = x.numel()
    launch_config = select_vector_launch_config(n, launch_policy, fixed_block_size)
    block_size = launch_config.block_size

    for name in ("relu", "relu_triton"):
        wrapper = namespace.get(name)
        if callable(wrapper) and not _is_triton_jit_function(wrapper):
            return wrapper(x)

    kernel = namespace.get("relu_kernel")
    if _is_triton_jit_function(kernel):
        z = torch.empty_like(x)
        grid = launch_config.grid
        try:
            kernel[grid](x, z, n, BLOCK_SIZE=block_size)
        except TypeError:
            kernel[grid](x, z, BLOCK_SIZE=block_size)
        return z

    raise KernelEvaluationError("no supported vector_relu wrapper or kernel found")


def _launch_scale(
    namespace: dict[str, Any],
    x: Any,
    stack: dict[str, Any],
    *,
    launch_policy: str,
    fixed_block_size: int,
) -> Any:
    torch = stack["torch"]
    triton = stack["triton"]
    n = x.numel()
    launch_config = select_vector_launch_config(n, launch_policy, fixed_block_size)
    block_size = launch_config.block_size
    alpha = SCALE_ALPHA

    for name in ("scale", "scalar_multiply"):
        wrapper = namespace.get(name)
        if callable(wrapper) and not _is_triton_jit_function(wrapper):
            return wrapper(x, alpha)

    multiply_vector = namespace.get("multiply_vector")
    if callable(multiply_vector) and not _is_triton_jit_function(multiply_vector):
        z = torch.empty_like(x)
        result = _call_wrapper(
            multiply_vector,
            (x, alpha, z, n, block_size),
            (x, alpha, z, n),
        )
        return z if result is None else result

    for name in ("scale_kernel", "scalar_multiply_kernel", "kernel"):
        kernel = namespace.get(name)
        if _is_triton_jit_function(kernel):
            z = torch.empty_like(x)
            grid = launch_config.grid
            kernel[grid](x, z, alpha, n, BLOCK_SIZE=block_size)
            return z

    raw_multiply_vector = namespace.get("multiply_vector")
    if _is_triton_jit_function(raw_multiply_vector):
        z = torch.empty_like(x)
        offsets = torch.arange(n, device=x.device, dtype=torch.int64)
        grid = (triton.cdiv(n, 128),)
        raw_multiply_vector[grid](alpha, x, z, offsets)
        return z

    raise KernelEvaluationError("no supported vector_scale wrapper or kernel found")


def _make_inputs(task_id: str, torch: Any, num_elements: int, seed: int) -> tuple[tuple[Any, ...], Any]:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)

    if task_id == "vector_add":
        x = torch.randn(num_elements, device="cuda", dtype=torch.float32, generator=generator)
        y = torch.randn(num_elements, device="cuda", dtype=torch.float32, generator=generator)
        return (x, y), x + y

    if task_id == "vector_relu":
        x = torch.randn(num_elements, device="cuda", dtype=torch.float32, generator=generator)
        return (x,), torch.relu(x)

    if task_id == "vector_scale":
        x = torch.randn(num_elements, device="cuda", dtype=torch.float32, generator=generator)
        return (x,), x * SCALE_ALPHA

    raise KernelEvaluationError(f"unsupported task_id: {task_id}")


def _launch_task(
    namespace: dict[str, Any],
    task_id: str,
    inputs: tuple[Any, ...],
    stack: dict[str, Any],
    *,
    launch_policy: str,
    fixed_block_size: int,
) -> Any:
    if task_id == "vector_add":
        return _launch_add(
            namespace,
            inputs[0],
            inputs[1],
            stack,
            launch_policy=launch_policy,
            fixed_block_size=fixed_block_size,
        )
    if task_id == "vector_relu":
        return _launch_relu(
            namespace,
            inputs[0],
            stack,
            launch_policy=launch_policy,
            fixed_block_size=fixed_block_size,
        )
    if task_id == "vector_scale":
        return _launch_scale(
            namespace,
            inputs[0],
            stack,
            launch_policy=launch_policy,
            fixed_block_size=fixed_block_size,
        )
    raise KernelEvaluationError(f"unsupported task_id: {task_id}")


def _reference_callable(task_id: str, inputs: tuple[Any, ...]) -> Callable[[], Any]:
    if task_id == "vector_add":
        return lambda: inputs[0] + inputs[1]
    if task_id == "vector_relu":
        return lambda: inputs[0].relu()
    if task_id == "vector_scale":
        return lambda: inputs[0] * SCALE_ALPHA
    raise KernelEvaluationError(f"unsupported task_id: {task_id}")


def _median_cuda_ms(fn: Callable[[], Any], torch: Any, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    timings: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        timings.append(float(start.elapsed_time(end)))
    torch.cuda.synchronize()
    return statistics.median(timings)


def _evaluate_row_in_process(
    row: dict[str, Any],
    row_number: int,
    stack: dict[str, Any],
    *,
    num_elements: int,
    warmup: int,
    repeats: int,
    launch_policy: str,
    fixed_block_size: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    torch = stack["torch"]
    diagnostic: dict[str, Any] = {
        "row_number": row_number,
        "task_id": row.get("task_id"),
        "sample_index": row.get("sample_index"),
        "status": "unknown",
        "reason": None,
        "max_abs_error": None,
        "pytorch_ms": None,
        "triton_ms": None,
        "shape_aware_block_size": None,
        "shape_aware_grid": None,
        "shape_aware_reason": None,
        "launch_policy": launch_policy,
    }
    enriched = dict(row)

    try:
        code = str(row["extracted_code"])
        safety = validate_generated_code_safety(code)
        if not safety.safe:
            raise KernelEvaluationError("; ".join(safety.errors))
        namespace = _exec_extracted_code(code, stack, row_number)
        inputs, expected = _make_inputs(str(row["task_id"]), torch, num_elements, seed=row_number)
        if str(row["task_id"]) in {"vector_add", "vector_relu", "vector_scale"}:
            launch_config = select_vector_launch_config(num_elements, launch_policy, fixed_block_size)
            diagnostic.update(
                {
                    "shape_aware_block_size": launch_config.block_size,
                    "shape_aware_grid": launch_config.grid[0],
                    "shape_aware_reason": launch_config.reason,
                }
            )
        actual = _launch_task(
            namespace,
            str(row["task_id"]),
            inputs,
            stack,
            launch_policy=launch_policy,
            fixed_block_size=fixed_block_size,
        )
        torch.cuda.synchronize()
    except Exception as exc:
        enriched.update(
            {
                "triton_compiles": False,
                "triton_numerically_correct": False,
                "triton_speedup": None,
            }
        )
        diagnostic.update(
            {
                "status": "compile_or_launch_failed",
                "reason": f"{type(exc).__name__}: {exc}",
            }
        )
        return enriched, diagnostic

    try:
        if actual is None:
            raise KernelEvaluationError("kernel returned None and no output tensor was available")
        if tuple(actual.shape) != tuple(expected.shape):
            raise KernelEvaluationError(f"shape mismatch: actual={tuple(actual.shape)} expected={tuple(expected.shape)}")
        diff = (actual - expected).abs()
        diagnostic["max_abs_error"] = 0.0 if diff.numel() == 0 else float(diff.max().detach().cpu())
        correct = bool(torch.allclose(actual, expected, atol=1e-5))
    except Exception as exc:
        correct = False
        diagnostic["reason"] = f"{type(exc).__name__}: {exc}"

    triton_ms: float | None = None
    pytorch_ms: float | None = None
    speedup: float | None = None
    if correct:
        try:
            triton_fn = lambda: _launch_task(
                namespace,
                str(row["task_id"]),
                inputs,
                stack,
                launch_policy=launch_policy,
                fixed_block_size=fixed_block_size,
            )
            torch_fn = _reference_callable(str(row["task_id"]), inputs)
            triton_ms = _median_cuda_ms(triton_fn, torch, warmup, repeats)
            pytorch_ms = _median_cuda_ms(torch_fn, torch, warmup, repeats)
            speedup = None if triton_ms == 0 else pytorch_ms / triton_ms
        except Exception as exc:
            diagnostic["reason"] = f"benchmark failed: {type(exc).__name__}: {exc}"

    enriched.update(
        {
            "triton_compiles": True,
            "triton_numerically_correct": correct,
            "triton_speedup": None if speedup is None else round(float(speedup), 6),
        }
    )
    diagnostic.update(
        {
            "status": "ok" if correct else "incorrect",
            "pytorch_ms": pytorch_ms,
            "triton_ms": triton_ms,
        }
    )
    if diagnostic["reason"] is None:
        diagnostic["reason"] = "outputs match" if correct else "outputs differ from PyTorch reference"
    return enriched, diagnostic


def _worker_environment() -> dict[str, str]:
    import os

    allowed = {
        "CUDA_CACHE_PATH",
        "CUDA_HOME",
        "CUDA_VISIBLE_DEVICES",
        "HOME",
        "LD_LIBRARY_PATH",
        "PATH",
        "PYTHONPATH",
        "TRITON_CACHE_DIR",
    }
    environment = {key: value for key, value in os.environ.items() if key in allowed}
    existing_pythonpath = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = (
        str(ROOT)
        if not existing_pythonpath
        else str(ROOT) + os.pathsep + existing_pythonpath
    )
    return environment


def _evaluate_row_isolated(
    row: dict[str, Any],
    row_number: int,
    *,
    num_elements: int,
    warmup: int,
    repeats: int,
    launch_policy: str,
    fixed_block_size: int,
    worker_timeout: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    request = {
        "row": row,
        "row_number": row_number,
        "num_elements": num_elements,
        "warmup": warmup,
        "repeats": repeats,
        "launch_policy": launch_policy,
        "fixed_block_size": fixed_block_size,
    }
    with tempfile.TemporaryDirectory(prefix="triton-kernel-worker-") as temp_dir:
        temp_path = Path(temp_dir)
        request_path = temp_path / "request.json"
        response_path = temp_path / "response.json"
        _json_write(request_path, request)
        command = [
            sys.executable,
            "-m",
            "benchmarks.kernel_worker",
            str(request_path),
            str(response_path),
        ]
        try:
            completed = subprocess.run(
                command,
                cwd=temp_path,
                env=_worker_environment(),
                capture_output=True,
                text=True,
                timeout=worker_timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return _worker_failure(row, row_number, f"worker timeout after {worker_timeout}s")

        if completed.returncode != 0 or not response_path.exists():
            reason = completed.stderr.strip() or completed.stdout.strip() or "worker failed without output"
            return _worker_failure(row, row_number, reason[-1000:])

        with response_path.open("r", encoding="utf-8") as file:
            response = json.load(file)
        return response["enriched"], response["diagnostic"]


def _worker_failure(
    row: dict[str, Any],
    row_number: int,
    reason: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    enriched = dict(row)
    enriched.update(
        {
            "triton_compiles": False,
            "triton_numerically_correct": False,
            "triton_speedup": None,
        }
    )
    diagnostic = {
        "row_number": row_number,
        "task_id": row.get("task_id"),
        "sample_index": row.get("sample_index"),
        "status": "isolated_worker_failed",
        "reason": reason,
        "max_abs_error": None,
        "pytorch_ms": None,
        "triton_ms": None,
    }
    return enriched, diagnostic


def evaluate_file(
    input_path: Path,
    output_path: Path,
    metadata_path: Path,
    diagnostics_path: Path,
    *,
    num_elements: int,
    warmup: int,
    repeats: int,
    launch_policy: str = "shape-aware",
    fixed_block_size: int = DEFAULT_FIXED_BLOCK_SIZE,
    worker_timeout: int = DEFAULT_WORKER_TIMEOUT,
) -> None:
    rows = _jsonl_load(input_path)
    stack, unavailable_reason = _load_stack()
    metadata = _stack_metadata(stack, unavailable_reason)
    metadata.update(
        {
            "input_path": str(input_path),
            "output_path": str(output_path),
            "diagnostics_path": str(diagnostics_path),
            "num_records": len(rows),
            "num_elements": num_elements,
            "warmup": warmup,
            "repeats": repeats,
            "allclose": "torch.allclose(actual, expected, atol=1e-5)",
            "speedup": "median_pytorch_ms / median_triton_ms",
            "launch_policy": launch_policy,
            "fixed_block_size": fixed_block_size,
            "worker_timeout_seconds": worker_timeout,
            "execution_isolation": "one subprocess per generated kernel",
            "shape_aware_policy": "generation.shape_aware_selector.select_vector_launch_config",
        }
    )

    diagnostics: list[dict[str, Any]] = []
    if stack is None:
        enriched_rows = []
        for row_number, row in enumerate(rows, start=1):
            enriched = dict(row)
            enriched.update({column: None for column in EXTRA_COLUMNS})
            enriched_rows.append(enriched)
            diagnostics.append(
                {
                    "row_number": row_number,
                    "task_id": row.get("task_id"),
                    "sample_index": row.get("sample_index"),
                    "status": "not_run",
                    "reason": unavailable_reason,
                    "max_abs_error": None,
                    "pytorch_ms": None,
                    "triton_ms": None,
                }
            )
    else:
        enriched_rows = []
        for row_number, row in enumerate(rows, start=1):
            enriched, diagnostic = _evaluate_row_isolated(
                row,
                row_number,
                num_elements=num_elements,
                warmup=warmup,
                repeats=repeats,
                launch_policy=launch_policy,
                fixed_block_size=fixed_block_size,
                worker_timeout=worker_timeout,
            )
            enriched_rows.append(enriched)
            diagnostics.append(diagnostic)

        metadata["summary"] = {
            "compiled": sum(row["triton_compiles"] is True for row in enriched_rows),
            "correct": sum(row["triton_numerically_correct"] is True for row in enriched_rows),
            "speedups_recorded": sum(row["triton_speedup"] is not None for row in enriched_rows),
        }

    _jsonl_write(output_path, enriched_rows)
    _jsonl_write(diagnostics_path, diagnostics)
    _json_write(metadata_path, metadata)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate generated Triton kernels from JSONL.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/generated_kernels_tritonbench.jsonl"))
    parser.add_argument("--metadata", type=Path, default=Path("artifacts/generated_kernels_tritonbench_metadata.json"))
    parser.add_argument("--diagnostics", type=Path, default=Path("artifacts/generated_kernels_tritonbench_diagnostics.jsonl"))
    parser.add_argument("--num-elements", type=int, default=DEFAULT_NUM_ELEMENTS)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--launch-policy", choices=("shape-aware", "fixed"), default="shape-aware")
    parser.add_argument("--fixed-block-size", type=int, default=DEFAULT_FIXED_BLOCK_SIZE)
    parser.add_argument("--worker-timeout", type=int, default=DEFAULT_WORKER_TIMEOUT)
    args = parser.parse_args()

    evaluate_file(
        args.input,
        args.output,
        args.metadata,
        args.diagnostics,
        num_elements=args.num_elements,
        warmup=args.warmup,
        repeats=args.repeats,
        launch_policy=args.launch_policy,
        fixed_block_size=args.fixed_block_size,
        worker_timeout=args.worker_timeout,
    )
    print(f"Wrote evaluated JSONL to {args.output}")
    print(f"Wrote reproducibility metadata to {args.metadata}")
    print(f"Wrote diagnostics to {args.diagnostics}")


if __name__ == "__main__":
    main()
