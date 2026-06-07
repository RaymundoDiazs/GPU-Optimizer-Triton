import time
from typing import Any


def profile_gpu() -> dict[str, Any]:
    """Profile a minimal Triton kernel when GPU dependencies are available."""
    try:
        import torch
    except ImportError:
        return {
            "gpu_available": False,
            "status": "not_run",
            "reason": "torch is not installed",
        }

    try:
        import triton
        import triton.language as tl
    except ImportError:
        return {
            "gpu_available": False,
            "status": "not_run",
            "reason": "triton is not installed",
        }

    if not torch.cuda.is_available():
        return {
            "gpu_available": False,
            "status": "not_run",
            "reason": "CUDA GPU is not available",
        }

    @triton.jit
    def elementwise_kernel(X, Y, Z, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        x = tl.load(X + offsets, mask=mask)
        y = tl.load(Y + offsets, mask=mask)
        tl.store(Z + offsets, x + y, mask=mask)

    size = 1024
    x = torch.rand(size, device="cuda")
    y = torch.rand(size, device="cuda")
    z = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(size, meta["BLOCK_SIZE"]),)

    torch.cuda.synchronize()
    start = time.perf_counter()
    elementwise_kernel[grid](x, y, z, size, BLOCK_SIZE=1024)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return {
        "gpu_available": True,
        "status": "ok",
        "device_name": torch.cuda.get_device_name(0),
        "elapsed_time": elapsed,
    }
