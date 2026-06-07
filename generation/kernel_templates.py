ELEMENTWISE_TEMPLATE = '''
import triton
import triton.language as tl
import torch

@triton.jit
def elementwise_kernel(X, Y, Z, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(X + offsets, mask=mask)
    y = tl.load(Y + offsets, mask=mask)
    tl.store(Z + offsets, x + y, mask=mask)

def _require_cuda_tensor(name, tensor):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be allocated on a CUDA device")
    return tensor.contiguous()

def launch_elementwise_add(X, Y, block_size=1024):
    X = _require_cuda_tensor("X", X)
    Y = _require_cuda_tensor("Y", Y)
    if X.shape != Y.shape:
        raise ValueError("X and Y must have the same shape")

    Z = torch.empty_like(X)
    N = Z.numel()
    if N == 0:
        return Z

    grid = (triton.cdiv(N, block_size),)
    elementwise_kernel[grid](X, Y, Z, N, BLOCK_SIZE=block_size)
    return Z
'''

REDUCTION_TEMPLATE = '''
import triton
import triton.language as tl
import torch

@triton.jit
def reduction_kernel(X, partials, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(X + offsets, mask=mask, other=0.0)
    acc = tl.sum(x, axis=0)
    tl.store(partials + pid, acc)

def _require_cuda_tensor(name, tensor):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be allocated on a CUDA device")
    return tensor.contiguous()

def launch_reduction_sum(X, block_size=1024):
    X = _require_cuda_tensor("X", X)
    N = X.numel()
    if N == 0:
        return torch.zeros((), device=X.device, dtype=X.dtype)

    num_blocks = triton.cdiv(N, block_size)
    partials = torch.empty((num_blocks,), device=X.device, dtype=X.dtype)
    grid = (num_blocks,)
    reduction_kernel[grid](X, partials, N, BLOCK_SIZE=block_size)
    return partials.sum()
'''

MATMUL_TEMPLATE = '''
import triton
import triton.language as tl
import torch

@triton.jit
def matmul_kernel(
    A,
    B,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + offs_k
        a = tl.load(
            A + offs_m[:, None] * K + k_offsets[None, :],
            mask=(offs_m[:, None] < M) & (k_offsets[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            B + k_offsets[:, None] * N + offs_n[None, :],
            mask=(k_offsets[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        )
        acc += tl.dot(a, b)

    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(C + offs_m[:, None] * N + offs_n[None, :], acc, mask=mask)

def _require_cuda_tensor(name, tensor):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be allocated on a CUDA device")
    return tensor.contiguous()

def launch_matmul(A, B, block_m=16, block_n=16, block_k=32):
    A = _require_cuda_tensor("A", A)
    B = _require_cuda_tensor("B", B)
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("A and B must be 2D tensors")
    if A.shape[1] != B.shape[0]:
        raise ValueError("A.shape[1] must match B.shape[0]")
    if A.device != B.device:
        raise ValueError("A and B must be on the same CUDA device")
    if A.dtype != B.dtype:
        raise ValueError("A and B must use the same dtype")
    if A.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("launch_matmul supports float16, bfloat16, and float32 tensors")

    M, K = A.shape
    _, N = B.shape
    C = torch.empty((M, N), device=A.device, dtype=A.dtype)
    if M == 0 or N == 0:
        return C

    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))
    matmul_kernel[grid](
        A,
        B,
        C,
        M,
        N,
        K,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
    )
    return C
'''

GENERIC_TEMPLATE = '''
# Generic Triton kernel for {problem_type}
# Derived from XGrammar representation:
{grammar_comment}

import triton
import triton.language as tl
import torch

@triton.jit
def generic_kernel(X, Y, Z, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(X + offsets, mask=mask)
    y = tl.load(Y + offsets, mask=mask)
    tl.store(Z + offsets, x + y, mask=mask)

def _require_cuda_tensor(name, tensor):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(name + " must be a torch.Tensor")
    if not tensor.is_cuda:
        raise ValueError(name + " must be allocated on a CUDA device")
    return tensor.contiguous()

def launch_generic(X, Y, block_size=1024):
    X = _require_cuda_tensor("X", X)
    Y = _require_cuda_tensor("Y", Y)
    if X.shape != Y.shape:
        raise ValueError("X and Y must have the same shape")

    Z = torch.empty_like(X)
    N = Z.numel()
    if N == 0:
        return Z

    grid = (triton.cdiv(N, block_size),)
    generic_kernel[grid](X, Y, Z, N, BLOCK_SIZE=block_size)
    return Z
'''
