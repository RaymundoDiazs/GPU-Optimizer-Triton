ELEMENTWISE_TEMPLATE = '''
import triton
import triton.language as tl

@triton.jit
def elementwise_kernel(X, Y, Z, N):
    pid = tl.program_id(0)
    offsets = pid * 1024 + tl.arange(0, 1024)
    mask = offsets < N
    x = tl.load(X + offsets, mask=mask)
    y = tl.load(Y + offsets, mask=mask)
    tl.store(Z + offsets, x + y, mask=mask)
'''

REDUCTION_TEMPLATE = '''
import triton
import triton.language as tl

@triton.jit
def reduction_kernel(X, output, N):
    pid = tl.program_id(0)
    offsets = pid * 1024 + tl.arange(0, 1024)
    mask = offsets < N
    x = tl.load(X + offsets, mask=mask)
    acc = tl.sum(x, axis=0)
    tl.store(output + pid, acc)
'''

MATMUL_TEMPLATE = '''
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(A, B, C, M, N, K):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    block_m = 128
    block_n = 128
    block_k = 32

    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_n = pid_n * block_n + tl.arange(0, block_n)

    A_block = tl.load(A + offs_m[:, None] * K + tl.arange(0, block_k)[None, :])
    B_block = tl.load(B + tl.arange(0, block_k)[:, None] * N + offs_n[None, :])
    C_block = tl.dot(A_block, B_block)
    tl.store(C + offs_m[:, None] * N + offs_n[None, :], C_block)
'''

GENERIC_TEMPLATE = '''
# Generic Triton kernel for {problem_type}
# Derived from XGrammar representation:
# {grammar}

import triton
import triton.language as tl

@triton.jit
def generic_kernel(X, Y, Z, N):
    pid = tl.program_id(0)
    offsets = pid * 1024 + tl.arange(0, 1024)
    mask = offsets < N
    x = tl.load(X + offsets, mask=mask)
    y = tl.load(Y + offsets, mask=mask)
    tl.store(Z + offsets, x + y, mask=mask)
'''
