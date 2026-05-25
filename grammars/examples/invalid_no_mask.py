import triton
import triton.language as tl


@triton.jit
def vector_add_kernel(X, Y, Z, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(X + offsets)
    y = tl.load(Y + offsets)
    tl.store(Z + offsets, x + y)

