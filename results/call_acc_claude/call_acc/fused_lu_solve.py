import torch
import triton
import triton.language as tl
from torch import Tensor

# Triton Kernel for Forward Substitution (Ly = Pb)
@triton.jit
def forward_substitution_kernel(
    y_ptr, L_ptr, b_ptr, P_ptr,
    n, stride_y, stride_L, stride_b, stride_P,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Solves Ly = Pb using forward substitution.
    y_ptr: output vector y
    L_ptr: lower triangular matrix L
    b_ptr: right-hand side vector b
    P_ptr: permutation matrix P
    """
    pid = tl.program_id(0)
    
    for i in range(pid * BLOCK_SIZE, n, BLOCK_SIZE * tl.num_programs(0)):
        if i < n:
            # Compute Pb (apply permutation)
            pb = tl.load(b_ptr + tl.cast(tl.load(P_ptr + i * stride_P), tl.int32) * stride_b)
            
            # Forward substitution: y[i] = (Pb[i] - sum(L[i,j]*y[j])) / L[i,i]
            sum_val = 0.0
            for j in range(i):
                L_ij = tl.load(L_ptr + i * stride_L + j)
                y_j = tl.load(y_ptr + j * stride_y)
                sum_val += L_ij * y_j
            
            L_ii = tl.load(L_ptr + i * stride_L + i)
            y_i = (pb - sum_val) / L_ii
            tl.store(y_ptr + i * stride_y, y_i)


# Triton Kernel for Backward Substitution (Ux = y)
@triton.jit
def backward_substitution_kernel(
    x_ptr, U_ptr, y_ptr,
    n, stride_x, stride_U, stride_y,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Solves Ux = y using backward substitution.
    x_ptr: output vector x
    U_ptr: upper triangular matrix U
    y_ptr: right-hand side vector y
    """
    pid = tl.program_id(0)
    
    for i in range(n - 1 - pid * BLOCK_SIZE, -1, -BLOCK_SIZE * tl.num_programs(0)):
        if i >= 0:
            # Backward substitution: x[i] = (y[i] - sum(U[i,j]*x[j])) / U[i,i]
            sum_val = 0.0
            for j in range(i + 1, n):
                U_ij = tl.load(U_ptr + i * stride_U + j)
                x_j = tl.load(x_ptr + j * stride_x)
                sum_val += U_ij * x_j
            
            U_ii = tl.load(U_ptr + i * stride_U + i)
            x_i = (tl.load(y_ptr + i * stride_y) - sum_val) / U_ii
            tl.store(x_ptr + i * stride_x, x_i)


# Wrapper Function
def fused_lu_solve(A: Tensor, b: Tensor) -> Tensor:
    """
    Solves the linear system Ax = b using LU decomposition.
    
    Args:
        A (Tensor): The input matrix of shape (n, n).
        b (Tensor): The right-hand side tensor of shape (n,).
        
    Returns:
        Tensor: The solution tensor x of shape (n,).
    """
    assert A.dim() == 2 and A.shape[0] == A.shape[1], "A must be a square matrix"
    assert b.dim() == 1 and b.shape[0] == A.shape[0], "b must be a 1D vector matching A's size"
    assert A.is_cuda and b.is_cuda, "Inputs must be on CUDA device"
    
    n = A.shape[0]
    device = A.device
    dtype = A.dtype
    
    # Perform LU decomposition using PyTorch
    P, L, U = torch.linalg.lu(A)
    
    # Initialize intermediate vector y for Ly = Pb
    y = torch.zeros(n, dtype=dtype, device=device)
    
    # Initialize output vector x
    x = torch.zeros(n, dtype=dtype, device=device)
    
    # Block size for Triton kernels
    BLOCK_SIZE = 32
    
    # Forward substitution: Ly = Pb
    forward_substitution_kernel[
        (triton.cdiv(n, BLOCK_SIZE),)
    ](
        y, L, b, P,
        n,
        y.stride(0), L.stride(0), b.stride(0), P.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Backward substitution: Ux = y
    backward_substitution_kernel[
        (triton.cdiv(n, BLOCK_SIZE),)
    ](
        x, U, y,
        n,
        x.stride(0), U.stride(0), y.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return x

##################################################################################################################################################



def test_fused_lu_solve():
    results = {}
    
    # Test case 1: Simple 2x2 system
    A1 = torch.tensor([[3.0, 1.0], [1.0, 2.0]], device='cuda')
    b1 = torch.tensor([9.0, 8.0], device='cuda')
    results["test_case_1"] = fused_lu_solve(A1, b1)
    
    # Test case 2: 3x3 system
    A2 = torch.tensor([[1.0, 2.0, 3.0], [0.0, 1.0, 4.0], [5.0, 6.0, 0.0]], device='cuda')
    b2 = torch.tensor([6.0, 4.0, 3.0], device='cuda')
    results["test_case_2"] = fused_lu_solve(A2, b2)
    
    # Test case 3: 4x4 system
    A3 = torch.tensor([[4.0, 3.0, 2.0, 1.0], [3.0, 2.0, 1.0, 4.0], [2.0, 1.0, 4.0, 3.0], [1.0, 4.0, 3.0, 2.0]], device='cuda')
    b3 = torch.tensor([10.0, 11.0, 12.0, 13.0], device='cuda')
    results["test_case_3"] = fused_lu_solve(A3, b3)
    
    # Test case 4: Singular matrix (should raise an error)
    A4 = torch.tensor([[1.0, 2.0], [2.0, 4.0]], device='cuda')
    b4 = torch.tensor([5.0, 10.0], device='cuda')
    try:
        results["test_case_4"] = fused_lu_solve(A4, b4)
    except RuntimeError as e:
        results["test_case_4"] = str(e)
    
    return results

test_results = test_fused_lu_solve()
