import torch
import triton
import triton.language as tl
from torch import Tensor

# Triton Kernel for Forward Substitution (Ly = b)
@triton.jit
def forward_substitution_kernel(
    y_ptr, L_ptr, b_ptr,
    n, k,
    stride_y_n, stride_y_k,
    stride_L_n, stride_L_m,
    stride_b_n, stride_b_k,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Solve Ly = b for y using forward substitution.
    L is lower triangular, b is (n, k).
    """
    pid_k = tl.program_id(0)
    
    if pid_k >= k:
        return
    
    # Process each row sequentially for forward substitution
    for i in range(n):
        # Load b[i, pid_k]
        b_idx = i * stride_b_n + pid_k * stride_b_k
        b_val = tl.load(b_ptr + b_idx)
        
        # Compute sum of L[i, j] * y[j, pid_k] for j < i
        sum_val = 0.0
        for j in range(i):
            L_idx = i * stride_L_n + j * stride_L_m
            y_idx = j * stride_y_n + pid_k * stride_y_k
            
            L_val = tl.load(L_ptr + L_idx)
            y_val = tl.load(y_ptr + y_idx)
            sum_val += L_val * y_val
        
        # y[i, pid_k] = (b[i, pid_k] - sum_val) / L[i, i]
        L_diag_idx = i * stride_L_n + i * stride_L_m
        L_diag = tl.load(L_ptr + L_diag_idx)
        
        y_val = (b_val - sum_val) / L_diag
        
        y_idx = i * stride_y_n + pid_k * stride_y_k
        tl.store(y_ptr + y_idx, y_val)


# Triton Kernel for Backward Substitution (L.T x = y)
@triton.jit
def backward_substitution_kernel(
    x_ptr, L_ptr, y_ptr,
    n, k,
    stride_x_n, stride_x_k,
    stride_L_n, stride_L_m,
    stride_y_n, stride_y_k,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Solve L.T x = y for x using backward substitution.
    L.T is upper triangular, y is (n, k).
    """
    pid_k = tl.program_id(0)
    
    if pid_k >= k:
        return
    
    # Process each row in reverse for backward substitution
    for i in range(n - 1, -1, -1):
        # Load y[i, pid_k]
        y_idx = i * stride_y_n + pid_k * stride_y_k
        y_val = tl.load(y_ptr + y_idx)
        
        # Compute sum of L[j, i] * x[j, pid_k] for j > i
        sum_val = 0.0
        for j in range(i + 1, n):
            L_idx = j * stride_L_n + i * stride_L_m  # L.T[i, j] = L[j, i]
            x_idx = j * stride_x_n + pid_k * stride_x_k
            
            L_val = tl.load(L_ptr + L_idx)
            x_val = tl.load(x_ptr + x_idx)
            sum_val += L_val * x_val
        
        # x[i, pid_k] = (y[i, pid_k] - sum_val) / L[i, i]
        L_diag_idx = i * stride_L_n + i * stride_L_m
        L_diag = tl.load(L_ptr + L_diag_idx)
        
        x_val = (y_val - sum_val) / L_diag
        
        x_idx = i * stride_x_n + pid_k * stride_x_k
        tl.store(x_ptr + x_idx, x_val)


# Wrapper Function
def fused_cholesky_solve(A: Tensor, b: Tensor) -> Tensor:
    """
    Solve the equation Ax = b using the Cholesky decomposition of the symmetric positive-definite matrix A.

    Args:
        A (torch.Tensor): The symmetric positive-definite matrix A of shape (n, n).
        b (torch.Tensor): The right-hand side tensor b of shape (n, k).

    Returns:
        torch.Tensor: The solution tensor x of shape (n, k).
    """
    assert A.dim() == 2 and A.shape[0] == A.shape[1], "A must be a square matrix"
    assert b.dim() == 2 and b.shape[0] == A.shape[0], "b must have shape (n, k)"
    assert A.is_cuda, "Tensors must be on CUDA device"
    
    n, _ = A.shape
    _, k = b.shape
    
    # Step 1: Cholesky decomposition using PyTorch
    L = torch.linalg.cholesky(A)
    
    # Step 2: Forward substitution - solve Ly = b
    y = torch.zeros_like(b)
    
    BLOCK_SIZE = 32
    grid = (k,)
    
    forward_substitution_kernel[grid](
        y, L, b,
        n, k,
        y.stride(0), y.stride(1),
        L.stride(0), L.stride(1),
        b.stride(0), b.stride(1),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Step 3: Backward substitution - solve L.T x = y
    x = torch.zeros_like(y)
    
    backward_substitution_kernel[grid](
        x, L, y,
        n, k,
        x.stride(0), x.stride(1),
        L.stride(0), L.stride(1),
        y.stride(0), y.stride(1),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return x

##################################################################################################################################################



import torch

def test_fused_cholesky_solve():
    results = {}

    # Test case 1: Simple 2x2 positive-definite matrix
    A1 = torch.tensor([[4.0, 1.0], [1.0, 3.0]], device='cuda')
    b1 = torch.tensor([[1.0], [2.0]], device='cuda')
    results["test_case_1"] = fused_cholesky_solve(A1, b1)

    # Test case 2: Larger 3x3 positive-definite matrix
    A2 = torch.tensor([[6.0, 2.0, 1.0], [2.0, 5.0, 2.0], [1.0, 2.0, 4.0]], device='cuda')
    b2 = torch.tensor([[1.0], [2.0], [3.0]], device='cuda')
    results["test_case_2"] = fused_cholesky_solve(A2, b2)

    # Test case 3: 2x2 matrix with multiple right-hand sides
    A3 = torch.tensor([[5.0, 2.0], [2.0, 3.0]], device='cuda')
    b3 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    results["test_case_3"] = fused_cholesky_solve(A3, b3)

    # Test case 4: 3x3 matrix with multiple right-hand sides
    A4 = torch.tensor([[7.0, 3.0, 1.0], [3.0, 6.0, 2.0], [1.0, 2.0, 5.0]], device='cuda')
    b4 = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], device='cuda')
    results["test_case_4"] = fused_cholesky_solve(A4, b4)

    return results

test_results = test_fused_cholesky_solve()
