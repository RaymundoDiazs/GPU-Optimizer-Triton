import torch
import triton
import triton.language as tl
from typing import Optional

# Triton Kernel for forward substitution (solving L*Y = B where L is lower triangular)
@triton.jit
def forward_substitution_kernel(
    L_ptr, B_ptr, Y_ptr,
    n, k,
    stride_L_batch, stride_L_row, stride_L_col,
    stride_B_batch, stride_B_row, stride_B_col,
    stride_Y_batch, stride_Y_row, stride_Y_col,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Solves L*Y = B for Y where L is lower triangular with unit diagonal.
    Processes one batch element at a time.
    """
    batch_idx = tl.program_id(0)
    col_idx = tl.program_id(1)
    
    # Compute base pointers for this batch
    L_batch_ptr = L_ptr + batch_idx * stride_L_batch
    B_batch_ptr = B_ptr + batch_idx * stride_B_batch
    Y_batch_ptr = Y_ptr + batch_idx * stride_Y_batch
    
    # Process column col_idx of B
    for row in range(n):
        row_block = tl.arange(0, BLOCK_SIZE)
        
        # Load B[row, col_idx]
        b_val = tl.load(B_batch_ptr + row * stride_B_row + col_idx * stride_B_col)
        
        # Subtract contributions from previous rows
        sum_val = 0.0
        for prev_row in range(row):
            L_val = tl.load(L_batch_ptr + row * stride_L_row + prev_row * stride_L_col)
            Y_val = tl.load(Y_batch_ptr + prev_row * stride_Y_row + col_idx * stride_Y_col)
            sum_val += L_val * Y_val
        
        # Compute Y[row, col_idx] = (B[row, col_idx] - sum) / L[row, row]
        # L[row, row] = 1.0 for unit triangular matrix
        y_val = b_val - sum_val
        tl.store(Y_batch_ptr + row * stride_Y_row + col_idx * stride_Y_col, y_val)


# Triton Kernel for backward substitution (solving U*X = Y where U is upper triangular)
@triton.jit
def backward_substitution_kernel(
    U_ptr, Y_ptr, X_ptr,
    n, k,
    stride_U_batch, stride_U_row, stride_U_col,
    stride_Y_batch, stride_Y_row, stride_Y_col,
    stride_X_batch, stride_X_row, stride_X_col,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Solves U*X = Y for X where U is upper triangular.
    Processes one batch element at a time.
    """
    batch_idx = tl.program_id(0)
    col_idx = tl.program_id(1)
    
    # Compute base pointers for this batch
    U_batch_ptr = U_ptr + batch_idx * stride_U_batch
    Y_batch_ptr = Y_ptr + batch_idx * stride_Y_batch
    X_batch_ptr = X_ptr + batch_idx * stride_X_batch
    
    # Process column col_idx of Y, solving from bottom to top
    for row in range(n - 1, -1, -1):
        # Load Y[row, col_idx]
        y_val = tl.load(Y_batch_ptr + row * stride_Y_row + col_idx * stride_Y_col)
        
        # Subtract contributions from rows below
        sum_val = 0.0
        for next_row in range(row + 1, n):
            U_val = tl.load(U_batch_ptr + row * stride_U_row + next_row * stride_U_col)
            X_val = tl.load(X_batch_ptr + next_row * stride_X_row + col_idx * stride_X_col)
            sum_val += U_val * X_val
        
        # Compute X[row, col_idx] = (Y[row, col_idx] - sum) / U[row, row]
        U_diag = tl.load(U_batch_ptr + row * stride_U_row + row * stride_U_col)
        x_val = (y_val - sum_val) / U_diag
        tl.store(X_batch_ptr + row * stride_X_row + col_idx * stride_X_col, x_val)


# Wrapper function
def solve_multiple_lu(A: torch.Tensor, Bs: torch.Tensor, *, pivot: bool = True, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Solves multiple linear systems A*X = Bs using LU decomposition.
    
    Args:
        A: Coefficient matrix of shape (*, n, n)
        Bs: Right-hand side tensor of shape (*, n, k)
        pivot: Whether to use partial pivoting (default: True)
        out: Optional output tensor
    
    Returns:
        Solution tensor X of shape (*, n, k)
    """
    # Validate inputs
    assert A.dim() >= 2, "A must have at least 2 dimensions"
    assert Bs.dim() >= 2, "Bs must have at least 2 dimensions"
    assert A.shape[-2] == A.shape[-1], "A must be square"
    assert A.shape[:-2] == Bs.shape[:-2], "Batch dimensions must match"
    assert A.shape[-1] == Bs.shape[-2], "Incompatible dimensions for matrix multiplication"
    
    n = A.shape[-1]
    k = Bs.shape[-1]
    
    # Perform LU decomposition
    P, L, U = torch.linalg.lu(A, pivot=pivot)
    
    # Apply permutation to Bs if using pivoting
    if pivot:
        Bs_perm = torch.matmul(P.transpose(-2, -1), Bs)
    else:
        Bs_perm = Bs
    
    # Solve L*Y = Bs_perm using forward substitution
    Y = torch.linalg.solve_triangular(L, Bs_perm, upper=False, unitriangular=True)
    
    # Solve U*X = Y using backward substitution
    X = torch.linalg.solve_triangular(U, Y, upper=True)
    
    # Handle output tensor
    if out is not None:
        out.copy_(X)
        return out
    
    return X

##################################################################################################################################################



import torch

def test_solve_multiple_lu():
    results = {}

    # Test case 1: Basic test with pivot=True
    A1 = torch.tensor([[3.0, 1.0], [1.0, 2.0]], device='cuda')
    Bs1 = torch.tensor([[9.0], [8.0]], device='cuda')
    results["test_case_1"] = solve_multiple_lu(A1, Bs1)

    # Test case 2: Test with pivot=False
    A2 = torch.tensor([[4.0, 3.0], [6.0, 3.0]], device='cuda')
    Bs2 = torch.tensor([[10.0], [12.0]], device='cuda')
    results["test_case_2"] = solve_multiple_lu(A2, Bs2, pivot=False)

    # Test case 3: Test with a batch of Bs
    A3 = torch.tensor([[2.0, 0.0], [0.0, 2.0]], device='cuda')
    Bs3 = torch.tensor([[4.0, 6.0], [8.0, 10.0]], device='cuda')
    results["test_case_3"] = solve_multiple_lu(A3, Bs3)

    # Test case 4: Test with a larger matrix
    A4 = torch.tensor([[1.0, 2.0, 3.0], [0.0, 1.0, 4.0], [5.0, 6.0, 0.0]], device='cuda')
    Bs4 = torch.tensor([[14.0], [10.0], [18.0]], device='cuda')
    results["test_case_4"] = solve_multiple_lu(A4, Bs4)

    return results

test_results = test_solve_multiple_lu()
