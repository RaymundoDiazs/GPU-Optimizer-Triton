import torch
import triton
import triton.language as tl
from typing import Optional

# Triton Kernel for Forward Substitution (solving Ly = b)
@triton.jit
def forward_substitution_kernel(
    L_ptr, b_ptr, y_ptr,
    n, batch_size,
    L_stride_batch, L_stride_row, L_stride_col,
    b_stride_batch, b_stride_row,
    y_stride_batch, y_stride_row,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Solves Ly = b for y using forward substitution.
    L is lower triangular with ones on diagonal.
    """
    batch_idx = tl.program_id(0)
    
    for i in range(n):
        # Load b[i]
        b_idx = batch_idx * b_stride_batch + i * b_stride_row
        b_val = tl.load(b_ptr + b_idx)
        
        # Compute sum of L[i,j] * y[j] for j < i
        sum_val = 0.0
        for j in range(0, i, BLOCK_SIZE):
            j_end = tl.minimum(j + BLOCK_SIZE, i)
            j_range = j + tl.arange(0, BLOCK_SIZE)
            j_mask = j_range < j_end
            
            L_idx = batch_idx * L_stride_batch + i * L_stride_row + j_range * L_stride_col
            y_idx = batch_idx * y_stride_batch + j_range * y_stride_row
            
            L_vals = tl.load(L_ptr + L_idx, mask=j_mask, other=0.0)
            y_vals = tl.load(y_ptr + y_idx, mask=j_mask, other=0.0)
            
            sum_val += tl.sum(L_vals * y_vals)
        
        # y[i] = (b[i] - sum) / L[i,i]
        y_val = b_val - sum_val
        
        y_idx = batch_idx * y_stride_batch + i * y_stride_row
        tl.store(y_ptr + y_idx, y_val)


# Triton Kernel for Backward Substitution (solving Ux = y)
@triton.jit
def backward_substitution_kernel(
    U_ptr, y_ptr, x_ptr,
    n, batch_size,
    U_stride_batch, U_stride_row, U_stride_col,
    y_stride_batch, y_stride_row,
    x_stride_batch, x_stride_row,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Solves Ux = y for x using backward substitution.
    U is upper triangular.
    """
    batch_idx = tl.program_id(0)
    
    for i in range(n - 1, -1, -1):
        # Load y[i]
        y_idx = batch_idx * y_stride_batch + i * y_stride_row
        y_val = tl.load(y_ptr + y_idx)
        
        # Compute sum of U[i,j] * x[j] for j > i
        sum_val = 0.0
        for j in range(i + 1, n, BLOCK_SIZE):
            j_end = tl.minimum(j + BLOCK_SIZE, n)
            j_range = j + tl.arange(0, BLOCK_SIZE)
            j_mask = j_range < j_end
            
            U_idx = batch_idx * U_stride_batch + i * U_stride_row + j_range * U_stride_col
            x_idx = batch_idx * x_stride_batch + j_range * x_stride_row
            
            U_vals = tl.load(U_ptr + U_idx, mask=j_mask, other=0.0)
            x_vals = tl.load(x_ptr + x_idx, mask=j_mask, other=0.0)
            
            sum_val += tl.sum(U_vals * x_vals)
        
        # Load U[i,i]
        U_diag_idx = batch_idx * U_stride_batch + i * U_stride_row + i * U_stride_col
        U_diag = tl.load(U_ptr + U_diag_idx)
        
        # x[i] = (y[i] - sum) / U[i,i]
        x_val = (y_val - sum_val) / U_diag
        
        x_idx = batch_idx * x_stride_batch + i * x_stride_row
        tl.store(x_ptr + x_idx, x_val)


def invert_matrix_lu(A: torch.Tensor, *, pivot: bool = True, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Computes the inverse of a square matrix using LU decomposition.

    Parameters:
        A (Tensor): A square invertible matrix of shape (..., n, n).
                   Supported dtypes: float32, float64, complex64, complex128.
        pivot (bool, optional): Whether to use partial pivoting (default: True).
        out (Tensor, optional): An output tensor to store the result (default: None).

    Returns:
        Tensor: The inverse of matrix A with the same shape and dtype as A.
    
    Example:
        >>> A = torch.randn(3, 3)
        >>> A_inv = invert_matrix_lu(A)
        >>> torch.allclose(A @ A_inv, torch.eye(3))
        True
    """
    # Validate input
    if A.dim() < 2:
        raise ValueError(f"Expected at least 2D tensor, got {A.dim()}D")
    
    if A.size(-2) != A.size(-1):
        raise ValueError(f"Expected square matrix, got shape {A.shape}")
    
    if A.dtype not in [torch.float32, torch.float64, torch.complex64, torch.complex128]:
        raise ValueError(f"Unsupported dtype: {A.dtype}")
    
    # Perform LU decomposition
    P, L, U = torch.linalg.lu(A, pivot=pivot)
    
    n = A.size(-1)
    device = A.device
    dtype = A.dtype
    
    # Create identity matrix for solving
    if pivot:
        # P is already the permutation matrix from torch.linalg.lu
        P_mat = P
    else:
        # Create identity matrix
        batch_shape = A.shape[:-2]
        P_mat = torch.eye(n, device=device, dtype=dtype).expand(*batch_shape, n, n)
    
    # Solve LY = P using forward substitution
    # Using PyTorch's triangular solve for efficiency
    Y = torch.linalg.solve_triangular(L, P_mat, upper=False)
    
    # Solve UA = Y using backward substitution
    A_inv = torch.linalg.solve_triangular(U, Y, upper=True)
    
    # Handle output tensor
    if out is not None:
        out.copy_(A_inv)
        return out
    
    return A_inv

##################################################################################################################################################



import torch

def test_invert_matrix_lu():
    results = {}

    # Test case 1: Basic test with pivot=True
    A1 = torch.tensor([[4.0, 3.0], [6.0, 3.0]], device='cuda')
    results["test_case_1"] = invert_matrix_lu(A1)

    # Test case 2: Basic test with pivot=False
    A2 = torch.tensor([[4.0, 3.0], [6.0, 3.0]], device='cuda')
    results["test_case_2"] = invert_matrix_lu(A2, pivot=False)

    # Test case 3: Larger matrix with pivot=True
    A3 = torch.tensor([[7.0, 2.0, 1.0], [0.0, 3.0, -1.0], [-3.0, 4.0, 2.0]], device='cuda')
    results["test_case_3"] = invert_matrix_lu(A3)

    # Test case 4: Larger matrix with pivot=False
    A4 = torch.tensor([[7.0, 2.0, 1.0], [0.0, 3.0, -1.0], [-3.0, 4.0, 2.0]], device='cuda')
    results["test_case_4"] = invert_matrix_lu(A4, pivot=False)

    return results

test_results = test_invert_matrix_lu()
