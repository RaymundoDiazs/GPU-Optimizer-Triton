import torch
import triton
import triton.language as tl
from typing import Optional

# Triton Kernel for computing product of diagonal elements
@triton.jit
def _diag_prod_kernel(
    diag_ptr,
    out_ptr,
    n: tl.int32,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Kernel to compute the product of diagonal elements.
    
    Args:
        diag_ptr: Pointer to diagonal elements
        out_ptr: Pointer to output (scalar)
        n: Size of diagonal
        BLOCK_SIZE: Block size for computation
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    block_end = tl.minimum(block_start + BLOCK_SIZE, n)
    
    # Load diagonal elements
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    
    diag_vals = tl.load(diag_ptr + offsets, mask=mask, other=1.0)
    
    # Compute product for this block
    prod = tl.reduce(diag_vals, axis=0, _op=tl.math.mul)
    
    # Store result (only first thread writes)
    if pid == 0:
        tl.store(out_ptr, prod)


def determinant_via_qr(
    A: torch.Tensor,
    *,
    mode: str = 'reduced',
    out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Computes the determinant of a square matrix using QR decomposition.
    
    The function performs QR decomposition of a square matrix A and computes
    the determinant as: det(A) = det(Q) * prod(diag(R))
    
    Parameters:
        A (Tensor): The input square matrix of shape (..., n, n).
        mode (str, optional): The mode for QR decomposition ('reduced' or 'complete'). 
                             Defaults to 'reduced'.
        out (Tensor, optional): The output tensor to store the result. Defaults to None.
    
    Returns:
        Tensor: The determinant of the matrix A with shape (...,).
    
    Raises:
        ValueError: If A is not a square matrix.
        RuntimeError: If QR decomposition fails.
    """
    # Validate input
    if A.shape[-2] != A.shape[-1]:
        raise ValueError(f"Input matrix must be square, got shape {A.shape}")
    
    # Perform QR decomposition
    Q, R = torch.linalg.qr(A, mode=mode)
    
    # Compute determinant of Q
    det_Q = torch.det(Q)
    
    # Extract diagonal elements of R
    diag_R = torch.diagonal(R, dim1=-2, dim2=-1)
    
    # Compute product of diagonal elements
    prod_diag_R = torch.prod(diag_R, dim=-1)
    
    # Compute determinant: det(A) = det(Q) * prod(diag(R))
    determinant = det_Q * prod_diag_R
    
    # Handle output tensor
    if out is not None:
        out.copy_(determinant)
        return out
    
    return determinant

##################################################################################################################################################



import torch

def test_determinant_via_qr():
    results = {}

    # Test case 1: 2x2 matrix, reduced mode
    A1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    results["test_case_1"] = determinant_via_qr(A1)

    # Test case 2: 3x3 matrix, reduced mode
    A2 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], device='cuda')
    results["test_case_2"] = determinant_via_qr(A2)

    # Test case 3: 2x2 matrix, complete mode
    A3 = torch.tensor([[2.0, 3.0], [1.0, 4.0]], device='cuda')
    results["test_case_3"] = determinant_via_qr(A3, mode='complete')

    # Test case 4: 3x3 matrix, complete mode
    A4 = torch.tensor([[2.0, 0.0, 1.0], [1.0, 3.0, 2.0], [4.0, 1.0, 3.0]], device='cuda')
    results["test_case_4"] = determinant_via_qr(A4, mode='complete')

    return results

test_results = test_determinant_via_qr()
