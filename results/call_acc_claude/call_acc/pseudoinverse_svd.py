import torch
import triton
import triton.language as tl
from typing import Optional

@triton.jit
def _pseudoinverse_svd_kernel(
    U_ptr, S_ptr, Vh_ptr,
    S_inv_ptr,
    m, n, k,
    rcond,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Kernel to invert singular values based on rcond threshold.
    
    Args:
        U_ptr: Pointer to U matrix (*, m, k)
        S_ptr: Pointer to S vector (*, k)
        Vh_ptr: Pointer to Vh matrix (*, k, n)
        S_inv_ptr: Pointer to output inverted S vector (*, k)
        m, n, k: Dimensions
        rcond: Relative condition number threshold
        BLOCK_SIZE: Block size for processing
    """
    # Get batch and position indices
    batch_idx = tl.program_id(0)
    block_id = tl.program_id(1)
    
    # Compute offsets
    offset = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Load singular values for this batch
    s_batch_ptr = S_ptr + batch_idx * k
    s_max = tl.load(s_batch_ptr)  # Assuming max is precomputed or we load first element
    
    # Load singular values
    mask = offset < k
    S_vals = tl.load(s_batch_ptr + offset, mask=mask, other=0.0)
    
    # Compute cutoff threshold
    cutoff = rcond * s_max
    
    # Invert singular values above threshold
    S_inv_vals = tl.where(S_vals > cutoff, 1.0 / S_vals, 0.0)
    
    # Store inverted singular values
    s_inv_batch_ptr = S_inv_ptr + batch_idx * k
    tl.store(s_inv_batch_ptr + offset, S_inv_vals, mask=mask)


def pseudoinverse_svd(
    A: torch.Tensor,
    *,
    full_matrices: bool = True,
    rcond: float = 1e-15,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Computes the Moore-Penrose pseudoinverse of a matrix using SVD.
    
    Args:
        A (Tensor): Input tensor of shape `(*, m, n)` where `*` is zero or more batch dimensions.
        full_matrices (bool, optional): If `True` (default), compute the full SVD. 
                                       If `False`, compute the reduced SVD. Default: `True`.
        rcond (float, optional): Relative condition number threshold. Singular values smaller 
                               than `rcond * largest_singular_value` are set to zero. Default: `1e-15`.
        out (Tensor, optional): Output tensor. If provided, result is written here. Default: `None`.
    
    Returns:
        Tensor: Pseudoinverse of A with shape `(*, n, m)`.
    """
    # Validate input
    if A.dim() < 2:
        raise ValueError(f"Input tensor must have at least 2 dimensions, got {A.dim()}")
    
    # Get dimensions
    *batch_dims, m, n = A.shape
    batch_size = 1
    for dim in batch_dims:
        batch_size *= dim
    
    # Reshape to (batch_size, m, n) for processing
    A_reshaped = A.reshape(batch_size, m, n)
    
    # Compute SVD
    U, S, Vh = torch.linalg.svd(A_reshaped, full_matrices=full_matrices)
    
    # Determine k (number of singular values)
    k = S.shape[-1]
    
    # Compute cutoff threshold: rcond * max(S) for each batch
    S_max = S.max(dim=-1, keepdim=True).values  # Shape: (batch_size, 1)
    cutoff = rcond * S_max
    
    # Invert singular values larger than cutoff
    S_inv = torch.where(S > cutoff, 1.0 / S, torch.zeros_like(S))
    
    # Create diagonal matrix of inverted singular values
    # Shape: (batch_size, k, k)
    S_inv_mat = torch.diag_embed(S_inv)
    
    # Compute pseudoinverse: A_pinv = Vh^H @ S_inv @ U^H
    # Vh shape: (batch_size, k, n)
    # S_inv_mat shape: (batch_size, k, k)
    # U shape: (batch_size, m, k)
    
    A_pinv = (
        Vh.transpose(-2, -1).conj() @ 
        S_inv_mat @ 
        U.transpose(-2, -1).conj()
    )
    
    # Reshape back to original batch dimensions
    output_shape = batch_dims + [n, m]
    A_pinv = A_pinv.reshape(output_shape)
    
    # Handle output tensor
    if out is not None:
        out.copy_(A_pinv)
        return out
    
    return A_pinv

##################################################################################################################################################



import torch

def test_pseudoinverse_svd():
    results = {}

    # Test case 1: Square matrix
    A1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    results["test_case_1"] = pseudoinverse_svd(A1)

    # Test case 4: Singular matrix
    A4 = torch.tensor([[1.0, 2.0], [2.0, 4.0]], device='cuda')
    results["test_case_4"] = pseudoinverse_svd(A4)

    return results

test_results = test_pseudoinverse_svd()
