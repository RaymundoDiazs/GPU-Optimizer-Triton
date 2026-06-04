import torch
import triton
import triton.language as tl
from typing import Optional

@triton.jit
def _low_rank_svd_approximation_kernel(
    U_ptr, S_ptr, Vh_ptr, out_ptr,
    m, n, k,
    U_stride_batch, U_stride_m, U_stride_n,
    S_stride_batch, S_stride_k,
    Vh_stride_batch, Vh_stride_k, Vh_stride_n,
    out_stride_batch, out_stride_m, out_stride_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    """
    Triton kernel for computing rank-k SVD approximation: A_k = U_k @ diag(S_k) @ Vh_k
    
    This kernel performs the matrix multiplication in a blocked fashion:
    - U_k: (m, k)
    - S_k: (k,) - treated as diagonal matrix
    - Vh_k: (k, n)
    - Output: (m, n)
    """
    # Get batch index
    batch_idx = tl.program_id(0)
    
    # Get output position
    m_idx = tl.program_id(1)
    n_idx = tl.program_id(2)
    
    # Compute offsets for batch dimension
    U_batch_offset = batch_idx * U_stride_batch
    S_batch_offset = batch_idx * S_stride_batch
    Vh_batch_offset = batch_idx * Vh_stride_batch
    out_batch_offset = batch_idx * out_stride_batch
    
    # Initialize accumulator for output block
    m_offsets = m_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = n_idx * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Mask for valid m and n indices
    m_mask = m_offsets < m
    n_mask = n_offsets < n
    
    # Accumulator for the result
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Loop over k dimension in blocks
    for k_block_idx in range(tl.cdiv(k, BLOCK_K)):
        k_offsets = k_block_idx * BLOCK_K + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < k
        
        # Load U_k block: (BLOCK_M, BLOCK_K)
        U_offsets = (
            U_batch_offset +
            m_offsets[:, None] * U_stride_m +
            k_offsets[None, :] * U_stride_n
        )
        U_block = tl.load(
            U_ptr + U_offsets,
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0
        )
        
        # Load S_k block: (BLOCK_K,)
        S_offsets = S_batch_offset + k_offsets * S_stride_k
        S_block = tl.load(
            S_ptr + S_offsets,
            mask=k_mask,
            other=0.0
        )
        
        # Load Vh_k block: (BLOCK_K, BLOCK_N)
        Vh_offsets = (
            Vh_batch_offset +
            k_offsets[:, None] * Vh_stride_k +
            n_offsets[None, :] * Vh_stride_n
        )
        Vh_block = tl.load(
            Vh_ptr + Vh_offsets,
            mask=k_mask[:, None] & n_mask[None, :],
            other=0.0
        )
        
        # Scale Vh_k by S_k: (BLOCK_K, BLOCK_N)
        S_Vh = S_block[:, None] * Vh_block
        
        # Accumulate: U_k @ (S_k * Vh_k)
        acc += tl.dot(U_block, S_Vh, allow_tf32=False)
    
    # Store result
    out_offsets = (
        out_batch_offset +
        m_offsets[:, None] * out_stride_m +
        n_offsets[None, :] * out_stride_n
    )
    tl.store(
        out_ptr + out_offsets,
        acc,
        mask=m_mask[:, None] & n_mask[None, :]
    )


def low_rank_svd_approximation(
    A: torch.Tensor,
    k: int,
    *,
    full_matrices: bool = True,
    out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Computes a rank-k approximation of a matrix using its Singular Value Decomposition (SVD).

    Args:
        A (Tensor): Tensor of shape `(*, m, n)` where `*` is zero or more batch dimensions.
        k (int): Rank of the approximation (must satisfy `1 <= k <= min(m, n)`).
        full_matrices (bool, optional): Controls whether to compute the full or reduced SVD. Default: `True`.
        out (Tensor, optional): Output tensor. Ignored if `None`. Default: `None`.

    Returns:
        Tensor: The rank-k approximation of A with shape `(*, m, n)`.
    """
    # Validate inputs
    assert A.dim() >= 2, "Input tensor must have at least 2 dimensions"
    assert isinstance(k, int) and k >= 1, "k must be a positive integer"
    
    m, n = A.shape[-2:]
    assert k <= min(m, n), f"k ({k}) must be <= min(m, n) ({min(m, n)})"
    
    # Supported dtypes
    assert A.dtype in [torch.float32, torch.float64, torch.complex64, torch.complex128], \
        f"Unsupported dtype: {A.dtype}"
    
    # Compute SVD
    U, S, Vh = torch.linalg.svd(A, full_matrices=full_matrices)
    
    # Extract top-k components
    U_k = U[..., :k]
    S_k = S[..., :k]
    Vh_k = Vh[..., :k, :]
    
    # Create output tensor
    if out is None:
        A_k = torch.empty_like(A)
    else:
        A_k = out
    
    # Get batch dimensions
    batch_shape = A.shape[:-2]
    num_batches = 1
    for dim in batch_shape:
        num_batches *= dim
    
    # Reshape for kernel processing
    U_k_flat = U_k.reshape(-1, m, k)
    S_k_flat = S_k.reshape(-1, k)
    Vh_k_flat = Vh_k.reshape(-1, k, n)
    A_k_flat = A_k.reshape(-1, m, n)
    
    # Kernel configuration
    BLOCK_M = 32
    BLOCK_N = 32
    BLOCK_K = 32
    
    grid = (
        num_batches,
        triton.cdiv(m, BLOCK_M),
        triton.cdiv(n, BLOCK_N)
    )
    
    # Launch kernel
    _low_rank_svd_approximation_kernel[grid](
        U_k_flat, S_k_flat, Vh_k_flat, A_k_flat,
        m, n, k,
        U_k_flat.stride(0), U_k_flat.stride(1), U_k_flat.stride(2),
        S_k_flat.stride(0), S_k_flat.stride(1),
        Vh_k_flat.stride(0), Vh_k_flat.stride(1), Vh_k_flat.stride(2),
        A_k_flat.stride(0), A_k_flat.stride(1), A_k_flat.stride(2),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
    )
    
    return A_k

##################################################################################################################################################



import torch

def test_low_rank_svd_approximation():
    results = {}

    # Test case 1: Basic rank-k approximation with full_matrices=True
    A = torch.randn(5, 4, device='cuda')
    k = 2
    results["test_case_1"] = low_rank_svd_approximation(A, k)

    # Test case 2: Basic rank-k approximation with full_matrices=False
    A = torch.randn(6, 3, device='cuda')
    k = 2
    results["test_case_2"] = low_rank_svd_approximation(A, k, full_matrices=False)

    # Test case 3: Batch matrix with full_matrices=True
    A = torch.randn(2, 5, 4, device='cuda')
    k = 3
    results["test_case_3"] = low_rank_svd_approximation(A, k)

    # Test case 4: Batch matrix with full_matrices=False
    A = torch.randn(3, 6, 3, device='cuda')
    k = 2
    results["test_case_4"] = low_rank_svd_approximation(A, k, full_matrices=False)

    return results

test_results = test_low_rank_svd_approximation()
