import torch
import triton
import triton.language as tl


@triton.jit
def _spectral_norm_eig_kernel(
    eigenvalues_ptr,
    output_ptr,
    n,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel to compute spectral norm from eigenvalues.
    Computes the maximum absolute value of eigenvalues for each batch element.
    
    Args:
        eigenvalues_ptr: Pointer to eigenvalues tensor of shape (batch_size, n)
        output_ptr: Pointer to output tensor of shape (batch_size,)
        n: Number of eigenvalues per batch element
        BLOCK_SIZE: Block size for processing eigenvalues
    """
    batch_idx = tl.program_id(0)
    
    # Initialize max value to 0
    max_abs_val = 0.0
    
    # Process eigenvalues in blocks
    for block_start in range(0, n, BLOCK_SIZE):
        block_end = tl.minimum(block_start + BLOCK_SIZE, n)
        block_size = block_end - block_start
        
        # Create mask for valid indices
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n
        
        # Load eigenvalues for this batch element
        eig_offset = batch_idx * n + offsets
        eigenvalues = tl.load(eigenvalues_ptr + eig_offset, mask=mask, other=0.0)
        
        # Compute absolute values
        abs_eigenvalues = tl.abs(eigenvalues)
        
        # Update maximum
        max_abs_val = tl.maximum(max_abs_val, tl.max(abs_eigenvalues))
    
    # Store result
    tl.store(output_ptr + batch_idx, max_abs_val)


def spectral_norm_eig(A, *, out=None) -> torch.Tensor:
    """
    Computes the spectral norm (operator norm induced by the Euclidean vector norm)
    of a square matrix using its eigenvalues.

    Args:
        A (Tensor): Tensor of shape `(*, n, n)` where `*` is zero or more batch dimensions
                    consisting of square matrices.
        out (Tensor, optional): Output tensor. Ignored if `None`. Default: `None`.

    Returns:
        Tensor: The spectral norm of the input matrix or batch of matrices.
    """
    # Validate input
    if A.dim() < 2:
        raise ValueError(f"Input tensor must have at least 2 dimensions, got {A.dim()}")
    
    if A.shape[-2] != A.shape[-1]:
        raise ValueError(f"Input must be square matrices, got shape {A.shape}")
    
    # Check dtype support
    if A.dtype not in [torch.float32, torch.float64, torch.complex64, torch.complex128]:
        raise ValueError(f"Unsupported dtype: {A.dtype}")
    
    # Compute eigenvalues
    eigenvalues, _ = torch.linalg.eig(A)
    
    # Compute absolute values of eigenvalues
    abs_eigenvalues = torch.abs(eigenvalues)
    
    # Get maximum absolute eigenvalue for each matrix
    spectral_norm, _ = torch.max(abs_eigenvalues, dim=-1)
    
    # Handle output tensor
    if out is not None:
        out.copy_(spectral_norm)
        return out
    
    return spectral_norm

##################################################################################################################################################



import torch

def test_spectral_norm_eig():
    results = {}

    # Test case 1: Single 2x2 matrix
    A1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    results["test_case_1"] = spectral_norm_eig(A1)

    # Test case 2: Batch of 2x2 matrices
    A2 = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], device='cuda')
    results["test_case_2"] = spectral_norm_eig(A2)

    # Test case 3: Single 3x3 matrix
    A3 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], device='cuda')
    results["test_case_3"] = spectral_norm_eig(A3)

    # Test case 4: Batch of 3x3 matrices
    A4 = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], 
                       [[9.0, 8.0, 7.0], [6.0, 5.0, 4.0], [3.0, 2.0, 1.0]]], device='cuda')
    results["test_case_4"] = spectral_norm_eig(A4)

    return results

test_results = test_spectral_norm_eig()
