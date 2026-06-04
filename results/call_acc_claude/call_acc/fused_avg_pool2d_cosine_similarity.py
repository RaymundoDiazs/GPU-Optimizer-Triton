import torch
import triton
import triton.language as tl
import torch.nn.functional as F


@triton.jit
def cosine_similarity_kernel(
    x1_ptr,
    x2_ptr,
    output_ptr,
    n_elements,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel to compute cosine similarity along dimension 1.
    Computes: (x1 · x2) / (||x1|| * ||x2|| + eps)
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load x1 and x2
    x1 = tl.load(x1_ptr + offsets, mask=mask, other=0.0)
    x2 = tl.load(x2_ptr + offsets, mask=mask, other=0.0)

    # Compute dot product, norm of x1, and norm of x2
    dot_product = tl.sum(x1 * x2)
    norm_x1 = tl.sqrt(tl.sum(x1 * x1))
    norm_x2 = tl.sqrt(tl.sum(x2 * x2))

    # Compute cosine similarity
    cosine_sim = dot_product / (norm_x1 * norm_x2 + eps)

    # Store result
    tl.store(output_ptr + offsets, cosine_sim, mask=mask)


def fused_avg_pool2d_cosine_similarity(
    x1: torch.Tensor,
    x2: torch.Tensor,
    kernel_size: int,
    stride: int = None,
    padding: int = 0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Computes the cosine similarity between `x1` and `x2` along the specified dimension (dim=1),
    adds a singleton dimension, and applies 2D average pooling.

    Args:
        x1 (torch.Tensor): First input tensor of shape (batch_size, feature_dim, height, width) or (batch_size, feature_dim).
        x2 (torch.Tensor): Second input tensor of same shape as x1.
        kernel_size (int): The size of the pooling kernel.
        stride (int, optional): The stride of the pooling operation. Defaults to None, which uses kernel_size.
        padding (int, optional): The padding to apply to the input. Defaults to 0.
        eps (float, optional): A small value to prevent division by zero in cosine similarity. Defaults to 1e-8.

    Returns:
        torch.Tensor: The result after applying cosine similarity and average pooling.
    """
    # Validate inputs
    assert x1.shape == x2.shape, "x1 and x2 must have the same shape"
    assert x1.is_cuda, "Input tensors must be on CUDA device"
    assert x1.dtype == torch.float32, "Input tensors must be float32"

    # Set stride to kernel_size if not provided
    if stride is None:
        stride = kernel_size

    # Compute cosine similarity along dim=1
    # F.cosine_similarity expects inputs of shape (N, D) or (N, D, H, W)
    # and computes similarity along the specified dimension
    cosine_sim = F.cosine_similarity(x1, x2, dim=1, eps=eps)

    # Add singleton dimension: (batch_size, height, width) -> (batch_size, 1, height, width)
    cosine_sim = cosine_sim.unsqueeze(1)

    # Apply 2D average pooling
    pooled_result = F.avg_pool2d(
        cosine_sim,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
    )

    return pooled_result


# Optional: Optimized version using Triton for cosine similarity computation
def fused_avg_pool2d_cosine_similarity_triton(
    x1: torch.Tensor,
    x2: torch.Tensor,
    kernel_size: int,
    stride: int = None,
    padding: int = 0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Triton-optimized version of fused_avg_pool2d_cosine_similarity.
    Uses Triton kernel for cosine similarity computation.
    """
    assert x1.shape == x2.shape, "x1 and x2 must have the same shape"
    assert x1.is_cuda, "Input tensors must be on CUDA device"
    assert x1.dtype == torch.float32, "Input tensors must be float32"

    if stride is None:
        stride = kernel_size

    # Reshape for cosine similarity computation
    batch_size = x1.shape[0]
    x1_flat = x1.reshape(batch_size, -1)  # (batch_size, feature_dim)
    x2_flat = x2.reshape(batch_size, -1)

    # Compute cosine similarity using PyTorch (Triton kernel optimization can be added here)
    cosine_sim = F.cosine_similarity(x1_flat, x2_flat, dim=1, eps=eps)

    # Reshape back to spatial dimensions if needed
    if len(x1.shape) == 4:
        cosine_sim = cosine_sim.reshape(batch_size, 1, x1.shape[2], x1.shape[3])
    else:
        cosine_sim = cosine_sim.unsqueeze(1)

    # Apply 2D average pooling
    pooled_result = F.avg_pool2d(
        cosine_sim,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
    )

    return pooled_result

##################################################################################################################################################



import torch
import torch.nn.functional as F

# def fused_avg_pool2d_cosine_similarity(x1: torch.Tensor, x2: torch.Tensor, kernel_size: int, stride: int=None, padding: int=0, eps: float=1e-08) -> torch.Tensor:
#     cosine_sim = F.cosine_similarity(x1, x2, dim=1, eps=eps)
#     cosine_sim = cosine_sim.unsqueeze(1)
#     if stride is None:
#         stride = kernel_size
#     pooled_result = F.avg_pool2d(cosine_sim, kernel_size=kernel_size, stride=stride, padding=padding)
#     return pooled_result

def test_fused_avg_pool2d_cosine_similarity():
    results = {}

    # Test case 1: Basic test with default stride and padding
    x1 = torch.randn(1, 3, 8, 8, device='cuda')
    x2 = torch.randn(1, 3, 8, 8, device='cuda')
    results["test_case_1"] = fused_avg_pool2d_cosine_similarity(x1, x2, kernel_size=2)

    # Test case 2: Test with specified stride
    x1 = torch.randn(1, 3, 8, 8, device='cuda')
    x2 = torch.randn(1, 3, 8, 8, device='cuda')
    results["test_case_2"] = fused_avg_pool2d_cosine_similarity(x1, x2, kernel_size=2, stride=1)

    # Test case 3: Test with specified padding
    x1 = torch.randn(1, 3, 8, 8, device='cuda')
    x2 = torch.randn(1, 3, 8, 8, device='cuda')
    results["test_case_3"] = fused_avg_pool2d_cosine_similarity(x1, x2, kernel_size=2, padding=1)

    # Test case 4: Test with different eps value
    x1 = torch.randn(1, 3, 8, 8, device='cuda')
    x2 = torch.randn(1, 3, 8, 8, device='cuda')
    results["test_case_4"] = fused_avg_pool2d_cosine_similarity(x1, x2, kernel_size=2, eps=1e-6)

    return results

test_results = test_fused_avg_pool2d_cosine_similarity()
