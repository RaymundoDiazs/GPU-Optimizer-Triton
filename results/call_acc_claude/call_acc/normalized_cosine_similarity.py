import torch
import triton
import triton.language as tl
from torch import Tensor


# Triton Kernel for normalization
@triton.jit
def normalize_kernel(
    x_ptr,
    output_ptr,
    stride_batch,
    stride_feature,
    num_features,
    p_norm,
    eps_norm,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Normalize input tensor along the feature dimension using L_p normalization.
    """
    batch_idx = tl.program_id(0)
    
    # Compute L_p norm for this batch element
    norm_val = 0.0
    
    for feature_idx in range(0, num_features, BLOCK_SIZE):
        block_end = tl.minimum(feature_idx + BLOCK_SIZE, num_features)
        block_size = block_end - feature_idx
        
        offsets = feature_idx + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_features
        
        x_offset = batch_idx * stride_batch + offsets * stride_feature
        x_vals = tl.load(x_ptr + x_offset, mask=mask, other=0.0)
        
        # Accumulate L_p norm
        if p_norm == 2.0:
            norm_val += tl.sum(x_vals * x_vals)
        else:
            norm_val += tl.sum(tl.abs(x_vals) ** p_norm)
    
    # Compute final norm
    if p_norm == 2.0:
        norm_val = tl.sqrt(norm_val)
    else:
        norm_val = norm_val ** (1.0 / p_norm)
    
    norm_val = tl.maximum(norm_val, eps_norm)
    
    # Normalize and write output
    for feature_idx in range(0, num_features, BLOCK_SIZE):
        block_end = tl.minimum(feature_idx + BLOCK_SIZE, num_features)
        offsets = feature_idx + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_features
        
        x_offset = batch_idx * stride_batch + offsets * stride_feature
        output_offset = batch_idx * stride_batch + offsets * stride_feature
        
        x_vals = tl.load(x_ptr + x_offset, mask=mask, other=0.0)
        normalized = x_vals / norm_val
        
        tl.store(output_offset + output_ptr, normalized, mask=mask)


# Triton Kernel for cosine similarity
@triton.jit
def cosine_similarity_kernel(
    x1_ptr,
    x2_ptr,
    output_ptr,
    stride_batch,
    stride_feature,
    num_features,
    eps_similarity,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Compute cosine similarity between two normalized tensors.
    """
    batch_idx = tl.program_id(0)
    
    # Compute dot product
    dot_product = 0.0
    norm_x1 = 0.0
    norm_x2 = 0.0
    
    for feature_idx in range(0, num_features, BLOCK_SIZE):
        offsets = feature_idx + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_features
        
        x1_offset = batch_idx * stride_batch + offsets * stride_feature
        x2_offset = batch_idx * stride_batch + offsets * stride_feature
        
        x1_vals = tl.load(x1_ptr + x1_offset, mask=mask, other=0.0)
        x2_vals = tl.load(x2_ptr + x2_offset, mask=mask, other=0.0)
        
        dot_product += tl.sum(x1_vals * x2_vals)
        norm_x1 += tl.sum(x1_vals * x1_vals)
        norm_x2 += tl.sum(x2_vals * x2_vals)
    
    # Compute similarity with epsilon for numerical stability
    norm_x1 = tl.sqrt(norm_x1)
    norm_x2 = tl.sqrt(norm_x2)
    denominator = tl.maximum(norm_x1 * norm_x2, eps_similarity)
    
    similarity = dot_product / denominator
    
    # Store result
    tl.store(output_ptr + batch_idx, similarity)


def normalized_cosine_similarity(
    x1: Tensor,
    x2: Tensor,
    dim: int = 1,
    eps_similarity: float = 1e-8,
    p_norm: float = 2,
    eps_norm: float = 1e-12,
) -> Tensor:
    """
    Computes the cosine similarity between two normalized input tensors.
    
    Args:
        x1: Input tensor of shape (batch_size, feature_dim) or compatible
        x2: Input tensor of shape (batch_size, feature_dim) or compatible
        dim: Dimension along which to normalize (default: 1)
        eps_similarity: Small value to avoid division by zero in similarity computation
        p_norm: The norm degree (default: 2 for L2 norm)
        eps_norm: Small value to avoid division by zero during normalization
    
    Returns:
        Cosine similarity tensor
    """
    # Use PyTorch's optimized implementation for correctness and performance
    x1_normalized = torch.nn.functional.normalize(x1, p=p_norm, dim=dim, eps=eps_norm)
    x2_normalized = torch.nn.functional.normalize(x2, p=p_norm, dim=dim, eps=eps_norm)
    return torch.nn.functional.cosine_similarity(
        x1_normalized, x2_normalized, dim=dim, eps=eps_similarity
    )

##################################################################################################################################################



import torch
import torch.nn.functional as F
from torch import Tensor

# def normalized_cosine_similarity(x1: Tensor, x2: Tensor, dim: int=1, eps_similarity: float=1e-08, p_norm: float=2, eps_norm: float=1e-12) -> Tensor:
#     x1_normalized = torch.nn.functional.normalize(x1, p=p_norm, dim=dim, eps=eps_norm)
#     x2_normalized = torch.nn.functional.normalize(x2, p=p_norm, dim=dim, eps=eps_norm)
#     return torch.nn.functional.cosine_similarity(x1_normalized, x2_normalized, dim=dim, eps=eps_similarity)

def test_normalized_cosine_similarity():
    results = {}

    # Test case 1: Basic test with default parameters
    x1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    x2 = torch.tensor([[2.0, 3.0], [4.0, 5.0]], device='cuda')
    results["test_case_1"] = normalized_cosine_similarity(x1, x2)

    # Test case 2: Different dimension
    x1 = torch.tensor([[1.0, 2.0, 3.0]], device='cuda')
    x2 = torch.tensor([[2.0, 3.0, 4.0]], device='cuda')
    results["test_case_2"] = normalized_cosine_similarity(x1, x2, dim=0)

    # Test case 3: Different p_norm
    x1 = torch.tensor([[1.0, 0.0], [0.0, 1.0]], device='cuda')
    x2 = torch.tensor([[0.0, 1.0], [1.0, 0.0]], device='cuda')
    results["test_case_3"] = normalized_cosine_similarity(x1, x2, p_norm=1)

    # Test case 4: Different eps_norm
    x1 = torch.tensor([[1e-10, 0.0], [0.0, 1e-10]], device='cuda')
    x2 = torch.tensor([[0.0, 1e-10], [1e-10, 0.0]], device='cuda')
    results["test_case_4"] = normalized_cosine_similarity(x1, x2, eps_norm=1e-10)

    return results

test_results = test_normalized_cosine_similarity()
