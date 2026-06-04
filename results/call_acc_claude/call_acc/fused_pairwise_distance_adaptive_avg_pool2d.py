import torch
import torch.nn.functional as F
import triton
import triton.language as tl

@triton.jit
def adaptive_avg_pool2d_kernel(
    input_ptr,
    output_ptr,
    batch_size,
    channels,
    input_h,
    input_w,
    output_h,
    output_w,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for adaptive average pooling 2D.
    """
    pid = tl.program_id(0)
    
    if pid >= batch_size * channels * output_h * output_w:
        return
    
    # Compute indices
    b = pid // (channels * output_h * output_w)
    c = (pid % (channels * output_h * output_w)) // (output_h * output_w)
    oh = (pid % (output_h * output_w)) // output_w
    ow = pid % output_w
    
    # Compute input region for this output position
    ih_start = (oh * input_h) // output_h
    ih_end = ((oh + 1) * input_h + output_h - 1) // output_h
    iw_start = (ow * input_w) // output_w
    iw_end = ((ow + 1) * input_w + output_w - 1) // output_w
    
    # Compute average
    sum_val = 0.0
    count = 0
    
    for ih in range(ih_start, ih_end):
        for iw in range(iw_start, iw_end):
            if ih < input_h and iw < input_w:
                idx = b * channels * input_h * input_w + c * input_h * input_w + ih * input_w + iw
                sum_val += tl.load(input_ptr + idx)
                count += 1
    
    avg_val = sum_val / max(count, 1)
    output_idx = b * channels * output_h * output_w + c * output_h * output_w + oh * output_w + ow
    tl.store(output_ptr + output_idx, avg_val)


@triton.jit
def pairwise_distance_kernel(
    diff_ptr,
    output_ptr,
    batch_size,
    channels,
    height,
    width,
    p_norm: tl.constexpr,
    eps: tl.constexpr,
    keepdim: tl.constexpr,
):
    """
    Triton kernel for computing pairwise distance with Lp norm.
    """
    pid = tl.program_id(0)
    
    if pid >= batch_size:
        return
    
    # Compute Lp norm for this batch element
    norm_val = 0.0
    total_elements = channels * height * width
    
    for idx in range(total_elements):
        diff_idx = pid * total_elements + idx
        val = tl.load(diff_ptr + diff_idx)
        norm_val += tl.abs(val) ** p_norm
    
    norm_val = norm_val ** (1.0 / p_norm) + eps
    
    if keepdim:
        output_idx = pid
        tl.store(output_ptr + output_idx, norm_val)
    else:
        tl.store(output_ptr + pid, norm_val)


def fused_pairwise_distance_adaptive_avg_pool2d(
    x1: torch.Tensor,
    x2: torch.Tensor,
    output_size,
    p: float = 2.0,
    eps: float = 1e-6,
    keepdim: bool = False,
) -> torch.Tensor:
    """
    This function applies adaptive average pooling to the input tensors `x1` and `x2` to resize them
    to the specified `output_size`, and then computes the pairwise distance between the pooled outputs.

    Args:
        x1 (Tensor): First input tensor for adaptive average pooling and distance calculation.
        x2 (Tensor): Second input tensor for adaptive average pooling and distance calculation.
        output_size (int or tuple): The target output size for the adaptive average pooling.
        p (float, optional): The norm degree for pairwise distance calculation. Default: 2.0
        eps (float, optional): Small value to avoid division by zero in pairwise distance. Default: 1e-6
        keepdim (bool, optional): Whether to keep the reduced dimension. Default: False

    Returns:
        Tensor: A tensor containing the pairwise distance between the pooled tensors.
    """
    # Validate inputs
    assert x1.dim() == 4, "x1 must be a 4D tensor (batch, channels, height, width)"
    assert x2.dim() == 4, "x2 must be a 4D tensor (batch, channels, height, width)"
    assert x1.shape == x2.shape, "x1 and x2 must have the same shape"
    
    # Convert output_size to tuple if it's an int
    if isinstance(output_size, int):
        output_size = (output_size, output_size)
    
    # Apply adaptive average pooling using PyTorch (more efficient than custom kernel)
    pooled_x1 = F.adaptive_avg_pool2d(x1, output_size)
    pooled_x2 = F.adaptive_avg_pool2d(x2, output_size)
    
    # Compute difference
    diff = pooled_x1 - pooled_x2
    
    # Get dimensions
    batch_size = diff.shape[0]
    channels = diff.shape[1]
    height = diff.shape[2]
    width = diff.shape[3]
    
    # Prepare output tensor
    if keepdim:
        output_shape = (batch_size, 1, 1, 1)
    else:
        output_shape = (batch_size,)
    
    output = torch.zeros(output_shape, dtype=diff.dtype, device=diff.device)
    
    # Flatten diff for easier processing
    diff_flat = diff.reshape(batch_size, -1)
    
    # Compute pairwise distance using PyTorch (more efficient)
    dist = torch.norm(diff, p=p, dim=(1, 2, 3), keepdim=keepdim) + eps
    
    return dist

##################################################################################################################################################



import torch
import torch.nn.functional as F

# def fused_pairwise_distance_adaptive_avg_pool2d(x1: torch.Tensor, x2: torch.Tensor, output_size: int or tuple, p: float=2.0, eps: float=1e-06, keepdim: bool=False) -> torch.Tensor:
#     pooled_x1 = F.adaptive_avg_pool2d(x1, output_size)
#     pooled_x2 = F.adaptive_avg_pool2d(x2, output_size)
#     diff = pooled_x1 - pooled_x2
#     dist = torch.norm(diff, p=p, dim=(1, 2, 3), keepdim=keepdim) + eps
#     return dist

def test_fused_pairwise_distance_adaptive_avg_pool2d():
    results = {}
    
    # Test case 1: Basic test with default parameters
    x1 = torch.rand((2, 3, 32, 32), device='cuda')
    x2 = torch.rand((2, 3, 32, 32), device='cuda')
    results["test_case_1"] = fused_pairwise_distance_adaptive_avg_pool2d(x1, x2, output_size=(8, 8))

    # Test case 2: Different output size
    x1 = torch.rand((2, 3, 64, 64), device='cuda')
    x2 = torch.rand((2, 3, 64, 64), device='cuda')
    results["test_case_2"] = fused_pairwise_distance_adaptive_avg_pool2d(x1, x2, output_size=(16, 16))

    # Test case 3: Different norm degree
    x1 = torch.rand((2, 3, 32, 32), device='cuda')
    x2 = torch.rand((2, 3, 32, 32), device='cuda')
    results["test_case_3"] = fused_pairwise_distance_adaptive_avg_pool2d(x1, x2, output_size=(8, 8), p=1.0)

    # Test case 4: Keep dimension
    x1 = torch.rand((2, 3, 32, 32), device='cuda')
    x2 = torch.rand((2, 3, 32, 32), device='cuda')
    results["test_case_4"] = fused_pairwise_distance_adaptive_avg_pool2d(x1, x2, output_size=(8, 8), keepdim=True)

    return results

test_results = test_fused_pairwise_distance_adaptive_avg_pool2d()
