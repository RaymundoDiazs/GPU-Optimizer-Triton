import torch
import triton
import triton.language as tl
import math

@triton.jit
def relu_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    Applies ReLU activation: output = max(input, 0)
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(input_ptr + offsets, mask=mask)
    y = tl.maximum(x, 0.0)
    tl.store(output_ptr + offsets, y, mask=mask)


@triton.jit
def fractional_max_pool2d_kernel(
    input_ptr,
    output_ptr,
    indices_ptr,
    batch_size,
    channels,
    input_h,
    input_w,
    output_h,
    output_w,
    kernel_h,
    kernel_w,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Applies fractional max pooling with random pooling regions.
    Computes max value and indices for each pooling region.
    """
    pid = tl.program_id(axis=0)
    
    # Each thread block processes one output element
    if pid >= batch_size * channels * output_h * output_w:
        return
    
    # Decompose linear index to (batch, channel, out_h, out_w)
    out_w_idx = pid % output_w
    out_h_idx = (pid // output_w) % output_h
    channel_idx = (pid // (output_w * output_h)) % channels
    batch_idx = pid // (channels * output_h * output_w)
    
    # Compute random pooling region boundaries
    # Fractional max pooling uses random alpha values for region boundaries
    alpha_h = input_h / output_h
    alpha_w = input_w / output_w
    
    start_h = tl.cast(out_h_idx * alpha_h, tl.int32)
    end_h = tl.cast((out_h_idx + 1) * alpha_h, tl.int32)
    start_w = tl.cast(out_w_idx * alpha_w, tl.int32)
    end_w = tl.cast((out_w_idx + 1) * alpha_w, tl.int32)
    
    # Clamp boundaries
    start_h = tl.maximum(start_h, 0)
    end_h = tl.minimum(end_h, input_h)
    start_w = tl.maximum(start_w, 0)
    end_w = tl.minimum(end_w, input_w)
    
    # Find max value and its index in the pooling region
    max_val = -float('inf')
    max_idx = 0
    
    for h in range(start_h, end_h):
        for w in range(start_w, end_w):
            idx = batch_idx * (channels * input_h * input_w) + channel_idx * (input_h * input_w) + h * input_w + w
            val = tl.load(input_ptr + idx)
            
            # Update max if current value is larger
            is_greater = val > max_val
            max_val = tl.where(is_greater, val, max_val)
            max_idx = tl.where(is_greater, idx, max_idx)
    
    # Store output and indices
    out_idx = batch_idx * (channels * output_h * output_w) + channel_idx * (output_h * output_w) + out_h_idx * output_w + out_w_idx
    tl.store(output_ptr + out_idx, max_val)
    tl.store(indices_ptr + out_idx, max_idx)


def fused_fractional_max_pool2d_with_relu(
    input: torch.Tensor,
    kernel_size,
    output_size=None,
    output_ratio=None,
    return_indices=False,
) -> torch.Tensor:
    """
    Applies ReLU activation followed by fractional max pooling over an input tensor.
    
    Args:
        input (torch.Tensor): Input tensor of shape (batch, channels, height, width).
        kernel_size (int or Tuple[int, int]): Size of the pooling window.
        output_size (Tuple[int, int], optional): Target output size (height, width).
        output_ratio (Tuple[float, float], optional): If set, output size is scaled as a ratio of the input size.
        return_indices (bool, optional): If True, return the max pooling indices along with the output.

    Returns:
        torch.Tensor or Tuple[torch.Tensor, torch.Tensor]: 
            The output tensor after ReLU and fractional max pooling.
            If return_indices=True, returns (output, indices).
    """
    assert input.dim() == 4, "Input must be 4D tensor (batch, channels, height, width)"
    assert input.is_cuda, "Input must be on CUDA device"
    
    # Normalize kernel_size
    if isinstance(kernel_size, int):
        kernel_h = kernel_w = kernel_size
    else:
        kernel_h, kernel_w = kernel_size
    
    batch_size, channels, input_h, input_w = input.shape
    
    # Compute output size
    if output_size is not None:
        output_h, output_w = output_size
    elif output_ratio is not None:
        output_h = int(input_h * output_ratio[0])
        output_w = int(input_w * output_ratio[1])
    else:
        raise ValueError("Either output_size or output_ratio must be specified")
    
    # Step 1: Apply ReLU
    relu_output = torch.empty_like(input)
    n_elements = input.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    relu_kernel[grid](input, relu_output, n_elements, BLOCK_SIZE=1024)
    
    # Step 2: Apply Fractional Max Pooling
    output = torch.empty(
        (batch_size, channels, output_h, output_w),
        dtype=input.dtype,
        device=input.device,
    )
    
    if return_indices:
        indices = torch.empty(
            (batch_size, channels, output_h, output_w),
            dtype=torch.int64,
            device=input.device,
        )
    else:
        indices = torch.empty(
            (batch_size, channels, output_h, output_w),
            dtype=torch.int64,
            device=input.device,
        )
    
    grid_size = batch_size * channels * output_h * output_w
    fractional_max_pool2d_kernel[(grid_size,)](
        relu_output,
        output,
        indices,
        batch_size,
        channels,
        input_h,
        input_w,
        output_h,
        output_w,
        kernel_h,
        kernel_w,
        BLOCK_SIZE=1024,
    )
    
    if return_indices:
        return output, indices
    else:
        return output

##################################################################################################################################################



import torch
import torch.nn.functional as F

# def fused_fractional_max_pool2d_with_relu(input: torch.Tensor, kernel_size, output_size=None, output_ratio=None, return_indices=False) -> torch.Tensor:
#     relu_output = F.relu(input)
#     pooled_output = F.fractional_max_pool2d(relu_output, kernel_size=kernel_size, output_size=output_size, output_ratio=output_ratio, return_indices=return_indices)
#     return pooled_output

def test_fused_fractional_max_pool2d_with_relu():
    results = {}
    
    # Test case 1: Basic functionality with kernel_size and output_size
    input_tensor = torch.randn(1, 1, 8, 8, device='cuda')
    kernel_size = (2, 2)
    output_size = (4, 4)
    results["test_case_1"] = fused_fractional_max_pool2d_with_relu(input_tensor, kernel_size, output_size=output_size)
    
    # Test case 2: Using output_ratio instead of output_size
    output_ratio = (0.5, 0.5)
    results["test_case_2"] = fused_fractional_max_pool2d_with_relu(input_tensor, kernel_size, output_ratio=output_ratio)
    
    # Test case 3: Return indices along with the pooled output
    results["test_case_3"] = fused_fractional_max_pool2d_with_relu(input_tensor, kernel_size, output_size=output_size, return_indices=True)
    
    # Test case 4: Larger kernel size
    kernel_size = (3, 3)
    results["test_case_4"] = fused_fractional_max_pool2d_with_relu(input_tensor, kernel_size, output_size=output_size)
    
    return results

test_results = test_fused_fractional_max_pool2d_with_relu()
