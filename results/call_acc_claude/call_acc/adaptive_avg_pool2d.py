import torch
import triton
import triton.language as tl
import math


@triton.jit
def adaptive_avg_pool2d_kernel(
    input_ptr,
    output_ptr,
    N, C, H_in, W_in,
    H_out, W_out,
    input_stride_n, input_stride_c, input_stride_h, input_stride_w,
    output_stride_n, output_stride_c, output_stride_h, output_stride_w,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for 2D adaptive average pooling.
    
    Each thread block processes one output element and computes the average
    of the corresponding input region.
    """
    pid = tl.program_id(0)
    
    # Compute output indices from linear pid
    # pid maps to (n, c, h_out, w_out)
    total_elements = N * C * H_out * W_out
    
    if pid >= total_elements:
        return
    
    # Decompose pid into (n, c, h_out, w_out)
    temp = pid
    w_out_idx = temp % W_out
    temp //= W_out
    h_out_idx = temp % H_out
    temp //= H_out
    c_idx = temp % C
    n_idx = temp // C
    
    # Compute input region boundaries for this output element
    # Using adaptive pooling formula: start = floor(i * H_in / H_out), end = ceil((i+1) * H_in / H_out)
    h_start = tl.cast(h_out_idx * H_in / H_out, tl.int32)
    h_end = tl.cast((h_out_idx + 1) * H_in / H_out, tl.int32)
    h_end = tl.minimum(h_end, H_in)
    
    w_start = tl.cast(w_out_idx * W_in / W_out, tl.int32)
    w_end = tl.cast((w_out_idx + 1) * W_in / W_out, tl.int32)
    w_end = tl.minimum(w_end, W_in)
    
    # Compute average over the input region
    sum_val = 0.0
    count = 0
    
    for h in range(h_start, h_end):
        for w in range(w_start, w_end):
            input_offset = (n_idx * input_stride_n + 
                          c_idx * input_stride_c + 
                          h * input_stride_h + 
                          w * input_stride_w)
            sum_val += tl.load(input_ptr + input_offset)
            count += 1
    
    # Compute average
    avg_val = sum_val / tl.cast(count, tl.float32)
    
    # Write output
    output_offset = (n_idx * output_stride_n + 
                    c_idx * output_stride_c + 
                    h_out_idx * output_stride_h + 
                    w_out_idx * output_stride_w)
    tl.store(output_ptr + output_offset, avg_val)


def adaptive_avg_pool2d(input: torch.Tensor, output_size) -> torch.Tensor:
    """
    Apply 2D adaptive average pooling over an input signal.

    Args:
        input (Tensor): The input tensor, either of shape (N, C, H_in, W_in) or (C, H_in, W_in).
        output_size (int or tuple): The target output size (single integer or tuple of two integers).
            - If an integer, the output will be square: (output_size, output_size).
            - If a tuple, the first element corresponds to the height, and the second element corresponds to the width of the output.

    Returns:
        Tensor: The output tensor with the specified output size.

    Example:
        >>> import torch
        >>> input = torch.randn(1, 64, 8, 9)
        >>> output = adaptive_avg_pool2d(input, (5, 7))
        >>> print(output.shape)  # Output shape: (1, 64, 5, 7)
    """
    # Ensure input is contiguous
    input = input.contiguous()
    
    # Handle output_size parameter
    if isinstance(output_size, int):
        H_out, W_out = output_size, output_size
    else:
        H_out, W_out = output_size[0], output_size[1]
    
    # Handle input shape (with or without batch dimension)
    if input.dim() == 3:
        # Shape: (C, H_in, W_in)
        C, H_in, W_in = input.shape
        N = 1
        input = input.unsqueeze(0)  # Add batch dimension
        squeeze_output = True
    else:
        # Shape: (N, C, H_in, W_in)
        N, C, H_in, W_in = input.shape
        squeeze_output = False
    
    # Create output tensor
    output = torch.zeros(
        (N, C, H_out, W_out),
        dtype=input.dtype,
        device=input.device
    ).contiguous()
    
    # Get strides
    input_stride_n, input_stride_c, input_stride_h, input_stride_w = input.stride()
    output_stride_n, output_stride_c, output_stride_h, output_stride_w = output.stride()
    
    # Launch kernel
    grid = (N * C * H_out * W_out,)
    
    adaptive_avg_pool2d_kernel[grid](
        input_ptr=input,
        output_ptr=output,
        N=N, C=C, H_in=H_in, W_in=W_in,
        H_out=H_out, W_out=W_out,
        input_stride_n=input_stride_n,
        input_stride_c=input_stride_c,
        input_stride_h=input_stride_h,
        input_stride_w=input_stride_w,
        output_stride_n=output_stride_n,
        output_stride_c=output_stride_c,
        output_stride_h=output_stride_h,
        output_stride_w=output_stride_w,
        BLOCK_SIZE=128,
    )
    
    # Remove batch dimension if input was 3D
    if squeeze_output:
        output = output.squeeze(0)
    
    return output

##################################################################################################################################################



import torch
from adaptive_avg_pool2d import adaptive_avg_pool2d

def test_adaptive_avg_pool2d():
    results = {}
    
    # Test case 1: input shape (N, C, H_in, W_in), output_size as integer
    input1 = torch.randn(1, 64, 8, 9).cuda()
    output1 = adaptive_avg_pool2d(input1, 5)
    results["test_case_1"] = output1
    
    # Test case 2: input shape (N, C, H_in, W_in), output_size as tuple
    input2 = torch.randn(1, 64, 8, 9).cuda()
    output2 = adaptive_avg_pool2d(input2, (5, 7))
    results["test_case_2"] = output2
    
    # Test case 3: input shape (C, H_in, W_in), output_size as integer
    input3 = torch.randn(64, 8, 9).cuda()
    output3 = adaptive_avg_pool2d(input3, 5)
    results["test_case_3"] = output3
    
    # Test case 4: input shape (C, H_in, W_in), output_size as tuple
    input4 = torch.randn(64, 8, 9).cuda()
    output4 = adaptive_avg_pool2d(input4, (5, 7))
    results["test_case_4"] = output4
    
    return results

test_results = test_adaptive_avg_pool2d()
