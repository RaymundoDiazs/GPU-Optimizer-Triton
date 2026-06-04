import triton
import triton.language as tl
import torch

@triton.jit
def adaptive_avg_pool2d_kernel(
    input_ptr, output_ptr,
    H_in, W_in, H_out, W_out,
    stride_h, stride_w,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    n = pid // (H_out * W_out)
    h_out = (pid // W_out) % H_out
    w_out = pid % W_out

    h_start = h_out * stride_h
    w_start = w_out * stride_w
    h_end = min(h_start + stride_h, H_in)
    w_end = min(w_start + stride_w, W_in)

    acc = 0.0
    count = 0
    for h in range(h_start, h_end):
        for w in range(w_start, w_end):
            acc += tl.load(input_ptr + n * H_in * W_in + h * W_in + w)
            count += 1

    avg = acc / count
    tl.store(output_ptr + n * H_out * W_out + h_out * W_out + w_out, avg)

def adaptive_avg_pool2d(input, output_size):
    # Determine input dimensions
    if input.dim() == 3:
        C, H_in, W_in = input.shape
        N = 1
    elif input.dim() == 4:
        N, C, H_in, W_in = input.shape
    else:
        raise ValueError("Input must be a 3D or 4D tensor")

    # Determine output dimensions
    if isinstance(output_size, int):
        H_out = W_out = output_size
    elif isinstance(output_size, tuple):
        H_out, W_out = output_size
    else:
        raise ValueError("Output size must be an int or a tuple of two ints")

    # Handle None case for output size
    H_out = H_out if H_out is not None else H_in
    W_out = W_out if W_out is not None else W_in

    # Calculate strides
    stride_h = H_in // H_out
    stride_w = W_in // W_out

    # Allocate output tensor
    output = torch.empty((N, C, H_out, W_out), device=input.device, dtype=input.dtype)

    # Launch Triton kernel
    grid = (N * H_out * W_out,)
    triton_kernel = adaptive_avg_pool2d_kernel[grid]
    triton_kernel(
        input, output,
        H_in, W_in, H_out, W_out,
        stride_h, stride_w,
        BLOCK_SIZE=1024
    )

    # Reshape output for 3D input
    if input.dim() == 3:
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
