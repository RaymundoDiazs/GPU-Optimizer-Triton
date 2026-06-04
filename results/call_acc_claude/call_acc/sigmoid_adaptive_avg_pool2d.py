import triton
import triton.language as tl
import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Union, Tuple
import math


@triton.jit
def sigmoid_kernel(
    output_ptr,
    input_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel to apply sigmoid activation function element-wise.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(input_ptr + offsets, mask=mask)
    
    # Apply sigmoid: 1 / (1 + exp(-x))
    output = 1.0 / (1.0 + tl.exp(-x))
    
    # Store output
    tl.store(output_ptr + offsets, output, mask=mask)


def sigmoid_adaptive_avg_pool2d(input: Tensor, output_size: Union[int, Tuple[int, int]]) -> Tensor:
    """
    Applies a 2D adaptive average pooling over an input tensor, followed by the sigmoid activation function applied element-wise.
    
    Args:
        input (Tensor): The input tensor with shape (batch_size, channels, height, width).
        output_size (Union[int, Tuple[int, int]]): The target output size of the pooled tensor.
    
    Returns:
        Tensor: The result tensor after applying adaptive average pooling and sigmoid activation.
    """
    # Validate input
    assert input.dim() == 4, "Input tensor must be 4-dimensional (batch_size, channels, height, width)"
    assert input.is_cuda, "Input tensor must be on CUDA device"
    
    # Normalize output_size to tuple
    if isinstance(output_size, int):
        output_size = (output_size, output_size)
    
    # Step 1: Apply adaptive average pooling using PyTorch
    pooled_output = F.adaptive_avg_pool2d(input, output_size)
    
    # Step 2: Apply sigmoid activation using Triton kernel
    output = torch.empty_like(pooled_output)
    n_elements = pooled_output.numel()
    
    # Ensure contiguous memory layout
    pooled_output = pooled_output.contiguous()
    output = output.contiguous()
    
    # Configure block size
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch Triton kernel
    sigmoid_kernel[grid](
        output,
        pooled_output,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

##################################################################################################################################################



def test_sigmoid_adaptive_avg_pool2d():
    # Initialize a dictionary to store the results of each test case
    results = {}

    # Test case 1: Basic test with a 4D tensor and output size as an integer
    input_tensor1 = torch.randn(1, 3, 8, 8, device='cuda')  # Batch size 1, 3 channels, 8x8 size
    output_size1 = 4
    result1 = sigmoid_adaptive_avg_pool2d(input_tensor1, output_size1)
    results["test_case_1"] = result1

    # Test case 2: Test with a 4D tensor and output size as a tuple
    input_tensor2 = torch.randn(2, 3, 10, 10, device='cuda')  # Batch size 2, 3 channels, 10x10 size
    output_size2 = (5, 5)
    result2 = sigmoid_adaptive_avg_pool2d(input_tensor2, output_size2)
    results["test_case_2"] = result2

    # Test case 3: Test with a larger batch size
    input_tensor3 = torch.randn(4, 3, 16, 16, device='cuda')  # Batch size 4, 3 channels, 16x16 size
    output_size3 = (8, 8)
    result3 = sigmoid_adaptive_avg_pool2d(input_tensor3, output_size3)
    results["test_case_3"] = result3

    # Test case 4: Test with a single channel
    input_tensor4 = torch.randn(1, 1, 12, 12, device='cuda')  # Batch size 1, 1 channel, 12x12 size
    output_size4 = (6, 6)
    result4 = sigmoid_adaptive_avg_pool2d(input_tensor4, output_size4)
    results["test_case_4"] = result4

    return results

test_results = test_sigmoid_adaptive_avg_pool2d()
