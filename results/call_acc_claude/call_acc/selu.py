import torch
import triton
import triton.language as tl
import math

# Triton Kernel
@triton.jit
def selu_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for SELU activation function.
    SELU(x) = scale * (max(0, x) + min(0, alpha * (exp(x) - 1)))
    """
    # Constants
    alpha = 1.6732632423543772
    scale = 1.0507009873554805
    
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Mask for valid elements
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(input_ptr + offsets, mask=mask)
    
    # Compute SELU: scale * (max(0, x) + min(0, alpha * (exp(x) - 1)))
    # max(0, x)
    positive_part = tl.maximum(x, 0.0)
    
    # alpha * (exp(x) - 1)
    exp_part = alpha * (tl.exp(x) - 1.0)
    
    # min(0, alpha * (exp(x) - 1))
    negative_part = tl.minimum(exp_part, 0.0)
    
    # scale * (max(0, x) + min(0, alpha * (exp(x) - 1)))
    output = scale * (positive_part + negative_part)
    
    # Store output
    tl.store(output_ptr + offsets, output, mask=mask)


# Wrapper Function
def selu(input: torch.Tensor, inplace: bool = False) -> torch.Tensor:
    """
    Applies the element-wise SELU (Scaled Exponential Linear Unit) function to the input tensor.
    
    The SELU function is defined as:
    SELU(x) = scale * (max(0, x) + min(0, alpha * (exp(x) - 1)))
    where alpha is approximately 1.673 and scale is approximately 1.051.
    
    Args:
    - input (torch.Tensor): The input tensor.
    - inplace (bool, optional): If set to True, will do the operation in-place. Default is False.

    Returns:
    - torch.Tensor: The resulting tensor after applying SELU function.
    """
    # Validate input
    if not input.is_cuda:
        raise RuntimeError("Input tensor must be on CUDA device")
    
    # Determine output tensor
    if inplace:
        output = input
    else:
        output = torch.empty_like(input)
    
    # Get number of elements
    n_elements = input.numel()
    
    # Define block size
    BLOCK_SIZE = 1024
    
    # Calculate grid size
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    selu_kernel[grid](
        input,
        output,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

##################################################################################################################################################



def test_selu():
    # Initialize a dictionary to store test results
    results = {}

    # Test case 1: Positive values
    input_tensor_1 = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    results["test_case_1"] = selu(input_tensor_1)

    # Test case 2: Negative values
    input_tensor_2 = torch.tensor([-1.0, -2.0, -3.0], device='cuda')
    results["test_case_2"] = selu(input_tensor_2)

    # Test case 3: Mixed values
    input_tensor_3 = torch.tensor([-1.0, 0.0, 1.0], device='cuda')
    results["test_case_3"] = selu(input_tensor_3)

    # Test case 4: Zero values
    input_tensor_4 = torch.tensor([0.0, 0.0, 0.0], device='cuda')
    results["test_case_4"] = selu(input_tensor_4)

    return results

test_results = test_selu()
