import torch
import triton
import triton.language as tl


@triton.jit
def exp_sqrt_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel that computes exp(sqrt(x)) for each element.
    
    Args:
        input_ptr: Pointer to input tensor
        output_ptr: Pointer to output tensor
        n_elements: Total number of elements
        BLOCK_SIZE: Block size for processing
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for valid elements
    mask = offsets < n_elements
    
    # Load input data
    x = tl.load(input_ptr + offsets, mask=mask)
    
    # Compute exp(x)
    exp_x = tl.exp(x)
    
    # Compute sqrt(exp(x))
    result = tl.sqrt(exp_x)
    
    # Store result
    tl.store(output_ptr + offsets, result, mask=mask)


def exp_sqrt(input: torch.Tensor, out: torch.Tensor = None) -> torch.Tensor:
    """
    Computes the exponential of each element in the input tensor,
    followed by calculating the square root of the result.
    
    Args:
        input (Tensor): The input tensor.
        out (Tensor, optional): The output tensor.

    Returns:
        Tensor: A tensor where each element is the result of applying 
                exponential followed by square root to each element of input.
    """
    # Validate input
    assert input.is_cuda, "Input tensor must be on CUDA device"
    
    # Create output tensor if not provided
    if out is None:
        out = torch.empty_like(input)
    else:
        assert out.shape == input.shape, "Output tensor shape must match input tensor shape"
        assert out.is_cuda, "Output tensor must be on CUDA device"
    
    # Get number of elements
    n_elements = input.numel()
    
    # Define block size
    BLOCK_SIZE = 1024
    
    # Calculate grid size
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    exp_sqrt_kernel[grid](
        input,
        out,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

##################################################################################################################################################



import torch

def test_exp_sqrt():
    results = {}

    # Test case 1: Basic functionality with a simple tensor
    input1 = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    results["test_case_1"] = exp_sqrt(input1)

    # Test case 2: Test with a tensor containing negative values
    input2 = torch.tensor([-1.0, -2.0, -3.0], device='cuda')
    results["test_case_2"] = exp_sqrt(input2)

    # Test case 3: Test with a tensor containing zero
    input3 = torch.tensor([0.0, 0.0, 0.0], device='cuda')
    results["test_case_3"] = exp_sqrt(input3)

    # Test case 4: Test with out parameter
    input4 = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    out4 = torch.empty(3, device='cuda')
    results["test_case_4"] = exp_sqrt(input4, out=out4)

    return results

test_results = test_exp_sqrt()
