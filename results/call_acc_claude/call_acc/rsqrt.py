import torch
import triton
import triton.language as tl

@triton.jit
def rsqrt_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for computing reciprocal of square root.
    rsqrt(x) = 1 / sqrt(x)
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for valid elements
    mask = offsets < n_elements
    
    # Load input values
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Compute rsqrt: 1 / sqrt(x)
    # Handle negative values by producing NaN
    result = 1.0 / tl.sqrt(x)
    
    # Store results
    tl.store(output_ptr + offsets, result, mask=mask)


def rsqrt(input: torch.Tensor, *, out: torch.Tensor = None) -> torch.Tensor:
    """
    Returns a new tensor with the reciprocal of the square-root of each 
    of the elements of the input tensor.
    
    Args:
        input (Tensor): the input tensor.
    
    Keyword args:
        out (Tensor, optional): the output tensor. Default is None.
    
    Returns:
        Tensor: A tensor with rsqrt of each element in the input.
    """
    # Validate input
    if not isinstance(input, torch.Tensor):
        raise TypeError(f"input must be a torch.Tensor, got {type(input)}")
    
    # Create output tensor if not provided
    if out is None:
        output = torch.empty_like(input)
    else:
        if not isinstance(out, torch.Tensor):
            raise TypeError(f"out must be a torch.Tensor, got {type(out)}")
        if out.shape != input.shape:
            raise RuntimeError(f"out shape {out.shape} does not match input shape {input.shape}")
        output = out
    
    # Ensure input is contiguous
    input = input.contiguous()
    output = output.contiguous()
    
    # Get number of elements
    n_elements = input.numel()
    
    # Define block size
    BLOCK_SIZE = 1024
    
    # Calculate grid size
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    rsqrt_kernel[grid](
        input,
        output,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

##################################################################################################################################################



import torch

def test_rsqrt():
    results = {}

    # Test case 1: Positive elements
    input1 = torch.tensor([4.0, 16.0, 25.0], device='cuda')
    results["test_case_1"] = rsqrt(input1)

    # Test case 2: Contains zero
    input2 = torch.tensor([0.0, 1.0, 4.0], device='cuda')
    results["test_case_2"] = rsqrt(input2)

    # Test case 3: Contains negative elements
    input3 = torch.tensor([-1.0, 4.0, 9.0], device='cuda')
    results["test_case_3"] = rsqrt(input3)

    # Test case 4: All elements are zero
    input4 = torch.tensor([0.0, 0.0, 0.0], device='cuda')
    results["test_case_4"] = rsqrt(input4)

    return results

test_results = test_rsqrt()
