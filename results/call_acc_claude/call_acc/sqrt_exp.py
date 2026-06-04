import torch
import triton
import triton.language as tl


@triton.jit
def sqrt_exp_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel that computes e^(sqrt(input)) for each element.
    
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
    
    # Load input values
    input_vals = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Compute sqrt(input)
    sqrt_vals = tl.sqrt(input_vals)
    
    # Compute exp(sqrt(input))
    output_vals = tl.exp(sqrt_vals)
    
    # Store results
    tl.store(output_ptr + offsets, output_vals, mask=mask)


def sqrt_exp(input, out=None) -> torch.Tensor:
    """
    Computes the square root of each element in :attr:`input`, 
    and then applies the exponential function to the square-rooted values.
    
    Args:
        input (Tensor): The input tensor.
        out (Tensor, optional): The output tensor.
    
    Returns:
        Tensor: A tensor containing e^(sqrt(input_i)) for each element in input.
    
    Example:
        >>> import torch
        >>> a = torch.tensor([0.25, 1.0, 4.0, 9.0])
        >>> result = sqrt_exp(a)
        >>> print(result)
        tensor([ 1.2840,  2.7183,  7.3891, 20.0855])
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
    grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    # Launch kernel
    sqrt_exp_kernel[grid](
        input,
        out,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

##################################################################################################################################################



import torch

def test_sqrt_exp():
    results = {}

    # Test case 1: Basic functionality with GPU tensor
    a = torch.tensor([0.25, 1.0, 4.0, 9.0], device='cuda')
    results["test_case_1"] = sqrt_exp(a)

    # Test case 2: Empty tensor
    b = torch.tensor([], device='cuda')
    results["test_case_2"] = sqrt_exp(b)

    # Test case 3: Tensor with zero values
    c = torch.tensor([0.0, 0.0, 0.0], device='cuda')
    results["test_case_3"] = sqrt_exp(c)

    # Test case 4: Using the out parameter
    d = torch.tensor([0.25, 1.0, 4.0, 9.0], device='cuda')
    out_tensor = torch.empty_like(d)
    results["test_case_4"] = sqrt_exp(d, out=out_tensor)

    return results

test_results = test_sqrt_exp()
