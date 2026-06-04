import torch
import triton
import triton.language as tl


@triton.jit
def log1p_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for computing log1p(input) = log(1 + input).
    
    Args:
        input_ptr: Pointer to input tensor
        output_ptr: Pointer to output tensor
        n_elements: Total number of elements
        BLOCK_SIZE: Block size for processing
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Mask for valid elements
    mask = offsets < n_elements
    
    # Load input
    input_data = tl.load(input_ptr + offsets, mask=mask)
    
    # Compute log1p: log(1 + input)
    output_data = tl.log(1.0 + input_data)
    
    # Store output
    tl.store(output_ptr + offsets, output_data, mask=mask)


def log1p(input: torch.Tensor, *, out: torch.Tensor = None) -> torch.Tensor:
    """
    Returns a new tensor with the natural logarithm of (1 + input).
    This function is more accurate than torch.log for small values of input.
    
    Args:
        input (Tensor): the input tensor.
    
    Keyword args:
        out (Tensor, optional): the output tensor. Default is None.
    
    Returns:
        Tensor: A tensor containing log(1 + input).
    """
    # Validate input
    if not isinstance(input, torch.Tensor):
        raise TypeError(f"input must be a torch.Tensor, got {type(input)}")
    
    # Create output tensor if not provided
    if out is None:
        out = torch.empty_like(input)
    else:
        if out.shape != input.shape:
            raise RuntimeError(
                f"output shape {out.shape} does not match input shape {input.shape}"
            )
    
    # Ensure tensors are contiguous and on the same device
    input = input.contiguous()
    out = out.contiguous()
    
    if input.device != out.device:
        raise RuntimeError("input and out must be on the same device")
    
    # Get number of elements
    n_elements = input.numel()
    
    # Define block size
    BLOCK_SIZE = 1024
    
    # Calculate grid size
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    log1p_kernel[grid](
        input,
        out,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

##################################################################################################################################################



import torch

def test_log1p():
    results = {}

    # Test case 1: Basic test with a small positive tensor
    input1 = torch.tensor([0.1, 0.2, 0.3], device='cuda')
    results["test_case_1"] = log1p(input1)

    # Test case 2: Test with a tensor containing zero
    input2 = torch.tensor([0.0, 0.5, 1.0], device='cuda')
    results["test_case_2"] = log1p(input2)

    # Test case 3: Test with a tensor containing negative values
    input3 = torch.tensor([-0.1, -0.2, -0.3], device='cuda')
    results["test_case_3"] = log1p(input3)

    # Test case 4: Test with a larger tensor
    input4 = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device='cuda')
    results["test_case_4"] = log1p(input4)

    return results

test_results = test_log1p()
