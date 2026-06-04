import torch
import triton
import triton.language as tl


@triton.jit
def _log_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for computing natural logarithm of tensor elements.
    
    Args:
        input_ptr: Pointer to input tensor
        output_ptr: Pointer to output tensor
        n_elements: Total number of elements to process
        BLOCK_SIZE: Block size for processing (compile-time constant)
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for boundary handling
    mask = offsets < n_elements
    
    # Load input values
    input_vals = tl.load(input_ptr + offsets, mask=mask)
    
    # Compute natural logarithm
    output_vals = tl.log(input_vals)
    
    # Store results
    tl.store(output_ptr + offsets, output_vals, mask=mask)


def log(input: torch.Tensor, *, out: torch.Tensor = None) -> torch.Tensor:
    """
    Computes the natural logarithm (base e) of each element in the input tensor.

    Args:
        input (Tensor): The input tensor containing the values to compute the log of.
        out (Tensor, optional): The output tensor to store the result. If not provided, a new tensor is returned.

    Returns:
        Tensor: A new tensor or the `out` tensor containing the natural logarithm of the input elements.
    """
    # Validate input
    if not isinstance(input, torch.Tensor):
        raise TypeError(f"input must be a torch.Tensor, got {type(input)}")
    
    # Create output tensor if not provided
    if out is None:
        out = torch.empty_like(input)
    else:
        if out.shape != input.shape:
            raise RuntimeError(f"output shape {out.shape} does not match input shape {input.shape}")
        if out.dtype != input.dtype:
            raise RuntimeError(f"output dtype {out.dtype} does not match input dtype {input.dtype}")
    
    # Ensure tensors are contiguous and on the same device
    input = input.contiguous()
    out = out.contiguous()
    
    if input.device != out.device:
        raise RuntimeError(f"input and output tensors must be on the same device")
    
    n_elements = input.numel()
    
    # Define block size
    BLOCK_SIZE = 1024
    
    # Calculate grid size
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    _log_kernel[grid](
        input,
        out,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

##################################################################################################################################################



import torch

def test_log():
    results = {}

    # Test case 1: Basic test with positive values
    input1 = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    results["test_case_1"] = log(input1)

    # Test case 2: Test with a tensor containing a zero
    input2 = torch.tensor([0.0, 1.0, 2.0], device='cuda')
    results["test_case_2"] = log(input2)

    # Test case 3: Test with a tensor containing negative values
    input3 = torch.tensor([-1.0, -2.0, -3.0], device='cuda')
    results["test_case_3"] = log(input3)

    # Test case 4: Test with a tensor containing a mix of positive, negative, and zero
    input4 = torch.tensor([-1.0, 0.0, 1.0], device='cuda')
    results["test_case_4"] = log(input4)

    return results

test_results = test_log()
