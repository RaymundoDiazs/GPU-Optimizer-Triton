import torch
import triton
import triton.language as tl


@triton.jit
def _cos_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for computing cosine of input tensor elements.
    
    Args:
        input_ptr: Pointer to input tensor
        output_ptr: Pointer to output tensor
        n_elements: Total number of elements
        BLOCK_SIZE: Block size for processing
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Mask to handle boundary cases
    mask = offsets < n_elements
    
    # Load input data
    x = tl.load(input_ptr + offsets, mask=mask)
    
    # Compute cosine
    y = tl.cos(x)
    
    # Store output data
    tl.store(output_ptr + offsets, y, mask=mask)


def cos(input, *, out=None) -> torch.Tensor:
    """
    Returns a new tensor with the cosine of the elements of the input tensor.
    
    Args:
        input (Tensor): the input tensor.
    
    Keyword args:
        out (Tensor, optional): the output tensor. Default: None
    
    Returns:
        Tensor: A new tensor with cosine of input elements.
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
        raise RuntimeError(
            f"input device {input.device} does not match output device {out.device}"
        )
    
    n_elements = input.numel()
    
    # Define block size
    BLOCK_SIZE = 1024
    
    # Calculate grid size
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    _cos_kernel[grid](
        input,
        out,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

##################################################################################################################################################



import torch

def test_cos():
    results = {}

    # Test case 1: Single positive value
    input_tensor_1 = torch.tensor([0.0], device='cuda')
    results["test_case_1"] = cos(input_tensor_1)

    # Test case 2: Single negative value
    input_tensor_2 = torch.tensor([-3.14159265 / 2], device='cuda')
    results["test_case_2"] = cos(input_tensor_2)

    # Test case 3: Multiple values
    input_tensor_3 = torch.tensor([0.0, 3.14159265 / 2, 3.14159265], device='cuda')
    results["test_case_3"] = cos(input_tensor_3)

    # Test case 4: Large tensor
    input_tensor_4 = torch.linspace(-3.14159265, 3.14159265, steps=1000, device='cuda')
    results["test_case_4"] = cos(input_tensor_4)

    return results

test_results = test_cos()
