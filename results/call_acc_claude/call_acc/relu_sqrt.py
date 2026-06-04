import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def relu_sqrt_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel that applies ReLU followed by sqrt to each element.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for valid elements
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(input_ptr + offsets, mask=mask)
    
    # Apply ReLU: max(x, 0)
    x_relu = tl.maximum(x, 0.0)
    
    # Apply sqrt
    output = tl.sqrt(x_relu)
    
    # Store output
    tl.store(output_ptr + offsets, output, mask=mask)


def relu_sqrt(input: Tensor, inplace: bool = False, out: Tensor = None) -> Tensor:
    """
    Applies the rectified linear unit (ReLU) function to each element in input,
    and then computes the square root of the result.
    
    Args:
        input (Tensor): The input tensor.
        inplace (bool, optional): If True, modifies input in-place (if possible). Default is False.
        out (Tensor, optional): The output tensor.
    
    Returns:
        Tensor: The result of applying relu followed by sqrt.
    
    Example:
        >>> import torch
        >>> a = torch.tensor([-1.0, 0.0, 4.0, 9.0])
        >>> result = relu_sqrt(a)
        >>> print(result)
        tensor([0.0000, 0.0000, 2.0000, 3.0000])
        >>> result = relu_sqrt(a, inplace=True)
        >>> print(result)
        tensor([0.0000, 0.0000, 2.0000, 3.0000])
    """
    # Ensure input is float32 or float64
    if input.dtype != torch.float32 and input.dtype != torch.float64:
        input = input.float()
    
    # Determine output tensor
    if inplace:
        output = input
    elif out is not None:
        output = out
    else:
        output = torch.empty_like(input)
    
    # Ensure input and output are contiguous
    input = input.contiguous()
    output = output.contiguous()
    
    # Get number of elements
    n_elements = input.numel()
    
    # Define block size
    BLOCK_SIZE = 1024
    
    # Calculate grid size
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    relu_sqrt_kernel[grid](
        input,
        output,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

##################################################################################################################################################



import torch
from torch import Tensor

# def relu_sqrt(input: Tensor, inplace: bool=False, out: Tensor=None) -> Tensor:
#     if input.dtype != torch.float32 and input.dtype != torch.float64:
#         input = input.float()
#     if inplace:
#         input.relu_()
#         input.sqrt_()
#         return input
#     elif out is not None:
#         out.copy_(torch.sqrt(torch.relu(input)))
#         return out
#     else:
#         return torch.sqrt(torch.relu(input))

def test_relu_sqrt():
    results = {}
    
    # Test case 1: Default parameters
    a = torch.tensor([-1.0, 0.0, 4.0, 9.0], device='cuda')
    results["test_case_1"] = relu_sqrt(a)
    
    # Test case 2: Inplace operation
    b = torch.tensor([-1.0, 0.0, 4.0, 9.0], device='cuda')
    results["test_case_2"] = relu_sqrt(b, inplace=True)
    
    # Test case 3: Out parameter
    c = torch.tensor([-1.0, 0.0, 4.0, 9.0], device='cuda')
    out = torch.empty_like(c)
    results["test_case_3"] = relu_sqrt(c, out=out)
    
    # Test case 4: Non-float input
    d = torch.tensor([-1, 0, 4, 9], device='cuda')
    results["test_case_4"] = relu_sqrt(d)
    
    return results

test_results = test_relu_sqrt()
