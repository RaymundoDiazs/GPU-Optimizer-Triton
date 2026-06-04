import torch
import triton
import triton.language as tl


@triton.jit
def erf_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel to compute the error function element-wise.
    
    Args:
        input_ptr: pointer to input tensor
        output_ptr: pointer to output tensor
        n_elements: total number of elements
        BLOCK_SIZE: block size for processing
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for valid elements
    mask = offsets < n_elements
    
    # Load input data
    x = tl.load(input_ptr + offsets, mask=mask)
    
    # Compute error function using Triton's built-in erf
    y = tl.math.erf(x)
    
    # Store output data
    tl.store(output_ptr + offsets, y, mask=mask)


def erf(input, *, out=None):
    """
    Computes the error function of the input tensor.
    
    The error function is used in probability, statistics, and partial 
    differential equations describing diffusion.
    
    Args:
        input (Tensor): the input tensor
        
    Keyword Args:
        out (Tensor, optional): the output tensor. If None, a new tensor is created.
        
    Returns:
        Tensor: the error function of the input tensor
        
    Example:
        >>> x = torch.randn(4, 4)
        >>> y = erf(x)
    """
    # Validate input
    if not isinstance(input, torch.Tensor):
        raise TypeError(f"input must be a Tensor, got {type(input)}")
    
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
    
    n_elements = input.numel()
    
    # Determine block size
    BLOCK_SIZE = 1024
    
    # Calculate grid size
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    erf_kernel[grid](
        input,
        out,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

##################################################################################################################################################



import torch

# def erf(input_tensor):
#     """
#     计算输入张量的误差函数（error function）。

#     参数：
#     input_tensor (Tensor): 输入的张量。

#     返回：
#     Tensor: 输入张量中每个元素的误差函数值。
#     """
#     return torch.special.erf(input_tensor)

def test_erf():
    results = {}
    
    # Test case 1: Single element tensor
    input_tensor = torch.tensor([0.5], device='cuda')
    results["test_case_1"] = erf(input_tensor)
    
    # Test case 2: Multi-element tensor
    input_tensor = torch.tensor([0.5, -1.0, 2.0], device='cuda')
    results["test_case_2"] = erf(input_tensor)
    
    # Test case 3: Large values tensor
    input_tensor = torch.tensor([10.0, -10.0], device='cuda')
    results["test_case_3"] = erf(input_tensor)
    
    # Test case 4: Zero tensor
    input_tensor = torch.tensor([0.0], device='cuda')
    results["test_case_4"] = erf(input_tensor)
    
    return results

test_results = test_erf()
