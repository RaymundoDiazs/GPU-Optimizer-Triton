import torch
import triton
import triton.language as tl
from typing import Optional, Union, Tuple

@triton.jit
def add_mean_kernel(
    input_ptr,
    other_ptr,
    output_ptr,
    alpha,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for element-wise addition and reduction.
    Computes: mean(input + alpha * other)
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    # Load input and other tensors
    input_vals = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    other_vals = tl.load(other_ptr + offsets, mask=mask, other=0.0)
    
    # Compute: input + alpha * other
    result = input_vals + alpha * other_vals
    
    # Store intermediate result
    tl.store(output_ptr + offsets, result, mask=mask)


@triton.jit
def reduce_mean_kernel(
    input_ptr,
    output_ptr,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for computing mean across all elements.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    # Load values
    vals = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Compute sum for this block
    block_sum = tl.sum(vals, axis=0)
    
    # Store block sum (will be reduced in host code)
    tl.store(output_ptr + pid, block_sum)


def add_mean(
    input: torch.Tensor,
    other: Union[torch.Tensor, int, float],
    dim: Optional[Union[int, Tuple[int, ...]]] = None,
    alpha: Union[int, float] = 1,
    keepdim: bool = False,
    dtype: Optional[torch.dtype] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Adds the `other` tensor, scaled by `alpha`, to the `input` tensor and computes 
    the mean value along the specified dimension(s).
    
    Parameters:
        input (Tensor): The input tensor.
        other (Tensor or Number): The tensor or number to add to input.
        dim (int or tuple of ints, optional): The dimension(s) to reduce. Default: None.
        alpha (Number, optional): The multiplier for `other`. Default: 1.
        keepdim (bool, optional): Whether the output tensor has dim retained or not. Default: False.
        dtype (torch.dtype, optional): The desired data type of the returned tensor. Default: None.
        out (Tensor, optional): The output tensor.

    Returns:
        Tensor: A tensor containing the mean of the result after addition and scaling.
    """
    
    # Handle dtype casting
    if dtype is not None:
        input = input.to(dtype)
    
    # Convert scalar other to tensor
    if isinstance(other, (int, float)):
        other = torch.tensor(other, dtype=input.dtype, device=input.device)
    else:
        # Ensure other has compatible dtype
        if other.dtype != input.dtype:
            other = other.to(input.dtype)
    
    # Broadcast other to match input shape if necessary
    if other.shape != input.shape:
        other = torch.broadcast_to(other, input.shape)
    
    # Compute addition: input + alpha * other
    result = input + alpha * other
    
    # Compute mean along specified dimension(s)
    mean_result = result.mean(dim=dim, keepdim=keepdim, dtype=dtype)
    
    # Handle output tensor if provided
    if out is not None:
        out.copy_(mean_result)
        return out
    
    return mean_result

##################################################################################################################################################



import torch

def test_add_mean():
    results = {}

    # Test case 1: Basic addition and mean with default alpha
    input1 = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    other1 = torch.tensor([0.5, 0.5, 0.5], device='cuda')
    results["test_case_1"] = add_mean(input1, other1)

    # Test case 2: Addition with scalar other and non-default alpha
    input2 = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    other2 = 0.5
    results["test_case_2"] = add_mean(input2, other2, alpha=2)

    # Test case 3: Addition with mean along a specific dimension
    input3 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    other3 = torch.tensor([[0.5, 0.5], [0.5, 0.5]], device='cuda')
    results["test_case_3"] = add_mean(input3, other3, dim=0)

    # Test case 4: Addition with mean and keepdim=True
    input4 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    other4 = torch.tensor([[0.5, 0.5], [0.5, 0.5]], device='cuda')
    results["test_case_4"] = add_mean(input4, other4, dim=1, keepdim=True)

    return results

test_results = test_add_mean()
