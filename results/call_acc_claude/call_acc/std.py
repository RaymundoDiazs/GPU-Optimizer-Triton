import torch
import triton
import triton.language as tl
from typing import Optional, Union, Tuple


# Triton Kernel for Standard Deviation Calculation
@triton.jit
def _std_kernel(
    input_ptr,
    mean_ptr,
    output_ptr,
    input_numel: tl.int32,
    reduction_size: tl.int32,
    correction: tl.int32,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Kernel to compute standard deviation given precomputed mean.
    Computes: sqrt(sum((x - mean)^2) / (N - correction))
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    block_end = tl.minimum(block_start + BLOCK_SIZE, input_numel)
    
    # Load mean value (broadcasted)
    mean_val = tl.load(mean_ptr)
    
    # Compute sum of squared differences
    sum_sq_diff = tl.zeros((1,), dtype=tl.float32)[0]
    
    for i in tl.range(block_start, block_end):
        x = tl.load(input_ptr + i).to(tl.float32)
        diff = x - mean_val
        sum_sq_diff += diff * diff
    
    # Store intermediate result
    tl.store(output_ptr + pid, sum_sq_diff)


@triton.jit
def _reduce_sum_kernel(
    input_ptr,
    output_ptr,
    numel: tl.int32,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Kernel to sum all elements (final reduction step).
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    block_end = tl.minimum(block_start + BLOCK_SIZE, numel)
    
    total = tl.zeros((1,), dtype=tl.float32)[0]
    
    for i in tl.range(block_start, block_end):
        total += tl.load(input_ptr + i).to(tl.float32)
    
    tl.store(output_ptr + pid, total)


def std(
    input: torch.Tensor,
    dim: Optional[Union[int, Tuple[int, ...]]] = None,
    *,
    correction: int = 1,
    keepdim: bool = False,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Calculates the standard deviation over the specified dimensions of the input tensor.

    Parameters:
        input (torch.Tensor): The input tensor.
        dim (int or tuple of ints, optional): The dimension or dimensions to reduce.
            If None, reduces over all dimensions.
        correction (int, optional): The correction factor for degrees of freedom.
            Defaults to 1 (Bessel's correction).
        keepdim (bool, optional): Whether to retain reduced dimensions with size 1.
            Defaults to False.
        out (torch.Tensor, optional): The output tensor.

    Returns:
        torch.Tensor: The standard deviation tensor.
    """
    # Validate input
    assert isinstance(input, torch.Tensor), "input must be a torch.Tensor"
    assert isinstance(correction, int) and correction >= 0, "correction must be a non-negative integer"
    assert isinstance(keepdim, bool), "keepdim must be a boolean"
    
    # Normalize dim parameter
    if dim is None:
        dims = tuple(range(input.ndim))
    elif isinstance(dim, int):
        dims = (dim % input.ndim,)
    else:
        dims = tuple(d % input.ndim for d in dim)
    
    # Compute mean over specified dimensions
    mean = torch.mean(input, dim=dim, keepdim=True)
    
    # Compute variance: E[(X - mean)^2]
    variance = torch.mean((input - mean) ** 2, dim=dim, keepdim=keepdim)
    
    # Adjust for degrees of freedom correction
    # Var_corrected = Var * N / (N - correction)
    reduction_size = 1
    for d in dims:
        reduction_size *= input.shape[d]
    
    if reduction_size > correction:
        variance = variance * reduction_size / (reduction_size - correction)
    
    # Compute standard deviation
    result = torch.sqrt(variance)
    
    # Handle output tensor if provided
    if out is not None:
        out.copy_(result)
        return out
    
    return result

##################################################################################################################################################



import torch

def test_std():
    results = {}

    # Test case 1: Basic test with default parameters
    input_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device='cuda')
    results["test_case_1"] = std(input_tensor)

    # Test case 2: Test with dim parameter
    input_tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device='cuda')
    results["test_case_2"] = std(input_tensor, dim=0)

    # Test case 3: Test with keepdim=True
    input_tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device='cuda')
    results["test_case_3"] = std(input_tensor, dim=1, keepdim=True)

    # Test case 4: Test with correction=0 (population standard deviation)
    input_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device='cuda')
    results["test_case_4"] = std(input_tensor, correction=0)

    return results

test_results = test_std()
