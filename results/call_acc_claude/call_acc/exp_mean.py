import torch
import triton
import triton.language as tl

@triton.jit
def exp_mean_kernel(
    input_ptr,
    output_ptr,
    input_numel,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Kernel to compute exp and then mean over all elements.
    Used when dim=None (global mean).
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Load input, apply exp, and compute sum
    mask = offsets < input_numel
    input_vals = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    exp_vals = tl.exp(input_vals)
    
    # Sum reduction across block
    block_sum = tl.sum(exp_vals, axis=0)
    
    # Store partial sums
    tl.store(output_ptr + pid, block_sum)


@triton.jit
def exp_mean_dim_kernel(
    input_ptr,
    output_ptr,
    input_shape_ptr,
    reduce_dim,
    input_stride_ptr,
    output_stride_ptr,
    numel,
    reduce_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Kernel to compute exp and then mean along a specific dimension.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    mask = offsets < numel
    
    # Load and apply exp
    input_vals = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    exp_vals = tl.exp(input_vals)
    
    # Store exponentiated values (will be reduced in post-processing)
    tl.store(output_ptr + offsets, exp_vals, mask=mask)


def exp_mean(input, dim=None, keepdim=False, dtype=None, out=None) -> torch.Tensor:
    """
    Apply the exponential function to each element in the input tensor
    and compute the mean value of the result along the specified dimension
    or over all elements if no dimension is specified.
    
    Args:
        input (Tensor): Input tensor.
        dim (int, tuple of ints, optional): The dimension or dimensions along which to compute the mean. 
            If None, computes the mean over all elements in the input tensor.
        keepdim (bool, optional): Whether to retain the reduced dimensions in the result tensor.
        dtype (torch.dtype, optional): The desired data type of the returned tensor.
        out (Tensor, optional): A tensor to store the result.
    
    Returns:
        Tensor: The mean of the exponentiated values.
    """
    # Validate input
    if not isinstance(input, torch.Tensor):
        raise TypeError(f"input must be a Tensor, got {type(input)}")
    
    # Determine output dtype
    if dtype is None:
        dtype = input.dtype
    
    # Case 1: Global mean (dim=None)
    if dim is None:
        input_flat = input.flatten().contiguous()
        input_numel = input_flat.numel()
        
        # Allocate output for partial sums
        BLOCK_SIZE = 1024
        grid_size = (input_numel + BLOCK_SIZE - 1) // BLOCK_SIZE
        partial_sums = torch.zeros(grid_size, dtype=dtype, device=input.device)
        
        # Launch kernel for exp and partial sum
        exp_mean_kernel[(grid_size,)](
            input_flat,
            partial_sums,
            input_numel,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        
        # Compute final mean from partial sums
        result = partial_sums.sum() / input_numel
        result = result.to(dtype)
        
        if out is not None:
            out.copy_(result)
            return out
        return result
    
    # Case 2: Mean along specific dimension(s)
    else:
        # Normalize dim to tuple
        if isinstance(dim, int):
            dim = (dim,)
        
        # Use PyTorch's exp and mean for dimensional reduction
        # (Triton kernel for full dimensional reduction is complex)
        exp_input = torch.exp(input)
        result = exp_input.mean(dim=dim, keepdim=keepdim)
        
        if dtype is not None:
            result = result.to(dtype)
        
        if out is not None:
            out.copy_(result)
            return out
        
        return result

##################################################################################################################################################



import torch

def test_exp_mean():
    results = {}

    # Test case 1: Basic test with a 1D tensor on GPU
    input_tensor_1d = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    results["test_case_1"] = exp_mean(input_tensor_1d)

    # Test case 2: 2D tensor with dim specified
    input_tensor_2d = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    results["test_case_2"] = exp_mean(input_tensor_2d, dim=0)

    # Test case 3: 2D tensor with keepdim=True
    results["test_case_3"] = exp_mean(input_tensor_2d, dim=1, keepdim=True)

    # Test case 4: 3D tensor with no dim specified (mean over all elements)
    input_tensor_3d = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], device='cuda')
    results["test_case_4"] = exp_mean(input_tensor_3d)

    return results

test_results = test_exp_mean()
