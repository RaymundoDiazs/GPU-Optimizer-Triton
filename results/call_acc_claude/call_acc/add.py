import torch
import triton
import triton.language as tl
from typing import Optional

@triton.jit
def _add_kernel(
    input_ptr,
    other_ptr,
    output_ptr,
    alpha,
    numel,
    input_stride,
    other_stride,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for element-wise addition with broadcasting support.
    
    Args:
        input_ptr: Pointer to input tensor
        other_ptr: Pointer to other tensor
        output_ptr: Pointer to output tensor
        alpha: Scaling factor for other
        numel: Total number of elements in output
        input_stride: Stride for input tensor
        other_stride: Stride for other tensor
        BLOCK_SIZE: Block size for processing
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Mask for boundary handling
    mask = offsets < numel
    
    # Load input with broadcasting (stride 0 means scalar/broadcast)
    input_offsets = offsets * input_stride
    input_vals = tl.load(input_ptr + input_offsets, mask=mask)
    
    # Load other with broadcasting (stride 0 means scalar/broadcast)
    other_offsets = offsets * other_stride
    other_vals = tl.load(other_ptr + other_offsets, mask=mask)
    
    # Compute: input + alpha * other
    output_vals = input_vals + alpha * other_vals
    
    # Store result
    tl.store(output_ptr + offsets, output_vals, mask=mask)


def add(
    input: torch.Tensor,
    other: torch.Tensor | float | int,
    *,
    alpha: float | int = 1,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Adds the tensor or number 'other', scaled by 'alpha', to the 'input' tensor.
    Supports broadcasting to a common shape, type promotion, and accepts integer, 
    float, and complex inputs.
    
    Args:
        input (Tensor): The input tensor.
        other (Tensor or Number): The tensor or number to add to input.
        alpha (Number, optional): The multiplier for 'other'. Default is 1.
        out (Tensor, optional): The output tensor. If provided, the result will be 
                               stored in this tensor.
        
    Returns:
        Tensor: The result of adding 'other' scaled by 'alpha' to 'input'.
    """
    # Handle scalar other
    if isinstance(other, (int, float, complex)):
        other = torch.tensor(other, dtype=input.dtype, device=input.device)
    
    # Determine output dtype (type promotion)
    output_dtype = torch.promote_types(input.dtype, other.dtype)
    
    # Broadcast tensors to common shape
    output_shape = torch.broadcast_shapes(input.shape, other.shape)
    
    # Create output tensor if not provided
    if out is None:
        output = torch.empty(output_shape, dtype=output_dtype, device=input.device)
    else:
        output = out
        if output.shape != output_shape:
            raise RuntimeError(
                f"Output shape {output.shape} does not match expected shape {output_shape}"
            )
    
    # Broadcast input and other to output shape
    input_broadcast = torch.broadcast_to(input, output_shape)
    other_broadcast = torch.broadcast_to(other, output_shape)
    
    # Ensure contiguity for kernel
    input_broadcast = input_broadcast.contiguous()
    other_broadcast = other_broadcast.contiguous()
    output = output.contiguous()
    
    # Convert to common dtype
    input_broadcast = input_broadcast.to(output_dtype)
    other_broadcast = other_broadcast.to(output_dtype)
    
    numel = output.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(numel, BLOCK_SIZE),)
    
    # Launch kernel
    _add_kernel[grid](
        input_broadcast,
        other_broadcast,
        output,
        alpha,
        numel,
        input_broadcast.stride(0) if input_broadcast.dim() > 0 else 0,
        other_broadcast.stride(0) if other_broadcast.dim() > 0 else 0,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

##################################################################################################################################################



import torch

def test_add():
    results = {}

    # Test case 1: Adding two tensors with default alpha
    input1 = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    other1 = torch.tensor([4.0, 5.0, 6.0], device='cuda')
    results["test_case_1"] = add(input1, other1)

    # Test case 2: Adding a tensor and a scalar with default alpha
    input2 = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    other2 = 2.0
    results["test_case_2"] = add(input2, other2)

    # Test case 3: Adding two tensors with a specified alpha
    input3 = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    other3 = torch.tensor([4.0, 5.0, 6.0], device='cuda')
    results["test_case_3"] = add(input3, other3, alpha=0.5)

    # Test case 4: Adding a tensor and a scalar with a specified alpha
    input4 = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    other4 = 2.0
    results["test_case_4"] = add(input4, other4, alpha=0.5)

    return results

test_results = test_add()
