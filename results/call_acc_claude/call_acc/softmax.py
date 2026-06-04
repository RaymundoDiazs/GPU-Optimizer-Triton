import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(
    output_ptr,
    input_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for computing softmax along the last dimension.
    
    Args:
        output_ptr: Pointer to output tensor
        input_ptr: Pointer to input tensor
        input_row_stride: Stride of input tensor rows
        output_row_stride: Stride of output tensor rows
        n_cols: Number of columns (size of softmax dimension)
        BLOCK_SIZE: Block size for processing
    """
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    
    # Load input row
    mask = col_offsets < n_cols
    row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
    
    # Compute softmax: subtract max for numerical stability
    row_max = tl.max(row, axis=0)
    row_exp = tl.exp(row - row_max)
    row_sum = tl.sum(row_exp, axis=0)
    softmax_output = row_exp / row_sum
    
    # Store output
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=mask)


def softmax(input: torch.Tensor, dim: int, dtype: torch.dtype = None) -> torch.Tensor:
    """
    Apply softmax function to the input tensor along the specified dimension.
    The elements in the tensor will be scaled to the range [0, 1] and sum to 1 along the specified dimension.

    Args:
        input (torch.Tensor): The input tensor to apply softmax to.
        dim (int): The dimension along which softmax will be computed.
        dtype (torch.dtype, optional): The desired data type of the returned tensor. 
            If specified, the input tensor is casted to dtype before the operation is performed. 
            This is useful for preventing data type overflows. Default: None.

    Returns:
        torch.Tensor: The tensor with softmax applied.
    """
    # Cast input if dtype is specified
    if dtype is not None:
        input = input.to(dtype)
    
    # Normalize dimension to handle negative indices
    if dim < 0:
        dim = input.ndim + dim
    
    # Move the softmax dimension to the last position
    input = input.transpose(dim, -1)
    original_shape = input.shape
    
    # Reshape to 2D: (batch_size, softmax_dim)
    batch_size = 1
    for i in range(len(original_shape) - 1):
        batch_size *= original_shape[i]
    softmax_dim = original_shape[-1]
    
    input_2d = input.reshape(batch_size, softmax_dim)
    
    # Allocate output tensor
    output = torch.empty_like(input_2d)
    
    # Determine block size
    BLOCK_SIZE = triton.next_power_of_2(softmax_dim)
    
    # Launch kernel
    grid = (batch_size,)
    softmax_kernel[grid](
        output,
        input_2d,
        input_2d.stride(0),
        output.stride(0),
        softmax_dim,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Reshape back to original shape
    output = output.reshape(original_shape)
    
    # Transpose back to original dimension order
    output = output.transpose(dim, -1)
    
    return output

##################################################################################################################################################



import torch
import torch.nn.functional as F

# def softmax(input: torch.Tensor, dim: int, dtype: torch.dtype=None) -> torch.Tensor:
#     return F.softmax(input, dim=dim, dtype=dtype)

def test_softmax():
    results = {}
    
    # Test case 1: Basic test with default dtype
    input1 = torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], device='cuda')
    results["test_case_1"] = softmax(input1, dim=1)
    
    # Test case 2: Test with different dimension
    input2 = torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], device='cuda')
    results["test_case_2"] = softmax(input2, dim=0)
    
    # Test case 3: Test with specified dtype
    input3 = torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], device='cuda')
    results["test_case_3"] = softmax(input3, dim=1, dtype=torch.float64)
    
    # Test case 4: Test with larger tensor
    input4 = torch.randn(100, 100, device='cuda')
    results["test_case_4"] = softmax(input4, dim=1)
    
    return results

test_results = test_softmax()
