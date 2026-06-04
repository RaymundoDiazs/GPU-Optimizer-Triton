import torch
import triton
import triton.language as tl


@triton.jit
def _reciprocal_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for computing reciprocal of tensor elements.
    
    Args:
        input_ptr: Pointer to input tensor
        output_ptr: Pointer to output tensor
        n_elements: Total number of elements
        BLOCK_SIZE: Block size for processing
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for valid elements
    mask = offsets < n_elements
    
    # Load input values
    x = tl.load(input_ptr + offsets, mask=mask)
    
    # Compute reciprocal: 1 / x
    y = 1.0 / x
    
    # Store output values
    tl.store(output_ptr + offsets, y, mask=mask)


def reciprocal(input, *, out=None):
    """
    Returns a new tensor with the reciprocal of the elements of the input.
    Unlike NumPy's reciprocal, this function supports integral inputs by 
    promoting them to the default scalar type.
    
    Args:
        input (Tensor): The input tensor whose reciprocal is computed.
        out (Tensor, optional): The output tensor. If provided, the result 
                               will be stored in it.
    
    Returns:
        Tensor: A new tensor containing the reciprocal of each element 
               in the input tensor.
    
    Example:
        >>> import torch
        >>> a = torch.randn(4)
        >>> a
        tensor([-0.4595, -2.1219, -1.4314,  0.7298])
        >>> reciprocal(a)
        tensor([-2.1763, -0.4713, -0.6986,  1.3702])
    """
    # Promote integral types to default float type
    if input.dtype in [torch.int32, torch.int64, torch.int16, torch.int8]:
        input = input.to(torch.float32)
    
    # Allocate output tensor if not provided
    if out is None:
        out = torch.empty_like(input)
    
    # Ensure output has compatible dtype
    if out.dtype != input.dtype:
        out = out.to(input.dtype)
    
    n_elements = input.numel()
    
    # Define block size
    BLOCK_SIZE = 1024
    
    # Calculate grid size
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    _reciprocal_kernel[grid](
        input,
        out,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

##################################################################################################################################################



import torch

def test_reciprocal():
    results = {}

    # Test case 1: Basic test with positive and negative numbers
    a = torch.tensor([-0.4595, -2.1219, -1.4314, 0.7298], device='cuda')
    results["test_case_1"] = reciprocal(a)

    # Test case 2: Test with a tensor containing zero (expecting inf)
    b = torch.tensor([0.0, 1.0, -1.0, 2.0], device='cuda')
    results["test_case_2"] = reciprocal(b)

    # Test case 3: Test with a tensor containing large numbers
    c = torch.tensor([1e10, -1e10, 1e-10, -1e-10], device='cuda')
    results["test_case_3"] = reciprocal(c)

    # Test case 4: Test with a tensor of ones (expecting ones)
    d = torch.ones(4, device='cuda')
    results["test_case_4"] = reciprocal(d)

    return results

test_results = test_reciprocal()
