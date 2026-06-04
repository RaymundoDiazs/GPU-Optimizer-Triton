import triton
import triton.language as tl
import torch

@triton.jit
def i0_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Compute the program index
    pid = tl.program_id(0)
    
    # Create a block of indices
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Load input data
    input_data = tl.load(input_ptr + offsets, mask=offsets < n_elements, other=0.0)
    
    # Compute the zeroth order modified Bessel function of the first kind
    # Using a series expansion or approximation for I_0(x)
    # Here we use a simple approximation for demonstration purposes
    # I_0(x) ≈ 1 + (x^2 / 4) + (x^4 / 64) + (x^6 / 2304) + ...
    x2 = input_data * input_data
    result = 1 + (x2 / 4) + (x2 * x2 / 64) + (x2 * x2 * x2 / 2304)
    
    # Store the result
    tl.store(output_ptr + offsets, result, mask=offsets < n_elements)

def i0(input, *, out=None):
    """
    Computes the zeroth order modified Bessel function of the first kind for each element of the input tensor.

    Args:
        input (Tensor): the input tensor
        out (Tensor, optional): the output tensor
    
    Returns:
        Tensor: the result tensor with I_0 applied to each element
    """
    # Ensure input is a contiguous tensor
    input = input.contiguous()
    
    # Allocate output tensor if not provided
    if out is None:
        out = torch.empty_like(input)
    
    # Get the number of elements
    n_elements = input.numel()
    
    # Define the block size
    BLOCK_SIZE = 1024  # You can adjust this based on your GPU's capabilities
    
    # Launch the Triton kernel
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    i0_kernel[grid](input, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    return out

##################################################################################################################################################



import torch

def test_i0():
    results = {}

    # Test case 1: Simple tensor on GPU
    input_tensor_1 = torch.tensor([0.0, 1.0, 2.0], device='cuda')
    results["test_case_1"] = i0(input_tensor_1)

    # Test case 2: Larger tensor with negative values on GPU
    input_tensor_2 = torch.tensor([-1.0, -2.0, 3.0, 4.0], device='cuda')
    results["test_case_2"] = i0(input_tensor_2)

    # Test case 3: Tensor with mixed positive and negative values on GPU
    input_tensor_3 = torch.tensor([-3.0, 0.0, 3.0], device='cuda')
    results["test_case_3"] = i0(input_tensor_3)

    # Test case 4: Tensor with fractional values on GPU
    input_tensor_4 = torch.tensor([0.5, 1.5, 2.5], device='cuda')
    results["test_case_4"] = i0(input_tensor_4)

    return results

test_results = test_i0()
