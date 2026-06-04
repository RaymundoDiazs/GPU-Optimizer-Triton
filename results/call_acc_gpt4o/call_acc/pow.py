import triton
import triton.language as tl

@triton.jit
def pow_kernel(
    input_ptr, exponent_ptr, output_ptr,
    n_elements, is_exponent_scalar: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    # Define the block index
    block_idx = tl.program_id(0)
    
    # Define the start and end indices for this block
    start_idx = block_idx * BLOCK_SIZE
    end_idx = tl.min(start_idx + BLOCK_SIZE, n_elements)
    
    # Create a range of indices for this block
    offsets = start_idx + tl.arange(0, BLOCK_SIZE)
    
    # Load input elements
    input_elements = tl.load(input_ptr + offsets, mask=offsets < n_elements)
    
    # Load exponent elements or broadcast scalar
    if is_exponent_scalar:
        exponent_value = tl.load(exponent_ptr)
        exponent_elements = tl.broadcast(exponent_value, input_elements.shape)
    else:
        exponent_elements = tl.load(exponent_ptr + offsets, mask=offsets < n_elements)
    
    # Compute the power
    result = input_elements ** exponent_elements
    
    # Store the result
    tl.store(output_ptr + offsets, result, mask=offsets < n_elements)

##################################################################################################################################################



import torch

def test_pow():
    results = {}

    # Test case 1: input_tensor and exponent are scalars
    input_tensor = torch.tensor([2.0], device='cuda')
    exponent = 3.0
    results["test_case_1"] = pow(input_tensor, exponent)

    # Test case 2: input_tensor is a tensor, exponent is a scalar
    input_tensor = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    exponent = 2.0
    results["test_case_2"] = pow(input_tensor, exponent)

    # Test case 3: input_tensor and exponent are tensors of the same shape
    input_tensor = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    exponent = torch.tensor([3.0, 2.0, 1.0], device='cuda')
    results["test_case_3"] = pow(input_tensor, exponent)

    # Test case 4: input_tensor is a tensor, exponent is a negative scalar
    input_tensor = torch.tensor([4.0, 9.0, 16.0], device='cuda')
    exponent = -0.5
    results["test_case_4"] = pow(input_tensor, exponent)

    return results

test_results = test_pow()
