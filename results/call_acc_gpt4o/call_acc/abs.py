import triton
import triton.language as tl

@triton.jit
def abs_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Define the block index and offset
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Mask to ensure we don't go out of bounds
    mask = offsets < n_elements
    
    # Load input data
    input_data = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Compute absolute value
    abs_data = tl.abs(input_data)
    
    # Store result
    tl.store(output_ptr + offsets, abs_data, mask=mask)

##################################################################################################################################################



import torch

def test_abs():
    results = {}

    # Test case 1: Simple positive and negative values
    input_tensor_1 = torch.tensor([-1.0, 2.0, -3.0], device='cuda')
    results["test_case_1"] = abs(input_tensor_1)

    # Test case 2: Zero values
    input_tensor_2 = torch.tensor([0.0, -0.0, 0.0], device='cuda')
    results["test_case_2"] = abs(input_tensor_2)

    # Test case 3: Mixed positive, negative, and zero values
    input_tensor_3 = torch.tensor([-5.0, 0.0, 5.0], device='cuda')
    results["test_case_3"] = abs(input_tensor_3)

    # Test case 4: Large positive and negative values
    input_tensor_4 = torch.tensor([-1e10, 1e10, -1e-10], device='cuda')
    results["test_case_4"] = abs(input_tensor_4)

    return results

test_results = test_abs()
