import torch
import triton
import triton.language as tl
from typing import Tuple

@triton.jit
def signbit_bitwise_and_kernel(
    input_ptr,
    other_ptr,
    signbit_out_ptr,
    bitwise_and_out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for computing signbit and bitwise AND operations.
    
    Args:
        input_ptr: Pointer to input tensor
        other_ptr: Pointer to other tensor
        signbit_out_ptr: Pointer to signbit output tensor
        bitwise_and_out_ptr: Pointer to bitwise AND output tensor
        n_elements: Total number of elements
        BLOCK_SIZE: Block size for processing
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input and other tensors
    input_vals = tl.load(input_ptr + offsets, mask=mask)
    other_vals = tl.load(other_ptr + offsets, mask=mask)
    
    # Compute signbit: check if sign bit is set (negative values)
    # For floating point, we can use < 0 comparison which handles -0.0 correctly
    signbit_result = input_vals < 0
    
    # Convert to int8 for bitwise AND operation
    input_int8 = input_vals.to(tl.int8)
    other_int8 = other_vals.to(tl.int8)
    
    # Compute bitwise AND
    bitwise_and_result = input_int8 & other_int8
    
    # Store results
    tl.store(signbit_out_ptr + offsets, signbit_result, mask=mask)
    tl.store(bitwise_and_out_ptr + offsets, bitwise_and_result, mask=mask)


def signbit_bitwise_and(input: torch.Tensor, other: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the sign bit check and the bitwise AND operation on the input tensors.
    
    Args:
        input (Tensor): The input tensor.
        other (Tensor): The second tensor for bitwise AND, should be of integral or boolean types.
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: 
            - The first tensor is a boolean tensor indicating whether the sign bit is set (True for negative values).
            - The second tensor is the result of performing bitwise AND on input and other.
    
    Example:
        >>> a = torch.tensor([0.7, -1.2, 0., 2.3])
        >>> b = torch.tensor([1, 0, 1, 1], dtype=torch.int8)
        >>> signbit_result, bitwise_and_result = signbit_bitwise_and(a, b)
        >>> signbit_result
        tensor([False, True, False, False])
        >>> bitwise_and_result
        tensor([0, 0, 0, 0], dtype=torch.int8)
    """
    # Ensure tensors are on the same device
    assert input.device == other.device, "Input tensors must be on the same device"
    assert input.shape == other.shape, "Input tensors must have the same shape"
    
    n_elements = input.numel()
    
    # Create output tensors
    signbit_out = torch.empty_like(input, dtype=torch.bool)
    bitwise_and_out = torch.empty_like(input, dtype=torch.int8)
    
    # Define block size
    BLOCK_SIZE = 1024
    
    # Calculate grid size
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    signbit_bitwise_and_kernel[grid](
        input,
        other,
        signbit_out,
        bitwise_and_out,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return (signbit_out, bitwise_and_out)

##################################################################################################################################################



import torch
from typing import Tuple

# def signbit_bitwise_and(input: torch.Tensor, other: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#     signbit_result = torch.signbit(input)
#     bitwise_and_result = input.to(torch.int8) & other.to(torch.int8)
#     return (signbit_result, bitwise_and_result)

def test_signbit_bitwise_and():
    results = {}

    # Test case 1: Positive and negative floats with integer tensor
    a = torch.tensor([0.7, -1.2, 0., 2.3], device='cuda')
    b = torch.tensor([1, 0, 1, 1], dtype=torch.int8, device='cuda')
    results["test_case_1"] = signbit_bitwise_and(a, b)

    # Test case 2: All negative floats with integer tensor
    a = torch.tensor([-0.7, -1.2, -0.1, -2.3], device='cuda')
    b = torch.tensor([1, 1, 1, 1], dtype=torch.int8, device='cuda')
    results["test_case_2"] = signbit_bitwise_and(a, b)

    # Test case 3: Mixed positive and zero floats with boolean tensor
    a = torch.tensor([0.0, 1.2, 0.0, 2.3], device='cuda')
    b = torch.tensor([True, False, True, True], dtype=torch.bool, device='cuda')
    results["test_case_3"] = signbit_bitwise_and(a, b)

    # Test case 4: All zero floats with integer tensor
    a = torch.tensor([0.0, 0.0, 0.0, 0.0], device='cuda')
    b = torch.tensor([1, 0, 1, 1], dtype=torch.int8, device='cuda')
    results["test_case_4"] = signbit_bitwise_and(a, b)

    return results

test_results = test_signbit_bitwise_and()
