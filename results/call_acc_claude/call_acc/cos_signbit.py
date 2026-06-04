import torch
import triton
import triton.language as tl
from typing import Tuple

@triton.jit
def cos_signbit_kernel(
    input_ptr,
    cos_output_ptr,
    signbit_output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel that computes cosine and sign bit for each element.
    
    Args:
        input_ptr: Pointer to input tensor
        cos_output_ptr: Pointer to cosine output tensor
        signbit_output_ptr: Pointer to sign bit output tensor
        n_elements: Total number of elements
        BLOCK_SIZE: Block size for processing
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for valid elements
    mask = offsets < n_elements
    
    # Load input values
    input_vals = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Compute cosine
    cos_vals = tl.cos(input_vals)
    
    # Compute sign bit (negative if < 0)
    signbit_vals = cos_vals < 0.0
    
    # Store results
    tl.store(cos_output_ptr + offsets, cos_vals, mask=mask)
    tl.store(signbit_output_ptr + offsets, signbit_vals, mask=mask)


def cos_signbit(input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the cosine of each element in the input tensor, followed by determining the sign bit 
    for each cosine result, indicating if it is positive or negative.
    
    Args:
        input (torch.Tensor): The input tensor for which the cosine and sign bit are computed.
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: 
            - cos_result: The cosine of each element in the input tensor.
            - sign_bit: A boolean tensor indicating whether the cosine result is positive (False) or negative (True).
            
    Example:
        >>> a = torch.tensor([1.4309, 1.2706, -0.8562, 0.9796])
        >>> cos_result, sign_bit = cos_signbit(a)
        >>> cos_result
        tensor([ 0.1395,  0.2957,  0.6553,  0.5574])
        >>> sign_bit
        tensor([False, False, False, False])
    """
    # Ensure input is contiguous and on CUDA
    input = input.contiguous()
    
    # Get number of elements
    n_elements = input.numel()
    
    # Create output tensors
    cos_result = torch.empty_like(input, dtype=input.dtype)
    sign_bit = torch.empty_like(input, dtype=torch.bool)
    
    # Define block size
    BLOCK_SIZE = 1024
    
    # Calculate grid size
    grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    # Launch kernel
    cos_signbit_kernel[grid](
        input,
        cos_result,
        sign_bit,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return (cos_result, sign_bit)

##################################################################################################################################################



import torch
from typing import Tuple

# def cos_signbit(input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#     cos_result = torch.cos(input)
#     sign_bit = torch.signbit(cos_result)
#     return (cos_result, sign_bit)

def test_cos_signbit():
    results = {}

    # Test case 1: Positive values
    input_tensor_1 = torch.tensor([0.0, 1.0, 2.0], device='cuda')
    cos_result_1, sign_bit_1 = cos_signbit(input_tensor_1)
    results["test_case_1"] = (cos_result_1.cpu(), sign_bit_1.cpu())

    # Test case 2: Negative values
    input_tensor_2 = torch.tensor([-1.0, -2.0, -3.0], device='cuda')
    cos_result_2, sign_bit_2 = cos_signbit(input_tensor_2)
    results["test_case_2"] = (cos_result_2.cpu(), sign_bit_2.cpu())

    # Test case 3: Mixed values
    input_tensor_3 = torch.tensor([-1.0, 0.0, 1.0], device='cuda')
    cos_result_3, sign_bit_3 = cos_signbit(input_tensor_3)
    results["test_case_3"] = (cos_result_3.cpu(), sign_bit_3.cpu())

    # Test case 4: Edge case with pi multiples
    input_tensor_4 = torch.tensor([torch.pi, -torch.pi, 2*torch.pi], device='cuda')
    cos_result_4, sign_bit_4 = cos_signbit(input_tensor_4)
    results["test_case_4"] = (cos_result_4.cpu(), sign_bit_4.cpu())

    return results

test_results = test_cos_signbit()
