import triton
import triton.language as tl
import torch

@triton.jit
def rsqrt_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Get the program ID and calculate the range of elements this program will handle
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Load input elements
    input_elements = tl.load(input_ptr + offsets, mask=offsets < n_elements, other=0.0)
    
    # Compute reciprocal of the square root
    result = 1.0 / tl.sqrt(input_elements)
    
    # Store the result
    tl.store(output_ptr + offsets, result, mask=offsets < n_elements)

def rsqrt(input: torch.Tensor, out: torch.Tensor = None) -> torch.Tensor:
    """
    Computes the reciprocal of the square root of each element in the input tensor.
    
    Args:
    - input (torch.Tensor): The input tensor.
    - out (torch.Tensor, optional): The output tensor to store the result. Default is None.
    
    Returns:
    - torch.Tensor: A tensor with the reciprocal of the square root of each element in the input.
    """
    # Ensure input is a CUDA tensor
    assert input.is_cuda, "Input tensor must be a CUDA tensor"
    
    # Allocate output tensor if not provided
    if out is None:
        out = torch.empty_like(input)
    
    # Define block size
    BLOCK_SIZE = 1024  # This can be tuned based on the hardware and problem size
    
    # Launch the Triton kernel
    n_elements = input.numel()
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    rsqrt_kernel[grid](input, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    return out

##################################################################################################################################################



import torch

def test_rsqrt():
    results = {}

    # Test case 1: Positive elements
    input1 = torch.tensor([4.0, 16.0, 25.0], device='cuda')
    results["test_case_1"] = rsqrt(input1)

    # Test case 2: Contains zero
    input2 = torch.tensor([0.0, 1.0, 4.0], device='cuda')
    results["test_case_2"] = rsqrt(input2)

    # Test case 3: Contains negative elements
    input3 = torch.tensor([-1.0, 4.0, 9.0], device='cuda')
    results["test_case_3"] = rsqrt(input3)

    # Test case 4: All elements are zero
    input4 = torch.tensor([0.0, 0.0, 0.0], device='cuda')
    results["test_case_4"] = rsqrt(input4)

    return results

test_results = test_rsqrt()
