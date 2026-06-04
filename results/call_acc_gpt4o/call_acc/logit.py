import triton
import triton.language as tl

@triton.jit
def logit_kernel(input_ptr, output_ptr, n_elements, eps, has_eps, BLOCK_SIZE: tl.constexpr):
    # Define the block index and offset
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Load input data
    input_data = tl.load(input_ptr + offsets, mask=offsets < n_elements, other=0.0)
    
    # Apply clamping if eps is provided
    if has_eps:
        input_data = tl.where(input_data < eps, eps, input_data)
        input_data = tl.where(input_data > 1 - eps, 1 - eps, input_data)
    
    # Compute logit
    logit_data = tl.log(input_data / (1.0 - input_data))
    
    # Store the result
    tl.store(output_ptr + offsets, logit_data, mask=offsets < n_elements)

def logit(input, eps=None, out=None):
    # Ensure input is a contiguous tensor
    input = input.contiguous()
    
    # Determine the number of elements
    n_elements = input.numel()
    
    # Allocate output tensor if not provided
    if out is None:
        out = torch.empty_like(input)
    
    # Determine block size
    BLOCK_SIZE = 1024  # You can adjust this based on your hardware capabilities
    
    # Launch the kernel
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    logit_kernel[grid](input, out, n_elements, eps if eps is not None else 0.0, eps is not None, BLOCK_SIZE=BLOCK_SIZE)
    
    return out

##################################################################################################################################################



import torch

def test_logit():
    results = {}

    # Test case 1: Basic test with input tensor in range [0, 1] without eps
    input1 = torch.tensor([0.2, 0.5, 0.8], device='cuda')
    results["test_case_1"] = logit(input1)

    # Test case 2: Test with input tensor in range [0, 1] with eps
    input2 = torch.tensor([0.0, 0.5, 1.0], device='cuda')
    eps = 1e-6
    results["test_case_2"] = logit(input2, eps=eps)

    # Test case 3: Test with input tensor in range [0, 1] with eps and out tensor
    input3 = torch.tensor([0.1, 0.9], device='cuda')
    out = torch.empty_like(input3)
    results["test_case_3"] = logit(input3, eps=eps, out=out)

    # Test case 4: Test with input tensor in range [0, 1] with out tensor
    input4 = torch.tensor([0.3, 0.7], device='cuda')
    out = torch.empty_like(input4)
    results["test_case_4"] = logit(input4, out=out)

    return results

test_results = test_logit()
