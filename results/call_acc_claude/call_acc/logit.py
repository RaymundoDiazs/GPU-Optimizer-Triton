import triton
import triton.language as tl
import torch


@triton.jit
def _logit_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for computing logit of input tensor elements.
    
    Args:
        input_ptr: Pointer to input tensor
        output_ptr: Pointer to output tensor
        n_elements: Total number of elements
        eps: Epsilon value for clamping (None is represented as -1.0)
        BLOCK_SIZE: Block size for processing
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(input_ptr + offsets, mask=mask)
    
    # Clamp if eps is provided (eps >= 0 indicates it was provided)
    if eps >= 0:
        x = tl.maximum(x, eps)
        x = tl.minimum(x, 1.0 - eps)
    
    # Compute logit: log(x / (1 - x))
    numerator = x
    denominator = 1.0 - x
    result = tl.log(numerator / denominator)
    
    # Store output
    tl.store(output_ptr + offsets, result, mask=mask)


def logit(input: torch.Tensor, eps: float = None, *, out: torch.Tensor = None) -> torch.Tensor:
    """
    Returns a new tensor with the logit of the elements of input.
    The input is clamped to [eps, 1 - eps] when eps is not None.
    When eps is None and input < 0 or input > 1, the function yields NaN.
    
    Args:
        input (Tensor): The input tensor.
        eps (float, optional): The epsilon for input clamp bound. Default: None.
        out (Tensor, optional): The output tensor. Default: None.
    
    Returns:
        Tensor: A new tensor with the logit of the elements of the input tensor.
    """
    # Validate input
    assert input.is_cuda, "Input tensor must be on CUDA device"
    assert input.dtype in [torch.float32, torch.float64], "Input must be float32 or float64"
    
    # Create output tensor if not provided
    if out is None:
        out = torch.empty_like(input)
    else:
        assert out.shape == input.shape, "Output tensor shape must match input shape"
        assert out.dtype == input.dtype, "Output tensor dtype must match input dtype"
    
    # Flatten tensors for easier processing
    input_flat = input.flatten()
    out_flat = out.flatten()
    n_elements = input_flat.numel()
    
    # Determine block size
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Convert eps to a value that can be passed to kernel
    # Use -1.0 to indicate eps is None
    eps_value = -1.0 if eps is None else float(eps)
    
    # Launch kernel
    _logit_kernel[grid](
        input_flat,
        out_flat,
        n_elements,
        eps_value,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Reshape output back to original shape
    return out_flat.reshape(input.shape)

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
