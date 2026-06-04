import torch
import triton
import triton.language as tl
import math


@triton.jit
def logspace_kernel(
    output_ptr,
    start_val,
    end_val,
    steps,
    base,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel to generate logarithmically spaced values.
    
    Args:
        output_ptr: pointer to output tensor
        start_val: starting exponent value
        end_val: ending exponent value
        steps: number of steps
        base: base of the logarithm
        BLOCK_SIZE: block size for parallel processing
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for valid indices
    mask = offsets < steps
    
    # Calculate the step size in exponent space
    if steps == 1:
        exponents = tl.where(mask, start_val, 0.0)
    else:
        step_size = (end_val - start_val) / (steps - 1)
        exponents = start_val + offsets.to(tl.float32) * step_size
    
    # Calculate base^exponent for each position
    # Using the formula: base^x = exp(x * ln(base))
    log_base = tl.log(base)
    values = tl.exp(exponents * log_base)
    
    # Store results
    output_offsets = offsets
    tl.store(output_ptr + output_offsets, values, mask=mask)


def logspace(
    start,
    end,
    steps,
    base=10.0,
    *,
    out=None,
    dtype=None,
    layout=torch.strided,
    device=None,
    requires_grad=False,
) -> torch.Tensor:
    """
    Creates a one-dimensional tensor of size 'steps' whose values are evenly spaced 
    from base^start to base^end, inclusive, on a logarithmic scale with a specified base.
    
    Args:
        start (float or Tensor): the starting value for the set of points. 
                                If `Tensor`, it must be 0-dimensional
        end (float or Tensor): the ending value for the set of points. 
                              If `Tensor`, it must be 0-dimensional
        steps (int): size of the constructed tensor
        base (float, optional): base of the logarithm function. Default: 10.0.
    
    Keyword arguments:
        out (Tensor, optional): the output tensor.
        dtype (torch.dtype, optional): the data type to perform the computation in.
        layout (torch.layout, optional): the desired layout of returned Tensor. 
                                        Default: torch.strided.
        device (torch.device, optional): the desired device of returned tensor.
        requires_grad (bool, optional): If autograd should record operations on the 
                                       returned tensor. Default: False.
    
    Returns:
        torch.Tensor: A tensor with logarithmically spaced values from base^start to base^end.
    """
    
    # Handle tensor inputs for start and end
    if isinstance(start, torch.Tensor):
        assert start.dim() == 0, "start must be 0-dimensional if Tensor"
        start = start.item()
    
    if isinstance(end, torch.Tensor):
        assert end.dim() == 0, "end must be 0-dimensional if Tensor"
        end = end.item()
    
    # Determine device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    
    # Determine dtype
    if dtype is None:
        dtype = torch.get_default_dtype()
    
    # Create output tensor
    if out is None:
        output = torch.empty(
            steps,
            dtype=dtype,
            layout=layout,
            device=device,
            requires_grad=requires_grad,
        )
    else:
        output = out
    
    # Handle edge case: steps == 0
    if steps == 0:
        return output
    
    # Handle edge case: steps == 1
    if steps == 1:
        output.fill_(base ** start)
        return output
    
    # Determine block size
    BLOCK_SIZE = 256
    grid = (triton.cdiv(steps, BLOCK_SIZE),)
    
    # Launch kernel
    logspace_kernel[grid](
        output,
        float(start),
        float(end),
        steps,
        float(base),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

##################################################################################################################################################



import torch

def test_logspace():
    results = {}

    # Test case 1: Basic functionality with default base (10.0)
    start = torch.tensor(1.0, device='cuda')
    end = torch.tensor(3.0, device='cuda')
    steps = 5
    results["test_case_1"] = logspace(start, end, steps)

    # Test case 2: Custom base (2.0)
    start = torch.tensor(0.0, device='cuda')
    end = torch.tensor(4.0, device='cuda')
    steps = 5
    base = 2.0
    results["test_case_2"] = logspace(start, end, steps, base=base)

    # Test case 3: Custom dtype (float64)
    start = torch.tensor(1.0, device='cuda')
    end = torch.tensor(2.0, device='cuda')
    steps = 4
    dtype = torch.float64
    results["test_case_3"] = logspace(start, end, steps, dtype=dtype)

    # Test case 4: Requires gradient
    start = torch.tensor(1.0, device='cuda')
    end = torch.tensor(3.0, device='cuda')
    steps = 3
    requires_grad = True
    results["test_case_4"] = logspace(start, end, steps, requires_grad=requires_grad)

    return results

test_results = test_logspace()
