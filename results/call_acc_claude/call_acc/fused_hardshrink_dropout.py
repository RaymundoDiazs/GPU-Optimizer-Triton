import torch
import triton
import triton.language as tl

@triton.jit
def fused_hardshrink_dropout_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    p,
    lambd,
    training,
    seed,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for fused dropout + hardshrink operation.
    
    Args:
        input_ptr: Pointer to input tensor
        output_ptr: Pointer to output tensor
        n_elements: Total number of elements
        p: Dropout probability
        lambd: Hard shrinkage lambda parameter
        training: Whether to apply dropout
        seed: Random seed for dropout
        BLOCK_SIZE: Block size for processing
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Load input
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Apply dropout if training
    if training:
        # Generate random values for dropout
        random_vals = tl.rand(seed, offsets)
        # Create dropout mask: keep if random > p, zero otherwise
        dropout_mask = random_vals > p
        # Scale by 1/(1-p) to maintain expected value
        scale = 1.0 / (1.0 - p)
        x = x * dropout_mask * scale
    
    # Apply hard shrinkage: shrink(x) = x if |x| > lambda, else 0
    abs_x = tl.abs(x)
    hardshrink_mask = abs_x > lambd
    x = x * hardshrink_mask
    
    # Store output
    tl.store(output_ptr + offsets, x, mask=mask)


def fused_hardshrink_dropout(
    input: torch.Tensor,
    p: float = 0.5,
    training: bool = True,
    inplace: bool = False,
    lambd: float = 0.5,
) -> torch.Tensor:
    """
    Applies a fused operation consisting of dropout followed by hard shrinkage on the input tensor.

    Args:
        input (Tensor): The input tensor.
        p (float, optional): Probability of an element to be zeroed in dropout. Default is 0.5.
        training (bool, optional): Apply dropout if True. Default is True.
        inplace (bool, optional): If set to True, dropout will be applied in-place. Default is False.
        lambd (float, optional): The lambda parameter for the hard shrinkage function. Default is 0.5.

    Returns:
        Tensor: Result after applying dropout and then hard shrinkage on the input.
    """
    # Validate inputs
    assert input.is_cuda, "Input tensor must be on CUDA device"
    assert 0.0 <= p < 1.0, "Dropout probability must be in [0, 1)"
    assert lambd >= 0.0, "Lambda parameter must be non-negative"
    
    # Prepare output tensor
    if inplace and training:
        output = input
    else:
        output = input.clone()
    
    # Flatten for processing
    n_elements = input.numel()
    output_flat = output.view(-1)
    input_flat = input.view(-1)
    
    # Determine block size
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Generate seed for random number generation
    seed = torch.randint(0, 2**31, (1,), device=input.device).item()
    
    # Launch kernel
    fused_hardshrink_dropout_kernel[grid](
        input_flat,
        output_flat,
        n_elements,
        p,
        lambd,
        training,
        seed,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output.view_as(input)

##################################################################################################################################################



import torch
import torch.nn.functional as F

# def fused_hardshrink_dropout(input: torch.Tensor, p: float=0.5, training: bool=True, inplace: bool=False, lambd: float=0.5) -> torch.Tensor:
#     """
#     Applies a fused operation consisting of dropout followed by hard shrinkage on the input tensor.

#     Args:
#         input (Tensor): The input tensor.
#         p (float, optional): Probability of an element to be zeroed in dropout. Default is 0.5.
#         training (bool, optional): Apply dropout if True. Default is True.
#         inplace (bool, optional): If set to True, dropout will be applied in-place. Default is False.
#         lambd (float, optional): The lambda parameter for the hard shrinkage function. Default is 0.5.

#     Returns:
#         Tensor: Result after applying dropout and then hard shrinkage on the input.
#     """
#     if training:
#         input = F.dropout(input, p=p, training=training, inplace=inplace)
#     return F.hardshrink(input, lambd)

def test_fused_hardshrink_dropout():
    results = {}
    
    # Test case 1: Default parameters
    input_tensor = torch.randn(5, 5).cuda()
    results["test_case_1"] = fused_hardshrink_dropout(input_tensor)
    
    # Test case 2: Dropout with p=0.3
    input_tensor = torch.randn(5, 5).cuda()
    results["test_case_2"] = fused_hardshrink_dropout(input_tensor, p=0.3)
    
    # Test case 3: Dropout with training=False
    input_tensor = torch.randn(5, 5).cuda()
    results["test_case_3"] = fused_hardshrink_dropout(input_tensor, training=False)
    
    # Test case 4: Hard shrinkage with lambd=0.7
    input_tensor = torch.randn(5, 5).cuda()
    results["test_case_4"] = fused_hardshrink_dropout(input_tensor, lambd=0.7)
    
    return results

test_results = test_fused_hardshrink_dropout()
