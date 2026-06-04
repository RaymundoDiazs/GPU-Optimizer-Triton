import triton
import triton.language as tl
import torch

@triton.jit
def erf_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Get the program ID and calculate the range of elements this program will handle
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Load input data
    input_data = tl.load(input_ptr + offsets, mask=offsets < n_elements, other=0.0)

    # Compute the error function using a polynomial approximation
    # This is a simple approximation, for demonstration purposes
    # More accurate approximations can be used if needed
    t = 1.0 / (1.0 + 0.5 * tl.abs(input_data))
    tau = t * tl.exp(-input_data * input_data - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))))
    erf_result = tl.where(input_data >= 0, 1 - tau, tau - 1)

    # Store the result
    tl.store(output_ptr + offsets, erf_result, mask=offsets < n_elements)

def erf(input_tensor, *, out=None):
    """
    Computes the error function of the input tensor.

    Args:
        input_tensor (Tensor): The input tensor.
        out (Tensor, optional): The output tensor.

    Returns:
        Tensor: A tensor containing the error function of each element in the input tensor.
    """
    # Ensure input is a contiguous tensor
    input_tensor = input_tensor.contiguous()

    # Allocate output tensor if not provided
    if out is None:
        out = torch.empty_like(input_tensor)

    # Get the number of elements in the input tensor
    n_elements = input_tensor.numel()

    # Define block size
    BLOCK_SIZE = 1024

    # Launch the Triton kernel
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    erf_kernel[grid](input_tensor, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)

    return out

##################################################################################################################################################



import torch

# def erf(input_tensor):
#     """
#     计算输入张量的误差函数（error function）。

#     参数：
#     input_tensor (Tensor): 输入的张量。

#     返回：
#     Tensor: 输入张量中每个元素的误差函数值。
#     """
#     return torch.special.erf(input_tensor)

def test_erf():
    results = {}
    
    # Test case 1: Single element tensor
    input_tensor = torch.tensor([0.5], device='cuda')
    results["test_case_1"] = erf(input_tensor)
    
    # Test case 2: Multi-element tensor
    input_tensor = torch.tensor([0.5, -1.0, 2.0], device='cuda')
    results["test_case_2"] = erf(input_tensor)
    
    # Test case 3: Large values tensor
    input_tensor = torch.tensor([10.0, -10.0], device='cuda')
    results["test_case_3"] = erf(input_tensor)
    
    # Test case 4: Zero tensor
    input_tensor = torch.tensor([0.0], device='cuda')
    results["test_case_4"] = erf(input_tensor)
    
    return results

test_results = test_erf()
