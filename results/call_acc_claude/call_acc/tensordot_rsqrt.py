import torch
import triton
import triton.language as tl


@triton.jit
def _tensordot_rsqrt_kernel(
    result_ptr,
    result_stride,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel to compute element-wise reciprocal square root.
    
    Args:
        result_ptr: Pointer to the tensordot result tensor
        result_stride: Stride of the result tensor
        n_elements: Total number of elements in result tensor
        BLOCK_SIZE: Block size for parallel processing
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Mask for boundary handling
    mask = offsets < n_elements
    
    # Load data
    result = tl.load(result_ptr + offsets * result_stride, mask=mask, other=0.0)
    
    # Compute reciprocal square root: 1 / sqrt(x)
    output = tl.rsqrt(result)
    
    # Store result
    tl.store(result_ptr + offsets * result_stride, output, mask=mask)


def tensordot_rsqrt(a: torch.Tensor, b: torch.Tensor, dims) -> torch.Tensor:
    """
    Perform a tensor contraction between two tensors a and b over specified dimensions and apply 
    reciprocal square root to the resulting tensor.

    Args:
        a (torch.Tensor): Left tensor to contract.
        b (torch.Tensor): Right tensor to contract.
        dims (int, Tuple[List[int], List[int]], or List[List[int]]): Dimensions for contraction, 
            as per `torch.tensordot`.

    Returns:
        torch.Tensor: The reciprocal square root of the tensordot product of tensors a and b.
    """
    # Perform tensor contraction using PyTorch
    result = torch.tensordot(a, b, dims=dims)
    
    # Ensure result is contiguous and on CUDA device
    result = result.contiguous()
    
    # Handle device placement
    if not result.is_cuda:
        result = result.cuda()
    
    # Get tensor properties
    n_elements = result.numel()
    
    # Define block size
    BLOCK_SIZE = 1024
    
    # Calculate grid size
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    _tensordot_rsqrt_kernel[grid](
        result,
        result.stride(0) if result.dim() > 0 else 1,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return result

##################################################################################################################################################



import torch

def test_tensordot_rsqrt():
    results = {}

    # Test case 1: Simple contraction with scalar result
    a = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    b = torch.tensor([4.0, 5.0, 6.0], device='cuda')
    dims = 1
    results["test_case_1"] = tensordot_rsqrt(a, b, dims)

    # Test case 2: Contraction with matrices
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    b = torch.tensor([[5.0, 6.0], [7.0, 8.0]], device='cuda')
    dims = ([1], [0])
    results["test_case_2"] = tensordot_rsqrt(a, b, dims)

    # Test case 3: Higher-dimensional tensors
    a = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], device='cuda')
    b = torch.tensor([[[9.0, 10.0], [11.0, 12.0]], [[13.0, 14.0], [15.0, 16.0]]], device='cuda')
    dims = ([2], [1])
    results["test_case_3"] = tensordot_rsqrt(a, b, dims)

    # Test case 4: Different dimensions for contraction
    a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device='cuda')
    b = torch.tensor([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]], device='cuda')
    dims = ([1], [0])
    results["test_case_4"] = tensordot_rsqrt(a, b, dims)

    return results

test_results = test_tensordot_rsqrt()
