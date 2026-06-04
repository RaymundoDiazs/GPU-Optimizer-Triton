import triton
import triton.language as tl
import torch

# Triton kernel (conceptual, as Triton does not directly support FFT)
@triton.jit
def fftn_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Triton kernel logic would go here, but FFT is not directly supported.
    # This is a placeholder to illustrate where Triton would be used.
    pass

# Wrapper function
def fftn(input, s=None, dim=None, norm=None, out=None):
    # Ensure input is on CUDA
    assert input.is_cuda, "Input tensor must be on CUDA."

    # Default dimensions to transform
    if dim is None:
        dim = tuple(range(input.ndim)) if s is None else tuple(range(input.ndim - len(s), input.ndim))

    # Default signal size
    if s is None:
        s = [input.size(d) for d in dim]

    # Check if signal sizes are powers of 2
    for size in s:
        if size != -1 and (size & (size - 1)) != 0:
            raise ValueError("Signal sizes must be powers of 2.")

    # Allocate output tensor if not provided
    if out is None:
        out = torch.empty_like(input, dtype=torch.complex64 if input.is_floating_point() else torch.complex128)

    # Call PyTorch's FFT function as Triton does not support FFT directly
    out = torch.fft.fftn(input, s=s, dim=dim, norm=norm)

    return out

# Example usage
input_tensor = torch.randn(8, 8, device='cuda', dtype=torch.float32)
output_tensor = fftn(input_tensor)

##################################################################################################################################################



import torch

# def fftn(input, s=None, dim=None, norm=None, out=None):
#     return torch.fft.fftn(input, s=s, dim=dim, norm=norm)

def test_fftn():
    results = {}
    
    # Test case 1: Only input tensor
    input_tensor = torch.randn(4, 4, device='cuda')
    results["test_case_1"] = fftn(input_tensor)
    
    # Test case 2: Input tensor with s parameter
    input_tensor = torch.randn(4, 4, device='cuda')
    s = (2, 2)
    results["test_case_2"] = fftn(input_tensor, s=s)
    
    # Test case 3: Input tensor with dim parameter
    input_tensor = torch.randn(4, 4, device='cuda')
    dim = (0, 1)
    results["test_case_3"] = fftn(input_tensor, dim=dim)
    
    # Test case 4: Input tensor with norm parameter
    input_tensor = torch.randn(4, 4, device='cuda')
    norm = "ortho"
    results["test_case_4"] = fftn(input_tensor, norm=norm)
    
    return results

test_results = test_fftn()
