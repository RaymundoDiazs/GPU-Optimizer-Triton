import triton
import triton.language as tl
import torch

@triton.jit
def logsumexp_kernel(
    input_ptr, output_ptr, stride, dim_size, keepdim,
    BLOCK_SIZE: tl.constexpr
):
    # Get the program ID and calculate the offset
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE

    # Create a pointer to the start of the block
    input_block_ptr = input_ptr + offset
    output_block_ptr = output_ptr + pid

    # Load the input block
    input_block = tl.load(input_block_ptr + tl.arange(0, BLOCK_SIZE))

    # Compute the maximum value in the block for numerical stability
    max_val = tl.max(input_block, axis=0)

    # Compute the exponential of the input elements minus the max value
    exp_block = tl.exp(input_block - max_val)

    # Sum the exponentials
    sum_exp = tl.sum(exp_block, axis=0)

    # Compute the logsumexp
    logsumexp_val = max_val + tl.log(sum_exp)

    # Store the result
    if keepdim:
        tl.store(output_block_ptr, logsumexp_val)
    else:
        tl.store(output_block_ptr, logsumexp_val)

def logsumexp(input, dim, keepdim=False, *, out=None):
    # Ensure the input is a contiguous tensor
    input = input.contiguous()

    # Get the shape and strides of the input tensor
    shape = input.shape
    strides = input.stride()

    # Calculate the number of elements along the specified dimension
    dim_size = shape[dim]

    # Calculate the number of blocks needed
    BLOCK_SIZE = 1024  # This can be tuned for performance
    num_blocks = (dim_size + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Allocate output tensor if not provided
    if out is None:
        out_shape = list(shape)
        if not keepdim:
            out_shape.pop(dim)
        out = torch.empty(out_shape, dtype=input.dtype, device=input.device)

    # Launch the Triton kernel
    logsumexp_kernel[(num_blocks,)](
        input_ptr=input,
        output_ptr=out,
        stride=strides[dim],
        dim_size=dim_size,
        keepdim=keepdim,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return out

# Example usage
input_tensor = torch.randn(1024, 1024, device='cuda')
result = logsumexp(input_tensor, dim=1, keepdim=True)

##################################################################################################################################################



import torch

def test_logsumexp():
    results = {}

    # Test case 1: Basic test with a 2D tensor on GPU
    input_tensor_1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    results["test_case_1"] = logsumexp(input_tensor_1, dim=0)

    # Test case 2: Test with keepdim=True
    input_tensor_2 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    results["test_case_2"] = logsumexp(input_tensor_2, dim=1, keepdim=True)

    # Test case 3: Test with a 3D tensor
    input_tensor_3 = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], device='cuda')
    results["test_case_3"] = logsumexp(input_tensor_3, dim=2)

    # Test case 4: Test with a negative dimension
    input_tensor_4 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    results["test_case_4"] = logsumexp(input_tensor_4, dim=-1)

    return results

test_results = test_logsumexp()
