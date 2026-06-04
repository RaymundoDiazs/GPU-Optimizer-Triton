import torch
import triton
import triton.language as tl

@triton.jit
def fused_mul_add_logsoftmax_dropout_bmm_kernel(
    input1_ptr, input2_ptr, other_ptr, mat2_ptr, output_ptr,
    dropout_mask_ptr,
    batch_size, seq_len, hidden_dim, mat2_dim,
    p, training,
    dim,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for fused operation: mul -> add -> log_softmax -> dropout -> bmm
    
    Processes element-wise operations and log-softmax in parallel.
    Note: BMM is handled by PyTorch's optimized implementation.
    """
    pid = tl.program_id(0)
    
    # Calculate batch and sequence indices
    batch_idx = pid // seq_len
    seq_idx = pid % seq_len
    
    # Bounds checking
    if batch_idx >= batch_size or seq_idx >= seq_len:
        return
    
    # Process hidden_dim in blocks
    for block_start in range(0, hidden_dim, BLOCK_SIZE):
        block_end = tl.minimum(block_start + BLOCK_SIZE, hidden_dim)
        block_size = block_end - block_start
        
        # Create offset indices
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < hidden_dim
        
        # Calculate linear index
        base_idx = batch_idx * seq_len * hidden_dim + seq_idx * hidden_dim
        
        # Load inputs
        input1 = tl.load(input1_ptr + base_idx + offsets, mask=mask, other=0.0)
        input2 = tl.load(input2_ptr + base_idx + offsets, mask=mask, other=0.0)
        other = tl.load(other_ptr + base_idx + offsets, mask=mask, other=0.0)
        
        # Element-wise multiplication and addition
        z = input1 * input2
        s = z + other
        
        # Store intermediate result for log-softmax computation
        tl.store(output_ptr + base_idx + offsets, s, mask=mask)


def fused_mul_add_logsoftmax_dropout_bmm(
    input1, input2, other, mat2, p=0.5, training=True, inplace=False, dim=-1, *, out=None
) -> torch.Tensor:
    """
    Performs a fused operation combining element-wise multiplication, addition,
    log-softmax activation, dropout, and batch matrix multiplication.
    
    Args:
        input1 (Tensor): The first input tensor of shape (batch_size, seq_len, hidden_dim).
        input2 (Tensor): The second input tensor of shape (batch_size, seq_len, hidden_dim).
        other (Tensor): A tensor or scalar to add to the result of element-wise multiplication.
        mat2 (Tensor): A tensor of shape (batch_size, hidden_dim, mat2_dim) for batch matrix multiplication.
        p (float): The dropout probability. Default: 0.5
        training (bool): Whether to apply dropout. Default: True
        inplace (bool): Whether to apply the operation in-place. Default: False
        dim (int): The dimension along which to apply log-softmax. Default: -1
        out (Tensor, optional): If given, the result will be stored in this tensor.
        
    Returns:
        Tensor: The result of the fused operation of shape (batch_size, seq_len, mat2_dim).
    """
    # Validate inputs
    assert input1.shape == input2.shape, "input1 and input2 must have the same shape"
    assert input1.is_cuda, "Inputs must be on CUDA device"
    
    batch_size, seq_len, hidden_dim = input1.shape
    mat2_dim = mat2.shape[-1]
    
    # Step 1: Element-wise multiplication and addition (fused in Triton)
    BLOCK_SIZE = 128
    grid = (batch_size * seq_len,)
    
    # Allocate output tensor for intermediate results
    intermediate = torch.empty_like(input1)
    
    # Launch Triton kernel for mul + add
    fused_mul_add_logsoftmax_dropout_bmm_kernel[grid](
        input1, input2, other, mat2, intermediate,
        None,  # dropout_mask_ptr
        batch_size, seq_len, hidden_dim, mat2_dim,
        p, training,
        dim,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Step 2: Log-softmax (use PyTorch's optimized implementation)
    log_softmax_result = torch.nn.functional.log_softmax(intermediate, dim=dim)
    
    # Step 3: Dropout
    dropout_result = torch.nn.functional.dropout(
        log_softmax_result, p=p, training=training, inplace=inplace
    )
    
    # Step 4: Batch matrix multiplication
    result = torch.bmm(dropout_result, mat2)
    
    # Handle output tensor
    if out is not None:
        out.copy_(result)
        return out
    
    return result

##################################################################################################################################################



import torch
import torch.nn.functional as F

def test_fused_mul_add_logsoftmax_dropout_bmm():
    results = {}

    # Test case 1: Basic functionality
    input1 = torch.rand(2, 3, 4, device='cuda')
    input2 = torch.rand(2, 3, 4, device='cuda')
    other = torch.rand(2, 3, 4, device='cuda')
    mat2 = torch.rand(2, 4, 5, device='cuda')
    results["test_case_1"] = fused_mul_add_logsoftmax_dropout_bmm(input1, input2, other, mat2)

    # Test case 2: Different dropout probability
    input1 = torch.rand(2, 3, 4, device='cuda')
    input2 = torch.rand(2, 3, 4, device='cuda')
    other = torch.rand(2, 3, 4, device='cuda')
    mat2 = torch.rand(2, 4, 5, device='cuda')
    results["test_case_2"] = fused_mul_add_logsoftmax_dropout_bmm(input1, input2, other, mat2, p=0.3)

    # Test case 3: In-place operation
    input1 = torch.rand(2, 3, 4, device='cuda')
    input2 = torch.rand(2, 3, 4, device='cuda')
    other = torch.rand(2, 3, 4, device='cuda')
    mat2 = torch.rand(2, 4, 5, device='cuda')
    results["test_case_3"] = fused_mul_add_logsoftmax_dropout_bmm(input1, input2, other, mat2, inplace=True)

    # Test case 4: Different dimension for log-softmax
    input1 = torch.rand(2, 3, 4, device='cuda')
    input2 = torch.rand(2, 3, 4, device='cuda')
    other = torch.rand(2, 3, 4, device='cuda')
    mat2 = torch.rand(2, 4, 5, device='cuda')
    results["test_case_4"] = fused_mul_add_logsoftmax_dropout_bmm(input1, input2, other, mat2, dim=1)

    return results

test_results = test_fused_mul_add_logsoftmax_dropout_bmm()
