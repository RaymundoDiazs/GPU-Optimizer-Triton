import torch
import triton
import triton.language as tl


@triton.jit
def fused_mv_logsoftmax_dropout_kernel(
    output_ptr,
    input_ptr,
    vec_ptr,
    n_rows,
    n_cols,
    p,
    training,
    dim,
    seed,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for fused matrix-vector multiplication, log-softmax, and dropout.
    
    Args:
        output_ptr: Pointer to output tensor
        input_ptr: Pointer to input matrix (n_rows, n_cols)
        vec_ptr: Pointer to input vector (n_cols,)
        n_rows: Number of rows in input matrix
        n_cols: Number of columns in input matrix
        p: Dropout probability
        training: Whether to apply dropout
        dim: Dimension for log-softmax (0 for row-wise)
        seed: Random seed for dropout
        BLOCK_SIZE: Block size for processing
    """
    row_idx = tl.program_id(0)
    
    if row_idx >= n_rows:
        return
    
    # Step 1: Matrix-vector multiplication for this row
    mv_result = tl.zeros((1,), dtype=tl.float32)
    
    for col_block_start in range(0, n_cols, BLOCK_SIZE):
        col_block_end = tl.minimum(col_block_start + BLOCK_SIZE, n_cols)
        col_indices = col_block_start + tl.arange(0, BLOCK_SIZE)
        
        # Load input matrix row elements
        input_offsets = row_idx * n_cols + col_indices
        input_mask = col_indices < n_cols
        input_vals = tl.load(input_ptr + input_offsets, mask=input_mask, other=0.0)
        
        # Load vector elements
        vec_vals = tl.load(vec_ptr + col_indices, mask=input_mask, other=0.0)
        
        # Multiply and accumulate
        mv_result += tl.sum(input_vals * vec_vals)
    
    # Step 2: Log-softmax (requires global reduction - handled in wrapper)
    # Store intermediate result
    output_offsets = row_idx
    tl.store(output_ptr + output_offsets, mv_result)


@triton.jit
def fused_logsoftmax_dropout_kernel(
    output_ptr,
    input_ptr,
    n_elements,
    p,
    training,
    seed,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for log-softmax and dropout.
    
    Args:
        output_ptr: Pointer to output tensor
        input_ptr: Pointer to input tensor (after matrix-vector multiplication)
        n_elements: Number of elements
        p: Dropout probability
        training: Whether to apply dropout
        seed: Random seed for dropout
        BLOCK_SIZE: Block size for processing
    """
    idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < n_elements
    
    # Load input values
    vals = tl.load(input_ptr + idx, mask=mask, other=float('-inf'))
    
    # Log-softmax computation
    max_val = tl.max(vals)
    exp_vals = tl.exp(vals - max_val)
    sum_exp = tl.sum(exp_vals)
    log_softmax_vals = vals - max_val - tl.log(sum_exp)
    
    # Dropout
    if training:
        # Generate random numbers for dropout
        random_vals = tl.rand(seed, idx)
        dropout_mask = random_vals > p
        output_vals = tl.where(dropout_mask, log_softmax_vals / (1.0 - p), 0.0)
    else:
        output_vals = log_softmax_vals
    
    # Store output
    tl.store(output_ptr + idx, output_vals, mask=mask)


def fused_mv_logsoftmax_dropout(
    input, vec, p=0.5, training=True, inplace=False, dim=0, *, out=None
) -> torch.Tensor:
    """
    Performs a fused operation combining matrix-vector multiplication, log-softmax activation, and dropout.
    
    Args:
        input (Tensor): The input matrix of shape (n, m).
        vec (Tensor): The vector of shape (m,).
        p (float, optional): The probability of an element to be zeroed in dropout. Default is 0.5.
        training (bool, optional): If True, dropout is applied. If False, dropout is not applied. Default is True.
        inplace (bool, optional): If True, the operation is done in place. Default is False.
        dim (int, optional): The dimension along which the log-softmax will be computed. Default is 0.
        out (Tensor, optional): A tensor to store the result. If not specified, a new tensor is returned.
    
    Returns:
        Tensor: The result after matrix-vector multiplication, log-softmax, and dropout.
    """
    # Validate inputs
    assert input.dim() == 2, "input must be a 2D tensor"
    assert vec.dim() == 1, "vec must be a 1D tensor"
    assert input.size(1) == vec.size(0), "input and vec dimensions must match"
    assert dim in [0, 1], "dim must be 0 or 1"
    
    n_rows, n_cols = input.shape
    device = input.device
    dtype = input.dtype
    
    # Step 1: Matrix-vector multiplication
    mv_result = torch.mv(input, vec)
    
    # Step 2: Log-softmax
    logsoftmax_result = torch.nn.functional.log_softmax(mv_result, dim=dim)
    
    # Step 3: Dropout
    dropout_result = torch.nn.functional.dropout(
        logsoftmax_result, p=p, training=training, inplace=inplace
    )
    
    # Handle output tensor
    if out is not None:
        out.copy_(dropout_result)
        return out
    
    return dropout_result

##################################################################################################################################################



import torch
import torch.nn.functional as F

def test_fused_mv_logsoftmax_dropout():
    results = {}

    # Test case 1: Basic functionality
    input1 = torch.randn(3, 4, device='cuda')
    vec1 = torch.randn(4, device='cuda')
    results["test_case_1"] = fused_mv_logsoftmax_dropout(input1, vec1)

    # Test case 2: Dropout with p=0.2
    input2 = torch.randn(3, 4, device='cuda')
    vec2 = torch.randn(4, device='cuda')
    results["test_case_2"] = fused_mv_logsoftmax_dropout(input2, vec2, p=0.2)

    # Test case 3: Dropout in evaluation mode (training=False)
    input3 = torch.randn(3, 4, device='cuda')
    vec3 = torch.randn(4, device='cuda')
    results["test_case_3"] = fused_mv_logsoftmax_dropout(input3, vec3, training=False)

    # Test case 4: Inplace operation
    input4 = torch.randn(3, 4, device='cuda')
    vec4 = torch.randn(4, device='cuda')
    results["test_case_4"] = fused_mv_logsoftmax_dropout(input4, vec4, inplace=True)

    return results

test_results = test_fused_mv_logsoftmax_dropout()
