import triton
import triton.language as tl
import torch
import torch.nn.functional as F


@triton.jit
def sigmoid_batch_norm_kernel(
    input_ptr,
    output_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    eps,
    N,  # batch size
    C,  # channels
    L,  # sequence length (1 for 2D input)
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for sigmoid batch normalization.
    Applies batch norm followed by sigmoid activation.
    """
    pid = tl.program_id(0)
    
    # Process each channel
    if pid < C:
        channel_idx = pid
        
        # Load running statistics for this channel
        mean = tl.load(running_mean_ptr + channel_idx)
        var = tl.load(running_var_ptr + channel_idx)
        
        # Load weight and bias if provided
        weight = tl.load(weight_ptr + channel_idx) if weight_ptr is not None else 1.0
        bias = tl.load(bias_ptr + channel_idx) if bias_ptr is not None else 0.0
        
        # Process all elements in this channel
        num_elements = N * L
        for offset in range(0, num_elements, BLOCK_SIZE):
            idx = offset + tl.arange(0, BLOCK_SIZE)
            mask = idx < num_elements
            
            # Calculate linear index: (batch_idx * C * L) + (channel_idx * L) + seq_idx
            batch_idx = idx // L
            seq_idx = idx % L
            linear_idx = batch_idx * C * L + channel_idx * L + seq_idx
            
            # Load input
            x = tl.load(input_ptr + linear_idx, mask=mask, other=0.0)
            
            # Batch normalization: (x - mean) / sqrt(var + eps)
            normalized = (x - mean) / tl.sqrt(var + eps)
            
            # Apply weight and bias (affine transformation)
            scaled = normalized * weight + bias
            
            # Apply sigmoid: 1 / (1 + exp(-x))
            output = 1.0 / (1.0 + tl.exp(-scaled))
            
            # Store output
            tl.store(output_ptr + linear_idx, output, mask=mask)


def sigmoid_batch_norm(
    input,
    running_mean,
    running_var,
    weight=None,
    bias=None,
    training=False,
    momentum=0.1,
    eps=1e-5,
) -> torch.Tensor:
    """
    Applies Batch Normalization over the input tensor, then applies the Sigmoid activation function element-wise.

    Args:
        input (Tensor): The input tensor of shape `(N, C)` or `(N, C, L)`, where `N` is batch size, 
                         `C` is the number of features or channels, and `L` is the sequence length.
        running_mean (Tensor): The running mean of the input channels.
        running_var (Tensor): The running variance of the input channels.
        weight (Tensor, optional): Learnable scaling factor for each channel, typically represented as `γ`. Default: None.
        bias (Tensor, optional): Learnable shift for each channel, typically represented as `β`. Default: None.
        training (bool, optional): If `True`, updates running statistics; if `False`, uses them for normalization. Default: False.
        momentum (float, optional): Value for updating the running mean and variance. Default: 0.1.
        eps (float, optional): A small value added for numerical stability. Default: 1e-5.

    Returns:
        Tensor: The output tensor after batch normalization followed by the sigmoid activation.
    """
    # Validate input dimensions
    assert input.dim() in [2, 3], "Input must be 2D (N, C) or 3D (N, C, L)"
    
    # Get dimensions
    N = input.shape[0]  # batch size
    C = input.shape[1]  # channels
    L = input.shape[2] if input.dim() == 3 else 1  # sequence length
    
    # Reshape input to (N, C, L) if needed
    if input.dim() == 2:
        input = input.unsqueeze(2)
    
    # Ensure tensors are contiguous and on the same device
    input = input.contiguous()
    running_mean = running_mean.contiguous()
    running_var = running_var.contiguous()
    
    if weight is not None:
        weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    # Create output tensor
    output = torch.empty_like(input)
    
    # Define block size
    BLOCK_SIZE = 128
    
    # Launch kernel with C blocks (one per channel)
    sigmoid_batch_norm_kernel[(C,)](
        input,
        output,
        running_mean,
        running_var,
        weight,
        bias,
        eps,
        N,
        C,
        L,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Reshape output back to original shape if input was 2D
    if output.shape[2] == 1:
        output = output.squeeze(2)
    
    return output

##################################################################################################################################################



import torch

def test_sigmoid_batch_norm():
    results = {}

    # Test case 1: Basic test with default parameters
    input_tensor = torch.randn(10, 5, device='cuda')
    running_mean = torch.zeros(5, device='cuda')
    running_var = torch.ones(5, device='cuda')
    results["test_case_1"] = sigmoid_batch_norm(input_tensor, running_mean, running_var)

    # Test case 2: With learnable parameters (weight and bias)
    weight = torch.ones(5, device='cuda') * 0.5
    bias = torch.zeros(5, device='cuda') + 0.1
    results["test_case_2"] = sigmoid_batch_norm(input_tensor, running_mean, running_var, weight=weight, bias=bias)

    # Test case 3: In training mode
    results["test_case_3"] = sigmoid_batch_norm(input_tensor, running_mean, running_var, training=True)

    # Test case 4: With a different momentum and eps
    results["test_case_4"] = sigmoid_batch_norm(input_tensor, running_mean, running_var, momentum=0.2, eps=1e-3)

    return results

test_results = test_sigmoid_batch_norm()
