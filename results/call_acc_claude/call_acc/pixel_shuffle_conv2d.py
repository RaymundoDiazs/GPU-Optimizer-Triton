import torch
import triton
import triton.language as tl
import torch.nn.functional as F

@triton.jit
def pixel_shuffle_kernel(
    output_ptr,
    input_ptr,
    batch_size,
    channels,
    height,
    width,
    upscale_factor,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for pixel shuffle operation.
    Rearranges elements from shape (N, C*r^2, H, W) to (N, C, H*r, W*r)
    """
    pid = tl.program_id(0)
    
    # Calculate output dimensions
    out_channels = channels // (upscale_factor * upscale_factor)
    out_height = height * upscale_factor
    out_width = width * upscale_factor
    
    # Process elements in blocks
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Total number of output elements
    total_elements = batch_size * out_channels * out_height * out_width
    
    # Mask for valid indices
    mask = idx < total_elements
    
    # Convert linear index to multi-dimensional indices
    # idx -> (batch, channel, height, width)
    temp = idx
    out_w = temp % out_width
    temp = temp // out_width
    out_h = temp % out_height
    temp = temp // out_height
    out_c = temp % out_channels
    batch = temp // out_channels
    
    # Map output indices back to input indices
    # Input shape: (batch, channels, height, width) where channels = out_channels * r^2
    r = upscale_factor
    
    # Determine which element in the r x r block
    h_offset = out_h % r
    w_offset = out_w % r
    
    # Input spatial coordinates
    in_h = out_h // r
    in_w = out_w // r
    
    # Input channel index
    in_c = out_c * r * r + h_offset * r + w_offset
    
    # Calculate linear indices
    in_idx = batch * channels * height * width + in_c * height * width + in_h * width + in_w
    out_idx = idx
    
    # Load and store
    input_val = tl.load(input_ptr + in_idx, mask=mask, other=0.0)
    tl.store(output_ptr + out_idx, input_val, mask=mask)


def pixel_shuffle_conv2d(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias=None,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
    upscale_factor=2,
) -> torch.Tensor:
    """
    Applies a 2D convolution followed by pixel shuffle upscaling to rearrange the spatial dimensions.

    Parameters:
    - input (Tensor): Input tensor of shape (minibatch, in_channels, iH, iW).
    - weight (Tensor): Convolution filter tensor of shape (out_channels, in_channels/groups, kH, kW).
    - bias (Tensor, optional): Optional bias tensor of shape (out_channels).
    - stride (int, optional): Stride of the convolving kernel. Default is 1.
    - padding (int, optional): Padding added to all four sides of the input. Default is 0.
    - dilation (int, optional): Spacing between kernel elements. Default is 1.
    - groups (int, optional): Number of blocked connections from input channels to output channels. Default is 1.
    - upscale_factor (int, optional): Factor by which to increase spatial resolution. Default is 2.

    Returns:
    - Tensor: The output tensor after applying the convolution and pixel shuffle.
    """
    # Validate inputs
    assert input.dim() == 4, "Input must be 4D tensor (batch, channels, height, width)"
    assert weight.dim() == 4, "Weight must be 4D tensor (out_channels, in_channels/groups, kH, kW)"
    assert upscale_factor > 0, "upscale_factor must be positive"
    
    # Apply 2D convolution
    conv_output = F.conv2d(
        input,
        weight,
        bias=bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )
    
    # Get dimensions
    batch_size, channels, height, width = conv_output.shape
    
    # Validate that channels is divisible by upscale_factor^2
    assert channels % (upscale_factor ** 2) == 0, \
        f"Number of channels ({channels}) must be divisible by upscale_factor^2 ({upscale_factor ** 2})"
    
    # Allocate output tensor
    out_channels = channels // (upscale_factor ** 2)
    out_height = height * upscale_factor
    out_width = width * upscale_factor
    output = torch.empty(
        (batch_size, out_channels, out_height, out_width),
        dtype=conv_output.dtype,
        device=conv_output.device,
    )
    
    # Flatten tensors for kernel processing
    conv_flat = conv_output.reshape(-1)
    output_flat = output.reshape(-1)
    
    # Configure grid and blocks
    BLOCK_SIZE = 256
    grid = (triton.cdiv(output_flat.numel(), BLOCK_SIZE),)
    
    # Launch kernel
    pixel_shuffle_kernel[grid](
        output_flat,
        conv_flat,
        batch_size,
        channels,
        height,
        width,
        upscale_factor,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

##################################################################################################################################################



import torch
import torch.nn.functional as F

# def pixel_shuffle_conv2d(input: torch.Tensor, weight: torch.Tensor, bias=None, stride=1, padding=0, dilation=1, groups=1, upscale_factor=2) -> torch.Tensor:
#     x = F.conv2d(input, weight, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
#     return F.pixel_shuffle(x, upscale_factor)

def test_pixel_shuffle_conv2d():
    results = {}
    
    # Test case 1: Basic test with default parameters
    input1 = torch.randn(1, 4, 8, 8, device='cuda')
    weight1 = torch.randn(16, 4, 3, 3, device='cuda')
    results["test_case_1"] = pixel_shuffle_conv2d(input1, weight1)
    
    # Test case 2: Test with bias
    input2 = torch.randn(1, 4, 8, 8, device='cuda')
    weight2 = torch.randn(16, 4, 3, 3, device='cuda')
    bias2 = torch.randn(16, device='cuda')
    results["test_case_2"] = pixel_shuffle_conv2d(input2, weight2, bias=bias2)
    
    # Test case 3: Test with stride
    input3 = torch.randn(1, 4, 16, 16, device='cuda')
    weight3 = torch.randn(16, 4, 3, 3, device='cuda')
    results["test_case_3"] = pixel_shuffle_conv2d(input3, weight3, stride=2)
    
    # Test case 4: Test with padding
    input4 = torch.randn(1, 4, 8, 8, device='cuda')
    weight4 = torch.randn(16, 4, 3, 3, device='cuda')
    results["test_case_4"] = pixel_shuffle_conv2d(input4, weight4, padding=1)
    
    return results

test_results = test_pixel_shuffle_conv2d()
