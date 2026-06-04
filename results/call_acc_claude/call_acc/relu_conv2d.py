import torch
import triton
import triton.language as tl


@triton.jit
def relu_conv2d_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    # Tensor dimensions
    batch_size,
    in_channels,
    input_h,
    input_w,
    out_channels,
    kernel_h,
    kernel_w,
    # Convolution parameters
    stride_h,
    stride_w,
    padding_h,
    padding_w,
    dilation_h,
    dilation_w,
    groups,
    # Output dimensions
    output_h,
    output_w,
    # Strides for memory access
    input_batch_stride,
    input_channel_stride,
    input_h_stride,
    input_w_stride,
    weight_out_stride,
    weight_in_stride,
    weight_h_stride,
    weight_w_stride,
    output_batch_stride,
    output_channel_stride,
    output_h_stride,
    output_w_stride,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for 2D convolution followed by ReLU activation.
    Processes output elements in parallel.
    """
    # Get thread block index
    pid = tl.program_id(0)
    
    # Calculate output position (batch, channel, h, w)
    total_elements = batch_size * out_channels * output_h * output_w
    
    # Distribute work across thread blocks
    start_idx = pid * BLOCK_SIZE
    offsets = start_idx + tl.arange(0, BLOCK_SIZE)
    
    # Mask for valid elements
    mask = offsets < total_elements
    
    # Convert linear index to (batch, out_channel, out_h, out_w)
    batch_idx = offsets // (out_channels * output_h * output_w)
    remainder = offsets % (out_channels * output_h * output_w)
    out_channel_idx = remainder // (output_h * output_w)
    remainder = remainder % (output_h * output_w)
    out_h_idx = remainder // output_w
    out_w_idx = remainder % output_w
    
    # Initialize accumulator for convolution result
    conv_result = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Perform convolution
    group_size = in_channels // groups
    group_id = out_channel_idx // (out_channels // groups)
    
    for kh in range(kernel_h):
        for kw in range(kernel_w):
            for ic in range(group_size):
                # Calculate input position
                in_h = out_h_idx * stride_h + kh * dilation_h - padding_h
                in_w = out_w_idx * stride_w + kw * dilation_w - padding_w
                
                # Check bounds
                in_bounds = (in_h >= 0) & (in_h < input_h) & (in_w >= 0) & (in_w < input_w)
                
                # Load input value
                input_offset = (
                    batch_idx * input_batch_stride +
                    (group_id * group_size + ic) * input_channel_stride +
                    in_h * input_h_stride +
                    in_w * input_w_stride
                )
                input_val = tl.load(input_ptr + input_offset, mask=in_bounds & mask, other=0.0)
                
                # Load weight value
                weight_offset = (
                    out_channel_idx * weight_out_stride +
                    ic * weight_in_stride +
                    kh * weight_h_stride +
                    kw * weight_w_stride
                )
                weight_val = tl.load(weight_ptr + weight_offset, mask=mask, other=0.0)
                
                # Accumulate
                conv_result += input_val * weight_val
    
    # Add bias if provided
    if bias_ptr is not None:
        bias_offset = out_channel_idx
        bias_val = tl.load(bias_ptr + bias_offset, mask=mask, other=0.0)
        conv_result += bias_val
    
    # Apply ReLU activation
    relu_result = tl.maximum(conv_result, 0.0)
    
    # Store output
    output_offset = (
        batch_idx * output_batch_stride +
        out_channel_idx * output_channel_stride +
        out_h_idx * output_h_stride +
        out_w_idx * output_w_stride
    )
    tl.store(output_ptr + output_offset, relu_result, mask=mask)


def relu_conv2d(
    input,
    weight,
    bias=None,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
    inplace=False,
):
    """
    Applies a 2D convolution followed by a ReLU activation function.

    Args:
        input (Tensor): The input tensor of shape (minibatch, in_channels, iH, iW).
        weight (Tensor): The convolution filters of shape (out_channels, in_channels / groups, kH, kW).
        bias (Tensor, optional): Optional bias tensor of shape (out_channels). Default: None.
        stride (int or tuple, optional): The stride of the convolution kernel. Default: 1.
        padding (int, tuple, or string, optional): Padding added to all sides of the input. Default: 0.
        dilation (int or tuple, optional): The spacing between kernel elements. Default: 1.
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1.
        inplace (bool, optional): If True, will perform ReLU operation in-place. Default: False.

    Returns:
        Tensor: A tensor resulting from the 2D convolution followed by ReLU activation.
    """
    # Validate inputs
    assert input.dim() == 4, "Input must be 4D (batch, channels, height, width)"
    assert weight.dim() == 4, "Weight must be 4D (out_channels, in_channels/groups, kH, kW)"
    assert input.is_cuda, "Input must be on CUDA device"
    assert weight.is_cuda, "Weight must be on CUDA device"
    
    # Convert stride, padding, dilation to tuples
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)
    
    # Extract dimensions
    batch_size, in_channels, input_h, input_w = input.shape
    out_channels, _, kernel_h, kernel_w = weight.shape
    stride_h, stride_w = stride
    padding_h, padding_w = padding
    dilation_h, dilation_w = dilation
    
    # Calculate output dimensions
    output_h = (input_h + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) // stride_h + 1
    output_w = (input_w + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) // stride_w + 1
    
    # Create output tensor
    output = torch.empty(
        (batch_size, out_channels, output_h, output_w),
        dtype=input.dtype,
        device=input.device,
    )
    
    # Get strides for memory access
    input_batch_stride = input.stride(0)
    input_channel_stride = input.stride(1)
    input_h_stride = input.stride(2)
    input_w_stride = input.stride(3)
    
    weight_out_stride = weight.stride(0)
    weight_in_stride = weight.stride(1)
    weight_h_stride = weight.stride(2)
    weight_w_stride = weight.stride(3)
    
    output_batch_stride = output.stride(0)
    output_channel_stride = output.stride(1)
    output_h_stride = output.stride(2)
    output_w_stride = output.stride(3)
    
    # Determine block size
    BLOCK_SIZE = 128
    
    # Calculate grid size
    total_elements = batch_size * out_channels * output_h * output_w
    grid = ((total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    # Launch kernel
    relu_conv2d_kernel[grid](
        input,
        weight,
        bias,
        output,
        batch_size,
        in_channels,
        input_h,
        input_w,
        out_channels,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        dilation_h,
        dilation_w,
        groups,
        output_h,
        output_w,
        input_batch_stride,
        input_channel_stride,
        input_h_stride,
        input_w_stride,
        weight_out_stride,
        weight_in_stride,
        weight_h_stride,
        weight_w_stride,
        output_batch_stride,
        output_channel_stride,
        output_h_stride,
        output_w_stride,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

##################################################################################################################################################



import torch
import torch.nn.functional as F

# def relu_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, inplace=False):
#     """
#     Applies a 2D convolution followed by a ReLU activation function.

#     Args:
#         input (Tensor): The input tensor of shape (minibatch, in_channels, iH, iW).
#         weight (Tensor): The convolution filters of shape (out_channels, in_channels / groups, kH, kW).
#         bias (Tensor, optional): Optional bias tensor of shape (out_channels). Default: None.
#         stride (int or tuple, optional): The stride of the convolution kernel. Default: 1.
#         padding (int, tuple, or string, optional): Padding added to all sides of the input. Default: 0.
#         dilation (int or tuple, optional): The spacing between kernel elements. Default: 1.
#         groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1.
#         inplace (bool, optional): If True, will perform ReLU operation in-place. Default: False.

#     Returns:
#         Tensor: A tensor resulting from the 2D convolution followed by ReLU activation.
#     """
#     conv_out = F.conv2d(input, weight, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
#     return F.relu(conv_out, inplace=inplace)

def test_relu_conv2d():
    results = {}
    
    # Test case 1: Basic test with default parameters
    input1 = torch.randn(1, 3, 5, 5, device='cuda')
    weight1 = torch.randn(2, 3, 3, 3, device='cuda')
    results["test_case_1"] = relu_conv2d(input1, weight1)
    
    # Test case 2: Test with bias
    input2 = torch.randn(1, 3, 5, 5, device='cuda')
    weight2 = torch.randn(2, 3, 3, 3, device='cuda')
    bias2 = torch.randn(2, device='cuda')
    results["test_case_2"] = relu_conv2d(input2, weight2, bias=bias2)
    
    # Test case 3: Test with stride and padding
    input3 = torch.randn(1, 3, 5, 5, device='cuda')
    weight3 = torch.randn(2, 3, 3, 3, device='cuda')
    results["test_case_3"] = relu_conv2d(input3, weight3, stride=2, padding=1)
    
    # Test case 4: Test with dilation and groups
    input4 = torch.randn(1, 4, 5, 5, device='cuda')
    weight4 = torch.randn(2, 2, 3, 3, device='cuda')
    results["test_case_4"] = relu_conv2d(input4, weight4, dilation=2, groups=2)
    
    return results

test_results = test_relu_conv2d()
