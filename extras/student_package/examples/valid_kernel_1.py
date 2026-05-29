# VALID KERNEL EXAMPLE 1: Vector Addition
#
# This kernel demonstrates:
# - @triton.jit decorator
# - Function parameters with type annotations
# - tl.program_id(), tl.arange(), tl.load(), tl.store()
# - Arithmetic operations
# - Mask-based memory access

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Get the program ID (which block we're processing)
    pid = tl.program_id(0)

    # Calculate the starting offset for this block
    block_start = pid * BLOCK_SIZE

    # Create offsets for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Create mask for valid elements (handle edge case at end)
    mask = offsets < n_elements

    # Load data from memory
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    # Perform addition
    output = x + y

    # Store result
    tl.store(output_ptr + offsets, output, mask=mask)
