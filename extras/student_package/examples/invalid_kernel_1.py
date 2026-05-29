# INVALID KERNEL EXAMPLE 1: Missing Colon
#
# ERROR: Missing colon after function definition
# Your parser should REJECT this kernel
#
# The correct syntax would be:
#   def kernel(x_ptr, n)
#                      ^-- MISSING COLON

@triton.jit
def kernel(x_ptr, n)
    x = tl.load(x_ptr)
    tl.store(x_ptr, x)
