# INVALID KERNEL EXAMPLE 2: Assignment to Literal
#
# ERROR: Cannot assign to a literal value
# Your parser should REJECT this kernel
#
# Only names, subscripts, and attributes can be assignment targets.
# "5 = x" is invalid because 5 is a literal, not a valid target.

@triton.jit
def kernel():
    5 = x
