# INVALID KERNEL EXAMPLE 3: Invalid Token
#
# ERROR: Invalid token '$' - not part of Python/Triton syntax
# Your lexer should fail on this invalid character
#
# Valid operators include: +, -, *, /, //, %, **, etc.
# But $ is not a valid operator in Python or Triton.

@triton.jit
def kernel():
    x = 5 $ 3
