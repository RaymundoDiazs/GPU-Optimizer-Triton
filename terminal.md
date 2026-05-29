=== Errors for mul_sub.py on GPU 0 ===
Traceback (most recent call last):
  File "/data/results/call_acc/mul_sub.py", line 92, in <module>
    test_results = test_mul_sub()
                   ^^^^^^^^^^^^^^
  File "/data/results/call_acc/mul_sub.py", line 76, in test_mul_sub
    results["test_case_1"] = mul_sub(input_tensor, other_mul_tensor, other_sub_tensor)
                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/results/call_acc/mul_sub.py", line 55, in mul_sub
    mul_sub_kernel[grid](
  File "/usr/local/lib/python3.12/site-packages/triton/runtime/jit.py", line 345, in <lambda>
    return lambda *args, **kwargs: self.run(grid=grid, warmup=False, *args, **kwargs)
                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/site-packages/triton/runtime/jit.py", line 662, in run
    kernel = self.compile(
             ^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/site-packages/triton/compiler/compiler.py", line 276, in compile
    module = src.make_ir(options, codegen_fns, context)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/site-packages/triton/compiler/compiler.py", line 113, in make_ir
    return ast_to_ttir(self.fn, self, context=context, options=options, codegen_fns=codegen_fns)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
triton.compiler.errors.CompilationError: at 14:21:
                   output_ptr, n_elements, alpha, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    input_vals = tl.load(input_ptr + offsets, mask=mask)

    if other_mul_is_tensor:
        other_mul_vals = tl.load(other_mul_ptr + offsets, mask=mask)
        mul_result = input_vals * other_mul_vals
    else:
        mul_result = input_vals * other_mul_ptr
                     ^
IncompatibleTypeErrorImpl('invalid operands of type pointer<fp32> and triton.language.float32')

=== Output for ldl_factor.py on GPU 0 ===

=== Errors for ldl_factor.py on GPU 0 ===
Traceback (most recent call last):
  File "/data/results/call_acc/ldl_factor.py", line 78, in <module>
    test_results = test_ldl_factor()
                   ^^^^^^^^^^^^^^^^^
  File "/data/results/call_acc/ldl_factor.py", line 62, in test_ldl_factor
    results["test_case_1"] = ldl_factor(A1)
                             ^^^^^^^^^^^^^^
  File "/data/results/call_acc/ldl_factor.py", line 41, in ldl_factor
    A_ = A.view(batch_size, n, n)
         ^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: view() received an invalid combination of arguments - got (float, int, int), but expected one of:
 * (torch.dtype dtype)
 * (tuple of ints size)


=== Output for abs.py on GPU 0 ===

=== Errors for abs.py on GPU 0 ===

=== Output for mul.py on GPU 0 ===

=== Errors for mul.py on GPU 0 ===
Traceback (most recent call last):
  File "/data/results/call_acc/mul.py", line 83, in <module>
    test_results = test_mul()
                   ^^^^^^^^^^
  File "/data/results/call_acc/mul.py", line 64, in test_mul
    results["test_case_1"] = mul(input1, other1)
                             ^^^^^^^^^^^^^^^^^^^
  File "/data/results/call_acc/mul.py", line 48, in mul
    mul_kernel[grid](input_broadcasted, other_broadcasted, output, n_elements, is_scalar, scalar_value, BLOCK_SIZE=1024)
  File "/usr/local/lib/python3.12/site-packages/triton/runtime/jit.py", line 345, in <lambda>
    return lambda *args, **kwargs: self.run(grid=grid, warmup=False, *args, **kwargs)
                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/site-packages/triton/runtime/jit.py", line 662, in run
    kernel = self.compile(
             ^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/site-packages/triton/compiler/compiler.py", line 276, in compile
    module = src.make_ir(options, codegen_fns, context)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/site-packages/triton/compiler/compiler.py", line 113, in make_ir
    return ast_to_ttir(self.fn, self, context=context, options=options, codegen_fns=codegen_fns)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
triton.compiler.errors.CompilationError: at 8:4:
def mul_kernel(input_ptr, other_ptr, output_ptr, n_elements, is_scalar, scalar_value: tl.float32, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    input_vals = tl.load(input_ptr + offsets, mask=mask)

    if is_scalar:
    ^
AssertionError('mismatched type for result between then block (<[1024], fp32>) and else block (<[1024], int64>)')

=== Output for softmax.py on GPU 0 ===

=== Errors for softmax.py on GPU 0 ===

=== Output for leaky_relu.py on GPU 0 ===

=== Errors for leaky_relu.py on GPU 0 ===
Traceback (most recent call last):
  File "/usr/local/lib/python3.12/site-packages/triton/language/core.py", line 35, in wrapper
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/site-packages/triton/language/core.py", line 2014, in _promote_bfloat16_to_float32
    scalar_ty = t.type.scalar
                ^^^^^^
AttributeError: 'constexpr' object has no attribute 'type'

The above exception was the direct cause of the following exception:

triton.compiler.errors.CompilationError: at 2:12:
def max(input, axis=None, return_indices=False, return_indices_tie_break_left=True, keep_dims=False):
    input = core._promote_bfloat16_to_float32(input)
            ^

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/data/results/call_acc/leaky_relu.py", line 55, in <module>
    test_results = test_leaky_relu()
                   ^^^^^^^^^^^^^^^^^
  File "/data/results/call_acc/leaky_relu.py", line 39, in test_leaky_relu
    results["test_case_1"] = leaky_relu(input_tensor_1)
                             ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/results/call_acc/leaky_relu.py", line 24, in leaky_relu
    leaky_relu_kernel[grid](input, output, n_elements, negative_slope, BLOCK_SIZE=1024)
  File "/usr/local/lib/python3.12/site-packages/triton/runtime/jit.py", line 345, in <lambda>
    return lambda *args, **kwargs: self.run(grid=grid, warmup=False, *args, **kwargs)
                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/site-packages/triton/runtime/jit.py", line 662, in run
    kernel = self.compile(
             ^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/site-packages/triton/compiler/compiler.py", line 276, in compile
    module = src.make_ir(options, codegen_fns, context)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/site-packages/triton/compiler/compiler.py", line 113, in make_ir
    return ast_to_ttir(self.fn, self, context=context, options=options, codegen_fns=codegen_fns)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
triton.compiler.errors.CompilationError: at 6:15:
def leaky_relu_kernel(input_ptr, output_ptr, n_elements, negative_slope, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask)
    positive = tl.max(0, x)
               ^

=== Output for invert_matrix_lu.py on GPU 0 ===

=== Errors for invert_matrix_lu.py on GPU 0 ===
Traceback (most recent call last):
  File "/data/results/call_acc/invert_matrix_lu.py", line 72, in <module>
    test_results = test_invert_matrix_lu()
                   ^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/results/call_acc/invert_matrix_lu.py", line 56, in test_invert_matrix_lu
    results["test_case_1"] = invert_matrix_lu(A1)
                             ^^^^^^^^^^^^^^^^^^^^
  File "/data/results/call_acc/invert_matrix_lu.py", line 31, in invert_matrix_lu
    batch_size, n, _ = A.shape
    ^^^^^^^^^^^^^^^^
ValueError: not enough values to unpack (expected 3, got 2)

=== Output for std.py on GPU 0 ===

=== Errors for std.py on GPU 0 ===
Traceback (most recent call last):
  File "/data/results/call_acc/std.py", line 66, in <module>
    test_results = test_std()
                   ^^^^^^^^^^
  File "/data/results/call_acc/std.py", line 50, in test_std
    results["test_case_1"] = std(input_tensor)
                             ^^^^^^^^^^^^^^^^^
  File "/data/results/call_acc/std.py", line 28, in std
    std_kernel[grid](input, mean_tensor, output, n_elements, correction, BLOCK_SIZE=1024)
  File "/usr/local/lib/python3.12/site-packages/triton/runtime/jit.py", line 345, in <lambda>
    return lambda *args, **kwargs: self.run(grid=grid, warmup=False, *args, **kwargs)
                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/site-packages/triton/runtime/jit.py", line 662, in run
    kernel = self.compile(
             ^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/site-packages/triton/compiler/compiler.py", line 276, in compile
    module = src.make_ir(options, codegen_fns, context)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/site-packages/triton/compiler/compiler.py", line 113, in make_ir
    return ast_to_ttir(self.fn, self, context=context, options=options, codegen_fns=codegen_fns)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
triton.compiler.errors.CompilationError: at 7:22:
def std_kernel(input_ptr, mean_ptr, output_ptr, n_elements, correction, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input_vals = tl.load(input_ptr + offsets, mask=mask)
    mean_vals = tl.load(mean_ptr + offsets, mask=mask)
    variance = tl.sum((input_vals - mean_vals) ** 2, axis=0) / (n_elements - correction)
                      ^
AttributeError("'tensor' object has no attribute '__pow__'")

=== Output for tril_mm_and_scale.py on GPU 0 ===

=== Errors for tril_mm_and_scale.py on GPU 0 ===

=== Output for solve.py on GPU 0 ===

=== Errors for solve.py on GPU 0 ===
Traceback (most recent call last):
  File "/data/results/call_acc/solve.py", line 98, in <module>
    test_results = test_solve()
                   ^^^^^^^^^^^^
  File "/data/results/call_acc/solve.py", line 73, in test_solve
    results["test_case_1"] = solve(A1, B1)
                             ^^^^^^^^^^^^^
  File "/data/results/call_acc/solve.py", line 37, in solve
    assert A.ndim >= 2 and B.ndim >= 2, "A and B must be at least 2-dimensional"
AssertionError: A and B must be at least 2-dimensional

=== Output for airy_ai.py on GPU 0 ===

=== Errors for airy_ai.py on GPU 0 ===

=== Output for signbit.py on GPU 0 ===

=== Errors for signbit.py on GPU 0 ===
Traceback (most recent call last):
  File "/data/results/call_acc/signbit.py", line 62, in <module>
    test_results = test_signbit()
                   ^^^^^^^^^^^^^^
  File "/data/results/call_acc/signbit.py", line 46, in test_signbit
    results["test_case_1"] = signbit(input_tensor_1)
                             ^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/results/call_acc/signbit.py", line 31, in signbit
    signbit_kernel[grid](input, out, n_elements, BLOCK_SIZE=1024)
  File "/usr/local/lib/python3.12/site-packages/triton/runtime/jit.py", line 345, in <lambda>
    return lambda *args, **kwargs: self.run(grid=grid, warmup=False, *args, **kwargs)
                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/site-packages/triton/runtime/jit.py", line 662, in run
    kernel = self.compile(
             ^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/site-packages/triton/compiler/compiler.py", line 240, in compile
    key = f"{triton_key()}-{src.hash()}-{backend.hash()}-{options.hash()}-{str(sorted(env_vars.items()))}"
                            ^^^^^^^^^^
  File "/usr/local/lib/python3.12/site-packages/triton/compiler/compiler.py", line 109, in hash
    key = f"{self.fn.cache_key}-{self.attrs.hash()}-{sorted_sig}-{sorted_constants}"
             ^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/site-packages/triton/runtime/jit.py", line 758, in cache_key
    dependencies_finder.visit(self.parse())
  File "/usr/local/lib/python3.12/ast.py", line 407, in visit
    return visitor(node)
           ^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/ast.py", line 415, in generic_visit
    self.visit(item)
  File "/usr/local/lib/python3.12/ast.py", line 407, in visit
    return visitor(node)
           ^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/site-packages/triton/runtime/jit.py", line 167, in visit_FunctionDef
    self.generic_visit(node)
  File "/usr/local/lib/python3.12/ast.py", line 415, in generic_visit
    self.visit(item)
  File "/usr/local/lib/python3.12/ast.py", line 407, in visit
    return visitor(node)
           ^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/site-packages/triton/runtime/jit.py", line 221, in visit_Assign
    self.generic_visit(node)
  File "/usr/local/lib/python3.12/ast.py", line 417, in generic_visit
    self.visit(value)
  File "/usr/local/lib/python3.12/ast.py", line 407, in visit
    return visitor(node)
           ^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/ast.py", line 417, in generic_visit
    self.visit(value)
  File "/usr/local/lib/python3.12/ast.py", line 407, in visit
    return visitor(node)
           ^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/site-packages/triton/runtime/jit.py", line 127, in visit_Call
    func = self.visit(node.func)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/ast.py", line 407, in visit
    return visitor(node)
           ^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/site-packages/triton/runtime/jit.py", line 117, in visit_Attribute
    return getattr(lhs, node.attr)
           ^^^^^^^^^^^^^^^^^^^^^^^
AttributeError: module 'triton.language' has no attribute 'bitwise_and'

=== Output for matrix_multiply_and_row_dot.py on GPU 0 ===

=== Errors for matrix_multiply_and_row_dot.py on GPU 0 ===
Traceback (most recent call last):
  File "/usr/local/lib/python3.12/site-packages/triton/language/core.py", line 35, in wrapper
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/site-packages/triton/language/core.py", line 1670, in store
    return semantic.store(pointer, value, mask, boundary_check, cache_modifier, eviction_policy, _builder)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/site-packages/triton/language/semantic.py", line 1141, in store
    return _store_legacy(ptr, val, mask, boundary_check, cache, eviction, builder)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/site-packages/triton/language/semantic.py", line 1099, in _store_legacy
    raise ValueError("Mask argument cannot be block type if pointer argument is not a block")
ValueError: Mask argument cannot be block type if pointer argument is not a block

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/data/results/call_acc/matrix_multiply_and_row_dot.py", line 111, in <module>
    test_results = test_matrix_multiply_and_row_dot()
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/results/call_acc/matrix_multiply_and_row_dot.py", line 83, in test_matrix_multiply_and_row_dot
    results["test_case_1"] = matrix_multiply_and_row_dot(A, B, alpha, beta, C).item()
                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/results/call_acc/matrix_multiply_and_row_dot.py", line 64, in matrix_multiply_and_row_dot
    dot_product_kernel[(1,)](output, result, p, BLOCK_SIZE=128)
  File "/usr/local/lib/python3.12/site-packages/triton/runtime/jit.py", line 345, in <lambda>
    return lambda *args, **kwargs: self.run(grid=grid, warmup=False, *args, **kwargs)
                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/site-packages/triton/runtime/jit.py", line 662, in run
    kernel = self.compile(
             ^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/site-packages/triton/compiler/compiler.py", line 276, in compile
    module = src.make_ir(options, codegen_fns, context)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/site-packages/triton/compiler/compiler.py", line 113, in make_ir
    return ast_to_ttir(self.fn, self, context=context, options=options, codegen_fns=codegen_fns)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
triton.compiler.errors.CompilationError: at 13:4:
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < p

    # Load first two rows
    row_0 = tl.load(output_ptr + col_offsets, mask=mask)
    row_1 = tl.load(output_ptr + p + col_offsets, mask=mask)

    # Compute dot product
    dot_product = tl.sum(row_0 * row_1, axis=0)

    # Store the result
    tl.store(result_ptr, dot_product, mask=col_offsets == 0)  # Only need to store one element
    ^

=== Output for polygamma.py on GPU 0 ===

=== Errors for polygamma.py on GPU 0 ===

=== Output for elu_linear.py on GPU 0 ===

=== Errors for elu_linear.py on GPU 0 ===
Traceback (most recent call last):
  File "/usr/local/lib/python3.12/site-packages/triton/language/core.py", line 35, in wrapper
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/site-packages/triton/language/core.py", line 1534, in dot
    return semantic.dot(input, other, acc, input_precision, max_num_imprecise_acc, out_dtype, _builder)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/site-packages/triton/language/semantic.py", line 1367, in dot
    assert lhs_rank == rhs_rank == 2 or lhs_rank == rhs_rank == 3, f"Both inputs must be either 2D or 3D; (lhs: {lhs.shape} vs rhs: {rhs.shape})"    
AssertionError: Both inputs must be either 2D or 3D; (lhs: [constexpr[1024]] vs rhs: [constexpr[1024]])

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/data/results/call_acc/elu_linear.py", line 76, in <module>
    test_results = test_elu_linear()
                   ^^^^^^^^^^^^^^^^^
  File "/data/results/call_acc/elu_linear.py", line 55, in test_elu_linear
    results["test_case_1"] = elu_linear(input1, weight1, bias1)
                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/results/call_acc/elu_linear.py", line 34, in elu_linear
    linear_elu_kernel[grid](input, weight, output, n_rows, n_cols, alpha, inplace, BLOCK_SIZE=1024)
  File "/usr/local/lib/python3.12/site-packages/triton/runtime/jit.py", line 345, in <lambda>
    return lambda *args, **kwargs: self.run(grid=grid, warmup=False, *args, **kwargs)
                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/site-packages/triton/runtime/jit.py", line 662, in run
    kernel = self.compile(
             ^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/site-packages/triton/compiler/compiler.py", line 276, in compile
    module = src.make_ir(options, codegen_fns, context)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/site-packages/triton/compiler/compiler.py", line 113, in make_ir
    return ast_to_ttir(self.fn, self, context=context, options=options, codegen_fns=codegen_fns)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
triton.compiler.errors.CompilationError: at 10:18:
def linear_elu_kernel(input_ptr, weight_ptr, output_ptr, n_rows, n_cols, alpha, inplace, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    row = tl.load(input_ptr + row_idx * n_cols + col_offsets, mask=mask)
    weight = tl.load(weight_ptr + col_offsets, mask=mask)

    # Perform linear transformation
    dot_product = tl.dot(row, weight)
                  ^

=== Output for fused_pairwise_distance_normalize.py on GPU 0 ===

=== Errors for fused_pairwise_distance_normalize.py on GPU 0 ===
Traceback (most recent call last):
  File "/data/results/call_acc/fused_pairwise_distance_normalize.py", line 94, in <module>
    test_results = test_fused_pairwise_distance_normalize()
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/results/call_acc/fused_pairwise_distance_normalize.py", line 75, in test_fused_pairwise_distance_normalize
    results["test_case_1"] = fused_pairwise_distance_normalize(x1, x2)
                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/results/call_acc/fused_pairwise_distance_normalize.py", line 52, in fused_pairwise_distance_normalize
    normalize_kernel[grid](x1, normalized_x1, norm_x1, n_elements, p_norm, eps_norm, BLOCK_SIZE=BLOCK_SIZE)
  File "/usr/local/lib/python3.12/site-packages/triton/runtime/jit.py", line 345, in <lambda>
    return lambda *args, **kwargs: self.run(grid=grid, warmup=False, *args, **kwargs)
                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/site-packages/triton/runtime/jit.py", line 662, in run
    kernel = self.compile(
             ^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/site-packages/triton/compiler/compiler.py", line 276, in compile
    module = src.make_ir(options, codegen_fns, context)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/site-packages/triton/compiler/compiler.py", line 113, in make_ir
    return ast_to_ttir(self.fn, self, context=context, options=options, codegen_fns=codegen_fns)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
triton.compiler.errors.CompilationError: at 7:14:
def normalize_kernel(x_ptr, output_ptr, norm_ptr, n_elements, p_norm, eps_norm, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)

    abs_x_p = tl.abs(x) ** p_norm
              ^
AttributeError("'tensor' object has no attribute '__pow__'")

=== Output for Adam.py on GPU 0 ===

=== Errors for Adam.py on GPU 0 ===
Traceback (most recent call last):
  File "/data/results/call_acc/Adam.py", line 144, in <module>
    test_results = test_Adam()
                   ^^^^^^^^^^^
  File "/data/results/call_acc/Adam.py", line 124, in test_Adam
    optimizer1 = Adam(params1)
                 ^^^^^^^^^^^^^
  File "/data/results/call_acc/Adam.py", line 85, in Adam
    param, grad, exp_avg, exp_avg_sq, max_exp_avg_sq = param_group
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ValueError: not enough values to unpack (expected 5, got 2)

=== Output for fused_hstack_div.py on GPU 0 ===

=== Errors for fused_hstack_div.py on GPU 0 ===
Traceback (most recent call last):
  File "/data/results/call_acc/fused_hstack_div.py", line 82, in <module>
    test_results = test_fused_hstack_div()
                   ^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/results/call_acc/fused_hstack_div.py", line 68, in test_fused_hstack_div
    results["test_case_1"] = fused_hstack_div(tensors1, divisor1)
                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/results/call_acc/fused_hstack_div.py", line 52, in fused_hstack_div
    hstack_div_kernel[grid](stacked_tensor_ptr, divisor_tensor_ptr, out, n_elems, rounding_mode_flag, BLOCK_SIZE=1024)
  File "/usr/local/lib/python3.12/site-packages/triton/runtime/jit.py", line 345, in <lambda>
    return lambda *args, **kwargs: self.run(grid=grid, warmup=False, *args, **kwargs)
                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/site-packages/triton/runtime/jit.py", line 691, in run
    kernel.run(grid_0, grid_1, grid_2, stream, kernel.function, kernel.packed_metadata, launch_metadata,
  File "/usr/local/lib/python3.12/site-packages/triton/backends/nvidia/driver.py", line 365, in __call__
    self.launch(*args, **kwargs)
ValueError: Pointer argument (at 1) cannot be accessed from Triton (cpu tensor?)

=== Output for broadcast_tensors.py on GPU 0 ===

=== Errors for broadcast_tensors.py on GPU 0 ===
Traceback (most recent call last):
  File "/data/results/call_acc/broadcast_tensors.py", line 93, in <module>
    test_results = test_broadcast_tensors()
                   ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/results/call_acc/broadcast_tensors.py", line 74, in test_broadcast_tensors
    results["test_case_1"] = broadcast_tensors(x1, y1)
                             ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/results/call_acc/broadcast_tensors.py", line 51, in broadcast_tensors
    broadcast_kernel[grid](
  File "/usr/local/lib/python3.12/site-packages/triton/runtime/jit.py", line 345, in <lambda>
    return lambda *args, **kwargs: self.run(grid=grid, warmup=False, *args, **kwargs)
                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/site-packages/triton/runtime/jit.py", line 618, in run
    bound_args, sig_and_spec, constexpr_vals, non_constexpr_vals, excess_kwargs = self.binder(*args, **kwargs)
                                                                                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "<string>", line 2, in dynamic_func
  File "/usr/local/lib/python3.12/site-packages/triton/runtime/jit.py", line 328, in mangle_type
    dsk = (arg.dtype, is_const)
           ^^^^^^^^^
AttributeError: 'torch.Size' object has no attribute 'dtype'

=== Output for relu_conv2d.py on GPU 0 ===

=== Errors for relu_conv2d.py on GPU 0 ===
Traceback (most recent call last):
  File "/data/results/call_acc/relu_conv2d.py", line 151, in <module>
    test_results = test_relu_conv2d()
                   ^^^^^^^^^^^^^^^^^^
  File "/data/results/call_acc/relu_conv2d.py", line 131, in test_relu_conv2d
    results["test_case_1"] = relu_conv2d(input1, weight1)
                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/results/call_acc/relu_conv2d.py", line 81, in relu_conv2d
    conv2d_relu_kernel[grid](
  File "/usr/local/lib/python3.12/site-packages/triton/runtime/jit.py", line 345, in <lambda>
    return lambda *args, **kwargs: self.run(grid=grid, warmup=False, *args, **kwargs)
                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/site-packages/triton/runtime/jit.py", line 662, in run
    kernel = self.compile(
             ^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/site-packages/triton/compiler/compiler.py", line 240, in compile
    key = f"{triton_key()}-{src.hash()}-{backend.hash()}-{options.hash()}-{str(sorted(env_vars.items()))}"
                            ^^^^^^^^^^
  File "/usr/local/lib/python3.12/site-packages/triton/compiler/compiler.py", line 109, in hash
    key = f"{self.fn.cache_key}-{self.attrs.hash()}-{sorted_sig}-{sorted_constants}"
             ^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/site-packages/triton/runtime/jit.py", line 758, in cache_key
    dependencies_finder.visit(self.parse())
  File "/usr/local/lib/python3.12/ast.py", line 407, in visit
    return visitor(node)
           ^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/ast.py", line 415, in generic_visit
    self.visit(item)
  File "/usr/local/lib/python3.12/ast.py", line 407, in visit
    return visitor(node)
           ^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/site-packages/triton/runtime/jit.py", line 167, in visit_FunctionDef
    self.generic_visit(node)
  File "/usr/local/lib/python3.12/ast.py", line 415, in generic_visit
    self.visit(item)
  File "/usr/local/lib/python3.12/ast.py", line 407, in visit
    return visitor(node)
           ^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/site-packages/triton/runtime/jit.py", line 233, in visit_For
    self.generic_visit(node)
  File "/usr/local/lib/python3.12/ast.py", line 415, in generic_visit
    self.visit(item)
  File "/usr/local/lib/python3.12/ast.py", line 407, in visit
    return visitor(node)
           ^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/site-packages/triton/runtime/jit.py", line 233, in visit_For
    self.generic_visit(node)
  File "/usr/local/lib/python3.12/ast.py", line 415, in generic_visit
    self.visit(item)
  File "/usr/local/lib/python3.12/ast.py", line 407, in visit
    return visitor(node)
           ^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/site-packages/triton/runtime/jit.py", line 233, in visit_For
    self.generic_visit(node)
  File "/usr/local/lib/python3.12/ast.py", line 415, in generic_visit
    self.visit(item)
  File "/usr/local/lib/python3.12/ast.py", line 407, in visit
    return visitor(node)
           ^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/site-packages/triton/runtime/jit.py", line 221, in visit_Assign
    self.generic_visit(node)
  File "/usr/local/lib/python3.12/ast.py", line 417, in generic_visit
    self.visit(value)
  File "/usr/local/lib/python3.12/ast.py", line 407, in visit
    return visitor(node)
           ^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/site-packages/triton/runtime/jit.py", line 134, in visit_Call
    for obj in itertools.chain(
  File "/usr/local/lib/python3.12/ast.py", line 407, in visit
    return visitor(node)
           ^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/ast.py", line 417, in generic_visit
    self.visit(value)
  File "/usr/local/lib/python3.12/ast.py", line 407, in visit
    return visitor(node)
           ^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/ast.py", line 417, in generic_visit
    self.visit(value)
  File "/usr/local/lib/python3.12/ast.py", line 407, in visit
    return visitor(node)
           ^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/site-packages/triton/runtime/jit.py", line 127, in visit_Call
    func = self.visit(node.func)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/ast.py", line 407, in visit
    return visitor(node)
           ^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/site-packages/triton/runtime/jit.py", line 117, in visit_Attribute
    return getattr(lhs, node.attr)
           ^^^^^^^^^^^^^^^^^^^^^^^
AttributeError: module 'triton.language' has no attribute 'arensor_id'

=== Output for log.py on GPU 0 ===

=== Errors for log.py on GPU 0 ===

=== Output for adaptive_avg_pool2d.py on GPU 0 ===

=== Errors for adaptive_avg_pool2d.py on GPU 0 ===
Traceback (most recent call last):
  File "/data/results/call_acc/adaptive_avg_pool2d.py", line 85, in <module>
    from adaptive_avg_pool2d import adaptive_avg_pool2d
  File "/__modal/volumes/vo-WvFtwf25UW7xHyNHsL4jq5/results/call_acc/adaptive_avg_pool2d.py", line 112, in <module>
    test_results = test_adaptive_avg_pool2d()
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/__modal/volumes/vo-WvFtwf25UW7xHyNHsL4jq5/results/call_acc/adaptive_avg_pool2d.py", line 92, in test_adaptive_avg_pool2d
    output1 = adaptive_avg_pool2d(input1, 5)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/__modal/volumes/vo-WvFtwf25UW7xHyNHsL4jq5/results/call_acc/adaptive_avg_pool2d.py", line 71, in adaptive_avg_pool2d
    adaptive_avg_pool2d_kernel[grid](
  File "/usr/local/lib/python3.12/site-packages/triton/runtime/jit.py", line 345, in <lambda>
    return lambda *args, **kwargs: self.run(grid=grid, warmup=False, *args, **kwargs)
                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/site-packages/triton/runtime/jit.py", line 662, in run
    kernel = self.compile(
             ^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/site-packages/triton/compiler/compiler.py", line 276, in compile
    module = src.make_ir(options, codegen_fns, context)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/site-packages/triton/compiler/compiler.py", line 113, in make_ir
    return ast_to_ttir(self.fn, self, context=context, options=options, codegen_fns=codegen_fns)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
triton.compiler.errors.CompilationError: at 26:11:

    in_w_start = ((ow_offsets * in_width) // out_width).to(tl.int32)
    in_w_end = (((ow_offsets + 1) * in_width) // out_width).to(tl.int32)

    in_h_range = in_h_end - in_h_start
    in_w_range = in_w_end - in_w_start

    base_idx = batch_id * (n_channels * in_height * in_width) + channel_id * (in_height * in_width)
    pool_area = in_h_range * in_w_range

    for ih_off in range(oh_offsets.shape[0]):
        if oh_mask[ih_off]:
           ^
ValueError('Did you forget to add @triton.jit ? (`_builder` argument must be provided outside of JIT functions.)')

=== Output for quantize_dynamic.py on GPU 0 ===

=== Errors for quantize_dynamic.py on GPU 0 ===
Traceback (most recent call last):
  File "/data/results/call_acc/quantize_dynamic.py", line 65, in <module>
    test_results = test_quantize_dynamic()
                   ^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/results/call_acc/quantize_dynamic.py", line 27, in test_quantize_dynamic
    class SimpleModel(nn.Module):
                      ^^
NameError: name 'nn' is not defined

=== Output for conv2d_add.py on GPU 0 ===

=== Errors for conv2d_add.py on GPU 0 ===
Traceback (most recent call last):
  File "/usr/local/lib/python3.12/site-packages/triton/language/core.py", line 35, in wrapper
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/site-packages/triton/language/core.py", line 1192, in arange
    return semantic.arange(start, end, _builder)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/site-packages/triton/language/semantic.py", line 503, in arange
    raise ValueError("arange's arguments must be of type tl.constexpr")
ValueError: arange's arguments must be of type tl.constexpr

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/data/results/call_acc/conv2d_add.py", line 134, in <module>
    test_results = test_conv2d_add()
                   ^^^^^^^^^^^^^^^^^
  File "/data/results/call_acc/conv2d_add.py", line 111, in test_conv2d_add
    results["test_case_1"] = conv2d_add(input_tensor, weight_tensor, bias=bias_tensor)
                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/results/call_acc/conv2d_add.py", line 91, in conv2d_add
    conv2d_add_kernel[grid](
  File "/usr/local/lib/python3.12/site-packages/triton/runtime/jit.py", line 345, in <lambda>
    return lambda *args, **kwargs: self.run(grid=grid, warmup=False, *args, **kwargs)
                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/site-packages/triton/runtime/jit.py", line 662, in run
    kernel = self.compile(
             ^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/site-packages/triton/compiler/compiler.py", line 276, in compile
    module = src.make_ir(options, codegen_fns, context)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/site-packages/triton/compiler/compiler.py", line 113, in make_ir
    return ast_to_ttir(self.fn, self, context=context, options=options, codegen_fns=codegen_fns)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
triton.compiler.errors.CompilationError: at 16:12:
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_OUT_C: tl.constexpr
):
    batch_id = tl.program_id(0)
    out_c_inner_id = tl.program_id(1)
    iN = batch_id

    out_c_outer_id = tl.arange(0, BLOCK_SIZE_OUT_C)
    out_c = out_c_inner_id * BLOCK_SIZE_OUT_C + out_c_outer_id

    out_offsets = out_c < out_channels

    out_h = tl.arange(0, outH)
            ^

=== Output for ifftshift.py on GPU 0 ===

=== Errors for ifftshift.py on GPU 0 ===
Traceback (most recent call last):
  File "/data/results/call_acc/ifftshift.py", line 89, in <module>
    test_results = test_ifftshift()
                   ^^^^^^^^^^^^^^^^
  File "/data/results/call_acc/ifftshift.py", line 74, in test_ifftshift
    results["test_case_1"] = ifftshift(input_tensor_1d)
                             ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/results/call_acc/ifftshift.py", line 50, in ifftshift
    ifftshift_kernel[grid](
  File "/usr/local/lib/python3.12/site-packages/triton/runtime/jit.py", line 345, in <lambda>
    return lambda *args, **kwargs: self.run(grid=grid, warmup=False, *args, **kwargs)
                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/site-packages/triton/runtime/jit.py", line 662, in run
    kernel = self.compile(
             ^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/site-packages/triton/compiler/compiler.py", line 276, in compile
    module = src.make_ir(options, codegen_fns, context)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/site-packages/triton/compiler/compiler.py", line 113, in make_ir
    return ast_to_ttir(self.fn, self, context=context, options=options, codegen_fns=codegen_fns)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
triton.compiler.errors.CompilationError: at 15:8:
    mask = offsets < n_elements

    # Load the sizes, strides and dims data
    sizes = tl.load(sizes_ptr + tl.arange(0, n_dims), mask=tl.arange(0, n_dims) < n_dims)
    strides = tl.load(strides_ptr + tl.arange(0, n_dims), mask=tl.arange(0, n_dims) < n_dims)
    dims = tl.load(dims_ptr + tl.arange(0, n_dims), mask=tl.arange(0, n_dims) < n_dims)

    # Calculate inverse shifted indices
    linear_idx = offsets
    coord = tl.zeros([n_dims], dtype=tl.int32)
    for i in range(n_dims):
        coord[i] = linear_idx // strides[i]
        ^
AssertionError()

=== Output for signbit_bitwise_and.py on GPU 0 ===

=== Errors for signbit_bitwise_and.py on GPU 0 ===
Traceback (most recent call last):
  File "/data/results/call_acc/signbit_bitwise_and.py", line 83, in <module>
    test_results = test_signbit_bitwise_and()
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/results/call_acc/signbit_bitwise_and.py", line 64, in test_signbit_bitwise_and
    results["test_case_1"] = signbit_bitwise_and(a, b)
                             ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/results/call_acc/signbit_bitwise_and.py", line 36, in signbit_bitwise_and
    signbit_bitwise_and_kernel[grid](
  File "/usr/local/lib/python3.12/site-packages/triton/runtime/jit.py", line 345, in <lambda>
    return lambda *args, **kwargs: self.run(grid=grid, warmup=False, *args, **kwargs)
                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/site-packages/triton/runtime/jit.py", line 691, in run
    kernel.run(grid_0, grid_1, grid_2, stream, kernel.function, kernel.packed_metadata, launch_metadata,
  File "/usr/local/lib/python3.12/site-packages/triton/backends/nvidia/driver.py", line 365, in __call__
    self.launch(*args, **kwargs)
ValueError: Pointer argument (at 2) cannot be accessed from Triton (cpu tensor?)

=== Output for fused_repeat_interleave_log_softmax.py on GPU 0 ===

=== Errors for fused_repeat_interleave_log_softmax.py on GPU 0 ===
Traceback (most recent call last):
  File "/data/results/call_acc/fused_repeat_interleave_log_softmax.py", line 101, in <module>
    test_results = test_fused_repeat_interleave_log_softmax()
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/results/call_acc/fused_repeat_interleave_log_softmax.py", line 78, in test_fused_repeat_interleave_log_softmax
    results["test_case_1"] = fused_repeat_interleave_log_softmax(input1, repeats1)
                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/results/call_acc/fused_repeat_interleave_log_softmax.py", line 37, in fused_repeat_interleave_log_softmax
    assert input.dim() == repeats.dim()
                          ^^^^^^^^^^^
AttributeError: 'int' object has no attribute 'dim'

=== Output for cholesky.py on GPU 0 ===

=== Errors for cholesky.py on GPU 0 ===
Traceback (most recent call last):
  File "/data/results/call_acc/cholesky.py", line 103, in <module>
    test_results = test_cholesky()
                   ^^^^^^^^^^^^^^^
  File "/data/results/call_acc/cholesky.py", line 80, in test_cholesky
    L1 = cholesky(A1)
         ^^^^^^^^^^^^
  File "/data/results/call_acc/cholesky.py", line 47, in cholesky
    batch_size, n, _ = A.shape
    ^^^^^^^^^^^^^^^^
ValueError: not enough values to unpack (expected 3, got 2)

=== Output for ones_like.py on GPU 0 ===

=== Errors for ones_like.py on GPU 0 ===

=== Output for autocast.py on GPU 0 ===

=== Errors for autocast.py on GPU 0 ===

=== Output for reciprocal.py on GPU 0 ===

=== Errors for reciprocal.py on GPU 0 ===

=== Output for cos_signbit.py on GPU 0 ===

=== Errors for cos_signbit.py on GPU 0 ===
Traceback (most recent call last):
  File "/data/results/call_acc/cos_signbit.py", line 62, in <module>
    test_results = test_cos_signbit()
                   ^^^^^^^^^^^^^^^^^^
  File "/data/results/call_acc/cos_signbit.py", line 42, in test_cos_signbit
    cos_result_1, sign_bit_1 = cos_signbit(input_tensor_1)
                               ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/results/call_acc/cos_signbit.py", line 22, in cos_signbit
    cos_signbit_kernel[grid](input, cos_output, signbit_output, n_elements, BLOCK_SIZE=1024)
  File "/usr/local/lib/python3.12/site-packages/triton/runtime/jit.py", line 345, in <lambda>
    return lambda *args, **kwargs: self.run(grid=grid, warmup=False, *args, **kwargs)
                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/site-packages/triton/runtime/jit.py", line 691, in run
    kernel.run(grid_0, grid_1, grid_2, stream, kernel.function, kernel.packed_metadata, launch_metadata,
  File "/usr/local/lib/python3.12/site-packages/triton/backends/nvidia/driver.py", line 365, in __call__
    self.launch(*args, **kwargs)
ValueError: Pointer argument (at 2) cannot be accessed from Triton (cpu tensor?)

=== Output for spectral_norm_eig.py on GPU 0 ===

=== Errors for spectral_norm_eig.py on GPU 0 ===
Traceback (most recent call last):
  File "/data/results/call_acc/spectral_norm_eig.py", line 77, in <module>
    test_results = test_spectral_norm_eig()
                   ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/results/call_acc/spectral_norm_eig.py", line 60, in test_spectral_norm_eig
    results["test_case_1"] = spectral_norm_eig(A1)
                             ^^^^^^^^^^^^^^^^^^^^^
  File "/data/results/call_acc/spectral_norm_eig.py", line 30, in spectral_norm_eig
    out = torch.empty(*result_shape, dtype=A.dtype, device=A.device)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: empty() received an invalid combination of arguments - got (device=torch.device, dtype=torch.dtype, ), but expected one of:
 * (tuple of ints size, *, tuple of names names, torch.memory_format memory_format = None, torch.dtype dtype = None, torch.layout layout = None, torch.device device = None, bool pin_memory = False, bool requires_grad = False)
 * (tuple of ints size, *, torch.memory_format memory_format = None, Tensor out = None, torch.dtype dtype = None, torch.layout layout = None, torch.device device = None, bool pin_memory = False, bool requires_grad = False)


=== Output for fftn.py on GPU 0 ===

=== Errors for fftn.py on GPU 0 ===
Traceback (most recent call last):
  File "/data/results/call_acc/fftn.py", line 109, in <module>
    test_results = test_fftn()
                   ^^^^^^^^^^^
  File "/data/results/call_acc/fftn.py", line 90, in test_fftn
    results["test_case_1"] = fftn(input_tensor)
                             ^^^^^^^^^^^^^^^^^^
  File "/data/results/call_acc/fftn.py", line 66, in fftn
    fft_kernel[grid](input, out, axes_ptr, s_ptr, norm_factor, n_elements, BLOCK_SIZE=1024)
  File "/usr/local/lib/python3.12/site-packages/triton/runtime/jit.py", line 345, in <lambda>
    return lambda *args, **kwargs: self.run(grid=grid, warmup=False, *args, **kwargs)
                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/site-packages/triton/runtime/jit.py", line 618, in run
    bound_args, sig_and_spec, constexpr_vals, non_constexpr_vals, excess_kwargs = self.binder(*args, **kwargs)
                                                                                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "<string>", line 2, in dynamic_func
  File "/usr/local/lib/python3.12/site-packages/triton/runtime/jit.py", line 331, in mangle_type
    res = ("*k" if dsk[1] else "*") + type_canonicalisation_dict[str(dsk[0]).split('.')[-1]]
                                      ~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^
KeyError: 'complex64'

Deleted Adam.py
Deleted SGD.py
Error deleting __pycache__: [Errno 21] Is a directory: '/data/results/call_acc/__pycache__'
Deleted adaptive_avg_pool2d.py
Deleted add_gelu.py
Deleted addmm.py
Deleted argmax.py
Deleted asin.py
Deleted batch_norm.py
Deleted bitwise_and.py
Deleted bitwise_and_binomial.py
Deleted broadcast_tensors.py
Deleted chebyshev_polynomial_t.py
Deleted cholesky.py
Deleted cholesky_solve.py
Deleted combined_activation.py
Deleted conv2d.py
Deleted conv2d_add.py
Deleted cos_avg_pool1d.py
Deleted cos_signbit.py
Deleted det.py
Deleted determinant_lu.py
Deleted determinant_via_qr.py
Deleted digamma.py
Deleted div.py
Deleted dropout_relu_batch_norm_conv2d.py
Deleted dropout_sigmoid_linear.py
Deleted eig.py
Deleted elu_linear.py
Deleted exp_mean.py
Deleted fftn.py
Deleted fused_add_mul_groupnorm.py
Deleted fused_avg_pool2d_cosine_similarity.py
Deleted fused_bmm_dropout_gelu.py
Deleted fused_bmm_rmsnorm_gelu_dropout.py
Deleted fused_bmm_rmsnorm_gelu_dropout_sub.py
Deleted fused_cosine_embedding_loss_with_normalization.py
Deleted fused_cross_entropy_log_softmax.py
Deleted fused_cross_entropy_softmax_layernorm.py
Deleted fused_embedding_add_tanh.py
Deleted fused_fractional_max_pool2d_with_relu.py
Deleted fused_hardshrink_dropout.py
Deleted fused_hardsigmoid_batch_norm.py
Deleted fused_hstack_div.py
Deleted fused_index_select_eq.py
Deleted fused_instance_norm_selu_conv2d.py
Deleted fused_layer_norm_relu_linear.py
Deleted fused_lu_solve.py
Deleted fused_masked_select_add_gelu.py
Deleted fused_mul_add_logsoftmax_dropout_bmm.py
Deleted fused_mv_logsoftmax_dropout.py
Deleted fused_mv_sigmoid_sub.py
Deleted fused_pairwise_distance_adaptive_avg_pool2d.py
Deleted fused_pairwise_distance_normalize.py
Deleted fused_qr_solve.py
Deleted fused_repeat_interleave_log_softmax.py
Deleted fused_silu_layer_norm_conv2d.py
Deleted fused_svd_reconstruct.py
Deleted fused_tile_exp.py
Deleted fused_transformer_block.py
Deleted gammaln.py
Deleted gelu.py
Deleted gelu_conv2d.py
Deleted gelu_min.py
Deleted gelu_std.py
Deleted grid_sample.py
Deleted grid_sample_with_affine.py
Deleted i0.py
Deleted ifftshift.py
Deleted index_fill_.py
Deleted invert_matrix_lu.py
Deleted ldl_factor.py
Deleted leaky_relu.py
Deleted leaky_relu_conv2d.py
Deleted least_squares_qr.py
Deleted log_softmax_linear.py
Deleted log_tanh.py
Deleted logspace.py
Deleted logsumexp.py
Deleted low_rank_svd_approximation.py
Deleted lu.py
Deleted matmul.py
Deleted matrix_multiply_and_row_dot.py
Deleted matrix_multiply_symmetric.py
Deleted matrix_power_eig.py
Deleted max.py
Deleted mean.py
Deleted min.py
Deleted min_gelu.py
Deleted mul.py
Deleted mul_relu.py
Deleted mul_sub.py
Deleted normalize_pairwise_distance.py
Deleted normalized_cosine_similarity.py
Deleted permute_copy.py
Deleted pixel_shuffle_conv2d.py
Deleted pow.py
Deleted pseudoinverse_svd.py
Deleted qr.py
Deleted quantize_dynamic.py
Deleted rad2deg_sqrt.py
Deleted relu_batch_norm_conv2d.py
Deleted relu_conv2d.py
Deleted relu_max_pool2d_conv2d.py
Deleted scaled_add_dot.py
Deleted scaled_add_norm.py
Deleted selu.py
Deleted sigmoid_adaptive_avg_pool2d.py
Deleted sigmoid_argmax.py
Deleted sigmoid_conv2d.py
Deleted signbit.py
Deleted signbit_bitwise_and.py
Deleted softmax_log.py
Deleted softmax_mul.py
Deleted softplus_linear.py
Deleted solve.py
Deleted solve_multiple_lu.py
Deleted solve_symmetric_ldl.py
Deleted spectral_norm_eig.py
Deleted sqrt_tanh.py
Deleted std.py
Deleted sub.py
Deleted sub_gelu.py
Deleted sum.py
Deleted sum_std.py
Deleted svd.py
Deleted symmetric_matrix_vector_norm.py
Deleted tanh_linear.py
Deleted tensordot.py
Deleted zeta.py

Correct execution rate: 22.29%
['tanh.py', 'relu_sqrt.py', 'sqrt.py', 'rsqrt.py', 'add.py', 'relu.py', 'silu_batch_norm.py', 'log1p.py', 'sigmoid_batch_norm.py', 'sqrt_exp.py', 'logit.py', 'exp_sqrt.py', 'add_mean.py', 'fused_cholesky_solve.py', 'fused_gather_masked_fill.py', 'cos.py', 'trunc.py', 'exp.py', 'erfc_sqrt.py', 'tensordot_rsqrt.py', 'bessel_j1.py', 'symmetric_mm_and_abs_sum.py', 'solve_and_add_scaled_vector.py', 'matrix_vector_dot.py', 'erf.py', 'sigmoid.py', 'floor.py', 'rand.py', 'abs.py', 'softmax.py', 'tril_mm_and_scale.py', 'airy_ai.py', 'polygamma.py', 'log.py', 'ones_like.py', 'autocast.py', 'reciprocal.py']
Above is call test for openai_gpt-4o_simp
================================================================================================================================================================

call_acc survivors: 37 / 166

======================================================================
=== Phase 2: execution accuracy ===
======================================================================

Correct execution rate: 100.00% = 37 / 37
above is the compare execution for /data/results/call_acc
================================================================================================================================================================================================================================================

exe_acc survivors: 37 / 166

======================================================================
=== Phase 3: efficiency ===
======================================================================
Process:   3%|▉                                  | 1/37 [00:27<16:14, 27.06s/it]/ Running (1/1 containers active)... View app at https://modal.com/ap
Process:   8%|██▊                                | 3/37 [01:24<15:58, 28.20s/it]/ Running (1/1 containers active)... View app at https://modal.com/ap
Process:  11%|███▊                               | 4/37 [01:50<15:06, 27.46s/it]- Running (1/1 containers active)... View app at https://modal.com/ap
Process:  14%|████▋                              | 5/37 [02:36<18:13, 34.17s/it]
\ Running (1/1 containers active)... View app at https://modal.com/apps/deoh02/main/ap-hmJhaZerXXXSAkamb9YP60