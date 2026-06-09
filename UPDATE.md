	
Content
Settings

======================================================================
=== Phase 1: call accuracy ===
======================================================================
instruction
=== Output for fused_bmm_rmsnorm_gelu_dropout_sub.py on GPU 0 ===

=== Errors for fused_bmm_rmsnorm_gelu_dropout_sub.py on GPU 0 ===
Traceback (most recent call last):
  File "/data/results_constrained_partial/call_acc/fused_bmm_rmsnorm_gelu_dropout_sub.py", line 105, in <module>
    test_results = test_fused_bmm_rmsnorm_gelu_dropout_sub()
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/fused_bmm_rmsnorm_gelu_dropout_sub.py", line 89, in test_fused_bmm_rmsnorm_gelu_dropout_sub
    results["test_case_1"] = fused_bmm_rmsnorm_gelu_dropout_sub(input1, input2, other, normalized_shape)
                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/fused_bmm_rmsnorm_gelu_dropout_sub.py", line 61, in fused_bmm_rmsnorm_gelu_dropout_sub
    F = other.shape[3]
        ~~~~~~~~~~~^^^
IndexError: tuple index out of range

=== Output for div.py on GPU 0 ===

=== Errors for div.py on GPU 0 ===
  File "/data/results_constrained_partial/call_acc/div.py", line 53
    tl.store(output_ptr + offsets, x / y, mask=mask)
    ^
IndentationError: expected an indented block after 'elif' statement on line 52

=== Output for sigmoid_conv2d.py on GPU 0 ===

=== Errors for sigmoid_conv2d.py on GPU 0 ===
  File "/data/results_constrained_partial/call_acc/sigmoid_conv2d.py", line 34
    mask = (offs_ih[:, None] >= 0) & (offs_ih[:, None] < iH - kh * block_height) &
                                                                                  ^
SyntaxError: invalid syntax

=== Output for solve_multiple_lu.py on GPU 0 ===

=== Errors for solve_multiple_lu.py on GPU 0 ===

=== Output for tanh.py on GPU 0 ===

=== Errors for tanh.py on GPU 0 ===
Traceback (most recent call last):
  File "/data/results_constrained_partial/call_acc/tanh.py", line 48, in <module>
    test_results = test_tanh()
                   ^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/tanh.py", line 32, in test_tanh
    results["test_case_1"] = tanh(input_tensor_1)
                             ^^^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/tanh.py", line 18, in tanh
    tanh_kernel[grid](input, output, n_elements, BLOCK_SIZE=1024)
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
  File "/usr/local/lib/python3.12/site-packages/triton/runtime/jit.py", line 127, in visit_Call
    func = self.visit(node.func)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/ast.py", line 407, in visit
    return visitor(node)
           ^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/site-packages/triton/runtime/jit.py", line 117, in visit_Attribute
    return getattr(lhs, node.attr)
           ^^^^^^^^^^^^^^^^^^^^^^^
AttributeError: module 'triton.language.math' has no attribute 'tanh'

=== Output for relu_sqrt.py on GPU 0 ===

=== Errors for relu_sqrt.py on GPU 0 ===

=== Output for sqrt.py on GPU 0 ===

=== Errors for sqrt.py on GPU 0 ===

=== Output for sigmoid_argmax.py on GPU 0 ===

=== Errors for sigmoid_argmax.py on GPU 0 ===
Traceback (most recent call last):
  File "/data/results_constrained_partial/call_acc/sigmoid_argmax.py", line 88, in <module>
    test_results = test_sigmoid_argmax()
                   ^^^^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/sigmoid_argmax.py", line 72, in test_sigmoid_argmax
    results["test_case_1"] = sigmoid_argmax(input1)
                             ^^^^^^^^^^^^^^
NameError: name 'sigmoid_argmax' is not defined. Did you mean: 'test_sigmoid_argmax'?

=== Output for sub.py on GPU 0 ===

=== Errors for sub.py on GPU 0 ===

=== Output for grid_sample.py on GPU 0 ===

=== Errors for grid_sample.py on GPU 0 ===
Traceback (most recent call last):
  File "/data/results_constrained_partial/call_acc/grid_sample.py", line 90, in <module>
    test_results = test_grid_sample()
                   ^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/grid_sample.py", line 75, in test_grid_sample
    results["test_case_1"] = grid_sample(input_4d, grid_4d)
                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/grid_sample.py", line 43, in grid_sample
    grid_ptr = grid.data.ptr
               ^^^^^^^^^^^^^
AttributeError: 'Tensor' object has no attribute 'ptr'

=== Output for svd.py on GPU 0 ===

=== Errors for svd.py on GPU 0 ===
Traceback (most recent call last):
  File "/usr/local/lib/python3.12/site-packages/triton/language/core.py", line 35, in wrapper
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/site-packages/triton/language/core.py", line 1220, in full
    shape = _shape_check_impl(shape)
            ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/site-packages/triton/language/core.py", line 1201, in _shape_check_impl
    raise TypeError(f"Shape element {i} must have type `constexpr`")
TypeError: Shape element 0 must have type `constexpr`

The above exception was the direct cause of the following exception:

triton.compiler.errors.CompilationError: at 10:11:
def zeros(shape, dtype):
    """
    Returns a tensor filled with the scalar value 0 for the given :code:`shape` and :code:`dtype`.

    :param shape: Shape of the new array, e.g., (8, 16) or (8, )
    :type shape: tuple of ints
    :param dtype: Data-type of the new array, e.g., :code:`tl.float16`
    :type dtype: DType
    """
    return core.full(shape, 0, dtype)
           ^

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/data/results_constrained_partial/call_acc/svd.py", line 62, in <module>
    test_results = test_svd()
                   ^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/svd.py", line 42, in test_svd
    U1, S1, Vh1 = svd(A1, full_matrices=True)
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/svd.py", line 28, in svd
    svd_kernel[grid](A, output, output, output, m, n, k, BLOCK_SIZE=1024)
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
triton.compiler.errors.CompilationError: at 6:8:
def svd_kernel(A_ptr, U_ptr, S_ptr, Vh_ptr, m, n, k, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < min(m, n)
    A = tl.load(A_ptr + offsets, mask=mask)
    U = tl.zeros((m, k), dtype=tl.float32)
        ^

=== Output for i0.py on GPU 0 ===

=== Errors for i0.py on GPU 0 ===
Traceback (most recent call last):
  File "/data/results_constrained_partial/call_acc/i0.py", line 51, in <module>
    test_results = test_i0()
                   ^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/i0.py", line 35, in test_i0
    results["test_case_1"] = i0(input_tensor_1)
                             ^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/i0.py", line 21, in i0
    i0_kernel[grid](input_tensor, output, n_elements, BLOCK_SIZE=1024)
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
triton.compiler.errors.CompilationError: at 6:26:
def i0_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    result = tl.math.exp(-x**2 / 2)
                          ^
AttributeError("'tensor' object has no attribute '__pow__'")

=== Output for rsqrt.py on GPU 0 ===

=== Errors for rsqrt.py on GPU 0 ===

=== Output for dropout_relu_batch_norm_conv2d.py on GPU 0 ===

=== Errors for dropout_relu_batch_norm_conv2d.py on GPU 0 ===
  File "/data/results_constrained_partial/call_acc/dropout_relu_batch_norm_conv2d.py", line 44
    , training, inplace)
                       ^
SyntaxError: closing parenthesis ')' does not match opening parenthesis '[' on line 43

=== Output for fused_mv_logsoftmax_dropout.py on GPU 0 ===

=== Errors for fused_mv_logsoftmax_dropout.py on GPU 0 ===

=== Output for add.py on GPU 0 ===

=== Errors for add.py on GPU 0 ===
Traceback (most recent call last):
  File "/usr/local/lib/python3.12/site-packages/triton/language/core.py", line 35, in wrapper
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/site-packages/triton/language/core.py", line 1597, in load
    return semantic.load(pointer, mask, other, boundary_check, padding_option, cache_modifier, eviction_policy,
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/site-packages/triton/language/semantic.py", line 1037, in load
    return _load_legacy(ptr, mask, other, boundary_check, padding, cache, eviction, is_volatile, builder)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/site-packages/triton/language/semantic.py", line 969, in _load_legacy
    raise ValueError(f"Unsupported ptr type {ptr.type.__repr__()} in `tl.load`")
ValueError: Unsupported ptr type <[1024], fp32> in `tl.load`

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/data/results_constrained_partial/call_acc/add.py", line 52, in <module>
    test_results = test_add()
                   ^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/add.py", line 38, in test_add
    results["test_case_2"] = add(input2, other2)
                             ^^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/add.py", line 18, in add
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
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
triton.compiler.errors.CompilationError: at 6:8:
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
        ^

=== Output for fused_silu_layer_norm_conv2d.py on GPU 0 ===

=== Errors for fused_silu_layer_norm_conv2d.py on GPU 0 ===
  File "/data/results_constrained_partial/call_acc/fused_silu_layer_norm_conv2d.py", line 17
    for k in range(0, tl.cdiv(K1024_000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
SyntaxError: invalid syntax. Perhaps you forgot a comma?

=== Output for fused_index_select_eq.py on GPU 0 ===

=== Errors for fused_index_select_eq.py on GPU 0 ===
Traceback (most recent call last):
  File "/data/results_constrained_partial/call_acc/fused_index_select_eq.py", line 81, in <module>
    test_results = test_fused_index_select_eq()
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/fused_index_select_eq.py", line 56, in test_fused_index_select_eq
    results["test_case_1"] = fused_index_select_eq(input_tensor, dim, index, other)
                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/fused_index_select_eq.py", line 32, in fused_index_select_eq
    INDEX_PTR = index.data_pointer
                ^^^^^^^^^^^^^^^^^^
AttributeError: 'Tensor' object has no attribute 'data_pointer'

=== Output for argmax.py on GPU 0 ===

=== Errors for argmax.py on GPU 0 ===
Traceback (most recent call last):
  File "/data/results_constrained_partial/call_acc/argmax.py", line 52, in <module>
    test_results = test_argmax()
                   ^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/argmax.py", line 38, in test_argmax
    results["test_case_1"] = argmax(tensor_2d, dim=0)
                             ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/argmax.py", line 24, in argmax
    argmax_kernel[(n_rows,)](output, x, x.stride(0), output.stride(0), n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE)
                                                     ^^^^^^^^^^^^^^^^
IndexError: Dimension specified as 0 but tensor has no dimensions

=== Output for fused_lu_solve.py on GPU 0 ===

=== Errors for fused_lu_solve.py on GPU 0 ===
  File "/data/results_constrained_partial/call_acc/fused_lu_solve.py", line 39
    @triton.jit
IndentationError: unexpected indent

=== Output for normalize_pairwise_distance.py on GPU 0 ===

=== Errors for normalize_pairwise_distance.py on GPU 0 ===
Traceback (most recent call last):
  File "/data/results_constrained_partial/call_acc/normalize_pairwise_distance.py", line 58, in <module>
    test_results = test_normalize_pairwise_distance()
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/normalize_pairwise_distance.py", line 52, in test_normalize_pairwise_distance
    results["test_case_1"] = normalize_pairwise_distance(x1, x2, p_distance=2.0, dim_norm=0)
                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/normalize_pairwise_distance.py", line 28, in normalize_pairwise_distance
    normalize_pairwise_distance_kernel[(n_rows,)](y, x1.data, x2.data, x1.stride(0), x2.stride(0), y.stride(0), n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE)
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
  File "/usr/local/lib/python3.12/site-packages/triton/runtime/jit.py", line 134, in visit_Call
    for obj in itertools.chain(
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
AttributeError: module 'triton.language' has no attribute 'pow'

=== Output for max.py on GPU 0 ===

=== Errors for max.py on GPU 0 ===

=== Output for log_softmax_linear.py on GPU 0 ===

=== Errors for log_softmax_linear.py on GPU 0 ===
Traceback (most recent call last):
  File "/data/results_constrained_partial/call_acc/log_softmax_linear.py", line 93, in <module>
    test_results = test_log_softmax_linear()
                   ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/log_softmax_linear.py", line 72, in test_log_softmax_linear
    results["test_case_1"] = log_softmax_linear(input1, weight1, bias1)
                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/log_softmax_linear.py", line 46, in log_softmax_linear
    log_softmax_linear_kernel[grid](input, weight, bias, out, M, N, K,
  File "/usr/local/lib/python3.12/site-packages/triton/runtime/jit.py", line 345, in <lambda>
    return lambda *args, **kwargs: self.run(grid=grid, warmup=False, *args, **kwargs)
                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/site-packages/triton/runtime/jit.py", line 618, in run
    bound_args, sig_and_spec, constexpr_vals, non_constexpr_vals, excess_kwargs = self.binder(*args, **kwargs)
                                                                                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: dynamic_func() got multiple values for argument 'BLOCK_M'

=== Output for relu.py on GPU 0 ===

=== Errors for relu.py on GPU 0 ===

=== Output for least_squares_qr.py on GPU 0 ===

=== Errors for least_squares_qr.py on GPU 0 ===
Traceback (most recent call last):
  File "/data/results_constrained_partial/call_acc/least_squares_qr.py", line 65, in <module>
    test_results = test_least_squares_qr()
                   ^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/least_squares_qr.py", line 55, in test_least_squares_qr
    results["test_case_1"] = least_squares_qr(A1, b1)
                             ^^^^^^^^^^^^^^^^
NameError: name 'least_squares_qr' is not defined. Did you mean: 'test_least_squares_qr'?

=== Output for determinant_via_qr.py on GPU 0 ===

=== Errors for determinant_via_qr.py on GPU 0 ===
Traceback (most recent call last):
  File "/data/results_constrained_partial/call_acc/determinant_via_qr.py", line 87, in <module>
    test_results = test_determinant_via_qr()
                   ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/determinant_via_qr.py", line 71, in test_determinant_via_qr
    results["test_case_1"] = determinant_via_qr(A1)
                             ^^^^^^^^^^^^^^^^^^
NameError: name 'determinant_via_qr' is not defined. Did you mean: 'test_determinant_via_qr'?

=== Output for fused_tile_exp.py on GPU 0 ===

=== Errors for fused_tile_exp.py on GPU 0 ===

=== Output for sqrt_tanh.py on GPU 0 ===

=== Errors for sqrt_tanh.py on GPU 0 ===
Traceback (most recent call last):
  File "/data/results_constrained_partial/call_acc/sqrt_tanh.py", line 50, in <module>
    test_results = test_sqrt_tanh()
                   ^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/sqrt_tanh.py", line 34, in test_sqrt_tanh
    results["test_case_1"] = sqrt_tanh(input1)
                             ^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/sqrt_tanh.py", line 20, in sqrt_tanh
    sqrt_tanh_kernel[grid](input.data, out.data, n_elements, BLOCK_SIZE=1024)
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
  File "/usr/local/lib/python3.12/site-packages/triton/runtime/jit.py", line 127, in visit_Call
    func = self.visit(node.func)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/ast.py", line 407, in visit
    return visitor(node)
           ^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/site-packages/triton/runtime/jit.py", line 117, in visit_Attribute
    return getattr(lhs, node.attr)
           ^^^^^^^^^^^^^^^^^^^^^^^
AttributeError: module 'triton.language' has no attribute 'tanh'. Did you mean: 'trans'?

=== Output for silu_batch_norm.py on GPU 0 ===

=== Errors for silu_batch_norm.py on GPU 0 ===
  File "/data/results_constrained_partial/call_acc/silu_batch_norm.py", line 69
    silu_batch_norm_kernel[grid](input.data, running_mean.data, running_var.data, weight.data, bias.data, output.data, M, N
                                ^
SyntaxError: '(' was never closed

=== Output for index_fill_.py on GPU 0 ===

=== Errors for index_fill_.py on GPU 0 ===
Traceback (most recent call last):
  File "/data/results_constrained_partial/call_acc/index_fill_.py", line 60, in <module>
    test_results = test_index_fill_()
                   ^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/index_fill_.py", line 38, in test_index_fill_
    results["test_case_1"] = index_fill_(0, x1, index1, value1).cpu()
                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/index_fill_.py", line 20, in index_fill_
    n_elements = self.numel()
                 ^^^^^^^^^^
AttributeError: 'int' object has no attribute 'numel'

=== Output for fused_cross_entropy_softmax_layernorm.py on GPU 0 ===

=== Errors for fused_cross_entropy_softmax_layernorm.py on GPU 0 ===
  File "/data/results_constrained_partial/call_acc/fused_cross_entropy_softmax_layernorm.py", line 80
    import torch
IndentationError: unexpected indent

=== Output for mean.py on GPU 0 ===

=== Errors for mean.py on GPU 0 ===
  File "/data/results_constrained_partial/call_acc/mean.py", line 55
    def mean(input
            ^
SyntaxError: '(' was never closed

=== Output for eig.py on GPU 0 ===

=== Errors for eig.py on GPU 0 ===

=== Output for logsumexp.py on GPU 0 ===

=== Errors for logsumexp.py on GPU 0 ===
Traceback (most recent call last):
  File "/data/results_constrained_partial/call_acc/logsumexp.py", line 57, in <module>
    test_results = test_logsumexp()
                   ^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/logsumexp.py", line 49, in test_logsumexp
    results["test_case_3"] = logsumexp(input_tensor_3, dim=2)
                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/logsumexp.py", line 22, in logsumexp
    n_rows, n_cols = input.shape
    ^^^^^^^^^^^^^^
ValueError: too many values to unpack (expected 2)

=== Output for fused_embedding_add_tanh.py on GPU 0 ===

=== Errors for fused_embedding_add_tanh.py on GPU 0 ===
  File "/data/results_constrained_partial/call_acc/fused_embedding_add_tanh.py", line 32
    fused_embedding_add_tanh_kernel[(n_rows,)](output, input_indices.data, input_indices.stride(0), weight.data, other.data, input_indices.stride(1), weight.stride(1), other.stride(1), output.data, n_rows
                                              ^
SyntaxError: '(' was never closed

=== Output for fused_mv_sigmoid_sub.py on GPU 0 ===

=== Errors for fused_mv_sigmoid_sub.py on GPU 0 ===
  File "/data/results_constrained_partial/call_acc/fused_mv_sigmoid_sub.py", line 34
    fused_mv_sigmoid_sub_kernel[input.stride(0), input.stride(1), output.stride(0), n_rows, n_cols, BLOCK_SIZE=1024]
                                                                                                    ^^^^^^^^^^^^^^^
SyntaxError: invalid syntax. Maybe you meant '==' or ':=' instead of '='?

=== Output for add_gelu.py on GPU 0 ===

=== Errors for add_gelu.py on GPU 0 ===
Traceback (most recent call last):
  File "/data/results_constrained_partial/call_acc/add_gelu.py", line 88, in <module>
    test_results = test_add_gelu()
                   ^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/add_gelu.py", line 72, in test_add_gelu
    results["test_case_1"] = add_gelu(input_tensor, other_tensor)
                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/add_gelu.py", line 48, in add_gelu
    M, K = input.shape
    ^^^^
ValueError: not enough values to unpack (expected 2, got 1)

=== Output for fused_cosine_embedding_loss_with_normalization.py on GPU 0 ===

=== Errors for fused_cosine_embedding_loss_with_normalization.py on GPU 0 ===
Traceback (most recent call last):
  File "/data/results_constrained_partial/call_acc/fused_cosine_embedding_loss_with_normalization.py", line 71, in <module>
    test_results = test_fused_cosine_embedding_loss_with_normalization()
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/fused_cosine_embedding_loss_with_normalization.py", line 55, in test_fused_cosine_embedding_loss_with_normalization
    results["test_case_1"] = fused_cosine_embedding_loss_with_normalization(input1, input2, target)
                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/fused_cosine_embedding_loss_with_normalization.py", line 30, in fused_cosine_embedding_loss_with_normalization
    cosine_embedding_loss_kernel[(n_rows,)](output, input1.data, input2.data, target.data, input1.stride(0), input1.stride(1), output.stride(0), target.stride(0), n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE)
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
AttributeError: module 'triton.language' has no attribute 'norm'

=== Output for fused_transformer_block.py on GPU 0 ===

=== Errors for fused_transformer_block.py on GPU 0 ===
Traceback (most recent call last):
  File "/data/results_constrained_partial/call_acc/fused_transformer_block.py", line 74, in <module>
    test_results = test_fused_transformer_block()
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/fused_transformer_block.py", line 49, in test_fused_transformer_block
    results["test_case_1"] = fused_transformer_block(input1, weight1_1, weight2_1, residual1)
                             ^^^^^^^^^^^^^^^^^^^^^^^
NameError: name 'fused_transformer_block' is not defined. Did you mean: 'test_fused_transformer_block'?

=== Output for log1p.py on GPU 0 ===

=== Errors for log1p.py on GPU 0 ===

=== Output for sigmoid_batch_norm.py on GPU 0 ===

=== Errors for sigmoid_batch_norm.py on GPU 0 ===
  File "/data/results_constrained_partial/call_acc/sigmoid_batch_norm.py", line 6
    def sigmoid_batch_norm_kernel(input_ptr, running_mean_ptr, running_var_ptr, weight_ptr, bias_ptr, output_ptr, M, N, C, stride_im, stride_in, stride_om, stride_onm1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1: tl.constexpr, stride_im1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d, stride_im1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d, stride_im1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d, stride_im1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d, stride_im1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d, stride_im1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d, stride_im1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d, stride_im1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d, stride_im1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d, stride_im1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d, stride_im1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d, stride_im1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d):
                                                                                                                                                                                                                                                                                                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
SyntaxError: duplicate argument 'stride_im1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d' in function definition

=== Output for fused_hardsigmoid_batch_norm.py on GPU 0 ===

=== Errors for fused_hardsigmoid_batch_norm.py on GPU 0 ===
  File "/data/results_constrained_partial/call_acc/fused_hardsigmoid_batch_norm.py", line 22
    running_mean = tl.load(running_mean_ptr + offs_out_running_mean_offset[:, None] * stride_out_running_mean_offset + offs_out_running_var_offset[:, None] * stride_out_running_var_offset, mask=offs_out_m
                          ^
SyntaxError: '(' was never closed

=== Output for zeta.py on GPU 0 ===

=== Errors for zeta.py on GPU 0 ===

=== Output for symmetric_matrix_vector_norm.py on GPU 0 ===

=== Errors for symmetric_matrix_vector_norm.py on GPU 0 ===
  File "/data/results_constrained_partial/call_acc/symmetric_matrix_vector_norm.py", line 6
    def symmetric_matrix_vector_norm_kernel(A_ptr, x_ptr, y_ptr, alpha, beta, p, M, N, stride_am, stride_an, stride_xn, stride_yn, stride_yb_ptr100000000000000000000000000000000000000000000000: tl.constexpr, stride_yb_ptr200000000000000000000000000000000000000000000000, stride_yb_ptr300000000000000000000000000000000000000000000000, stride_yb_ptr400000000000000000000000000000000000000000000000, stride_yb_ptr500000000000000000000000000000000000000000000000, stride_yb_ptr600000000000000000000000000000000000000000000000, stride_yb_ptr700000000000000000000000000000000000000000000000, stride_yb_ptr800000000000000000000000000000000000000000000000, stride_yb_ptr900000000000000000000000000000000000000000000000, stride_yb_ptr100000000000000000000000000000000000000000000000, stride_yb_ptr110000000000000000000000000000000000000000000000, stride_yb_ptr120000000000000000000000000000000000000000000000, stride_yb_ptr130000000000000000000000000000000000000000000000):
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
SyntaxError: duplicate argument 'stride_yb_ptr100000000000000000000000000000000000000000000000' in function definition

=== Output for softplus_linear.py on GPU 0 ===

=== Errors for softplus_linear.py on GPU 0 ===
  File "/data/results_constrained_partial/call_acc/softplus_linear.py", line 16
    w_ptrs = weight_ptr + offs_wt[:, None] * stride_wt + offs_biases_ptr01234567890123456789012345678901234567890123: tl.constexpr
                                                                                                                    ^
SyntaxError: invalid syntax

=== Output for fused_svd_reconstruct.py on GPU 0 ===

=== Errors for fused_svd_reconstruct.py on GPU 0 ===
Traceback (most recent call last):
  File "/data/results_constrained_partial/call_acc/fused_svd_reconstruct.py", line 86, in <module>
    test_results = test_fused_svd_reconstruct()
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/fused_svd_reconstruct.py", line 70, in test_fused_svd_reconstruct
    results["test_case_1"] = fused_svd_reconstruct(A1)
                             ^^^^^^^^^^^^^^^^^^^^^
NameError: name 'fused_svd_reconstruct' is not defined. Did you mean: 'test_fused_svd_reconstruct'?

=== Output for fused_mul_add_logsoftmax_dropout_bmm.py on GPU 0 ===

=== Errors for fused_mul_add_logsoftmax_dropout_bmm.py on GPU 0 ===
Traceback (most recent call last):
  File "/data/results_constrained_partial/call_acc/fused_mul_add_logsoftmax_dropout_bmm.py", line 79, in <module>
    test_results = test_fused_mul_add_logsoftmax_dropout_bmm()
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/fused_mul_add_logsoftmax_dropout_bmm.py", line 54, in test_fused_mul_add_logsoftmax_dropout_bmm
    results["test_case_1"] = fused_mul_add_logsoftmax_dropout_bmm(input1, input2, other, mat2)
                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/fused_mul_add_logsoftmax_dropout_bmm.py", line 28, in fused_mul_add_logsoftmax_dropout_bmm
    M, K = input1.shape
    ^^^^
ValueError: too many values to unpack (expected 2)

=== Output for selu.py on GPU 0 ===

=== Errors for selu.py on GPU 0 ===

=== Output for scaled_add_norm.py on GPU 0 ===

=== Errors for scaled_add_norm.py on GPU 0 ===
Traceback (most recent call last):
  File "/data/results_constrained_partial/call_acc/scaled_add_norm.py", line 63, in <module>
    test_results = test_scaled_add_norm()
                   ^^^^^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/scaled_add_norm.py", line 41, in test_scaled_add_norm
    results["test_case_1"] = scaled_add_norm(y1, x1, alpha1).item()
                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/scaled_add_norm.py", line 22, in scaled_add_norm
    y_ptr = y.data_pointer()
            ^^^^^^^^^^^^^^
AttributeError: 'Tensor' object has no attribute 'data_pointer'

=== Output for leaky_relu_conv2d.py on GPU 0 ===

=== Errors for leaky_relu_conv2d.py on GPU 0 ===
Traceback (most recent call last):
  File "/data/results_constrained_partial/call_acc/leaky_relu_conv2d.py", line 89, in <module>
    test_results = test_leaky_relu_conv2d()
                   ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/leaky_relu_conv2d.py", line 67, in test_leaky_relu_conv2d
    results["test_case_1"] = leaky_relu_conv2d(input, weight, bias)
                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/leaky_relu_conv2d.py", line 30, in leaky_relu_conv2d
    assert C_in == C_out, "Input and output channels must match."
AssertionError: Input and output channels must match.

=== Output for sqrt_exp.py on GPU 0 ===

=== Errors for sqrt_exp.py on GPU 0 ===

=== Output for cos_avg_pool1d.py on GPU 0 ===

=== Errors for cos_avg_pool1d.py on GPU 0 ===
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
  File "/data/results_constrained_partial/call_acc/cos_avg_pool1d.py", line 55, in <module>
    test_results = test_cos_avg_pool1d()
                   ^^^^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/cos_avg_pool1d.py", line 39, in test_cos_avg_pool1d
    results['test_case_1'] = cos_avg_pool1d(input_tensor_1, kernel_size=2)
                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/cos_avg_pool1d.py", line 20, in cos_avg_pool1d
    cos_avg_pool1d_kernel[grid](input, input, output, n_elements, kernel_size, stride, padding, ceil_mode, count_include_pad, BLOCK_SIZE=1024)
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
triton.compiler.errors.CompilationError: at 3:33:
def cos_avg_pool1d_kernel(x_ptr, y_ptr, output_ptr, n_elements, kernel_size: tl.constexpr, stride, padding, ceil_mode, count_include_pad, BLOCK_SIZE):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                                 ^

=== Output for sum_std.py on GPU 0 ===

=== Errors for sum_std.py on GPU 0 ===
Traceback (most recent call last):
  File "/data/results_constrained_partial/call_acc/sum_std.py", line 59, in <module>
    test_results = test_sum_std()
                   ^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/sum_std.py", line 43, in test_sum_std
    results["test_case_1"] = sum_std(input1)
                             ^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/sum_std.py", line 29, in sum_std
    sum_std_kernel[(n_rows,)](y, input, input.stride(0), y.stride(0), n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE)
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
  File "/usr/local/lib/python3.12/site-packages/triton/runtime/jit.py", line 128, in visit_Call
    assert func is None or is_triton_builtin(func) or isinstance(
AssertionError: Function "tensor" is being called from a Triton function but is not a Triton function itself. Decorate it with @triton.jit to fix this

=== Output for mul_relu.py on GPU 0 ===

=== Errors for mul_relu.py on GPU 0 ===
Traceback (most recent call last):
  File "/data/results_constrained_partial/call_acc/mul_relu.py", line 71, in <module>
    test_results = test_mul_relu()
                   ^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/mul_relu.py", line 57, in test_mul_relu
    results["test_case_2"] = mul_relu(input2, other2)
                             ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/mul_relu.py", line 32, in mul_relu
    mul_relu_kernel[grid](input.data, other.data, output.data, n_elements, BLOCK_SIZE=1024)
                                      ^^^^^^^^^^
AttributeError: 'float' object has no attribute 'data'

=== Output for gelu_conv2d.py on GPU 0 ===

=== Errors for gelu_conv2d.py on GPU 0 ===
  File "/data/results_constrained_partial/call_acc/gelu_conv2d.py", line 32
    def gelu_conv2d(input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]=None, stride: Union[int, Tuple[int, int]]=1, padding: Union[int, Tuple[int, int], str]=0, dilation: Union[int, Tuple[int, int]]=1, groups: int=1, approximate: str='none', out: T=):
                                                                                                                                                                                                                                                                             ^
SyntaxError: expected default value expression

=== Output for fused_instance_norm_selu_conv2d.py on GPU 0 ===

=== Errors for fused_instance_norm_selu_conv2d.py on GPU 0 ===
Traceback (most recent call last):
  File "/data/results_constrained_partial/call_acc/fused_instance_norm_selu_conv2d.py", line 91, in <module>
    test_results = test_fused_instance_norm_selu_conv2d()
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/fused_instance_norm_selu_conv2d.py", line 81, in test_fused_instance_norm_selu_conv2d
    results["test_case_1"] = fused_instance_norm_selu_conv2d(input_tensor, weight_tensor)
                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/fused_instance_norm_selu_conv2d.py", line 38, in fused_instance_norm_selu_conv2d
    H, W, C_in = input.shape
    ^^^^^^^^^^
ValueError: too many values to unpack (expected 3)

=== Output for fused_fractional_max_pool2d_with_relu.py on GPU 0 ===

=== Errors for fused_fractional_max_pool2d_with_relu.py on GPU 0 ===
  File "/data/results_constrained_partial/call_acc/fused_fractional_max_pool2d_with_relu.py", line 62
    import torch
IndentationError: expected an indented block after 'else' statement on line 56

=== Output for chebyshev_polynomial_t.py on GPU 0 ===

=== Errors for chebyshev_polynomial_t.py on GPU 0 ===
Traceback (most recent call last):
  File "/data/results_constrained_partial/call_acc/chebyshev_polynomial_t.py", line 69, in <module>
    test_results = test_chebyshev_polynomial_t()
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/chebyshev_polynomial_t.py", line 50, in test_chebyshev_polynomial_t
    results["test_case_1"] = chebyshev_polynomial_t(input_tensor_1, n_1)
                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/chebyshev_polynomial_t.py", line 35, in chebyshev_polynomial_t
    chebyshev_polynomial_t_kernel[grid](input, out, n_elements, BLOCK_SIZE=1024)
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
  File "/usr/local/lib/python3.12/ast.py", line 415, in generic_visit
    self.visit(item)
  File "/usr/local/lib/python3.12/ast.py", line 407, in visit
    return visitor(node)
           ^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/ast.py", line 415, in generic_visit
    self.visit(item)
  File "/usr/local/lib/python3.12/ast.py", line 407, in visit
    return visitor(node)
           ^^^^^^^^^^^^^
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
  File "/usr/local/lib/python3.12/site-packages/triton/runtime/jit.py", line 127, in visit_Call
    func = self.visit(node.func)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/ast.py", line 407, in visit
    return visitor(node)
           ^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/site-packages/triton/runtime/jit.py", line 117, in visit_Attribute
    return getattr(lhs, node.attr)
           ^^^^^^^^^^^^^^^^^^^^^^^
AttributeError: module 'triton.language' has no attribute 'atan'

=== Output for logit.py on GPU 0 ===

=== Errors for logit.py on GPU 0 ===
Traceback (most recent call last):
  File "/data/results_constrained_partial/call_acc/logit.py", line 60, in <module>
    test_results = test_logit()
                   ^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/logit.py", line 41, in test_logit
    results["test_case_1"] = logit(input1)
                             ^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/logit.py", line 27, in logit
    logit_kernel[grid](input, output, n_elements, BLOCK_SIZE=1024)
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
triton.compiler.errors.CompilationError: at 6:7:
def logit_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    if eps is not None:
       ^
NameError('eps is not defined')

=== Output for solve_symmetric_ldl.py on GPU 0 ===

=== Errors for solve_symmetric_ldl.py on GPU 0 ===
Traceback (most recent call last):
  File "/data/results_constrained_partial/call_acc/solve_symmetric_ldl.py", line 88, in <module>
    test_results = test_solve_symmetric_ldl()
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/solve_symmetric_ldl.py", line 56, in test_solve_symmetric_ldl
    results["test_case_1"] = solve_symmetric_ldl(A1, b1)
                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/solve_symmetric_ldl.py", line 33, in solve_symmetric_ldl
    solve_symmetric_ldl_kernel[grid](A, b, C, M, N, K,
  File "/usr/local/lib/python3.12/site-packages/triton/runtime/jit.py", line 345, in <lambda>
    return lambda *args, **kwargs: self.run(grid=grid, warmup=False, *args, **kwargs)
                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/site-packages/triton/runtime/jit.py", line 618, in run
    bound_args, sig_and_spec, constexpr_vals, non_constexpr_vals, excess_kwargs = self.binder(*args, **kwargs)
                                                                                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: dynamic_func() got multiple values for argument 'BLOCK_M'

=== Output for exp_sqrt.py on GPU 0 ===

=== Errors for exp_sqrt.py on GPU 0 ===

=== Output for combined_activation.py on GPU 0 ===

=== Errors for combined_activation.py on GPU 0 ===
Traceback (most recent call last):
  File "/data/results_constrained_partial/call_acc/combined_activation.py", line 125, in <module>
    test_results = test_combined_activation()
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/combined_activation.py", line 100, in test_combined_activation
    results["test_case_1"] = combined_activation(input1, weight1_1, weight2_1, bias1)
                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/combined_activation.py", line 45, in combined_activation
    M, N, D_in = input.shape
    ^^^^^^^^^^
ValueError: not enough values to unpack (expected 3, got 2)

=== Output for scaled_add_dot.py on GPU 0 ===

=== Errors for scaled_add_dot.py on GPU 0 ===
Traceback (most recent call last):
  File "/data/results_constrained_partial/call_acc/scaled_add_dot.py", line 27, in <module>
    def scaled_add_dot(y: Tensor, x: Tensor, alpha: float):
                          ^^^^^^
NameError: name 'Tensor' is not defined

=== Output for tensordot.py on GPU 0 ===

=== Errors for tensordot.py on GPU 0 ===
Traceback (most recent call last):
  File "/data/results_constrained_partial/call_acc/tensordot.py", line 14, in <module>
    def tensordot(a: torch.Tensor, b: torch.Tensor, dims: Union[int, Tuple[List[int], List[int]], List[List[int]]]):
                                                          ^^^^^
NameError: name 'Union' is not defined

=== Output for qr.py on GPU 0 ===

=== Errors for qr.py on GPU 0 ===
Traceback (most recent call last):
  File "/usr/local/lib/python3.12/site-packages/triton/language/core.py", line 35, in wrapper
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/site-packages/triton/language/core.py", line 1220, in full
    shape = _shape_check_impl(shape)
            ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/site-packages/triton/language/core.py", line 1201, in _shape_check_impl
    raise TypeError(f"Shape element {i} must have type `constexpr`")
TypeError: Shape element 0 must have type `constexpr`

The above exception was the direct cause of the following exception:

triton.compiler.errors.CompilationError: at 10:11:
def zeros(shape, dtype):
    """
    Returns a tensor filled with the scalar value 0 for the given :code:`shape` and :code:`dtype`.

    :param shape: Shape of the new array, e.g., (8, 16) or (8, )
    :type shape: tuple of ints
    :param dtype: Data-type of the new array, e.g., :code:`tl.float16`
    :type dtype: DType
    """
    return core.full(shape, 0, dtype)
           ^

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/data/results_constrained_partial/call_acc/qr.py", line 97, in <module>
    test_results = test_qr()
                   ^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/qr.py", line 77, in test_qr
    Q1, R1 = qr(A1, mode='reduced')
             ^^^^^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/qr.py", line 63, in qr
    qr_kernel[grid](A.data, Q.data, R.data, m, n, BLOCK_SIZE=1024)
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
triton.compiler.errors.CompilationError: at 6:8:
def qr_kernel(A_ptr, Q_ptr, R_ptr, m, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < min(m, n)
    a = tl.load(A_ptr + offsets, mask=mask)
    q = tl.zeros((m, BLOCK_SIZE), dtype=tl.float32)
        ^

=== Output for asin.py on GPU 0 ===

=== Errors for asin.py on GPU 0 ===
Traceback (most recent call last):
  File "/data/results_constrained_partial/call_acc/asin.py", line 57, in <module>
    test_results = test_asin()
                   ^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/asin.py", line 41, in test_asin
    results["test_case_1"] = asin(input_tensor_1)
                             ^^^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/asin.py", line 27, in asin
    asin_kernel[grid](input_tensor.data, output.data, n_elements, BLOCK_SIZE=1024)
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
  File "/usr/local/lib/python3.12/site-packages/triton/runtime/jit.py", line 127, in visit_Call
    func = self.visit(node.func)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/ast.py", line 407, in visit
    return visitor(node)
           ^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/site-packages/triton/runtime/jit.py", line 117, in visit_Attribute
    return getattr(lhs, node.attr)
           ^^^^^^^^^^^^^^^^^^^^^^^
AttributeError: module 'triton.language.math' has no attribute 'asin'. Did you mean: 'sin'?

=== Output for fused_masked_select_add_gelu.py on GPU 0 ===

=== Errors for fused_masked_select_add_gelu.py on GPU 0 ===
Traceback (most recent call last):
  File "/data/results_constrained_partial/call_acc/fused_masked_select_add_gelu.py", line 95, in <module>
    test_results = test_fused_masked_select_add_gelu()
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/fused_masked_select_add_gelu.py", line 72, in test_fused_masked_select_add_gelu
    results["test_case_1"] = fused_masked_select_add_gelu(input1, mask1, other1)
                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
NameError: name 'fused_masked_select_add_gelu' is not defined. Did you mean: 'test_fused_masked_select_add_gelu'?

=== Output for fused_pairwise_distance_adaptive_avg_pool2d.py on GPU 0 ===

=== Errors for fused_pairwise_distance_adaptive_avg_pool2d.py on GPU 0 ===
  File "/data/results_constrained_partial/call_acc/fused_pairwise_distance_adaptive_avg_pool2d.py", line 71
    adaptive_avg_pool2d_kernel[(batch_size, output_height, output_width)](output_ptr, input_ptr, input_row_stride, input_col_stride, output_row_stride, output_col_stride, output_height, output_width,BLOCK
                                                                         ^
SyntaxError: '(' was never closed

=== Output for add_mean.py on GPU 0 ===

=== Errors for add_mean.py on GPU 0 ===
Traceback (most recent call last):
  File "/data/results_constrained_partial/call_acc/add_mean.py", line 74, in <module>
    test_results = test_add_mean()
                   ^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/add_mean.py", line 55, in test_add_mean
    results["test_case_1"] = add_mean(input1, other1)
                             ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/add_mean.py", line 37, in add_mean
    result = add_mean_kernel[triton.CONFIGS['add_mean_config']](out, input, input.stride(0), out.stride(0), input.numel(), input.numel(), None, alpha, keepdim, input.dtype)
                             ^^^^^^^^^^^^^^
AttributeError: module 'triton' has no attribute 'CONFIGS'

=== Output for fused_layer_norm_relu_linear.py on GPU 0 ===

=== Errors for fused_layer_norm_relu_linear.py on GPU 0 ===
  File "/data/results_constrained_partial/call_acc/fused_layer_norm_relu_linear.py", line 25
    mask = (offs
           ^
SyntaxError: '(' was never closed

=== Output for fused_add_mul_groupnorm.py on GPU 0 ===

=== Errors for fused_add_mul_groupnorm.py on GPU 0 ===
Traceback (most recent call last):
  File "/data/results_constrained_partial/call_acc/fused_add_mul_groupnorm.py", line 88, in <module>
    test_results = test_fused_add_mul_groupnorm()
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/fused_add_mul_groupnorm.py", line 62, in test_fused_add_mul_groupnorm
    results["test_case_1"] = fused_add_mul_groupnorm(input1, input2, weight, bias, num_groups)
                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/fused_add_mul_groupnorm.py", line 37, in fused_add_mul_groupnorm
    n_rows, n_cols = input1.shape
    ^^^^^^^^^^^^^^
ValueError: too many values to unpack (expected 2)

=== Output for SGD.py on GPU 0 ===

=== Errors for SGD.py on GPU 0 ===
Traceback (most recent call last):
  File "/data/results_constrained_partial/call_acc/SGD.py", line 130, in <module>
    test_results = test_SGD()
                   ^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/SGD.py", line 101, in test_SGD
    loss = SGD(model, input, target, loss_fn)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: SGD() missing 8 required positional arguments: 'momentum', 'weight_decay', 'dampening', 'nesterov', 'maximize', 'foreach', 'differentiable', and 'fused'

=== Output for relu_batch_norm_conv2d.py on GPU 0 ===

=== Errors for relu_batch_norm_conv2d.py on GPU 0 ===
  File "/data/results_constrained_partial/call_acc/relu_batch_norm_conv2d.py", line 37
    def relu_batch_norm_conv2d(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, running_mean: torch.Tensor = None, running_var: torch.Tensor = None, bn_weight: torch.Tensor = None, bn_bias:):
                                                                                                                                                                                                                                                                                         ^
SyntaxError: invalid syntax

=== Output for conv2d.py on GPU 0 ===

=== Errors for conv2d.py on GPU 0 ===
  File "/data/results_constrained_partial/call_acc/conv2d.py", line 38
    conv2d_kernel[grid](input.data, weight.data, output.data, batch_size, channels_in, height_in, width_in, channels_out, kernel_height, kernel_width, stride, stride, padding_h_start_pad_end_h_start_pad_0
                       ^
SyntaxError: '(' was never closed

=== Output for normalized_cosine_similarity.py on GPU 0 ===

=== Errors for normalized_cosine_similarity.py on GPU 0 ===
Traceback (most recent call last):
  File "/data/results_constrained_partial/call_acc/normalized_cosine_similarity.py", line 22, in <module>
    def normalized_cosine_similarity(x1: Tensor, x2: Tensor, dim: int=1, eps_similarity: float=1e-08, p_norm: float=2, eps_norm: float=1e-12):
                                         ^^^^^^
NameError: name 'Tensor' is not defined

=== Output for fused_cholesky_solve.py on GPU 0 ===

=== Errors for fused_cholesky_solve.py on GPU 0 ===
Traceback (most recent call last):
  File "/data/results_constrained_partial/call_acc/fused_cholesky_solve.py", line 91, in <module>
    test_results = test_fused_cholesky_solve()
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/fused_cholesky_solve.py", line 72, in test_fused_cholesky_solve
    results["test_case_1"] = fused_cholesky_solve(A1, b1)
                             ^^^^^^^^^^^^^^^^^^^^
NameError: name 'fused_cholesky_solve' is not defined. Did you mean: 'test_fused_cholesky_solve'?

=== Output for matmul.py on GPU 0 ===

=== Errors for matmul.py on GPU 0 ===
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
  File "/data/results_constrained_partial/call_acc/matmul.py", line 69, in <module>
    test_results = test_matmul()
                   ^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/matmul.py", line 50, in test_matmul
    results["test_case_1"] = matmul(tensor1, tensor2)
                             ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/matmul.py", line 32, in matmul
    matmul_kernel[grid](a, b, c, M, N, K,
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
triton.compiler.errors.CompilationError: at 7:32:
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, BLOCK_M: tl.constexpr, BLOCK_N, BLOCK_K):
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
                                ^

=== Output for fused_gather_masked_fill.py on GPU 0 ===

=== Errors for fused_gather_masked_fill.py on GPU 0 ===
Traceback (most recent call last):
  File "/data/results_constrained_partial/call_acc/fused_gather_masked_fill.py", line 69, in <module>
    test_results = test_fused_gather_masked_fill()
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/fused_gather_masked_fill.py", line 44, in test_fused_gather_masked_fill
    results["test_case_1"] = fused_gather_masked_fill(input1, 1, index1, mask1, value1)
                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/fused_gather_masked_fill.py", line 18, in fused_gather_masked_fill
    N, M, K, DIM = input.shape
    ^^^^^^^^^^^^
ValueError: not enough values to unpack (expected 4, got 2)

=== Output for fused_cross_entropy_log_softmax.py on GPU 0 ===

=== Errors for fused_cross_entropy_log_softmax.py on GPU 0 ===
Traceback (most recent call last):
  File "/data/results_constrained_partial/call_acc/fused_cross_entropy_log_softmax.py", line 94, in <module>
    test_results = test_fused_cross_entropy_log_softmax()
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/fused_cross_entropy_log_softmax.py", line 74, in test_fused_cross_entropy_log_softmax
    results["test_case_1"] = fused_cross_entropy_log_softmax(input, target)
                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
NameError: name 'fused_cross_entropy_log_softmax' is not defined. Did you mean: 'test_fused_cross_entropy_log_softmax'?

=== Output for addmm.py on GPU 0 ===

=== Errors for addmm.py on GPU 0 ===
Traceback (most recent call last):
  File "/data/results_constrained_partial/call_acc/addmm.py", line 64, in <module>
    test_results = test_addmm()
                   ^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/addmm.py", line 48, in test_addmm
    results["test_case_1"] = addmm(input1, mat1_1, mat2_1)
                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/addmm.py", line 32, in addmm
    addmm_kernel[grid](mat1, mat2, C, M, N, K, alpha, beta, mat1.stride(0), mat1.stride(1), mat2.stride(0), mat2.stride(1), C.stride(0), C.stride(1), BLOCK_M=64, BLOCK_N=64, BLOCK_K=32)
  File "/usr/local/lib/python3.12/site-packages/triton/runtime/jit.py", line 345, in <lambda>
    return lambda *args, **kwargs: self.run(grid=grid, warmup=False, *args, **kwargs)
                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/site-packages/triton/runtime/jit.py", line 636, in run
    raise KeyError("Keyword argument %s was specified but unrecognised" % k)
KeyError: 'Keyword argument BLOCK_M was specified but unrecognised'

=== Output for fused_qr_solve.py on GPU 0 ===

=== Errors for fused_qr_solve.py on GPU 0 ===
Traceback (most recent call last):
  File "/data/results_constrained_partial/call_acc/fused_qr_solve.py", line 106, in <module>
    test_results = test_fused_qr_solve()
                   ^^^^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/fused_qr_solve.py", line 87, in test_fused_qr_solve
    results["test_case_1"] = fused_qr_solve(A1, b1)
                             ^^^^^^^^^^^^^^
NameError: name 'fused_qr_solve' is not defined. Did you mean: 'test_fused_qr_solve'?

=== Output for sigmoid_adaptive_avg_pool2d.py on GPU 0 ===

=== Errors for sigmoid_adaptive_avg_pool2d.py on GPU 0 ===
Traceback (most recent call last):
  File "/data/results_constrained_partial/call_acc/sigmoid_adaptive_avg_pool2d.py", line 22, in <module>
    def sigmoid_adaptive_avg_pool2d(input: Tensor, output_size: Union[int, Tuple[int, int]]):
                                           ^^^^^^
NameError: name 'Tensor' is not defined

=== Output for cos.py on GPU 0 ===

=== Errors for cos.py on GPU 0 ===

=== Output for fused_bmm_dropout_gelu.py on GPU 0 ===

=== Errors for fused_bmm_dropout_gelu.py on GPU 0 ===
Traceback (most recent call last):
  File "/data/results_constrained_partial/call_acc/fused_bmm_dropout_gelu.py", line 90, in <module>
    test_results = test_fused_bmm_dropout_gelu()
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/fused_bmm_dropout_gelu.py", line 71, in test_fused_bmm_dropout_gelu
    results["test_case_1"] = fused_bmm_dropout_gelu(input1, input2)
                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/fused_bmm_dropout_gelu.py", line 32, in fused_bmm_dropout_gelu
    assert M * N == P, "Input dimensions do not match."
AssertionError: Input dimensions do not match.

=== Output for trunc.py on GPU 0 ===

=== Errors for trunc.py on GPU 0 ===

=== Output for matrix_power_eig.py on GPU 0 ===

=== Errors for matrix_power_eig.py on GPU 0 ===
Traceback (most recent call last):
  File "/data/results_constrained_partial/call_acc/matrix_power_eig.py", line 76, in <module>
    test_results = test_matrix_power_eig()
                   ^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/matrix_power_eig.py", line 62, in test_matrix_power_eig
    results["test_case_1"] = matrix_power_eig(A1, k1)
                             ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/matrix_power_eig.py", line 45, in matrix_power_eig
    matrix_power_eig_kernel[grid](A, k, out, n_elements, BLOCK_SIZE=1024)
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
  File "/usr/local/lib/python3.12/site-packages/triton/runtime/jit.py", line 127, in visit_Call
    func = self.visit(node.func)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/ast.py", line 407, in visit
    return visitor(node)
           ^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/site-packages/triton/runtime/jit.py", line 112, in visit_Attribute
    lhs = self.visit(node.value)
          ^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/ast.py", line 407, in visit
    return visitor(node)
           ^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/site-packages/triton/runtime/jit.py", line 117, in visit_Attribute
    return getattr(lhs, node.attr)
           ^^^^^^^^^^^^^^^^^^^^^^^
AttributeError: module 'triton.language' has no attribute 'linalg'

=== Output for log_tanh.py on GPU 0 ===

=== Errors for log_tanh.py on GPU 0 ===
Traceback (most recent call last):
  File "/data/results_constrained_partial/call_acc/log_tanh.py", line 70, in <module>
    test_results = test_log_tanh()
                   ^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/log_tanh.py", line 49, in test_log_tanh
    results["test_case_1"] = log_tanh(input1)
                             ^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/log_tanh.py", line 22, in log_tanh
    log_tanh_kernel[grid](input, result, n_elements, BLOCK_SIZE=1024)
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
  File "/usr/local/lib/python3.12/site-packages/triton/runtime/jit.py", line 127, in visit_Call
    func = self.visit(node.func)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/ast.py", line 407, in visit
    return visitor(node)
           ^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/site-packages/triton/runtime/jit.py", line 117, in visit_Attribute
    return getattr(lhs, node.attr)
           ^^^^^^^^^^^^^^^^^^^^^^^
AttributeError: module 'triton.language.math' has no attribute 'tanh'

=== Output for exp.py on GPU 0 ===

=== Errors for exp.py on GPU 0 ===

=== Output for matrix_multiply_symmetric.py on GPU 0 ===

=== Errors for matrix_multiply_symmetric.py on GPU 0 ===
Traceback (most recent call last):
  File "/data/results_constrained_partial/call_acc/matrix_multiply_symmetric.py", line 79, in <module>
    test_results = test_matrix_multiply_symmetric()
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/matrix_multiply_symmetric.py", line 54, in test_matrix_multiply_symmetric
    results["test_case_1"] = matrix_multiply_symmetric(A, B, C, alpha, beta)
                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/matrix_multiply_symmetric.py", line 37, in matrix_multiply_symmetric
    matrix_multiply_symmetric_kernel[grid](A, B, c, M, N, P, alpha, beta, A.stride(0), A.stride(1), B.stride(0), B.stride(1), C.stride(0), C.stride(1), BLOCK_M=64, BLOCK_N=64, BLOCK_P=64)
  File "/usr/local/lib/python3.12/site-packages/triton/runtime/jit.py", line 345, in <lambda>
    return lambda *args, **kwargs: self.run(grid=grid, warmup=False, *args, **kwargs)
                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/site-packages/triton/runtime/jit.py", line 618, in run
    bound_args, sig_and_spec, constexpr_vals, non_constexpr_vals, excess_kwargs = self.binder(*args, **kwargs)
                                                                                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: dynamic_func() got multiple values for argument 'BLOCK_M'

=== Output for fused_avg_pool2d_cosine_similarity.py on GPU 0 ===

=== Errors for fused_avg_pool2d_cosine_similarity.py on GPU 0 ===
Traceback (most recent call last):
  File "/data/results_constrained_partial/call_acc/fused_avg_pool2d_cosine_similarity.py", line 100, in <module>
    test_results = test_fused_avg_pool2d_cosine_similarity()
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/fused_avg_pool2d_cosine_similarity.py", line 81, in test_fused_avg_pool2d_cosine_similarity
    results["test_case_1"] = fused_avg_pool2d_cosine_similarity(x1, x2, kernel_size=2)
                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
NameError: name 'fused_avg_pool2d_cosine_similarity' is not defined. Did you mean: 'test_fused_avg_pool2d_cosine_similarity'?

=== Output for fused_hardshrink_dropout.py on GPU 0 ===

=== Errors for fused_hardshrink_dropout.py on GPU 0 ===
Traceback (most recent call last):
  File "/data/results_constrained_partial/call_acc/fused_hardshrink_dropout.py", line 103, in <module>
    test_results = test_fused_hardshrink_dropout()
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/fused_hardshrink_dropout.py", line 87, in test_fused_hardshrink_dropout
    results["test_case_1"] = fused_hardshrink_dropout(input_tensor)
                             ^^^^^^^^^^^^^^^^^^^^^^^^
NameError: name 'fused_hardshrink_dropout' is not defined. Did you mean: 'test_fused_hardshrink_dropout'?

=== Output for erfc_sqrt.py on GPU 0 ===

=== Errors for erfc_sqrt.py on GPU 0 ===
Traceback (most recent call last):
  File "/data/results_constrained_partial/call_acc/erfc_sqrt.py", line 71, in <module>
    test_results = test_erfc_sqrt()
                   ^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/erfc_sqrt.py", line 55, in test_erfc_sqrt
    results["test_case_1"] = erfc_sqrt(input1)
                             ^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/erfc_sqrt.py", line 22, in erfc_sqrt
    erfc_sqrt_kernel[grid](input.data, erfc_result.data, sqrt_result.data, n_elements, BLOCK_SIZE=1024)
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
  File "/usr/local/lib/python3.12/site-packages/triton/runtime/jit.py", line 127, in visit_Call
    func = self.visit(node.func)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/ast.py", line 407, in visit
    return visitor(node)
           ^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/site-packages/triton/runtime/jit.py", line 117, in visit_Attribute
    return getattr(lhs, node.attr)
           ^^^^^^^^^^^^^^^^^^^^^^^
AttributeError: module 'triton.language.math' has no attribute 'erfc'. Did you mean: 'erf'?

=== Output for tensordot_rsqrt.py on GPU 0 ===

=== Errors for tensordot_rsqrt.py on GPU 0 ===
Traceback (most recent call last):
  File "/data/results_constrained_partial/call_acc/tensordot_rsqrt.py", line 58, in <module>
    test_results = test_tensordot_rsqrt()
                   ^^^^^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/tensordot_rsqrt.py", line 36, in test_tensordot_rsqrt
    results["test_case_1"] = tensordot_rsqrt(a, b, dims)
                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/tensordot_rsqrt.py", line 20, in tensordot_rsqrt
    tensordot_rsqrt_kernel[grid](a, b, output, n_elements, BLOCK_SIZE=1024)
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
  File "/usr/local/lib/python3.12/site-packages/triton/runtime/jit.py", line 127, in visit_Call
    func = self.visit(node.func)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/ast.py", line 407, in visit
    return visitor(node)
           ^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/site-packages/triton/runtime/jit.py", line 117, in visit_Attribute
    return getattr(lhs, node.attr)
           ^^^^^^^^^^^^^^^^^^^^^^^
AttributeError: module 'triton.language' has no attribute 'tensordot'. Did you mean: 'tensor'?

=== Output for softmax_log.py on GPU 0 ===

=== Errors for softmax_log.py on GPU 0 ===
Traceback (most recent call last):
  File "/data/results_constrained_partial/call_acc/softmax_log.py", line 62, in <module>
    test_results = test_softmax_log()
                   ^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/softmax_log.py", line 46, in test_softmax_log
    results["test_case_1"] = softmax_log(input_tensor)
                             ^^^^^^^^^^^
NameError: name 'softmax_log' is not defined. Did you mean: 'test_softmax_log'?

=== Output for dropout_sigmoid_linear.py on GPU 0 ===

=== Errors for dropout_sigmoid_linear.py on GPU 0 ===
  File "/data/results_constrained_partial/call_acc/dropout_sigmoid_linear.py", line 17
    w_ptrs = weight_ptr + offs_wm1000000000000000000000000000000000000000000000000000: tl.constexpr
                                                                                     ^
SyntaxError: invalid syntax

=== Output for batch_norm.py on GPU 0 ===

=== Errors for batch_norm.py on GPU 0 ===
  File "/data/results_constrained_partial/call_acc/batch_norm.py", line 20
    b_ptrs = running_mean_ptr + (offs_h[:, None] * D * E * F * G + offs_w[None, :] * E * F * G + offs_c[:, None] * F * G + offs_d[:, None] * G + offs_e[:, None] * G + offs_f[:, None] * G + offs_g[:, None]
                                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
SyntaxError: invalid syntax. Perhaps you forgot a comma?

=== Output for gammaln.py on GPU 0 ===

=== Errors for gammaln.py on GPU 0 ===
Traceback (most recent call last):
  File "/data/results_constrained_partial/call_acc/gammaln.py", line 61, in <module>
    test_results = test_gammaln()
                   ^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/gammaln.py", line 45, in test_gammaln
    results["test_case_1"] = gammaln(input1)
                             ^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/gammaln.py", line 18, in gammaln
    gammaln_kernel[grid](input, output, n_elements, BLOCK_SIZE=1024)
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
  File "/usr/local/lib/python3.12/site-packages/triton/runtime/jit.py", line 127, in visit_Call
    func = self.visit(node.func)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/ast.py", line 407, in visit
    return visitor(node)
           ^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/site-packages/triton/runtime/jit.py", line 117, in visit_Attribute
    return getattr(lhs, node.attr)
           ^^^^^^^^^^^^^^^^^^^^^^^
AttributeError: module 'triton.language.math' has no attribute 'gamma'

=== Output for bitwise_and.py on GPU 0 ===

=== Errors for bitwise_and.py on GPU 0 ===

=== Output for sub_gelu.py on GPU 0 ===

=== Errors for sub_gelu.py on GPU 0 ===
Traceback (most recent call last):
  File "/data/results_constrained_partial/call_acc/sub_gelu.py", line 98, in <module>
    test_results = test_sub_gelu()
                   ^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/sub_gelu.py", line 83, in test_sub_gelu
    results["test_case_1"] = sub_gelu(input_tensor, other_tensor)
                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/sub_gelu.py", line 47, in sub_gelu
    M, K = input.shape
    ^^^^
ValueError: not enough values to unpack (expected 2, got 1)

=== Output for gelu_std.py on GPU 0 ===

=== Errors for gelu_std.py on GPU 0 ===
Traceback (most recent call last):
  File "/data/results_constrained_partial/call_acc/gelu_std.py", line 64, in <module>
    test_results = test_gelu_std()
                   ^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/gelu_std.py", line 48, in test_gelu_std
    results["test_case_1"] = gelu_std(input1)
                             ^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/gelu_std.py", line 24, in gelu_std
    M, N = input.shape
    ^^^^
ValueError: not enough values to unpack (expected 2, got 1)

=== Output for permute_copy.py on GPU 0 ===

=== Errors for permute_copy.py on GPU 0 ===
Traceback (most recent call last):
  File "/usr/local/lib/python3.12/site-packages/triton/language/core.py", line 69, in wrapper
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/site-packages/triton/language/core.py", line 35, in wrapper
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/site-packages/triton/language/core.py", line 1313, in permute
    return semantic.permute(input, dims, _builder)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/site-packages/triton/language/semantic.py", line 616, in permute
    if len(input.shape) != len(dims):
                           ^^^^^^^^^
TypeError: object of type 'tensor' has no len()

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/data/results_constrained_partial/call_acc/permute_copy.py", line 48, in <module>
    test_results = test_permute_copy()
                   ^^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/permute_copy.py", line 32, in test_permute_copy
    results["test_case_1"] = permute_copy(tensor_2d, [1, 0])
                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/permute_copy.py", line 18, in permute_copy
    permute_copy_kernel[grid](input.data, output.data, output.data, n_elements, BLOCK_SIZE=1024)
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
triton.compiler.errors.CompilationError: at 7:35:
def permute_copy_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x.permute(y), mask=mask)
                                   ^

=== Output for digamma.py on GPU 0 ===

=== Errors for digamma.py on GPU 0 ===

=== Output for softmax_mul.py on GPU 0 ===

=== Errors for softmax_mul.py on GPU 0 ===
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
  File "/data/results_constrained_partial/call_acc/softmax_mul.py", line 77, in <module>
    test_results = test_softmax_mul()
                   ^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/softmax_mul.py", line 57, in test_softmax_mul
    results["test_case_1"] = softmax_mul(input1, other1, dim=1)
                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/softmax_mul.py", line 27, in softmax_mul
    softmax_mul_kernel[(n_rows,)](y, input, input.stride(0), y.stride(0), n_rows, n_cols, dim=dim)
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
triton.compiler.errors.CompilationError: at 4:18:
def softmax_mul_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_rows, n_cols, dim: tl.constexpr):
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    col_offsets = tl.arange(0, n_cols)
                  ^

=== Output for bitwise_and_binomial.py on GPU 0 ===

=== Errors for bitwise_and_binomial.py on GPU 0 ===
Traceback (most recent call last):
  File "/data/results_constrained_partial/call_acc/bitwise_and_binomial.py", line 52, in <module>
    test_results = test_bitwise_and_binomial()
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/bitwise_and_binomial.py", line 37, in test_bitwise_and_binomial
    results["test_case_1"] = bitwise_and_binomial(input_tensor, other_tensor, total_count, probs=probs)
                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/bitwise_and_binomial.py", line 19, in bitwise_and_binomial
    bitwise_and_binomial_kernel[grid](input.data, other.data, output.data, total_count.data, probs.data, logits.data, n_elements, BLOCK_SIZE=1024)
                                                                                                         ^^^^^^^^^^^
AttributeError: 'NoneType' object has no attribute 'data'

=== Output for rad2deg_sqrt.py on GPU 0 ===

=== Errors for rad2deg_sqrt.py on GPU 0 ===
Traceback (most recent call last):
  File "/data/results_constrained_partial/call_acc/rad2deg_sqrt.py", line 60, in <module>
    test_results = test_rad2deg_sqrt()
                   ^^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/rad2deg_sqrt.py", line 40, in test_rad2deg_sqrt
    deg_result, sqrt_result = rad2deg_sqrt(a)
                              ^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/rad2deg_sqrt.py", line 19, in rad2deg_sqrt
    rad2deg_sqrt_kernel[grid](input, output[:, 0], n_elements, BLOCK_SIZE=1024)
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
  File "/usr/local/lib/python3.12/site-packages/triton/runtime/jit.py", line 127, in visit_Call
    func = self.visit(node.func)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/ast.py", line 407, in visit
    return visitor(node)
           ^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/site-packages/triton/runtime/jit.py", line 117, in visit_Attribute
    return getattr(lhs, node.attr)
           ^^^^^^^^^^^^^^^^^^^^^^^
AttributeError: module 'triton.language' has no attribute 'degrees'

=== Output for bessel_j1.py on GPU 0 ===

=== Errors for bessel_j1.py on GPU 0 ===
Traceback (most recent call last):
  File "/data/results_constrained_partial/call_acc/bessel_j1.py", line 49, in <module>
    test_results = test_bessel_j1()
                   ^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/bessel_j1.py", line 33, in test_bessel_j1
    results["test_case_1"] = bessel_j1(input1)
                             ^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/bessel_j1.py", line 19, in bessel_j1
    bessel_j1_kernel[grid](input, output, n_elements, BLOCK_SIZE=1024)
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
  File "/usr/local/lib/python3.12/site-packages/triton/runtime/jit.py", line 127, in visit_Call
    func = self.visit(node.func)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/ast.py", line 407, in visit
    return visitor(node)
           ^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/site-packages/triton/runtime/jit.py", line 117, in visit_Attribute
    return getattr(lhs, node.attr)
           ^^^^^^^^^^^^^^^^^^^^^^^
AttributeError: module 'triton.language.math' has no attribute 'bessel_j1'

=== Output for lu.py on GPU 0 ===

=== Errors for lu.py on GPU 0 ===
Traceback (most recent call last):
  File "/usr/local/lib/python3.12/site-packages/triton/language/core.py", line 35, in wrapper
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/site-packages/triton/language/core.py", line 1220, in full
    shape = _shape_check_impl(shape)
            ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/site-packages/triton/language/core.py", line 1201, in _shape_check_impl
    raise TypeError(f"Shape element {i} must have type `constexpr`")
TypeError: Shape element 0 must have type `constexpr`

The above exception was the direct cause of the following exception:

triton.compiler.errors.CompilationError: at 10:11:
def zeros(shape, dtype):
    """
    Returns a tensor filled with the scalar value 0 for the given :code:`shape` and :code:`dtype`.

    :param shape: Shape of the new array, e.g., (8, 16) or (8, )
    :type shape: tuple of ints
    :param dtype: Data-type of the new array, e.g., :code:`tl.float16`
    :type dtype: DType
    """
    return core.full(shape, 0, dtype)
           ^

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/data/results_constrained_partial/call_acc/lu.py", line 95, in <module>
    test_results = test_lu()
                   ^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/lu.py", line 75, in test_lu
    P1, L1, U1 = lu(A1)
                 ^^^^^^
  File "/data/results_constrained_partial/call_acc/lu.py", line 35, in lu
    lu_kernel[grid](A.data, P.data, L.data, U.data, A.shape[0], A.shape[1], BLOCK_SIZE=1024)
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
triton.compiler.errors.CompilationError: at 6:8:
def lu_kernel(A_ptr, P_ptr, L_ptr, U_ptr, m, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < min(m, n)
    A = tl.load(A_ptr + offsets, mask=mask)
    P = tl.zeros((m, m), dtype=tl.float32)
        ^

=== Output for gelu_min.py on GPU 0 ===

=== Errors for gelu_min.py on GPU 0 ===
  File "/data/results_constrained_partial/call_acc/gelu_min.py", line 50
    def gelu_min(input, approximate='none', dim=None, keepdim=False, out=None):
IndentationError: unexpected indent

=== Output for grid_sample_with_affine.py on GPU 0 ===

=== Errors for grid_sample_with_affine.py on GPU 0 ===
  File "/data/results_constrained_partial/call_acc/grid_sample_with_affine.py", line 43
    grid_sample_with_affine_kernel[grid](input, theta, c, M, C, H_in, W_in, H_out, W_out, mode, padding_mode, align_corners, stride_im_in_h, stride_im_in_w, stride_im_out_h, stride_im_out_w, stride_thetaH
                                                                                                                                                                                               ^^^^^^^^^^^^^
SyntaxError: invalid syntax. Perhaps you forgot a comma?

=== Output for pseudoinverse_svd.py on GPU 0 ===

=== Errors for pseudoinverse_svd.py on GPU 0 ===
Traceback (most recent call last):
  File "/data/results_constrained_partial/call_acc/pseudoinverse_svd.py", line 61, in <module>
    test_results = test_pseudoinverse_svd()
                   ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/pseudoinverse_svd.py", line 53, in test_pseudoinverse_svd
    results["test_case_1"] = pseudoinverse_svd(A1)
                             ^^^^^^^^^^^^^^^^^
NameError: name 'pseudoinverse_svd' is not defined. Did you mean: 'test_pseudoinverse_svd'?

=== Output for exp_mean.py on GPU 0 ===

=== Errors for exp_mean.py on GPU 0 ===
Traceback (most recent call last):
  File "/data/results_constrained_partial/call_acc/exp_mean.py", line 54, in <module>
    test_results = test_exp_mean()
                   ^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/exp_mean.py", line 50, in test_exp_mean
    results["test_case_4"] = exp_mean(input_tensor_3d)
                             ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/exp_mean.py", line 22, in exp_mean
    n_rows, n_cols = input.shape
    ^^^^^^^^^^^^^^
ValueError: too many values to unpack (expected 2)

=== Output for low_rank_svd_approximation.py on GPU 0 ===

=== Errors for low_rank_svd_approximation.py on GPU 0 ===
Traceback (most recent call last):
  File "/data/results_constrained_partial/call_acc/low_rank_svd_approximation.py", line 69, in <module>
    test_results = test_low_rank_svd_approximation()
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/low_rank_svd_approximation.py", line 50, in test_low_rank_svd_approximation
    results["test_case_1"] = low_rank_svd_approximation(A, k)
                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/low_rank_svd_approximation.py", line 30, in low_rank_svd_approximation
    Vh_k = Vh[:, :k, :]
           ~~^^^^^^^^^^
IndexError: too many indices for tensor of dimension 2

=== Output for min.py on GPU 0 ===

=== Errors for min.py on GPU 0 ===
Traceback (most recent call last):
  File "/data/results_constrained_partial/call_acc/min.py", line 58, in <module>
    test_results = test_min()
                   ^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/min.py", line 50, in test_min
    results["test_case_3"] = min(input_tensor, dim=2, keepdim=True)
                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/min.py", line 18, in min
    n_rows, n_cols = input_tensor.shape
    ^^^^^^^^^^^^^^
ValueError: too many values to unpack (expected 2)

=== Output for symmetric_mm_and_abs_sum.py on GPU 0 ===

=== Errors for symmetric_mm_and_abs_sum.py on GPU 0 ===
Traceback (most recent call last):
  File "/data/results_constrained_partial/call_acc/symmetric_mm_and_abs_sum.py", line 75, in <module>
    test_results = test_symmetric_mm_and_abs_sum()
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/symmetric_mm_and_abs_sum.py", line 50, in test_symmetric_mm_and_abs_sum
    results["test_case_1"] = symmetric_mm_and_abs_sum(A1, C1, alpha1, beta1).item()
                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/symmetric_mm_and_abs_sum.py", line 32, in symmetric_mm_and_abs_sum
    symmetric_mm_and_abs_sum_kernel[grid](A, A.T, c, M, N, K, alpha, beta, A.stride(0), A.stride(1), A.stride(0), A.stride(1), C.stride(0), C.stride(1), BLOCK_M=64, BLOCK_N=64, BLOCK_K=32)
  File "/usr/local/lib/python3.12/site-packages/triton/runtime/jit.py", line 345, in <lambda>
    return lambda *args, **kwargs: self.run(grid=grid, warmup=False, *args, **kwargs)
                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/site-packages/triton/runtime/jit.py", line 618, in run
    bound_args, sig_and_spec, constexpr_vals, non_constexpr_vals, excess_kwargs = self.binder(*args, **kwargs)
                                                                                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: dynamic_func() got multiple values for argument 'BLOCK_M'

=== Output for determinant_lu.py on GPU 0 ===

=== Errors for determinant_lu.py on GPU 0 ===

=== Output for tanh_linear.py on GPU 0 ===

=== Errors for tanh_linear.py on GPU 0 ===
Traceback (most recent call last):
  File "/data/results_constrained_partial/call_acc/tanh_linear.py", line 45, in <module>
    from tanh_linear import tanh_linear
  File "/__modal/volumes/vo-WvFtwf25UW7xHyNHsL4jq5/results_constrained_partial/call_acc/tanh_linear.py", line 79, in <module>
    test_results = test_tanh_linear()
                   ^^^^^^^^^^^^^^^^^^
  File "/__modal/volumes/vo-WvFtwf25UW7xHyNHsL4jq5/results_constrained_partial/call_acc/tanh_linear.py", line 54, in test_tanh_linear
    result1 = tanh_linear(input1, weight1, bias1)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/__modal/volumes/vo-WvFtwf25UW7xHyNHsL4jq5/results_constrained_partial/call_acc/tanh_linear.py", line 34, in tanh_linear
    tanh_linear_kernel[grid](input.data, weight.data, bias.data if bias is not None else None, out.data, M, N, K,
  File "/usr/local/lib/python3.12/site-packages/triton/runtime/jit.py", line 345, in <lambda>
    return lambda *args, **kwargs: self.run(grid=grid, warmup=False, *args, **kwargs)
                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/site-packages/triton/runtime/jit.py", line 618, in run
    bound_args, sig_and_spec, constexpr_vals, non_constexpr_vals, excess_kwargs = self.binder(*args, **kwargs)
                                                                                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: dynamic_func() got multiple values for argument 'BLOCK_M'

=== Output for sum.py on GPU 0 ===

=== Errors for sum.py on GPU 0 ===
Traceback (most recent call last):
  File "/data/results_constrained_partial/call_acc/sum.py", line 52, in <module>
    test_results = test_sum()
                   ^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/sum.py", line 44, in test_sum
    results["test_case_3"] = sum(input_tensor_3d, dim=(0, 2))
                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/sum.py", line 18, in sum
    n_rows, n_cols = input.shape
    ^^^^^^^^^^^^^^
ValueError: too many values to unpack (expected 2)

=== Output for logspace.py on GPU 0 ===

=== Errors for logspace.py on GPU 0 ===
Traceback (most recent call last):
  File "/data/results_constrained_partial/call_acc/logspace.py", line 65, in <module>
    test_results = test_logspace()
                   ^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/logspace.py", line 40, in test_logspace
    results["test_case_1"] = logspace(start, end, steps)
                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/logspace.py", line 24, in logspace
    logspace_kernel[grid](output.data, base_tensor.data, output.data, n_elements, BLOCK_SIZE=1024)
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
triton.compiler.errors.CompilationError: at 7:19:
def logspace_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    base = tl.load(base_ptr, mask=mask)
                   ^
NameError('base_ptr is not defined')

=== Output for solve_and_add_scaled_vector.py on GPU 0 ===

=== Errors for solve_and_add_scaled_vector.py on GPU 0 ===
Traceback (most recent call last):
  File "/data/results_constrained_partial/call_acc/solve_and_add_scaled_vector.py", line 57, in <module>
    test_results = test_solve_and_add_scaled_vector()
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/solve_and_add_scaled_vector.py", line 54, in test_solve_and_add_scaled_vector
    results["test_case_1"] = solve_and_add_scaled_vector(A1, b1, y1, alpha1)
                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/solve_and_add_scaled_vector.py", line 34, in solve_and_add_scaled_vector
    solve_and_add_scaled_vector_kernel[grid](A, b, y, C, M, N, K, alpha,
  File "/usr/local/lib/python3.12/site-packages/triton/runtime/jit.py", line 345, in <lambda>
    return lambda *args, **kwargs: self.run(grid=grid, warmup=False, *args, **kwargs)
                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/site-packages/triton/runtime/jit.py", line 618, in run
    bound_args, sig_and_spec, constexpr_vals, non_constexpr_vals, excess_kwargs = self.binder(*args, **kwargs)
                                                                                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: dynamic_func() takes 13 positional arguments but 14 were given

=== Output for pixel_shuffle_conv2d.py on GPU 0 ===

=== Errors for pixel_shuffle_conv2d.py on GPU 0 ===
Traceback (most recent call last):
  File "/data/results_constrained_partial/call_acc/pixel_shuffle_conv2d.py", line 106, in <module>
    test_results = test_pixel_shuffle_conv2d()
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/pixel_shuffle_conv2d.py", line 86, in test_pixel_shuffle_conv2d
    results["test_case_1"] = pixel_shuffle_conv2d(input1, weight1)
                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/pixel_shuffle_conv2d.py", line 33, in pixel_shuffle_conv2d
    C_per_block = C_out // BLOCK_K
                           ^^^^^^^
NameError: name 'BLOCK_K' is not defined

=== Output for matrix_vector_dot.py on GPU 0 ===

=== Errors for matrix_vector_dot.py on GPU 0 ===

=== Output for min_gelu.py on GPU 0 ===

=== Errors for min_gelu.py on GPU 0 ===
  File "/data/results_constrained_partial/call_acc/min_gelu.py", line 53
    tl.store(y_ptrs[r], y)
    ^
IndentationError: expected an indented block after 'for' statement on line 52

=== Output for pow.py on GPU 0 ===

=== Errors for pow.py on GPU 0 ===
Traceback (most recent call last):
  File "/usr/local/lib/python3.12/site-packages/triton/language/core.py", line 35, in wrapper
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/site-packages/triton/language/core.py", line 1597, in load
    return semantic.load(pointer, mask, other, boundary_check, padding_option, cache_modifier, eviction_policy,
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/site-packages/triton/language/semantic.py", line 1037, in load
    return _load_legacy(ptr, mask, other, boundary_check, padding, cache, eviction, is_volatile, builder)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/site-packages/triton/language/semantic.py", line 969, in _load_legacy
    raise ValueError(f"Unsupported ptr type {ptr.type.__repr__()} in `tl.load`")
ValueError: Unsupported ptr type <[1024], fp32> in `tl.load`

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/data/results_constrained_partial/call_acc/pow.py", line 60, in <module>
    test_results = test_pow()
                   ^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/pow.py", line 41, in test_pow
    results["test_case_1"] = pow(input_tensor, exponent)
                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/pow.py", line 24, in pow
    pow_kernel[grid](input_tensor.data, exponent, output.data, n_elements, BLOCK_SIZE=1024)
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
triton.compiler.errors.CompilationError: at 6:8:
def pow_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
        ^

=== Output for relu_max_pool2d_conv2d.py on GPU 0 ===

=== Errors for relu_max_pool2d_conv2d.py on GPU 0 ===
Traceback (most recent call last):
  File "/data/results_constrained_partial/call_acc/relu_max_pool2d_conv2d.py", line 87, in <module>
    test_results = test_relu_max_pool2d_conv2d()
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/relu_max_pool2d_conv2d.py", line 73, in test_relu_max_pool2d_conv2d
    results["test_case_1"] = relu_max_pool2d_conv2d(input, weight)
                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/relu_max_pool2d_conv2d.py", line 41, in relu_max_pool2d_conv2d
    input.contiguous(), weight.contiguous(), bias.contiguous(),
                                             ^^^^^^^^^^^^^^^
AttributeError: 'NoneType' object has no attribute 'contiguous'

=== Output for erf.py on GPU 0 ===

=== Errors for erf.py on GPU 0 ===

=== Output for sigmoid.py on GPU 0 ===

=== Errors for sigmoid.py on GPU 0 ===

=== Output for gelu.py on GPU 0 ===

=== Errors for gelu.py on GPU 0 ===
Traceback (most recent call last):
  File "/data/results_constrained_partial/call_acc/gelu.py", line 58, in <module>
    test_results = test_gelu()
                   ^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/gelu.py", line 42, in test_gelu
    results["test_case_1"] = gelu(input_tensor_1)
                             ^^^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/gelu.py", line 24, in gelu
    gelu_kernel[grid](input, output, n_elements, BLOCK_SIZE=1024)
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
  File "/usr/local/lib/python3.12/ast.py", line 417, in generic_visit
    self.visit(value)
  File "/usr/local/lib/python3.12/ast.py", line 407, in visit
    return visitor(node)
           ^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/site-packages/triton/runtime/jit.py", line 128, in visit_Call
    assert func is None or is_triton_builtin(func) or isinstance(
AssertionError: Function "tanh" is being called from a Triton function but is not a Triton function itself. Decorate it with @triton.jit to fix this

=== Output for det.py on GPU 0 ===

=== Errors for det.py on GPU 0 ===
Traceback (most recent call last):
  File "/data/results_constrained_partial/call_acc/det.py", line 55, in <module>
    test_results = test_det()
                   ^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/det.py", line 39, in test_det
    results["test_case_1"] = det(A1).item()
                             ^^^^^^^
  File "/data/results_constrained_partial/call_acc/det.py", line 22, in det
    det_kernel[grid](A, output, n_elements, BLOCK_SIZE=1024)
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
  File "/usr/local/lib/python3.12/site-packages/triton/runtime/jit.py", line 127, in visit_Call
    func = self.visit(node.func)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/ast.py", line 407, in visit
    return visitor(node)
           ^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/site-packages/triton/runtime/jit.py", line 117, in visit_Attribute
    return getattr(lhs, node.attr)
           ^^^^^^^^^^^^^^^^^^^^^^^
AttributeError: module 'triton.language' has no attribute 'lu'

=== Output for fused_bmm_rmsnorm_gelu_dropout.py on GPU 0 ===

=== Errors for fused_bmm_rmsnorm_gelu_dropout.py on GPU 0 ===
Traceback (most recent call last):
  File "/data/results_constrained_partial/call_acc/fused_bmm_rmsnorm_gelu_dropout.py", line 76, in <module>
    test_results = test_fused_bmm_rmsnorm_gelu_dropout()
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/fused_bmm_rmsnorm_gelu_dropout.py", line 63, in test_fused_bmm_rmsnorm_gelu_dropout
    results["test_case_1"] = fused_bmm_rmsnorm_gelu_dropout(input1, input2, normalized_shape=5)
                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/fused_bmm_rmsnorm_gelu_dropout.py", line 34, in fused_bmm_rmsnorm_gelu_dropout
    fused_bmm_rmsnorm_gelu_dropout_kernel[grid](input1.data, input2.data, c.data, M, N, P, normalized_shape, dropout_p, eps, training, approximate, c.data, BLOCK_M=64, BLOCK_N=64, BLOCK_P=32)
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
AttributeError: 'str' object has no attribute 'dtype'

=== Output for floor.py on GPU 0 ===

=== Errors for floor.py on GPU 0 ===
Traceback (most recent call last):
  File "/usr/local/lib/python3.12/site-packages/triton/language/core.py", line 35, in wrapper
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/site-packages/triton/language/math.py", line 26, in check
    raise ValueError(f"Expected dtype {dtypes} but got {arg.type.scalar.name}")
ValueError: Expected dtype ['fp32', 'fp64'] but got int64

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/data/results_constrained_partial/call_acc/floor.py", line 47, in <module>
    test_results = test_floor()
                   ^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/floor.py", line 35, in test_floor
    results["test_case_2"] = floor(input2)
                             ^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/floor.py", line 17, in floor
    floor_kernel[grid](input, output, n_elements, BLOCK_SIZE=1024)
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
triton.compiler.errors.CompilationError: at 6:35:
def floor_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, tl.floor(x), mask=mask)
                                   ^

=== Output for rand.py on GPU 0 ===

=== Errors for rand.py on GPU 0 ===
Traceback (most recent call last):
  File "/data/results_constrained_partial/call_acc/rand.py", line 60, in <module>
    test_results = test_rand()
                   ^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/rand.py", line 45, in test_rand
    results["test_case_1"] = rand(2, 3, device='cuda')
                             ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/rand.py", line 29, in rand
    size_ptr = output.data.ptr
               ^^^^^^^^^^^^^^^
AttributeError: 'Tensor' object has no attribute 'ptr'

=== Output for cholesky_solve.py on GPU 0 ===

=== Errors for cholesky_solve.py on GPU 0 ===
Traceback (most recent call last):
  File "/data/results_constrained_partial/call_acc/cholesky_solve.py", line 72, in <module>
    test_results = test_cholesky_solve()
                   ^^^^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/cholesky_solve.py", line 53, in test_cholesky_solve
    results["test_case_1"] = cholesky_solve(B1, L1)
                             ^^^^^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/cholesky_solve.py", line 35, in cholesky_solve
    C = torch.empty((M, N), device=B.device, dtype=complex128)
                                                   ^^^^^^^^^^
NameError: name 'complex128' is not defined. Did you mean: 'complex'?

=== Output for mul_sub.py on GPU 0 ===

=== Errors for mul_sub.py on GPU 0 ===
Traceback (most recent call last):
  File "/data/results_constrained_partial/call_acc/mul_sub.py", line 53, in <module>
    test_results = test_mul_sub()
                   ^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/mul_sub.py", line 41, in test_mul_sub
    results["test_case_2"] = mul_sub(input_tensor, other_mul_number, other_sub_tensor)
                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/mul_sub.py", line 18, in mul_sub
    mul_sub_kernel[grid](input.data, other_mul.data, output.data, n_elements, BLOCK_SIZE=1024)
                                     ^^^^^^^^^^^^^^
AttributeError: 'float' object has no attribute 'data'

=== Output for ldl_factor.py on GPU 0 ===

=== Errors for ldl_factor.py on GPU 0 ===
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
  File "/data/results_constrained_partial/call_acc/ldl_factor.py", line 102, in <module>
    test_results = test_ldl_factor()
                   ^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/ldl_factor.py", line 86, in test_ldl_factor
    results["test_case_1"] = ldl_factor(A1)
                             ^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/ldl_factor.py", line 53, in ldl_factor
    ldl_factor_kernel[grid](C.data, C.data, A.data, M, N, K,
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
triton.compiler.errors.CompilationError: at 7:32:
def ldl_factor_kernel(LD_ptr, pivots_ptr, A_ptr, M, N, K, stride_AM, stride_AK, stride_BK, stride_AN, stride_LD, stride_PIVOTS, BLOCK_M: tl.constexpr, BLOCK_N, BLOCK_K):
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    offs_AM = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_AN = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
                                ^

=== Output for abs.py on GPU 0 ===

=== Errors for abs.py on GPU 0 ===

=== Output for mul.py on GPU 0 ===

=== Errors for mul.py on GPU 0 ===
Traceback (most recent call last):
  File "/data/results_constrained_partial/call_acc/mul.py", line 52, in <module>
    test_results = test_mul()
                   ^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/mul.py", line 33, in test_mul
    results["test_case_1"] = mul(input1, other1)
                             ^^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/mul.py", line 18, in mul
    mul_kernel[grid](input.data_PTR(), other.data_PTR(), output.data_PTR(), n_elements, BLOCK_SIZE=1024)
                     ^^^^^^^^^^^^^^
AttributeError: 'Tensor' object has no attribute 'data_PTR'. Did you mean: 'data_ptr'?

=== Output for softmax.py on GPU 0 ===

=== Errors for softmax.py on GPU 0 ===

=== Output for leaky_relu.py on GPU 0 ===

=== Errors for leaky_relu.py on GPU 0 ===
Traceback (most recent call last):
  File "/data/results_constrained_partial/call_acc/leaky_relu.py", line 55, in <module>
    test_results = test_leaky_relu()
                   ^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/leaky_relu.py", line 39, in test_leaky_relu
    results["test_case_1"] = leaky_relu(input_tensor_1)
                             ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/leaky_relu.py", line 25, in leaky_relu
    leaky_relu_kernel[(input.numel(),)](output.data, input.data, input.stride(0), output.stride(0), input.size(0), input.size(1), BLOCK_SIZE=32)
                                                                                                                   ^^^^^^^^^^^^^
IndexError: Dimension out of range (expected to be in range of [-1, 0], but got 1)

=== Output for invert_matrix_lu.py on GPU 0 ===

=== Errors for invert_matrix_lu.py on GPU 0 ===
Traceback (most recent call last):
  File "/data/results_constrained_partial/call_acc/invert_matrix_lu.py", line 69, in <module>
    test_results = test_invert_matrix_lu()
                   ^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/invert_matrix_lu.py", line 53, in test_invert_matrix_lu
    results["test_case_1"] = invert_matrix_lu(A1)
                             ^^^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/invert_matrix_lu.py", line 33, in invert_matrix_lu
    invert_matrix_lu_kernel[grid](A, A, C, M, N, K,
  File "/usr/local/lib/python3.12/site-packages/triton/runtime/jit.py", line 345, in <lambda>
    return lambda *args, **kwargs: self.run(grid=grid, warmup=False, *args, **kwargs)
                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/site-packages/triton/runtime/jit.py", line 618, in run
    bound_args, sig_and_spec, constexpr_vals, non_constexpr_vals, excess_kwargs = self.binder(*args, **kwargs)
                                                                                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: dynamic_func() got multiple values for argument 'BLOCK_M'

=== Output for std.py on GPU 0 ===

=== Errors for std.py on GPU 0 ===
Traceback (most recent call last):
  File "/data/results_constrained_partial/call_acc/std.py", line 68, in <module>
    test_results = test_std()
                   ^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/std.py", line 52, in test_std
    results["test_case_1"] = std(input_tensor)
                             ^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/std.py", line 36, in std
    std_kernel[grid](input, input, output, n_elements, BLOCK_SIZE=1024)
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
triton.compiler.errors.CompilationError: at 9:19:
def std_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    mean_x = tl.sum(x, axis=0) / n_elements
    mean_y = tl.sum(y, axis=0) / n_elements
    var_x = tl.sum((x - mean_x) ** 2, axis=0) / (n_elements - 1)
                   ^
AttributeError("'tensor' object has no attribute '__pow__'")

=== Output for tril_mm_and_scale.py on GPU 0 ===

=== Errors for tril_mm_and_scale.py on GPU 0 ===
Traceback (most recent call last):
  File "/data/results_constrained_partial/call_acc/tril_mm_and_scale.py", line 76, in <module>
    test_results = test_tril_mm_and_scale()
                   ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/tril_mm_and_scale.py", line 51, in test_tril_mm_and_scale
    results["test_case_1"] = tril_mm_and_scale(A1, B1, alpha1, beta1)
                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/tril_mm_and_scale.py", line 34, in tril_mm_and_scale
    tril_mm_and_scale_kernel[grid](A, B, C, M, N, K, alpha, beta, A.stride(0), A.stride(1), B.stride(0), B.stride(1), C.stride(0), C.stride(1), BLOCK_M=64, BLOCK_N=64, BLOCK_K=32)
  File "/usr/local/lib/python3.12/site-packages/triton/runtime/jit.py", line 345, in <lambda>
    return lambda *args, **kwargs: self.run(grid=grid, warmup=False, *args, **kwargs)
                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/site-packages/triton/runtime/jit.py", line 618, in run
    bound_args, sig_and_spec, constexpr_vals, non_constexpr_vals, excess_kwargs = self.binder(*args, **kwargs)
                                                                                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: dynamic_func() got multiple values for argument 'BLOCK_M'

=== Output for solve.py on GPU 0 ===

=== Errors for solve.py on GPU 0 ===
Traceback (most recent call last):
  File "/data/results_constrained_partial/call_acc/solve.py", line 79, in <module>
    test_results = test_solve()
                   ^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/solve.py", line 54, in test_solve
    results["test_case_1"] = solve(A1, B1)
                             ^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/solve.py", line 29, in solve
    K, N = B.shape
    ^^^^
ValueError: not enough values to unpack (expected 2, got 1)

=== Output for airy_ai.py on GPU 0 ===

=== Errors for airy_ai.py on GPU 0 ===
Traceback (most recent call last):
  File "/data/results_constrained_partial/call_acc/airy_ai.py", line 48, in <module>
    test_results = test_airy_ai()
                   ^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/airy_ai.py", line 32, in test_airy_ai
    results["test_case_1"] = airy_ai(input1)
                             ^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/airy_ai.py", line 18, in airy_ai
    airy_ai_kernel[grid](input, output, n_elements, BLOCK_SIZE=1024)
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
  File "/usr/local/lib/python3.12/site-packages/triton/runtime/jit.py", line 127, in visit_Call
    func = self.visit(node.func)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/ast.py", line 407, in visit
    return visitor(node)
           ^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/site-packages/triton/runtime/jit.py", line 117, in visit_Attribute
    return getattr(lhs, node.attr)
           ^^^^^^^^^^^^^^^^^^^^^^^
AttributeError: module 'triton.language.math' has no attribute 'airy_ai'

=== Output for signbit.py on GPU 0 ===

=== Errors for signbit.py on GPU 0 ===

=== Output for matrix_multiply_and_row_dot.py on GPU 0 ===

=== Errors for matrix_multiply_and_row_dot.py on GPU 0 ===
Traceback (most recent call last):
  File "/data/results_constrained_partial/call_acc/matrix_multiply_and_row_dot.py", line 78, in <module>
    test_results = test_matrix_multiply_and_row_dot()
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/matrix_multiply_and_row_dot.py", line 50, in test_matrix_multiply_and_row_dot
    results["test_case_1"] = matrix_multiply_and_row_dot(A, B, alpha, beta, C).item()
                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/matrix_multiply_and_row_dot.py", line 32, in matrix_multiply_and_row_dot
    matrix_multiply_and_row_dot_kernel[grid](A, B, c, M, N, K, alpha, beta, A.stride(0), A.stride(1), B.stride(0), B.stride(1), C.stride(0), C.stride(1), BLOCK_M=64, BLOCK_N=64, BLOCK_K=32)
  File "/usr/local/lib/python3.12/site-packages/triton/runtime/jit.py", line 345, in <lambda>
    return lambda *args, **kwargs: self.run(grid=grid, warmup=False, *args, **kwargs)
                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/site-packages/triton/runtime/jit.py", line 618, in run
    bound_args, sig_and_spec, constexpr_vals, non_constexpr_vals, excess_kwargs = self.binder(*args, **kwargs)
                                                                                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: dynamic_func() got multiple values for argument 'BLOCK_M'

=== Output for polygamma.py on GPU 0 ===

=== Errors for polygamma.py on GPU 0 ===
Traceback (most recent call last):
  File "/data/results_constrained_partial/call_acc/polygamma.py", line 53, in <module>
    test_results = test_polygamma()
                   ^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/polygamma.py", line 40, in test_polygamma
    results["test_case_1"] = polygamma(1, a)
                             ^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/polygamma.py", line 24, in polygamma
    polygamma_kernel[grid](input, input, result, n_elements, BLOCK_SIZE=1024)
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
AttributeError: module 'triton.language.math' has no attribute 'gamma'

=== Output for elu_linear.py on GPU 0 ===

=== Errors for elu_linear.py on GPU 0 ===
Traceback (most recent call last):
  File "/data/results_constrained_partial/call_acc/elu_linear.py", line 100, in <module>
    test_results = test_elu_linear()
                   ^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/elu_linear.py", line 79, in test_elu_linear
    results["test_case_1"] = elu_linear(input1, weight1, bias1)
                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/elu_linear.py", line 52, in elu_linear
    elu_linear_kernel[grid](input.data, weight.data, bias.data, y.data, M, N, K,
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
triton.compiler.errors.CompilationError: at 3:27:
def elu_linear_kernel(x_ptr, w_ptr, b_ptr, y_ptr, M, N, K, stride_x, stride_w, stride_b, stride_y, stride_m, stride_n: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
                           ^
NameError('BLOCK_N is not defined')

=== Output for fused_pairwise_distance_normalize.py on GPU 0 ===

=== Errors for fused_pairwise_distance_normalize.py on GPU 0 ===
Traceback (most recent call last):
  File "/data/results_constrained_partial/call_acc/fused_pairwise_distance_normalize.py", line 92, in <module>
    test_results = test_fused_pairwise_distance_normalize()
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/fused_pairwise_distance_normalize.py", line 73, in test_fused_pairwise_distance_normalize
    results["test_case_1"] = fused_pairwise_distance_normalize(x1, x2)
                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
NameError: name 'fused_pairwise_distance_normalize' is not defined. Did you mean: 'test_fused_pairwise_distance_normalize'?

=== Output for Adam.py on GPU 0 ===

=== Errors for Adam.py on GPU 0 ===
Traceback (most recent call last):
  File "/data/results_constrained_partial/call_acc/Adam.py", line 115, in <module>
    test_results = test_Adam()
                   ^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/Adam.py", line 95, in test_Adam
    optimizer1 = Adam(params1)
                 ^^^^
NameError: name 'Adam' is not defined. Did you mean: 'adam'?

=== Output for fused_hstack_div.py on GPU 0 ===

=== Errors for fused_hstack_div.py on GPU 0 ===
  File "/data/results_constrained_partial/call_acc/fused_hstack_div.py", line 21
    @triton.jit
IndentationError: unexpected indent

=== Output for broadcast_tensors.py on GPU 0 ===

=== Errors for broadcast_tensors.py on GPU 0 ===
Traceback (most recent call last):
  File "/data/results_constrained_partial/call_acc/broadcast_tensors.py", line 57, in <module>
    test_results = test_broadcast_tensors()
                   ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/broadcast_tensors.py", line 38, in test_broadcast_tensors
    results["test_case_1"] = broadcast_tensors(x1, y1)
                             ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/broadcast_tensors.py", line 21, in broadcast_tensors
    broadcast_tensor_kernel[grid](tensors[0], tensors[1], output, n_elements, BLOCK_SIZE=1024)
  File "/usr/local/lib/python3.12/site-packages/triton/runtime/jit.py", line 345, in <lambda>
    return lambda *args, **kwargs: self.run(grid=grid, warmup=False, *args, **kwargs)
                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/site-packages/triton/runtime/jit.py", line 691, in run
    kernel.run(grid_0, grid_1, grid_2, stream, kernel.function, kernel.packed_metadata, launch_metadata,
  File "/usr/local/lib/python3.12/site-packages/triton/backends/nvidia/driver.py", line 365, in __call__
    self.launch(*args, **kwargs)
ValueError: Pointer argument (at 2) cannot be accessed from Triton (cpu tensor?)

=== Output for relu_conv2d.py on GPU 0 ===

=== Errors for relu_conv2d.py on GPU 0 ===
Traceback (most recent call last):
  File "/data/results_constrained_partial/call_acc/relu_conv2d.py", line 122, in <module>
    test_results = test_relu_conv2d()
                   ^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/relu_conv2d.py", line 102, in test_relu_conv2d
    results["test_case_1"] = relu_conv2d(input1, weight1)
                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/relu_conv2d.py", line 30, in relu_conv2d
    assert C_in == C_out * groups, "Input and output channels must match."
AssertionError: Input and output channels must match.

=== Output for log.py on GPU 0 ===

=== Errors for log.py on GPU 0 ===

=== Output for adaptive_avg_pool2d.py on GPU 0 ===

=== Errors for adaptive_avg_pool2d.py on GPU 0 ===
Traceback (most recent call last):
  File "/data/results_constrained_partial/call_acc/adaptive_avg_pool2d.py", line 39, in <module>
    from adaptive_avg_pool2d import adaptive_avg_pool2d
  File "/__modal/volumes/vo-WvFtwf25UW7xHyNHsL4jq5/results_constrained_partial/call_acc/adaptive_avg_pool2d.py", line 66, in <module>
    test_results = test_adaptive_avg_pool2d()
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/__modal/volumes/vo-WvFtwf25UW7xHyNHsL4jq5/results_constrained_partial/call_acc/adaptive_avg_pool2d.py", line 46, in test_adaptive_avg_pool2d
    output1 = adaptive_avg_pool2d(input1, 5)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: adaptive_avg_pool2d() takes 1 positional argument but 2 were given

=== Output for quantize_dynamic.py on GPU 0 ===

=== Errors for quantize_dynamic.py on GPU 0 ===
Traceback (most recent call last):
  File "/data/results_constrained_partial/call_acc/quantize_dynamic.py", line 78, in <module>
    test_results = test_quantize_dynamic()
                   ^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/quantize_dynamic.py", line 40, in test_quantize_dynamic
    class SimpleModel(nn.Module):
                      ^^
NameError: name 'nn' is not defined

=== Output for conv2d_add.py on GPU 0 ===

=== Errors for conv2d_add.py on GPU 0 ===
  File "/data/results_constrained_partial/call_acc/conv2d_add.py", line 55
    def conv2d_add(input: torch.Tensor, weight: torch.Tensor, bias=None, other=None, stride=1, padding=0, dilation=1, groups=1, alpha=1, out=None):
    ^^^
SyntaxError: invalid syntax

=== Output for ifftshift.py on GPU 0 ===

=== Errors for ifftshift.py on GPU 0 ===
Traceback (most recent call last):
  File "/data/results_constrained_partial/call_acc/ifftshift.py", line 64, in <module>
    test_results = test_ifftshift()
                   ^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/ifftshift.py", line 49, in test_ifftshift
    results["test_case_1"] = ifftshift(input_tensor_1d)
                             ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/ifftshift.py", line 35, in ifftshift
    ifftshift_kernel[grid](input, input, n_elements, BLOCK_SIZE=1024)
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
triton.compiler.errors.UnsupportedLanguageConstruct: at 7:27:
def ifftshift_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    # Reverse the order of dimensions
    reversed_x = x.permute(*range(len(x.shape))[::-1])
                           ^
unsupported AST node type: Starred

=== Output for signbit_bitwise_and.py on GPU 0 ===

=== Errors for signbit_bitwise_and.py on GPU 0 ===
Traceback (most recent call last):
  File "/data/results_constrained_partial/call_acc/signbit_bitwise_and.py", line 93, in <module>
    test_results = test_signbit_bitwise_and()
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/signbit_bitwise_and.py", line 74, in test_signbit_bitwise_and
    results["test_case_1"] = signbit_bitwise_and(a, b)
                             ^^^^^^^^^^^^^^^^^^^
NameError: name 'signbit_bitwise_and' is not defined. Did you mean: 'test_signbit_bitwise_and'?

=== Output for fused_repeat_interleave_log_softmax.py on GPU 0 ===

=== Errors for fused_repeat_interleave_log_softmax.py on GPU 0 ===
Traceback (most recent call last):
  File "/data/results_constrained_partial/call_acc/fused_repeat_interleave_log_softmax.py", line 78, in <module>
    test_results = test_fused_repeat_interleave_log_softmax()
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/fused_repeat_interleave_log_softmax.py", line 55, in test_fused_repeat_interleave_log_softmax
    results["test_case_1"] = fused_repeat_interleave_log_softmax(input1, repeats1)
                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: fused_repeat_interleave_log_softmax() missing 1 required positional argument: 'dim'

=== Output for cholesky.py on GPU 0 ===

=== Errors for cholesky.py on GPU 0 ===
Traceback (most recent call last):
  File "/data/results_constrained_partial/call_acc/cholesky.py", line 98, in <module>
    test_results = test_cholesky()
                   ^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/cholesky.py", line 75, in test_cholesky
    L1 = cholesky(A1)
         ^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/cholesky.py", line 60, in cholesky
    cholesky_kernel[grid](A, A, out, n_elements, BLOCK_SIZE=1024)
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
triton.compiler.errors.CompilationError: at 48:13:
    # https://github.com/pytorch/pytorch/blob/master/torch/linalg/decomposition.py#L407
    # For a more detailed implementation, see the official documentation for the Cholesky decomposition algorithm
    # https://github.com/pytorch/pytorch/blob/master/torch/linalg/decomposition.py#L457
    # For a more thorough implementation, see the official documentation for the Cholesky decomposition algorithm
    # https://github.com/pytorch/pytorch/blob/master/torch/linalg/decomposition.py#L507
    # For a more robust implementation, see the official documentation for the Cholesky decomposition algorithm
    # https://github.com/pytorch/pytorch/blob/master/torch/linalg/decomposition.py#L557
    # For a more complete implementation, see the official documentation for the Cholesky decomposition algorithm
    # https://github.com/pytorch/pytorch/blob/master/torch/linalg/decomposition.py#L607
    # For a more detailed implementation, see the official documentation for the Cholesky decomposition algorithm
    # https://github.com/pytorch/pytorch/blob/master/torch/linalg/decomposition.py#L657
    tl.store(output_ptr + offsets, x + y, mask=mask)
             ^
AssertionError("cannot convert None of type <class 'NoneType'> to tensor")

=== Output for ones_like.py on GPU 0 ===

=== Errors for ones_like.py on GPU 0 ===

=== Output for autocast.py on GPU 0 ===

=== Errors for autocast.py on GPU 0 ===
Traceback (most recent call last):
  File "/data/results_constrained_partial/call_acc/autocast.py", line 113, in <module>
    test_results = test_autocast()
                   ^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/autocast.py", line 92, in test_autocast
    with autocast('cuda'):
         ^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/autocast.py", line 58, in autocast
    device_type = tl.array([device_type], dtype=tl.int32)
                  ^^^^^^^^
AttributeError: module 'triton.language' has no attribute 'array'

=== Output for reciprocal.py on GPU 0 ===

=== Errors for reciprocal.py on GPU 0 ===
Traceback (most recent call last):
  File "/data/results_constrained_partial/call_acc/reciprocal.py", line 55, in <module>
    test_results = test_reciprocal()
                   ^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/reciprocal.py", line 39, in test_reciprocal
    results["test_case_1"] = reciprocal(a)
                             ^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/reciprocal.py", line 25, in reciprocal
    reciprocal_kernel[grid](input.data, out.data, n_elements, BLOCK_SIZE=1024)
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
  File "/usr/local/lib/python3.12/site-packages/triton/runtime/jit.py", line 127, in visit_Call
    func = self.visit(node.func)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/ast.py", line 407, in visit
    return visitor(node)
           ^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/site-packages/triton/runtime/jit.py", line 117, in visit_Attribute
    return getattr(lhs, node.attr)
           ^^^^^^^^^^^^^^^^^^^^^^^
AttributeError: module 'triton.language.math' has no attribute 'reciprocal'

=== Output for cos_signbit.py on GPU 0 ===

=== Errors for cos_signbit.py on GPU 0 ===
Traceback (most recent call last):
  File "/data/results_constrained_partial/call_acc/cos_signbit.py", line 59, in <module>
    test_results = test_cos_signbit()
                   ^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/cos_signbit.py", line 39, in test_cos_signbit
    cos_result_1, sign_bit_1 = cos_signbit(input_tensor_1)
                               ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/cos_signbit.py", line 19, in cos_signbit
    cos_signbit_kernel[grid](input, output, n_elements, BLOCK_SIZE=1024)
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
  File "/usr/local/lib/python3.12/site-packages/triton/runtime/jit.py", line 127, in visit_Call
    func = self.visit(node.func)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/ast.py", line 407, in visit
    return visitor(node)
           ^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/site-packages/triton/runtime/jit.py", line 117, in visit_Attribute
    return getattr(lhs, node.attr)
           ^^^^^^^^^^^^^^^^^^^^^^^
AttributeError: module 'triton.language' has no attribute 'signbit'

=== Output for spectral_norm_eig.py on GPU 0 ===

=== Errors for spectral_norm_eig.py on GPU 0 ===
Traceback (most recent call last):
  File "/data/results_constrained_partial/call_acc/spectral_norm_eig.py", line 59, in <module>
    test_results = test_spectral_norm_eig()
                   ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/spectral_norm_eig.py", line 46, in test_spectral_norm_eig
    results["test_case_2"] = spectral_norm_eig(A2)
                             ^^^^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/spectral_norm_eig.py", line 22, in spectral_norm_eig
    n_rows, n_cols = A.shape
    ^^^^^^^^^^^^^^
ValueError: too many values to unpack (expected 2)

=== Output for fftn.py on GPU 0 ===

=== Errors for fftn.py on GPU 0 ===
Traceback (most recent call last):
  File "/data/results_constrained_partial/call_acc/fftn.py", line 64, in <module>
    test_results = test_fftn()
                   ^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/fftn.py", line 45, in test_fftn
    results["test_case_1"] = fftn(input_tensor)
                             ^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/fftn.py", line 28, in fftn
    fftn_kernel[grid](input.data, out.data, out.data, n_elements, BLOCK_SIZE=1024)
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
  File "/usr/local/lib/python3.12/site-packages/triton/runtime/jit.py", line 127, in visit_Call
    func = self.visit(node.func)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/ast.py", line 407, in visit
    return visitor(node)
           ^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/site-packages/triton/runtime/jit.py", line 117, in visit_Attribute
    return getattr(lhs, node.attr)
           ^^^^^^^^^^^^^^^^^^^^^^^
AttributeError: module 'triton.language' has no attribute 'complex'

Deleted Adam.py
Deleted SGD.py
Error deleting __pycache__: [Errno 21] Is a directory: '/data/results_constrained_partial/call_acc/__pycache__'
Deleted adaptive_avg_pool2d.py
Deleted add.py
Deleted add_gelu.py
Deleted add_mean.py
Deleted addmm.py
Deleted airy_ai.py
Deleted argmax.py
Deleted asin.py
Deleted autocast.py
Deleted batch_norm.py
Deleted bessel_j1.py
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
Deleted determinant_via_qr.py
Deleted div.py
Deleted dropout_relu_batch_norm_conv2d.py
Deleted dropout_sigmoid_linear.py
Deleted elu_linear.py
Deleted erfc_sqrt.py
Deleted exp_mean.py
Deleted fftn.py
Deleted floor.py
Deleted fused_add_mul_groupnorm.py
Deleted fused_avg_pool2d_cosine_similarity.py
Deleted fused_bmm_dropout_gelu.py
Deleted fused_bmm_rmsnorm_gelu_dropout.py
Deleted fused_bmm_rmsnorm_gelu_dropout_sub.py
Deleted fused_cholesky_solve.py
Deleted fused_cosine_embedding_loss_with_normalization.py
Deleted fused_cross_entropy_log_softmax.py
Deleted fused_cross_entropy_softmax_layernorm.py
Deleted fused_embedding_add_tanh.py
Deleted fused_fractional_max_pool2d_with_relu.py
Deleted fused_gather_masked_fill.py
Deleted fused_hardshrink_dropout.py
Deleted fused_hardsigmoid_batch_norm.py
Deleted fused_hstack_div.py
Deleted fused_index_select_eq.py
Deleted fused_instance_norm_selu_conv2d.py
Deleted fused_layer_norm_relu_linear.py
Deleted fused_lu_solve.py
Deleted fused_masked_select_add_gelu.py
Deleted fused_mul_add_logsoftmax_dropout_bmm.py
Deleted fused_mv_sigmoid_sub.py
Deleted fused_pairwise_distance_adaptive_avg_pool2d.py
Deleted fused_pairwise_distance_normalize.py
Deleted fused_qr_solve.py
Deleted fused_repeat_interleave_log_softmax.py
Deleted fused_silu_layer_norm_conv2d.py
Deleted fused_svd_reconstruct.py
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
Deleted logit.py
Deleted logspace.py
Deleted logsumexp.py
Deleted low_rank_svd_approximation.py
Deleted lu.py
Deleted matmul.py
Deleted matrix_multiply_and_row_dot.py
Deleted matrix_multiply_symmetric.py
Deleted matrix_power_eig.py
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
Deleted polygamma.py
Deleted pow.py
Deleted pseudoinverse_svd.py
Deleted qr.py
Deleted quantize_dynamic.py
Deleted rad2deg_sqrt.py
Deleted rand.py
Deleted reciprocal.py
Deleted relu_batch_norm_conv2d.py
Deleted relu_conv2d.py
Deleted relu_max_pool2d_conv2d.py
Deleted scaled_add_dot.py
Deleted scaled_add_norm.py
Deleted sigmoid_adaptive_avg_pool2d.py
Deleted sigmoid_argmax.py
Deleted sigmoid_batch_norm.py
Deleted sigmoid_conv2d.py
Deleted signbit_bitwise_and.py
Deleted silu_batch_norm.py
Deleted softmax_log.py
Deleted softmax_mul.py
Deleted softplus_linear.py
Deleted solve.py
Deleted solve_and_add_scaled_vector.py
Deleted solve_symmetric_ldl.py
Deleted spectral_norm_eig.py
Deleted sqrt_tanh.py
Deleted std.py
Deleted sub_gelu.py
Deleted sum.py
Deleted sum_std.py
Deleted svd.py
Deleted symmetric_matrix_vector_norm.py
Deleted symmetric_mm_and_abs_sum.py
Deleted tanh.py
Deleted tanh_linear.py
Deleted tensordot.py
Deleted tensordot_rsqrt.py
Deleted tril_mm_and_scale.py

Correct execution rate: 17.47%
['solve_multiple_lu.py', 'relu_sqrt.py', 'sqrt.py', 'sub.py', 'rsqrt.py', 'fused_mv_logsoftmax_dropout.py', 'max.py', 'relu.py', 'fused_tile_exp.py', 'eig.py', 'log1p.py', 'zeta.py', 'selu.py', 'sqrt_exp.py', 'exp_sqrt.py', 'cos.py', 'trunc.py', 'exp.py', 'bitwise_and.py', 'digamma.py', 'determinant_lu.py', 'matrix_vector_dot.py', 'erf.py', 'sigmoid.py', 'abs.py', 'softmax.py', 'signbit.py', 'log.py', 'ones_like.py']
Above is call test for predictions_qwen_constrained
================================================================================================================================================================

call_acc survivors: 29 / 166

======================================================================
=== Phase 2: execution accuracy ===
======================================================================

Correct execution rate: 100.00% = 29 / 29
above is the compare execution for /data/results_constrained_partial/call_acc
================================================================================================================================================================================================================================================

exe_acc survivors: 29 / 166

======================================================================
=== Phase 3: efficiency ===
======================================================================
Process:   0%|                                           | 0/29 [00:00<?, ?it/s]