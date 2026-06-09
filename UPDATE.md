=== Output for div.py on GPU 0 ===

=== Errors for div.py on GPU 0 ===
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
ValueError: Unsupported ptr type <[1024], int64> in `tl.load`

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/data/results_constrained_partial/call_acc/div.py", line 54, in <module>
    test_results = test_div()
                   ^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/div.py", line 35, in test_div
    results["test_case_1"] = div(input1, other1)
                             ^^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/div.py", line 20, in div
    div_kernel[grid](input.data_ptr(), other.data_ptr(), output.data_ptr(), n_elements, BLOCK_SIZE=1024)
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
def div_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
        ^

=== Output for sigmoid_conv2d.py on GPU 0 ===

=== Errors for sigmoid_conv2d.py on GPU 0 ===
  File "/data/results_constrained_partial/call_acc/sigmoid_conv2d.py", line 71
    assert out.shape == (minibatch, out_channels, iH + 2 * padding_h, iW + 2 * padding_w), "Output shape must be (minibatch, out_ch, i
                                                                                           ^
SyntaxError: unterminated string literal (detected at line 71)

=== Output for solve_multiple_lu.py on GPU 0 ===

=== Errors for solve_multiple_lu.py on GPU 0 ===

=== Output for tanh.py on GPU 0 ===

=== Errors for tanh.py on GPU 0 ===
Traceback (most recent call last):
  File "/data/results_constrained_partial/call_acc/tanh.py", line 57, in <module>
    test_results = test_tanh()
                   ^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/tanh.py", line 41, in test_tanh
    results["test_case_1"] = tanh(input_tensor_1)
                             ^^^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/tanh.py", line 27, in tanh
    tanh_kernel[grid](input_tensor.data_ptr(), output.data_ptr(), n_elements, BLOCK_SIZE=1024)
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
Traceback (most recent call last):
  File "/data/results_constrained_partial/call_acc/relu_sqrt.py", line 102, in <module>
    test_results = test_relu_sqrt()
                   ^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/relu_sqrt.py", line 85, in test_relu_sqrt
    results["test_case_1"] = relu_sqrt(a)
                             ^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/relu_sqrt.py", line 34, in relu_sqrt
    M, N = input.shape
    ^^^^
ValueError: not enough values to unpack (expected 2, got 1)

=== Output for sqrt.py on GPU 0 ===

=== Errors for sqrt.py on GPU 0 ===
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
ValueError: Unsupported ptr type <[1024], int64> in `tl.load`

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/data/results_constrained_partial/call_acc/sqrt.py", line 49, in <module>
    test_results = test_sqrt()
                   ^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/sqrt.py", line 33, in test_sqrt
    results["test_case_1"] = sqrt(input1)
                             ^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/sqrt.py", line 19, in sqrt
    sqrt_kernel[grid](input.data_ptr(), output.data_ptr(), n_elements, BLOCK_SIZE=1024)
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
def sqrt_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
        ^

=== Output for sigmoid_argmax.py on GPU 0 ===

=== Errors for sigmoid_argmax.py on GPU 0 ===
Traceback (most recent call last):
  File "/data/results_constrained_partial/call_acc/sigmoid_argmax.py", line 92, in <module>
    test_results = test_sigmoid_argmax()
                   ^^^^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/sigmoid_argmax.py", line 76, in test_sigmoid_argmax
    results["test_case_1"] = sigmoid_argmax(input1)
                             ^^^^^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/sigmoid_argmax.py", line 61, in sigmoid_argmax
    softmax_kernel[(n_rows,)](y, input, input.stride(0), y.stride(0),
    ^^^^^^^^^^^^^^
NameError: name 'softmax_kernel' is not defined

=== Output for sub.py on GPU 0 ===

=== Errors for sub.py on GPU 0 ===

=== Output for grid_sample.py on GPU 0 ===

=== Errors for grid_sample.py on GPU 0 ===
Traceback (most recent call last):
  File "/data/results_constrained_partial/call_acc/grid_sample.py", line 101, in <module>
    test_results = test_grid_sample()
                   ^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/grid_sample.py", line 86, in test_grid_sample
    results["test_case_1"] = grid_sample(input_4d, grid_4d)
                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/grid_sample.py", line 37, in grid_sample
    M, K = input.shape
    ^^^^
ValueError: too many values to unpack (expected 2)

=== Output for svd.py on GPU 0 ===

=== Errors for svd.py on GPU 0 ===
  File "/data/results_constrained_partial/call_acc/svd.py", line 46
    assert A.stride(-1) // BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE == A.shape[-1], "Block size must divide last
                                                                                                          ^
SyntaxError: unterminated string literal (detected at line 46)

=== Output for i0.py on GPU 0 ===

=== Errors for i0.py on GPU 0 ===
Traceback (most recent call last):
  File "/data/results_constrained_partial/call_acc/i0.py", line 50, in <module>
    test_results = test_i0()
                   ^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/i0.py", line 34, in test_i0
    results["test_case_1"] = i0(input_tensor_1)
                             ^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/i0.py", line 20, in i0
    i0_kernel[grid](input_tensor.data_ptr(), output.data_ptr(), output.data_ptr(), n_elements, BLOCK_SIZE=1024)
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
AttributeError: module 'triton.language.math' has no attribute 'i0'

=== Output for rsqrt.py on GPU 0 ===

=== Errors for rsqrt.py on GPU 0 ===
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
ValueError: Unsupported ptr type <[1024], int64> in `tl.load`

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/data/results_constrained_partial/call_acc/rsqrt.py", line 53, in <module>
    test_results = test_rsqrt()
                   ^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/rsqrt.py", line 37, in test_rsqrt
    results["test_case_1"] = rsqrt(input1)
                             ^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/rsqrt.py", line 20, in rsqrt
    rsqrt_kernel[grid](input.data_ptr(), output.data_ptr(), n_elements, BLOCK_SIZE=1024)
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
def rsqrt_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
        ^

=== Output for dropout_relu_batch_norm_conv2d.py on GPU 0 ===

=== Errors for dropout_relu_batch_norm_conv2d.py on GPU 0 ===
  File "/data/results_constrained_partial/call_acc/dropout_relu_batch_norm_conv2d.py", line 35
    def dropout_relu_batch_norm_conv2d(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor=None, stride_h: int=1, stride_w: int=1, padding_h_starting_from_zero_starting_from_zero_starting_from: int=0, padding_w_starting_from_zero_starting_from_zero_starting_from: int=0, dilation):
                                                                                                                                                                                                                                                                                         ^^^^^^^^
SyntaxError: parameter without a default follows parameter with a default

=== Output for fused_mv_logsoftmax_dropout.py on GPU 0 ===

=== Errors for fused_mv_logsoftmax_dropout.py on GPU 0 ===
Traceback (most recent call last):
  File "/data/results_constrained_partial/call_acc/fused_mv_logsoftmax_dropout.py", line 124, in <module>
    test_results = test_fused_mv_logsoftmax_dropout()
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/fused_mv_logsoftmax_dropout.py", line 105, in test_fused_mv_logsoftmax_dropout
    results["test_case_1"] = fused_mv_logsoftmax_dropout(input1, vec1)
                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/fused_mv_logsoftmax_dropout.py", line 61, in fused_mv_logsoftmax_dropout
    fused_mv_logsoftmax_dropout_kernel[(n_rows,)](y, input.data_ptr(), input.stride(0), y.stride(0),
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
UnboundLocalError: cannot access local variable 'fused_mv_logsoftmax_dropout_kernel' where it is not associated with a value

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
ValueError: Unsupported ptr type <[1024], int64> in `tl.load`

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/data/results_constrained_partial/call_acc/add.py", line 54, in <module>
    test_results = test_add()
                   ^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/add.py", line 35, in test_add
    results["test_case_1"] = add(input1, other1)
                             ^^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/add.py", line 20, in add
    add_kernel[grid](input.data_ptr(), other.data_ptr(), output.data_ptr(), n_elements, BLOCK_SIZE=1024)
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
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
        ^

=== Output for fused_silu_layer_norm_conv2d.py on GPU 0 ===

=== Errors for fused_silu_layer_norm_conv2d.py on GPU 0 ===
Traceback (most recent call last):
  File "/data/results_constrained_partial/call_acc/fused_silu_layer_norm_conv2d.py", line 64, in <module>
    test_results = test_fused_silu_layer_norm_conv2d()
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/fused_silu_layer_norm_conv2d.py", line 50, in test_fused_silu_layer_norm_conv2d
    results['test_case_1'] = fused_silu_layer_norm_conv2d(x, None, conv_weight)
                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
NameError: name 'fused_silu_layer_norm_conv2d' is not defined. Did you mean: 'test_fused_silu_layer_norm_conv2d'?

=== Output for fused_index_select_eq.py on GPU 0 ===

=== Errors for fused_index_select_eq.py on GPU 0 ===
Traceback (most recent call last):
  File "/data/results_constrained_partial/call_acc/fused_index_select_eq.py", line 91, in <module>
    test_results = test_fused_index_select_eq()
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/fused_index_select_eq.py", line 66, in test_fused_index_select_eq
    results["test_case_1"] = fused_index_select_eq(input_tensor, dim, index, other)
                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/fused_index_select_eq.py", line 27, in fused_index_select_eq
    assert isinstance(index, (torch.IntTensor, torch.LongTensor)), "Index must be IntTensor or LongTensor"
AssertionError: Index must be IntTensor or LongTensor

=== Output for argmax.py on GPU 0 ===

=== Errors for argmax.py on GPU 0 ===
Traceback (most recent call last):
  File "/usr/local/lib/python3.12/site-packages/triton/language/core.py", line 35, in wrapper
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/site-packages/triton/language/math.py", line 26, in check
    raise ValueError(f"Expected dtype {dtypes} but got {arg.type.scalar.name}")
ValueError: Expected dtype ['fp32', 'fp64'] but got int64

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/data/results_constrained_partial/call_acc/argmax.py", line 92, in <module>
    test_results = test_argmax()
                   ^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/argmax.py", line 78, in test_argmax
    results["test_case_1"] = argmax(tensor_2d, dim=0)
                             ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/argmax.py", line 61, in argmax
    argmax_kernel[(n_rows,)](y, input_tensor, input_tensor.stride(0), y.stride(0),
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
triton.compiler.errors.CompilationError: at 11:20:
def argmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_rows, n_cols, BLOCK_SIZE: tl.constexpr, num_stages):
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        row_start_ptr = input_ptr + row_idx * input_row_stride
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
        row_minus_max = row - tl.max(row, axis=0)
        numerator = tl.exp(row_minus_max)
                    ^

=== Output for fused_lu_solve.py on GPU 0 ===

=== Errors for fused_lu_solve.py on GPU 0 ===
Traceback (most recent call last):
  File "/data/results_constrained_partial/call_acc/fused_lu_solve.py", line 90, in <module>
    test_results = test_fused_lu_solve()
                   ^^^^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/fused_lu_solve.py", line 68, in test_fused_lu_solve
    results["test_case_1"] = fused_lu_solve(A1, b1)
                             ^^^^^^^^^^^^^^
NameError: name 'fused_lu_solve' is not defined. Did you mean: 'test_fused_lu_solve'?

=== Output for normalize_pairwise_distance.py on GPU 0 ===

=== Errors for normalize_pairwise_distance.py on GPU 0 ===
Traceback (most recent call last):
  File "/data/results_constrained_partial/call_acc/normalize_pairwise_distance.py", line 90, in <module>
    test_results = test_normalize_pairwise_distance()
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/normalize_pairwise_distance.py", line 84, in test_normalize_pairwise_distance
    results["test_case_1"] = normalize_pairwise_distance(x1, x2, p_distance=2.0, dim_norm=0)
                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: normalize_pairwise_distance() got multiple values for argument 'p_distance'

=== Output for max.py on GPU 0 ===

=== Errors for max.py on GPU 0 ===
  File "/data/results_constrained_partial/call_acc/max.py", line 49
    else:
         ^
IndentationError: unindent does not match any outer indentation level

=== Output for log_softmax_linear.py on GPU 0 ===

=== Errors for log_softmax_linear.py on GPU 0 ===
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
  File "/data/results_constrained_partial/call_acc/log_softmax_linear.py", line 88, in <module>
    test_results = test_log_softmax_linear()
                   ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/log_softmax_linear.py", line 67, in test_log_softmax_linear
    results["test_case_1"] = log_softmax_linear(input1, weight1, bias1)
                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/log_softmax_linear.py", line 42, in log_softmax_linear
    log_softmax_linear_kernel[grid](input, weight, bias, out, M, N, K, dim, dtype,
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
triton.compiler.errors.CompilationError: at 12:38:
def log_softmax_linear_kernel(input_ptr, weight_ptr, bias_ptr, out_ptr, M, N, K, dim, dtype, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
                                      ^

=== Output for relu.py on GPU 0 ===

=== Errors for relu.py on GPU 0 ===
Traceback (most recent call last):
  File "/data/results_constrained_partial/call_acc/relu.py", line 66, in <module>
    test_results = test_relu()
                   ^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/relu.py", line 50, in test_relu
    results["test_case_1"] = relu(input1)
                             ^^^^^^^^^^^^
  File "/data/results_constrained_partial/call_acc/relu.py", line 27, in relu
    M, N = x.shape
    ^^^^
ValueError: not enough values to unpack (expected 2, got 1)

Deleted add.py
Deleted argmax.py
Deleted div.py
Deleted dropout_relu_batch_norm_conv2d.py
Deleted fused_bmm_rmsnorm_gelu_dropout_sub.py
Deleted fused_index_select_eq.py
Deleted fused_lu_solve.py
Deleted fused_mv_logsoftmax_dropout.py
Deleted fused_silu_layer_norm_conv2d.py
Deleted grid_sample.py
Deleted i0.py
Deleted log_softmax_linear.py
Deleted max.py
Deleted normalize_pairwise_distance.py
Deleted relu.py
Deleted relu_sqrt.py
Deleted rsqrt.py
Deleted sigmoid_argmax.py
Deleted sigmoid_conv2d.py
Deleted sqrt.py
Deleted svd.py
Deleted tanh.py

Correct execution rate: 8.33%
['solve_multiple_lu.py', 'sub.py']
Above is call test for predictions_qwen_constrained
================================================================================================================================================================

call_acc survivors: 2 / 24

======================================================================
=== Phase 2: execution accuracy ===
======================================================================

Correct execution rate: 100.00% = 2 / 2
above is the compare execution for /data/results_constrained_partial/call_acc
================================================================================================================================================================================================================================================

exe_acc survivors: 2 / 24

======================================================================
=== Phase 3: efficiency ===
======================================================================
Process:   0%|                                            | 0/2 [00:00<?, ?it/s]
✅ finished 1/2: solve_multiple_lu_perf.py
✅ finished 2/2: sub_perf.py
Process: 100%|████████████████████████████████████| 2/2 [00:37<00:00, 18.53s/it]