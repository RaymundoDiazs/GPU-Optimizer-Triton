Stopping app - local entrypoint completed.
✓ App completed. View run at https://modal.com/apps/deoh02/main/ap-zDfqgKQWcDswBDLokP3Mxe
PS C:\Users\Polou\OneDrive\Desktop\Projects\GPU-Optimizer-Triton\extras\TritonBench4Modal-main> cd "C:/Users/Polou/OneDrive/Desktop/Projects/GPU-Optimizer-Triton/extras/TritonBench4Modal-main"; modal run 
modal_app.py::evaluate_only --predictions "../../evaluation/predictions_qwen_constrained.jsonl" --output-subdir results_constrained_partial
al;a2da899d-3b14-44cd-978e-3abad764bb29C:\Users\Polou\AppData\Local\Programs\Python\Python314\Lib\site-packages\modal\_utils\async_utils.py:4
5: DeprecationWarning: 'asyncio.WindowsSelectorEventLoopPolicy' is deprecated and slated for removal in Python 3.16
  asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
C:\Users\Polou\AppData\Local\Programs\Python\Python314\Lib\site-packages\modal\_utils\async_utils.py:45: DeprecationWarning: 'asyncio.set_event_loop_policy' is deprecated and slated for removal in Python 
3.16
  asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
✓ Initialized. View run at https://modal.com/apps/deoh02/main/ap-BMf99Pvc7xxoL08HNwM0fy
✓ Created objects.
├── 🔨 Created mount
│   C:\Users\Polou\OneDrive\Desktop\Projects\GPU-Optimizer-Triton\extras\TritonBench4Modal-main\modal_│   app.py
├── 🔨 Created function generate_predictions.
└── 🔨 Created function evaluate.
uploading ..\..\evaluation\predictions_qwen_constrained.jsonl ->
volume://uploads/predictions_qwen_constrained.jsonl

==========
== CUDA ==
==========

CUDA Version 12.4.1

Container image Copyright (c) 2016-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.        

This container image and its contents are governed by the NVIDIA Deep Learning Container License.     
By pulling and using the container, you accept the terms and conditions of this license:
https://developer.nvidia.com/ngc/nvidia-deep-learning-container-license

A copy of this license is made available in this container at /NGC-DL-CONTAINER-LICENSE for your convenience.


======================================================================
=== Phase 1: call accuracy ===
======================================================================
Traceback (most recent call last):
  File "/pkg/modal/_runtime/container_io_manager.py", line 945, in handle_input_exception
    yield
  File "/pkg/modal/_container_entrypoint.py", line 189, in run_input_sync
    values = io_context.call_function_sync()
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/pkg/modal/_runtime/container_io_manager.py", line 225, in call_function_sync
    expected_value_or_values = self.finalized_function.callable(*args, **kwargs)
                               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/modal_app.py", line 391, in evaluate
    call_acc.call_4file(str(pred_full), str(call_acc_dir), gpus=[0])
  File "/opt/TritonBench/EVAL/eval_T/call_acc.py", line 167, in call_4file
    pred, test, files = get_codes_for_test(path)
                        ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/TritonBench/EVAL/eval_T/call_acc.py", line 67, in get_codes_for_test
    data = [json.loads(line) for line in open(path, 'r', encoding='utf-8').readlines()]
            ^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/json/__init__.py", line 346, in loads
    return _default_decoder.decode(s)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/json/decoder.py", line 337, in decode
    obj, end = self.raw_decode(s, idx=_w(s, 0).end())
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/json/decoder.py", line 355, in raw_decode
    raise JSONDecodeError("Expecting value", s, err.value) from None
json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)
Stopping app - uncaught exception raised locally: JSONDecodeError('Expecting value: line 1 column 1 (char 0)').
╭─────────────────────────────── Traceback (most recent call last) ────────────────────────────────╮
│ C:\Users\Polou\OneDrive\Desktop\Projects\GPU-Optimizer-Triton\extras\TritonBench4Modal-main\moda │  
│ l_app.py:541 in evaluate_only                                                                    │  
│                                                                                                  │  
│   540 │   remote = _upload_local_predictions(Path(predictions))                                  │  
│ ❱ 541 │   summary = evaluate.remote(predictions_path=remote, output_subdir=output_subdir)        │  
│   542 │   print(json.dumps(summary, indent=2))                                                   │  
│                                                                                                  │  
│ C:\Users\Polou\AppData\Local\Programs\Python\Python314\Lib\site-packages\modal\_object.py:46 in  │  
│ wrapped                                                                                          │  
│                                                                                                  │  
│    45 │   │   await self.hydrate()                                                               │  
│ ❱  46 │   │   return await method(self, *args, **kwargs)                                         │  
│    47                                                                                            │  
│                                                                                                  │  
│ C:\Users\Polou\AppData\Local\Programs\Python\Python314\Lib\site-packages\modal\_functions.py:169 │  
│ 6 in remote                                                                                      │  
│                                                                                                  │  
│   1695 │   │                                                                                     │  
│ ❱ 1696 │   │   return await self._call_function(args, kwargs)                                    │  
│   1697                                                                                           │  
│                                                                                                  │  
│ C:\Users\Polou\AppData\Local\Programs\Python\Python314\Lib\site-packages\modal\_functions.py:164 │  
│ 0 in _call_function                                                                              │  
│                                                                                                  │  
│   1639 │   │                                                                                     │  
│ ❱ 1640 │   │   return await invocation.run_function()                                            │  
│   1641                                                                                           │  
│                                                                                                  │  
│ C:\Users\Polou\AppData\Local\Programs\Python\Python314\Lib\site-packages\modal\_functions.py:290 │  
│ in run_function                                                                                  │  
│                                                                                                  │  
│    289 │   │   │   item = await self._get_single_output()                                        │  
│ ❱  290 │   │   │   return await _process_result(item.result, item.data_format, self.stub, self.  │  
│    291                                                                                           │  
│                                                                                                  │  
│ C:\Users\Polou\AppData\Local\Programs\Python\Python314\Lib\site-packages\modal\_utils\function_u │  
│ tils.py:548 in _process_result                                                                   │  
│                                                                                                  │  
│   547 │   │   │                                                                                  │  
│ ❱ 548 │   │   │   raise exc_with_hints(exc)                                                      │  
│   549                                                                                            │  
│                                                                                                  │  
│               ...Remote call to Modal Function (ta-01KTNHX0NHNG4FAYR6H2QSG4QA)...                │  
│                                                                                                  │  
│ /root/modal_app.py:391 in evaluate                                                               │  
│                                                                                                  │  
│ ❱ 391 call_acc.call_4file(str(pred_full), str(call_acc_dir), gpus=[0])                           │  
│                                                                                                  │  
│                                                                                                  │  
│ /opt/TritonBench/EVAL/eval_T/call_acc.py:167 in call_4file                                       │  
│                                                                                                  │  
│ ❱ 167 pred, test, files = get_codes_for_test(path)                                               │  
│                                                                                                  │  
│                                                                                                  │  
│ /opt/TritonBench/EVAL/eval_T/call_acc.py:67 in get_codes_for_test                                │  
│                                                                                                  │  
│ ❱ 67 data = [json.loads(line) for line in open(path, 'r', encoding='utf-8').readlines()]         │  
│                                                                                                  │  
│                                                                                                  │  
│ /usr/local/lib/python3.12/json/__init__.py:346 in loads                                          │  
│                                                                                                  │  
│ ❱ 346 return _default_decoder.decode(s)                                                          │
│                                                                                                  │  
│                                                                                                  │  
│ /usr/local/lib/python3.12/json/decoder.py:337 in decode                                          │  
│                                                                                                  │  
│ ❱ 337 obj, end = self.raw_decode(s, idx=_w(s, 0).end())                                          │  
│                                                                                                  │  
│                                                                                                  │  
│ /usr/local/lib/python3.12/json/decoder.py:355 in raw_decode                                      │  
│                                                                                                  │  
│ ❱ 355 raise JSONDecodeError("Expecting value", s, err.value) from None                           │  
│                                                                                                  │  
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯  
JSONDecodeError: Expecting value: line 1 column 1 (char 0)
PS C:\Users\Polou\OneDrive\Desktop\Projects\GPU-Optimizer-Triton\extras\TritonBench4Modal-main> 