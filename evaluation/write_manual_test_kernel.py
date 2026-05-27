import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_PATH = BASE_DIR / "artifacts" / "manual_test_kernel.jsonl"

valid_kernel_code = """
import triton
import triton.language as tl

@triton.jit
def add_vectors(x_ptr, y_ptr, z_ptr, n_elements, BLOCK_SIZE: tl.constexpr = 256):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(z_ptr + offsets, output, mask=mask)
"""

test_data = [
    {
        "task_id": "vector_add",
        "sample_index": 0,
        "extracted_code": valid_kernel_code,
    }
]

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with OUTPUT_PATH.open("w", encoding="utf-8") as f:
    for entry in test_data:
        f.write(json.dumps(entry) + "\n")

print(f"Manual test file updated with BLOCK_SIZE: tl.constexpr at {OUTPUT_PATH}")
