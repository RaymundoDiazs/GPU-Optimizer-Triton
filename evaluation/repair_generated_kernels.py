import json
import re
import os
import pandas as pd
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def repair_code(code):
    if not code:
        return code
    # 1. Fix missing colons in def statements
    code = re.sub(r'(def\s+\w+\(.*?\))(?![\s:]|$)', r'\1:', code)
    # 2. Force function name to add_vectors
    code = re.sub(r'def\s+vector_add_kernel', 'def add_vectors', code)
    # 3. Add default for BLOCK_SIZE if it's in the signature but missing a default
    code = re.sub(r'BLOCK_SIZE:\s*tl\.constexpr(?![\s=])', 'BLOCK_SIZE: tl.constexpr = 1024', code)
    # 4. Ensure N is constexpr if used in arange(0, N)
    if 'tl.arange(0, N)' in code and 'N: tl.constexpr' not in code:
        code = code.replace('N)', 'N: tl.constexpr)')
    return code


def main() -> None:
    base_dir = REPO_ROOT
    input_file = base_dir / 'evaluation' / 'artifacts' / 'generated_kernels.jsonl'
    repaired_file = base_dir / 'artifacts' / 'repaired_kernels_v2.jsonl'
    results_file = base_dir / 'artifacts' / 'benchmark_results_repaired_v2.jsonl'

    os.makedirs(base_dir / 'artifacts', exist_ok=True)

    with input_file.open('r', encoding='utf-8') as f_in, repaired_file.open('w', encoding='utf-8') as f_out:
        for line in f_in:
            data = json.loads(line)
            data['extracted_code'] = repair_code(data.get('extracted_code', ''))
            f_out.write(json.dumps(data) + '\n')

    print('Running benchmark on v2 repaired set (with default BLOCK_SIZE and constexpr fixes)...')
    # Placeholder for benchmark integration.
    print(f'Repaired kernels written to: {repaired_file}')
    print(f'Benchmark results should be written to: {results_file}')


if __name__ == '__main__':
    main()
