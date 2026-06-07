from __future__ import annotations

import json
import sys
from pathlib import Path

from benchmarks.run_generated_kernels import _evaluate_row_in_process, _load_stack


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: python -m benchmarks.kernel_worker REQUEST_JSON RESPONSE_JSON", file=sys.stderr)
        return 2

    request_path = Path(sys.argv[1])
    response_path = Path(sys.argv[2])
    with request_path.open("r", encoding="utf-8") as file:
        request = json.load(file)

    stack, reason = _load_stack()
    if stack is None:
        print(reason, file=sys.stderr)
        return 3

    enriched, diagnostic = _evaluate_row_in_process(
        request["row"],
        int(request["row_number"]),
        stack,
        num_elements=int(request["num_elements"]),
        warmup=int(request["warmup"]),
        repeats=int(request["repeats"]),
        launch_policy=str(request["launch_policy"]),
        fixed_block_size=int(request["fixed_block_size"]),
    )
    with response_path.open("w", encoding="utf-8") as file:
        json.dump(
            {"enriched": enriched, "diagnostic": diagnostic},
            file,
            ensure_ascii=False,
        )
        file.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
