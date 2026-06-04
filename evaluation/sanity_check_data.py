"""
Paso 1 — Sanity check del cruce de datos.
Verifica que TritonBench_T_simp_alpac_v1.json y TritonBench_T_v1.jsonl
tienen 166 entradas alineadas por posición.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
SIMP_PATH = ROOT / "extras/TritonBench-main/data/TritonBench_T_simp_alpac_v1.json"
T_PATH    = ROOT / "extras/TritonBench-main/data/TritonBench_T_v1.jsonl"


def run_sanity_check():
    with open(SIMP_PATH, encoding="utf-8") as f:
        simp_data = json.load(f)
    with open(T_PATH, encoding="utf-8") as f:
        t_data = json.load(f)

    print(f"simp_alpac entries: {len(simp_data)}")
    print(f"T_v1 entries:       {len(t_data)}")

    if len(simp_data) != 166:
        print(f"ERROR: simp_alpac tiene {len(simp_data)} entradas, esperaba 166")
        sys.exit(1)
    if len(t_data) != 166:
        print(f"ERROR: T_v1 tiene {len(t_data)} entradas, esperaba 166")
        sys.exit(1)

    mismatches = []
    for i, (s, t) in enumerate(zip(simp_data, t_data)):
        fname      = t["file"].replace(".py", "")
        short_name = t["name"].split(".")[-1]
        desc_match = t["description"][:40] in s["instruction"]
        ok = fname in s["instruction"] or short_name in s["instruction"] or desc_match
        if not ok:
            mismatches.append(
                f"  [{i}] name={t['name']}, file={t['file']}\n"
                f"       instruction[:80]={s['instruction'][:80]!r}"
            )

    if mismatches:
        print(f"ERROR: {len(mismatches)} mismatches encontrados:")
        for m in mismatches:
            print(m)
        sys.exit(1)

    print("OK — 166 operadores alineados por posición")
    return simp_data, t_data


if __name__ == "__main__":
    run_sanity_check()
