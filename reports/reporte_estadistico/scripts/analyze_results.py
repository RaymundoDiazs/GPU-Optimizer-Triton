from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
REPORT_DIR = ROOT / "reports" / "reporte_estadistico"
SOURCE = ROOT / "evaluation" / "artifacts" / "model_eval_results.csv"
RESULTS_DIR = REPORT_DIR / "results"
FIGURES_DIR = REPORT_DIR / "figures"


BOOLEAN_METRICS = ["syntax_valid", "kernel_shape_valid", "correctness_proxy"]


def _to_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = successes / n
    denominator = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denominator
    margin = z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def two_proportion_z_test(success_a: int, n_a: int, success_b: int, n_b: int) -> dict[str, float | str]:
    if n_a == 0 or n_b == 0:
        return {"z": float("nan"), "p_value": float("nan"), "note": "empty group"}

    p_a = success_a / n_a
    p_b = success_b / n_b
    pooled = (success_a + success_b) / (n_a + n_b)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n_a + 1 / n_b))
    if se == 0:
        return {"z": float("nan"), "p_value": float("nan"), "note": "zero standard error"}

    z = (p_b - p_a) / se
    p_value = math.erfc(abs(z) / math.sqrt(2))
    return {"z": z, "p_value": p_value, "note": "normal approximation; use cautiously for small n"}


def cohen_h(rate_a: float, rate_b: float) -> float:
    """Effect size for two proportions."""
    rate_a = min(max(rate_a, 0.0), 1.0)
    rate_b = min(max(rate_b, 0.0), 1.0)
    return 2 * math.asin(math.sqrt(rate_b)) - 2 * math.asin(math.sqrt(rate_a))


def holm_bonferroni(p_values: list[float], alpha: float = 0.05) -> list[dict[str, float | bool]]:
    valid = [(index, p) for index, p in enumerate(p_values) if not math.isnan(p)]
    ordered = sorted(valid, key=lambda item: item[1])
    adjusted: list[dict[str, float | bool]] = [
        {"holm_threshold": float("nan"), "holm_reject": False, "holm_rank": float("nan")}
        for _ in p_values
    ]
    for rank, (original_index, p_value) in enumerate(ordered, start=1):
        threshold = alpha / (len(ordered) - rank + 1)
        adjusted[original_index] = {
            "holm_threshold": threshold,
            "holm_reject": p_value <= threshold,
            "holm_rank": rank,
        }
    return adjusted


def load_results(path: Path = SOURCE) -> pd.DataFrame:
    frame = pd.read_csv(path)
    for metric in BOOLEAN_METRICS:
        frame[metric] = _to_bool(frame[metric])
    return frame


def build_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    grouped = frame.groupby(["model_id", "model_display_name", "tier", "provider", "mode"], dropna=False)
    for keys, group in grouped:
        model_id, display_name, tier, provider, mode = keys
        row = {
            "model_id": model_id,
            "model_display_name": display_name,
            "tier": tier,
            "provider": provider,
            "mode": mode,
            "n": len(group),
            "latency_mean": group["latency_seconds"].mean(),
            "latency_std": group["latency_seconds"].std(ddof=1) if len(group) > 1 else 0.0,
            "latency_min": group["latency_seconds"].min(),
            "latency_max": group["latency_seconds"].max(),
        }
        for metric in BOOLEAN_METRICS:
            successes = int(group[metric].sum())
            lower, upper = wilson_interval(successes, len(group))
            row[f"{metric}_successes"] = successes
            row[f"{metric}_rate"] = successes / len(group)
            row[f"{metric}_ci95_low"] = lower
            row[f"{metric}_ci95_high"] = upper
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["model_id", "mode"])


def build_pairwise_tests(frame: pd.DataFrame) -> pd.DataFrame:
    comparisons = [
        ("small_qwen25_coder_1_5b", "baseline", "small_qwen25_coder_1_5b", "constrained", "kernel_shape_valid"),
        ("small_qwen25_coder_1_5b", "constrained", "frontier_openai", "constrained", "correctness_proxy"),
        ("small_qwen25_coder_1_5b", "constrained", "frontier_anthropic", "constrained", "correctness_proxy"),
        ("frontier_openai", "baseline", "frontier_openai", "constrained", "correctness_proxy"),
        ("frontier_anthropic", "baseline", "frontier_anthropic", "constrained", "correctness_proxy"),
    ]
    rows = []
    for model_a, mode_a, model_b, mode_b, metric in comparisons:
        group_a = frame[(frame["model_id"] == model_a) & (frame["mode"] == mode_a)]
        group_b = frame[(frame["model_id"] == model_b) & (frame["mode"] == mode_b)]
        success_a = int(group_a[metric].sum())
        success_b = int(group_b[metric].sum())
        n_a = len(group_a)
        n_b = len(group_b)
        stats = two_proportion_z_test(success_a, n_a, success_b, n_b)
        rate_a = success_a / n_a if n_a else float("nan")
        rate_b = success_b / n_b if n_b else float("nan")
        rows.append(
            {
                "comparison": f"{model_a}:{mode_a} vs {model_b}:{mode_b}",
                "metric": metric,
                "success_a": success_a,
                "n_a": n_a,
                "rate_a": rate_a,
                "success_b": success_b,
                "n_b": n_b,
                "rate_b": rate_b,
                "difference_b_minus_a": rate_b - rate_a,
                "effect_size_cohen_h": cohen_h(rate_a, rate_b),
                **stats,
            }
        )
    frame = pd.DataFrame(rows)
    corrections = holm_bonferroni(frame["p_value"].tolist())
    for column in ["holm_threshold", "holm_reject", "holm_rank"]:
        frame[column] = [item[column] for item in corrections]
    return frame


def plot_metric_rates(summary: pd.DataFrame, metric: str, output: Path) -> None:
    labels = summary["model_id"] + "\n" + summary["mode"]
    values = summary[f"{metric}_rate"]

    plt.figure(figsize=(10, 4.8))
    plt.bar(labels, values, color="#2f6f6d")
    plt.ylim(0, 1.05)
    plt.ylabel("Tasa")
    plt.title(metric.replace("_", " ").title())
    plt.xticks(rotation=35, ha="right")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output)
    plt.close()


def plot_latency(frame: pd.DataFrame, output: Path) -> None:
    frame = frame.copy()
    frame["label"] = frame["model_id"] + "\n" + frame["mode"]
    labels = list(dict.fromkeys(frame["label"].tolist()))
    data = [frame.loc[frame["label"] == label, "latency_seconds"].tolist() for label in labels]

    plt.figure(figsize=(10, 4.8))
    plt.boxplot(data, tick_labels=labels, showmeans=True)
    plt.ylabel("Latencia de generacion (s)")
    plt.title("Latencia por modelo y modo")
    plt.xticks(rotation=35, ha="right")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output)
    plt.close()


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    frame = load_results()
    summary = build_summary(frame)
    tests = build_pairwise_tests(frame)

    summary.to_csv(RESULTS_DIR / "summary_by_model_mode.csv", index=False)
    tests.to_csv(RESULTS_DIR / "pairwise_proportion_tests.csv", index=False)
    build_interpretation_notes(summary, tests).to_csv(
        RESULTS_DIR / "interpretation_notes.csv",
        index=False,
    )

    for metric in BOOLEAN_METRICS:
        plot_metric_rates(summary, metric, FIGURES_DIR / f"{metric}_rate.png")
    plot_latency(frame, FIGURES_DIR / "latency_by_model_mode.png")

    print(f"Analisis generado en {REPORT_DIR}")
    print(f"Filas analizadas: {len(frame)}")
    print("Nota: los resultados actuales son preliminares por n pequeno.")


def build_interpretation_notes(summary: pd.DataFrame, tests: pd.DataFrame) -> pd.DataFrame:
    notes = [
        {
            "tema": "tamano_muestra",
            "interpretacion": "El n actual por grupo es 3; no debe usarse para conclusiones inferenciales fuertes.",
        },
        {
            "tema": "intervalos_confianza",
            "interpretacion": "Los intervalos Wilson son amplios por el n pequeno; esto refleja alta incertidumbre.",
        },
        {
            "tema": "qwen_shape",
            "interpretacion": "Qwen mejora en kernel_shape_valid bajo constrained, pero el resultado es preliminar.",
        },
        {
            "tema": "constructo",
            "interpretacion": "correctness_proxy no equivale a ejecucion real PyTorch-vs-Triton.",
        },
        {
            "tema": "proximo_paso",
            "interpretacion": "El reporte final debe reemplazar o complementar proxies con compile_success, equivalence_rate y speedup.",
        },
    ]
    return pd.DataFrame(notes)


if __name__ == "__main__":
    main()
