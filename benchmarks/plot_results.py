import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_results(
    input_path: Path | None = None,
    output_path: Path | None = None,
) -> Path:
    """Create a summary plot of generation time by problem type."""
    source = input_path or Path(__file__).resolve().parent / "results.csv"
    destination = output_path or Path(__file__).resolve().parent / "results_summary.png"

    frame = pd.read_csv(source)
    summary = frame.groupby("problem_type", as_index=True)["generation_time"].mean().sort_index()

    ax = summary.plot(kind="bar", color="#2f6f6d", figsize=(8, 4))
    ax.set_title("Average kernel generation time by problem type")
    ax.set_xlabel("Problem type")
    ax.set_ylabel("Generation time (s)")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(destination)
    plt.close()
    print(f"Wrote plot to {destination}")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot benchmark results.")
    parser.add_argument("--input", type=Path, default=None, help="Optional benchmark CSV path.")
    parser.add_argument("--output", type=Path, default=None, help="Optional PNG output path.")
    args = parser.parse_args()
    plot_results(input_path=args.input, output_path=args.output)


if __name__ == "__main__":
    main()
