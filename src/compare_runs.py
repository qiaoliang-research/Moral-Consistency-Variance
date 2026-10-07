from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


THEME_ORDER = [
    "truth_vs_harm",
    "privacy_vs_safety",
    "loyalty_vs_fairness",
    "rules_vs_rescue",
    "autonomy_vs_beneficence",
    "scarcity_allocation",
    "promise_vs_public_good",
    "transparency_vs_stability",
    "fairness_vs_efficiency",
    "consent_vs_speed",
]

DISPLAY_LABELS = {
    "grok-4-fast-non-reasoning": "Grok 4",
    "grok-4-1-fast-non-reasoning": "Grok 4.1",
    "HuggingFaceTB/SmolLM2-360M-Instruct": "SmolLM2-360M",
    "gemini-2.5-flash": "Gemini 2.5 Flash",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare multiple MCV experiment runs and generate plots.")
    parser.add_argument("--runs", nargs="+", required=True, help="JSON files produced by run_openai_mcv.py")
    parser.add_argument("--output-dir", default="results/reproduced/comparison", help="Directory for summary artifacts")
    parser.add_argument(
        "--presentation-labels",
        action="store_true",
        help="Use concise display labels and a wider layout for presentation figures",
    )
    return parser.parse_args()


def load_run(path: str | Path) -> Dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload


def short_label(model_name: str, presentation_labels: bool = False) -> str:
    if presentation_labels and model_name in DISPLAY_LABELS:
        return DISPLAY_LABELS[model_name]
    return model_name.replace("-non-reasoning", "").replace("grok-", "grok ")


def write_summary_csv(runs: List[Dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["model", "scenario_count", "mean_mcv", "median_mcv", "mean_flip_rate", "fallback_rate"],
        )
        writer.writeheader()
        for run in runs:
            summary = run["summary"]
            writer.writerow(
                {
                    "model": run["metadata"]["model"],
                    "scenario_count": summary["scenario_count"],
                    "mean_mcv": f"{summary['mean_mcv']:.12f}",
                    "median_mcv": f"{summary['median_mcv']:.12f}",
                    "mean_flip_rate": f"{summary['mean_flip_rate']:.12f}",
                    "fallback_rate": f"{summary['fallback_rate']:.12f}",
                }
            )


def write_theme_csv(runs: List[Dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "theme", "scenario_count", "mean_mcv", "mean_flip_rate"])
        writer.writeheader()
        for run in runs:
            model = run["metadata"]["model"]
            for theme in THEME_ORDER:
                row = run["summary"]["theme_summary"][theme]
                writer.writerow(
                    {
                        "model": model,
                        "theme": theme,
                        "scenario_count": row["scenario_count"],
                        "mean_mcv": f"{row['mean_mcv']:.12f}",
                        "mean_flip_rate": f"{row['mean_flip_rate']:.12f}",
                    }
                )


def make_boxplot(
    runs: List[Dict[str, object]], output_path: Path, presentation_labels: bool = False
) -> None:
    labels = [short_label(run["metadata"]["model"], presentation_labels) for run in runs]
    series = []
    for run in runs:
        values = [max(item["mcv"], 1e-9) for item in run["results"]]
        series.append(values)

    figsize = (10.5, 4.8) if presentation_labels else (7.0, 4.2)
    fig, ax = plt.subplots(figsize=figsize)
    box = ax.boxplot(series, patch_artist=True, tick_labels=labels)
    colors = ["#1f5aa6", "#c95f1d", "#3f8f5c", "#7b4fa3"]
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)

    ax.set_yscale("log")
    ax.set_ylabel("Scenario-level MCV (log scale)")
    ax.set_title("Distribution of Moral Consistency Variance by model")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    if presentation_labels:
        ax.tick_params(axis="x", labelsize=10, pad=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def make_theme_plot(
    runs: List[Dict[str, object]], output_path: Path, presentation_labels: bool = False
) -> None:
    labels = [short_label(run["metadata"]["model"], presentation_labels) for run in runs]
    x = np.arange(len(THEME_ORDER))
    width = 0.36 if len(runs) == 2 else 0.25

    figsize = (12.5, 5.6) if presentation_labels else (9.5, 4.8)
    fig, ax = plt.subplots(figsize=figsize)
    colors = ["#1f5aa6", "#c95f1d", "#3f8f5c", "#7b4fa3"]
    for index, run in enumerate(runs):
        values = [
            max(run["summary"]["theme_summary"][theme]["mean_mcv"], 1e-9)
            for theme in THEME_ORDER
        ]
        offset = (index - (len(runs) - 1) / 2) * width
        ax.bar(x + offset, values, width=width, label=labels[index], color=colors[index], alpha=0.8)

    ax.set_yscale("log")
    ax.set_ylabel("Theme mean MCV (log scale)")
    if presentation_labels:
        ax.set_title("Theme-level distributional sensitivity", pad=36)
    else:
        ax.set_title("Theme-level distributional sensitivity")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [
            "truth/\nharm",
            "privacy/\nsafety",
            "loyalty/\nfairness",
            "rules/\nrescue",
            "autonomy/\nbeneficence",
            "scarcity",
            "promise/\npublic",
            "transparency/\nstability",
            "fairness/\nefficiency",
            "consent/\nspeed",
        ],
        fontsize=9,
    )
    if presentation_labels:
        ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.03), ncol=4)
        fig.subplots_adjust(top=0.78, bottom=0.22, left=0.07, right=0.99)
    else:
        ax.legend(frameon=False)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    if not presentation_labels:
        fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def write_markdown_summary(runs: List[Dict[str, object]], output_path: Path) -> None:
    lines = ["# Run comparison", ""]
    for run in runs:
        model = run["metadata"]["model"]
        summary = run["summary"]
        lines.append(f"## {model}")
        lines.append(f"- Scenario count: {summary['scenario_count']}")
        lines.append(f"- Mean MCV: {summary['mean_mcv']:.6f}")
        lines.append(f"- Median MCV: {summary['median_mcv']:.6f}")
        lines.append(f"- Mean flip rate: {summary['mean_flip_rate']:.6f}")
        lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    runs = [load_run(path) for path in args.runs]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    write_summary_csv(runs, output_dir / "model_summary.csv")
    write_theme_csv(runs, output_dir / "theme_summary.csv")
    write_markdown_summary(runs, output_dir / "summary.md")
    make_boxplot(runs, output_dir / "mcv_boxplot.png", args.presentation_labels)
    make_theme_plot(runs, output_dir / "theme_mcv.png", args.presentation_labels)
    print(f"Wrote comparison artifacts to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
