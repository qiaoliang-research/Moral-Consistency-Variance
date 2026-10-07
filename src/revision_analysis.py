from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from scipy import stats

from scenarios import build_prompt_variants, generate_scenarios


LABELS = ("A", "B")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate revision-oriented statistics for Moral Mimicry runs.")
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="JSON files produced by run_openai_mcv.py or run_local_mcv.py",
    )
    parser.add_argument(
        "--output-dir",
        default="results/reproduced/revision",
        help="Directory for revision artifacts",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Scenario generation seed used by the experiments",
    )
    parser.add_argument(
        "--audit-sample-size",
        type=int,
        default=20,
        help="How many scenario/variant pairs to export for manual semantic-equivalence audit",
    )
    return parser.parse_args()


def load_run(path: str | Path) -> Dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def kl_divergence(p: Dict[str, float], q: Dict[str, float], epsilon: float = 1e-12) -> float:
    score = 0.0
    for label in LABELS:
        p_value = max(p[label], epsilon)
        q_value = max(q[label], epsilon)
        score += p_value * math.log(p_value / q_value)
    return score


def jensen_shannon_divergence(p: Dict[str, float], q: Dict[str, float]) -> float:
    midpoint = {label: 0.5 * (p[label] + q[label]) for label in LABELS}
    return 0.5 * kl_divergence(p, midpoint) + 0.5 * kl_divergence(q, midpoint)


def total_variation_distance(p: Dict[str, float], q: Dict[str, float]) -> float:
    return 0.5 * sum(abs(p[label] - q[label]) for label in LABELS)


def bootstrap_mean_ci(values: Iterable[float], *, seed: int = 7, n_boot: int = 5000, alpha: float = 0.05) -> Tuple[float, float]:
    array = np.asarray(list(values), dtype=float)
    if len(array) == 0:
        return 0.0, 0.0
    if len(array) == 1:
        return float(array[0]), float(array[0])
    rng = np.random.default_rng(seed)
    samples = rng.choice(array, size=(n_boot, len(array)), replace=True)
    means = samples.mean(axis=1)
    lower = np.quantile(means, alpha / 2)
    upper = np.quantile(means, 1 - alpha / 2)
    return float(lower), float(upper)


def scenario_metric_rows(run: Dict[str, object]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for row in run["results"]:
        baseline = row["variants"]["baseline"]["distribution"]
        jsd_values = []
        tvd_values = []
        for variant_name, variant in row["variants"].items():
            if variant_name == "baseline":
                continue
            distribution = variant["distribution"]
            jsd_values.append(jensen_shannon_divergence(baseline, distribution))
            tvd_values.append(total_variation_distance(baseline, distribution))
        rows.append(
            {
                "scenario_id": row["scenario_id"],
                "theme": row["theme"],
                "context": row["context"],
                "mcv": row["mcv"],
                "jsd": float(np.mean(jsd_values)),
                "tvd": float(np.mean(tvd_values)),
                "decision_flip_rate": row["decision_flip_rate"],
                "variants": row["variants"],
            }
        )
    return rows


def summarize_run(run: Dict[str, object], *, seed: int) -> Dict[str, object]:
    rows = scenario_metric_rows(run)
    mcv_values = [row["mcv"] for row in rows]
    jsd_values = [row["jsd"] for row in rows]
    flip_values = [row["decision_flip_rate"] for row in rows]
    return {
        "model": run["metadata"]["model"],
        "scenario_count": len(rows),
        "mean_mcv": float(np.mean(mcv_values)),
        "std_mcv": float(np.std(mcv_values, ddof=1)) if len(mcv_values) > 1 else 0.0,
        "median_mcv": float(np.median(mcv_values)),
        "mcv_ci_low": bootstrap_mean_ci(mcv_values, seed=seed)[0],
        "mcv_ci_high": bootstrap_mean_ci(mcv_values, seed=seed)[1],
        "mean_jsd": float(np.mean(jsd_values)),
        "std_jsd": float(np.std(jsd_values, ddof=1)) if len(jsd_values) > 1 else 0.0,
        "jsd_ci_low": bootstrap_mean_ci(jsd_values, seed=seed + 101)[0],
        "jsd_ci_high": bootstrap_mean_ci(jsd_values, seed=seed + 101)[1],
        "mean_flip_rate": float(np.mean(flip_values)),
        "rows": rows,
    }


def pairwise_test(run_a: Dict[str, object], run_b: Dict[str, object], metric: str) -> Dict[str, object]:
    rows_a = {row["scenario_id"]: row for row in run_a["rows"]}
    rows_b = {row["scenario_id"]: row for row in run_b["rows"]}
    shared_ids = sorted(set(rows_a) & set(rows_b))
    values_a = np.asarray([rows_a[row_id][metric] for row_id in shared_ids], dtype=float)
    values_b = np.asarray([rows_b[row_id][metric] for row_id in shared_ids], dtype=float)
    deltas = values_a - values_b
    if len(shared_ids) > 0 and not np.allclose(deltas, 0.0):
        statistic, p_value = stats.wilcoxon(values_a, values_b, zero_method="wilcox", alternative="two-sided")
    else:
        statistic, p_value = 0.0, 1.0
    ci_low, ci_high = bootstrap_mean_ci(deltas, seed=17)
    return {
        "model_a": run_a["model"],
        "model_b": run_b["model"],
        "metric": metric,
        "shared_scenarios": len(shared_ids),
        "mean_a": float(np.mean(values_a)) if len(values_a) else 0.0,
        "mean_b": float(np.mean(values_b)) if len(values_b) else 0.0,
        "mean_delta_a_minus_b": float(np.mean(deltas)) if len(deltas) else 0.0,
        "delta_ci_low": ci_low,
        "delta_ci_high": ci_high,
        "wilcoxon_statistic": float(statistic),
        "wilcoxon_p": float(p_value),
    }


def write_model_summary(path: Path, summaries: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model",
                "scenario_count",
                "mean_mcv",
                "std_mcv",
                "median_mcv",
                "mcv_ci_low",
                "mcv_ci_high",
                "mean_jsd",
                "std_jsd",
                "jsd_ci_low",
                "jsd_ci_high",
                "mean_flip_rate",
            ],
        )
        writer.writeheader()
        for row in summaries:
            writer.writerow({key: row[key] for key in writer.fieldnames})


def write_pairwise_summary(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model_a",
                "model_b",
                "metric",
                "shared_scenarios",
                "mean_a",
                "mean_b",
                "mean_delta_a_minus_b",
                "delta_ci_low",
                "delta_ci_high",
                "wilcoxon_statistic",
                "wilcoxon_p",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_case_studies(path: Path, summaries: List[Dict[str, object]], *, seed: int) -> None:
    scenario_lookup = {scenario["scenario_id"]: scenario for scenario in generate_scenarios(limit=500, seed=seed)}
    lines = ["# High-MCV Case Studies", ""]
    for summary in summaries:
        model_name = summary["model"]
        top_rows = sorted(summary["rows"], key=lambda item: item["mcv"], reverse=True)[:3]
        lines.append(f"## {model_name}")
        lines.append("")
        for row in top_rows:
            scenario = scenario_lookup[row["scenario_id"]]
            prompts = build_prompt_variants(scenario)
            lines.append(f"### {row['scenario_id']}")
            lines.append(f"- Theme: `{row['theme']}`")
            lines.append(f"- MCV: `{row['mcv']:.6f}`")
            lines.append(f"- Mean JSD: `{row['jsd']:.6f}`")
            lines.append(f"- Flip rate: `{row['decision_flip_rate']:.2f}`")
            lines.append(f"- Context: {row['context']}")
            lines.append("")
            for variant_name in ("baseline", "audit", "review"):
                distribution = row["variants"][variant_name]["distribution"]
                lines.append(f"#### {variant_name}")
                lines.append("")
                lines.append("```text")
                lines.append(prompts[variant_name])
                lines.append("```")
                lines.append(
                    f"Distribution: A={distribution['A']:.4f}, B={distribution['B']:.4f}"
                )
                lines.append("")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_semantic_audit_sample(path: Path, *, seed: int, sample_size: int) -> None:
    rng = np.random.default_rng(seed)
    scenarios = generate_scenarios(limit=500, seed=seed)
    indices = rng.choice(len(scenarios), size=min(sample_size, len(scenarios)), replace=False)
    lines = [
        "# Semantic-Equivalence Audit Sample",
        "",
        "These pairs are intended for manual review. The scenario facts and action options are held fixed; only the introductory wrapper changes.",
        "",
    ]
    for sample_index, scenario_index in enumerate(sorted(indices), start=1):
        scenario = scenarios[scenario_index]
        prompts = build_prompt_variants(scenario)
        lines.append(f"## Sample {sample_index}: {scenario['scenario_id']}")
        lines.append("")
        lines.append(f"- Theme: `{scenario['theme']}`")
        lines.append(f"- Context: {scenario['context']}")
        lines.append(f"- Option A: {scenario['options']['A']}")
        lines.append(f"- Option B: {scenario['options']['B']}")
        lines.append("")
        for variant_name in ("baseline", "formal", "audit"):
            lines.append(f"### {variant_name}")
            lines.append("")
            lines.append("```text")
            lines.append(prompts[variant_name])
            lines.append("```")
            lines.append("")
        lines.append("Manual judgement: [ ] semantically equivalent  [ ] not equivalent")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_markdown_summary(path: Path, summaries: List[Dict[str, object]], pairwise_rows: List[Dict[str, object]]) -> None:
    lines = ["# Revision Summary", ""]
    lines.append("## Model-Level Results")
    lines.append("")
    for row in summaries:
        lines.append(f"### {row['model']}")
        lines.append(f"- Scenario count: {row['scenario_count']}")
        lines.append(f"- Mean MCV: {row['mean_mcv']:.6f} (SD {row['std_mcv']:.6f}; 95% CI [{row['mcv_ci_low']:.6f}, {row['mcv_ci_high']:.6f}])")
        lines.append(f"- Median MCV: {row['median_mcv']:.6f}")
        lines.append(f"- Mean JSD: {row['mean_jsd']:.6f} (SD {row['std_jsd']:.6f}; 95% CI [{row['jsd_ci_low']:.6f}, {row['jsd_ci_high']:.6f}])")
        lines.append(f"- Mean flip rate: {row['mean_flip_rate']:.6f}")
        lines.append("")
    lines.append("## Pairwise Tests")
    lines.append("")
    for row in pairwise_rows:
        lines.append(f"- {row['metric']}: {row['model_a']} vs {row['model_b']} | Δ={row['mean_delta_a_minus_b']:.6f} | 95% CI [{row['delta_ci_low']:.6f}, {row['delta_ci_high']:.6f}] | Wilcoxon p={row['wilcoxon_p']:.6g}")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_runs = [load_run(path) for path in args.runs]
    summaries = [summarize_run(run, seed=args.seed) for run in raw_runs]

    pairwise_rows: List[Dict[str, object]] = []
    for index, run_a in enumerate(summaries):
        for run_b in summaries[index + 1 :]:
            pairwise_rows.append(pairwise_test(run_a, run_b, "mcv"))
            pairwise_rows.append(pairwise_test(run_a, run_b, "jsd"))

    write_model_summary(output_dir / "model_revision_summary.csv", summaries)
    write_pairwise_summary(output_dir / "pairwise_tests.csv", pairwise_rows)
    write_markdown_summary(output_dir / "revision_summary.md", summaries, pairwise_rows)
    write_case_studies(output_dir / "case_studies.md", summaries, seed=args.seed)
    write_semantic_audit_sample(
        output_dir / "semantic_audit_sample.md",
        seed=args.seed,
        sample_size=args.audit_sample_size,
    )

    print(f"Wrote revision artifacts to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
