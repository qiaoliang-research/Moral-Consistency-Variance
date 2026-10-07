from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize MCV experiment results.")
    parser.add_argument("input", help="Path to the JSON file produced by run_openai_mcv.py")
    parser.add_argument("--csv", help="Optional CSV path for scenario-level metrics.")
    return parser.parse_args()


def load_payload(path: str | Path) -> Dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(path: str | Path, results: List[Dict[str, object]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["scenario_id", "theme", "mcv", "decision_flip_rate", "context"],
        )
        writer.writeheader()
        for row in results:
            writer.writerow(
                {
                    "scenario_id": row["scenario_id"],
                    "theme": row["theme"],
                    "mcv": f"{row['mcv']:.6f}",
                    "decision_flip_rate": f"{row['decision_flip_rate']:.6f}",
                    "context": row["context"],
                }
            )


def main() -> int:
    args = parse_args()
    payload = load_payload(args.input)
    metadata = payload["metadata"]
    summary = payload["summary"]
    results = payload["results"]

    print(f"Model: {metadata['model']}")
    print(f"Scenario count: {summary['scenario_count']}")
    print(f"Mean MCV: {summary['mean_mcv']:.6f}")
    print(f"Median MCV: {summary['median_mcv']:.6f}")
    print(f"Mean flip rate: {summary['mean_flip_rate']:.6f}")
    print(f"Fallback rate: {summary['fallback_rate']:.6f}")
    print("\nPer-theme summary:")
    for theme, row in sorted(summary["theme_summary"].items()):
        print(
            f"- {theme}: scenarios={row['scenario_count']}, "
            f"mean_mcv={row['mean_mcv']:.6f}, "
            f"mean_flip_rate={row['mean_flip_rate']:.6f}"
        )

    top_rows = sorted(results, key=lambda item: item["mcv"], reverse=True)[:5]
    print("\nMost unstable scenarios:")
    for row in top_rows:
        print(f"- {row['scenario_id']}: MCV={row['mcv']:.6f}, flip_rate={row['decision_flip_rate']:.6f}")

    if args.csv:
        write_csv(args.csv, results)
        print(f"\nWrote CSV: {args.csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
