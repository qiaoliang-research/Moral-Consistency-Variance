from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import requests
from google.auth import default as google_auth_default
from google.auth.transport.requests import Request as GoogleAuthRequest

from run_openai_mcv import (
    SYSTEM_PROMPT,
    kl_divergence,
    normalize_label_token,
    persist_snapshot,
    summarize_results,
)
from scenarios import build_prompt_variants, export_dataset, generate_scenarios


class VertexAccessTokenProvider:
    def __init__(self) -> None:
        self.credentials, self.project_id = google_auth_default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        self.request = GoogleAuthRequest()

    def get_token(self) -> str:
        if not self.credentials.valid or not self.credentials.token:
            self.credentials.refresh(self.request)
        return self.credentials.token


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Moral Consistency Variance experiments against Gemini on Vertex AI."
    )
    parser.add_argument("--model", default="gemini-2.5-flash", help="Vertex AI Gemini model ID.")
    parser.add_argument(
        "--project",
        default=os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GCLOUD_PROJECT"),
        help="Google Cloud project ID. Defaults to GOOGLE_CLOUD_PROJECT.",
    )
    parser.add_argument(
        "--location",
        default=os.getenv("GOOGLE_CLOUD_LOCATION", "global"),
        help="Vertex AI location. Use global unless you need a regional endpoint.",
    )
    parser.add_argument("--scenario-count", type=int, default=20, help="Number of generated scenarios to evaluate.")
    parser.add_argument("--seed", type=int, default=7, help="Dataset generation seed.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--top-logprobs", type=int, default=20, help="How many candidate token logprobs to request.")
    parser.add_argument("--pause-sec", type=float, default=0.2, help="Sleep duration between calls.")
    parser.add_argument("--output", default="results/reproduced/gemini_run.json", help="Where to write the experiment JSON.")
    parser.add_argument(
        "--dataset-output",
        default="results/reproduced/gemini_dataset.jsonl",
        help="Where to export the generated dataset.",
    )
    parser.add_argument("--max-retries", type=int, default=5, help="Maximum retry attempts for transient API errors.")
    return parser.parse_args()


def build_endpoint(project: str, location: str, model: str) -> str:
    if location == "global":
        host = "https://aiplatform.googleapis.com"
    else:
        host = f"https://{location}-aiplatform.googleapis.com"
    return (
        f"{host}/v1/projects/{project}/locations/{location}"
        f"/publishers/google/models/{model}:generateContent"
    )


def call_vertex_generate_content(
    *,
    token_provider: VertexAccessTokenProvider,
    endpoint: str,
    prompt: str,
    temperature: float,
    top_logprobs: int,
    max_retries: int,
) -> Dict[str, object]:
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "text": (
                            f"{SYSTEM_PROMPT}\n\n"
                            f"{prompt}"
                        )
                    }
                ],
            }
        ],
        "generationConfig": {
            "temperature": temperature,
            # Gemini 2.5 models may spend output budget on thinking unless disabled.
            # We want one constrained label token with usable logprobs.
            "maxOutputTokens": 8,
            "responseLogprobs": True,
            "logprobs": top_logprobs,
            "thinkingConfig": {"thinkingBudget": 0},
        },
    }

    for attempt in range(1, max_retries + 1):
        try:
            access_token = token_provider.get_token()
            response = requests.post(
                endpoint,
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=120,
            )
            if response.status_code >= 400:
                body = response.text
                if response.status_code in {429, 500, 502, 503, 504} and attempt < max_retries:
                    if response.status_code == 429:
                        time.sleep(min(10 * attempt, 60))
                    else:
                        time.sleep(min(2 ** (attempt - 1), 8))
                    continue
                raise RuntimeError(f"Vertex AI API error {response.status_code}: {body}")
            return response.json()
        except requests.RequestException as exc:
            if attempt < max_retries:
                time.sleep(min(2 ** (attempt - 1), 8))
                continue
            raise RuntimeError(f"Network error while calling Vertex AI: {exc}") from exc

    raise RuntimeError("Vertex AI call exhausted retries without returning a response.")


def extract_vertex_label_distribution(response: Dict[str, object]) -> Tuple[Dict[str, float], str | None, str | None, bool]:
    candidates = response.get("candidates", [])
    if not candidates:
        raise RuntimeError(f"Vertex AI response did not contain any candidates: {json.dumps(response)}")

    candidate = candidates[0]
    content = candidate.get("content", {})
    message_parts = content.get("parts", [])
    message = "".join(part.get("text", "") for part in message_parts if isinstance(part, dict))
    sampled_label = normalize_label_token(message)

    logprobs_result = candidate.get("logprobsResult", {})
    top_candidates = logprobs_result.get("topCandidates", [])
    chosen_candidates = logprobs_result.get("chosenCandidates", [])

    normalized_candidates: List[Dict[str, object]] = []
    if top_candidates:
        for top_candidate in top_candidates[0].get("candidates", []):
            normalized_candidates.append(
                {
                    "token": top_candidate.get("token"),
                    "logprob": top_candidate.get("logProbability"),
                }
            )
    if chosen_candidates:
        chosen = chosen_candidates[0]
        normalized_candidates.append(
            {
                "token": chosen.get("token"),
                "logprob": chosen.get("logProbability"),
            }
        )

    probability_mass = {"A": 0.0, "B": 0.0}
    for entry in normalized_candidates:
        token = entry.get("token")
        logprob = entry.get("logprob")
        if token is None or logprob is None:
            continue
        label = normalize_label_token(str(token))
        if label is None:
            continue
        probability_mass[label] += math.exp(float(logprob))

    fallback_used = False
    if probability_mass["A"] == 0.0 and probability_mass["B"] == 0.0:
        fallback_used = True
        if sampled_label is None:
            probability_mass = {"A": 0.5, "B": 0.5}
        else:
            other = "B" if sampled_label == "A" else "A"
            probability_mass = {sampled_label: 0.999, other: 0.001}

    total = probability_mass["A"] + probability_mass["B"]
    distribution = {
        "A": probability_mass["A"] / total,
        "B": probability_mass["B"] / total,
    }
    sampled_token = None
    if chosen_candidates:
        sampled_token = chosen_candidates[0].get("token")
    return distribution, sampled_label, sampled_token, fallback_used


def main() -> int:
    args = parse_args()
    if not args.project:
        raise SystemExit("Missing Google Cloud project ID. Set GOOGLE_CLOUD_PROJECT or pass --project.")

    token_provider = VertexAccessTokenProvider()
    endpoint = build_endpoint(args.project, args.location, args.model)

    output_path = Path(args.output)
    dataset_path = Path(args.dataset_output)
    export_dataset(dataset_path, limit=args.scenario_count, seed=args.seed)
    scenarios = generate_scenarios(limit=args.scenario_count, seed=args.seed)

    payload: Dict[str, object] = {
        "metadata": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "provider": "vertex_ai",
            "location": args.location,
            "model": args.model,
            "scenario_count": args.scenario_count,
            "seed": args.seed,
            "temperature": args.temperature,
            "top_logprobs": args.top_logprobs,
            "dataset_path": dataset_path.name,
        },
        "results": [],
        "summary": {},
    }

    results: List[Dict[str, object]] = []
    completed_ids = set()
    if output_path.exists():
        existing_payload = json.loads(output_path.read_text(encoding="utf-8"))
        existing_results = existing_payload.get("results", [])
        if isinstance(existing_results, list):
            results = existing_results
            payload["results"] = results
            payload["summary"] = summarize_results(results)
            for row in results:
                scenario_id = row.get("scenario_id")
                if scenario_id:
                    completed_ids.add(scenario_id)

    pending_scenarios = [scenario for scenario in scenarios if scenario["scenario_id"] not in completed_ids]
    start_index = len(results) + 1

    for offset, scenario in enumerate(pending_scenarios, start=0):
        index = start_index + offset
        prompts = build_prompt_variants(scenario)
        variant_records: Dict[str, Dict[str, object]] = {}

        for variant_name, prompt in prompts.items():
            response = call_vertex_generate_content(
                token_provider=token_provider,
                endpoint=endpoint,
                prompt=prompt,
                temperature=args.temperature,
                top_logprobs=args.top_logprobs,
                max_retries=args.max_retries,
            )
            distribution, sampled_label, sampled_token, fallback_used = extract_vertex_label_distribution(response)
            variant_records[variant_name] = {
                "distribution": distribution,
                "sampled_label": sampled_label,
                "sampled_token": sampled_token,
                "fallback_used": fallback_used,
                "response_metadata": {
                    "usageMetadata": response.get("usageMetadata"),
                    "modelVersion": response.get("modelVersion"),
                },
            }
            if args.pause_sec:
                time.sleep(args.pause_sec)

        baseline_distribution = variant_records["baseline"]["distribution"]
        baseline_label = max(baseline_distribution, key=baseline_distribution.get)
        variant_items = {name: data for name, data in variant_records.items() if name != "baseline"}
        divergences = [kl_divergence(baseline_distribution, item["distribution"]) for item in variant_items.values()]
        flips = [
            1
            if max(item["distribution"], key=item["distribution"].get) != baseline_label
            else 0
            for item in variant_items.values()
        ]

        row = {
            "scenario_id": scenario["scenario_id"],
            "theme": scenario["theme"],
            "context": scenario["context"],
            "options": scenario["options"],
            "variants": variant_records,
            "mcv": statistics.fmean(divergences),
            "decision_flip_rate": statistics.fmean(flips) if flips else 0.0,
        }
        results.append(row)
        payload["results"] = results
        payload["summary"] = summarize_results(results)
        persist_snapshot(payload, output_path)
        print(
            f"[{index}/{len(scenarios)}] {scenario['scenario_id']} "
            f"MCV={row['mcv']:.4f} flip_rate={row['decision_flip_rate']:.2f}"
        )

    payload["summary"] = summarize_results(results)
    persist_snapshot(payload, output_path)
    print(json.dumps(payload["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
