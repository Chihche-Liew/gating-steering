#!/usr/bin/env python3
"""Evaluate steering vectors on triples to see if they improve predictions."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
from tqdm import tqdm

from model_wrapper import ModelWrapper
from setup import (
    configure_hf_caches,
    setup_logging,
    extract_final_answer,
    answers_match,
    reconstruct_prompt,
)

LOGGER = logging.getLogger("evaluate_steering")


# ===================== ARGS =========================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--triples-path", type=Path, required=True)
    parser.add_argument("--steering-vectors-dir", type=Path, default=Path("artifacts/steering_vectors"))
    parser.add_argument("--model-name", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--layers", nargs="+", required=True, type=int)
    parser.add_argument("--steering-coefficient", type=float, default=1.0)
    parser.add_argument("--skip-baseline", action="store_true")     # always passed from sweep
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--output-name", type=str, default=None)
    parser.add_argument(
        "--system-prompt",
        default="You are a careful reasoning assistant. Think step by step and end with 'Final answer: <choice>'."
    )
    parser.add_argument("--device", default=None)
    return parser.parse_args()


# ===================== HELPERS ======================
def load_triples(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return [row for row in data if isinstance(row, dict) and row.get("correct_chain")]


def load_steering_vectors(steering_dir: Path, layers: List[int]) -> Dict[int, np.ndarray]:
    steering_vectors = {}
    for layer in layers:
        fp = steering_dir / f"layer{layer}_steering_vector.npy"
        if not fp.exists():
            LOGGER.warning("Missing steering vector for layer %d", layer)
            continue
        arr = np.load(fp)
        steering_vectors[layer] = arr
        LOGGER.info("Loaded vector for layer %d (norm %.4f)", layer, np.linalg.norm(arr))
    return steering_vectors


# ================= MAIN EVAL =========================
def main():
    setup_logging()
    configure_hf_caches()
    args = parse_args()

    # ----------------- Load triples -----------------
    triples = load_triples(args.triples_path)
    if args.sample_offset:
        triples = triples[args.sample_offset:]
    if args.max_samples:
        triples = triples[:args.max_samples]
    LOGGER.info("Loaded %d triples", len(triples))

    # ----------------- Load steering vectors --------
    steering_vectors = load_steering_vectors(args.steering_vectors_dir, args.layers)
    if not steering_vectors:
        LOGGER.error("No steering vectors loaded.")
        return

    # ----------------- Decide baseline --------------
    # For alpha=0: baseline is meaningful => run it
    # For alpha>0: skip baseline, but we still save WITH predictions
    if args.skip_baseline and args.steering_coefficient != 0:
        LOGGER.info(f"Skipping baseline for alpha={args.steering_coefficient}")
    else:
        args.skip_baseline = False   # ensure baseline for alpha=0

    # ----------------- Init model -------------------
    model = ModelWrapper(args.model_name, device=args.device)

    # ----------------- Init results -----------------
    results = {
        "model_name": args.model_name,
        "layers": args.layers,
        "steering_coefficient": args.steering_coefficient,
        "total": len(triples),
        "without_steering": {
            "correct": 0,
            "incorrect": 0,
            "accuracy": None,
            "predictions": []  # will remain empty if skip_baseline is True for alpha>0
        },
        "with_steering": {
            "correct": 0,
            "incorrect": 0,
            "accuracy": None,
            "predictions": []
        }
    }

    # ----------------- Evaluate ----------------------
    for triple in tqdm(triples, desc=f"Evaluating α={args.steering_coefficient}"):
        prompt = reconstruct_prompt(triple.get("metadata", {}), triple.get("prompt", ""))
        correct_answer = triple.get("correct_answer", "")
        sample_id = triple.get("sample_id", "")

        # === Without steering (baseline) ===
        if not args.skip_baseline:
            resp = model.generate(
                prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=0.0,
                strip_prompt=True,
                system_prompt=args.system_prompt,
            )
            pred = extract_final_answer(resp)
            ok = answers_match(pred, correct_answer)
            if ok: results["without_steering"]["correct"] += 1
            else: results["without_steering"]["incorrect"] += 1

            results["without_steering"]["predictions"].append({
                "sample_id": sample_id,
                "predicted": pred,
                "correct": correct_answer,
                "is_correct": ok,
                "response": resp[:500]
            })

        # === With steering ===
        resp2 = model.generate_with_steering(
            prompt,
            steering_vectors=steering_vectors,
            steering_coefficient=args.steering_coefficient,
            max_new_tokens=args.max_new_tokens,
            temperature=0.0,
            strip_prompt=True,
            system_prompt=args.system_prompt,
        )
        pred2 = extract_final_answer(resp2)
        ok2 = answers_match(pred2, correct_answer)
        if ok2: results["with_steering"]["correct"] += 1
        else:   results["with_steering"]["incorrect"] += 1

        results["with_steering"]["predictions"].append({
            "sample_id": sample_id,
            "predicted": pred2,
            "correct": correct_answer,
            "is_correct": ok2,
            "response": resp2[:500]
        })

    # ---------- Accuracy ----------
    total = results["total"]
    results["with_steering"]["accuracy"] = results["with_steering"]["correct"] / total
    if not args.skip_baseline:
        results["without_steering"]["accuracy"] = results["without_steering"]["correct"] / total

    # ---------- Save JSON ----------
    alpha_tag = f"alpha_{args.steering_coefficient}".replace(".", "_")
    suffix = f"{alpha_tag}_offset{args.sample_offset}"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / f"{suffix}.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    LOGGER.info("Saved evaluation results to %s", out)



if __name__ == "__main__":
    main()
