#!/usr/bin/env python3
"""
Run steering evaluation across multiple layers (layer sweep).
Baseline is NOT recomputed (skip-baseline=True).
"""

import argparse
import subprocess
import os
import json

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--triples-path", default="artifacts/manual_review/10232025_human_review.json")
    p.add_argument("--model-name", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--layers", nargs="+", type=int, default=[20, 22, 24, 26, 28, 30])
    p.add_argument("--steering-vectors-dir", default="artifacts/steering_vectors")
    p.add_argument("--output-dir", default="results/layer_sweep")
    p.add_argument("--max-samples", type=int, default=100)
    p.add_argument("--sample-offset", type=int, default=0)
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--system-prompt",
        default="You are a careful reasoning assistant. Think step by step and end with 'Final answer: <choice>'.")
    p.add_argument("--alpha", type=float, default=1.0)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    for L in args.layers:
        print(f"\n================ Running Layer = {L} ================\n")

        out_json = os.path.join(
            args.output_dir,
            f"layer_{L}_alpha{str(args.alpha).replace('.','_')}_offset{args.sample_offset}.json"
        )

        cmd = [
            "python", "scripts/evaluate_steering_alpha_sweep.py",
            "--triples-path", args.triples_path,
            "--model-name", args.model_name,
            "--layers", str(L),
            "--steering-vectors-dir", args.steering_vectors_dir,
            "--steering-coefficient", str(args.alpha),
            "--max-samples", str(args.max_samples),
            "--max-new-tokens", str(args.max_new_tokens),
            "--sample-offset", str(args.sample_offset),
            "--output-dir", args.output_dir,
            "--system-prompt", args.system_prompt,
            "--skip-baseline"             # 🔥 ALWAYS skip baseline
        ]

        subprocess.run(cmd, check=True)

    print("\n===== ALL LAYERS FINISHED =====")
    print(f"Results saved in: {args.output_dir}/")


if __name__ == "__main__":
    main()
