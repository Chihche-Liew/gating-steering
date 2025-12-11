#!/usr/bin/env python3
"""
Ablation Study from Existing alpha_sweep Results.

Using:
- Baseline (no steering):    from WITHOUT block in alpha_0_0_offset0.json
- Steering loaded α=0:       from WITH block in alpha_0_0_offset0.json
- Full steering α=1:         from WITH block in alpha_1_0_offset0.json
"""

import json, os
import matplotlib.pyplot as plt

RESULT_DIR = "results/alpha_sweep"
ABLATION_DIR = "results/ablation"
os.makedirs(ABLATION_DIR, exist_ok=True)

# ==== Helper to load accuracy from sweep file ====
def load_acc(alpha, mode):
    """
    mode = "with"  uses with_steering
    mode = "without" uses without_steering
    """
    tag = str(alpha).replace(".", "_")
    file = f"{RESULT_DIR}/alpha_{tag}_offset0.json"
    with open(file, "r") as f:
        r = json.load(f)
    if mode == "with":
        return r["with_steering"]["accuracy"]
    elif mode == "without":
        return r["without_steering"]["accuracy"]
    else:
        raise ValueError("mode must be 'with' or 'without'")

# ==== Extract ablation accuracy ====
acc_baseline      = load_acc(0.0, "without")  # no steering at all
acc_alpha0        = load_acc(0.0, "with")     # vector loaded but disabled
acc_alpha1        = load_acc(1.0, "with")     # full steering

# ==== Save readable results ====
ablation_results = {
    "Baseline (no steering)": acc_baseline,
    "Steering loaded α=0": acc_alpha0,
    "Full steering α=1": acc_alpha1
}

with open(f"{ABLATION_DIR}/ablation_summary.json", "w") as f:
    json.dump(ablation_results, f, indent=2)

print("\n>>> Ablation Results (from sweep):\n")
for k,v in ablation_results.items():
    print(f"{k:25s} : {v:.3f}")

# ==== Plot ====
names = list(ablation_results.keys())
values = list(ablation_results.values())

plt.figure(figsize=(6,4))
plt.bar(names, values, color=["gray","orange","green"])
plt.ylabel("Accuracy")
plt.title("Ablation Study: Steering Causal Effect")
plt.xticks(rotation=10)
plt.tight_layout()
plt.savefig(f"{ABLATION_DIR}/ablation_plot.png")

print("\n📌 Saved:", f"{ABLATION_DIR}/ablation_plot.png")
