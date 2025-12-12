#!/usr/bin/env python3
"""
Compare steering vector training sample size (10 / 30 / 50)
using fixed evaluation set (100 triples) with bootstrap 95% CI.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt

# ================= CONFIG =================
BOOTSTRAP_N = 3000
OUT_FIG = "results/sample_size_compare/sample_size_vs_accuracy.png"

FILES = {
    10: "results/sample_size_compare/ss10_steering_evaluation_results_offset0.json",
    30: "results/alpha_sweep/alpha_1_0_offset0.json",   
    50: "results/sample_size_compare/ss50_steering_evaluation_results_offset0.json",
}
# ==========================================


def bootstrap_mean_ci(binary, n_boot=3000, alpha=0.05):
    arr = np.asarray(binary, dtype=float)
    n = len(arr)

    boots = []
    for _ in range(n_boot):
        boots.append(np.random.choice(arr, size=n, replace=True).mean())
    boots = np.asarray(boots)

    mean = boots.mean()
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return mean, mean - lo, hi - mean


# ============== LOAD DATA =================
sample_sizes = []
means, lower_err, upper_err = [], [], []

for ss, path in FILES.items():
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    with open(path, "r") as f:
        r = json.load(f)

    flags = [int(p["is_correct"]) for p in r["with_steering"]["predictions"]]
    m, lo, hi = bootstrap_mean_ci(flags, BOOTSTRAP_N)

    sample_sizes.append(ss)
    means.append(m)
    lower_err.append(lo)
    upper_err.append(hi)

# sort by sample size
sample_sizes, means, lower_err, upper_err = zip(
    *sorted(zip(sample_sizes, means, lower_err, upper_err))
)

# ================== PLOT ==================
plt.figure(figsize=(6.5, 4.2))

plt.errorbar(
    sample_sizes,
    means,
    yerr=[lower_err, upper_err],
    fmt="o-",
    linewidth=3,
    markersize=8,
    capsize=6,
    color="#1f77b4",
)

for x, y in zip(sample_sizes, means):
    plt.text(x, y + 0.01, f"{y*100:.1f}%", ha="center", fontsize=11)

plt.xlabel("Steering Vector Training Sample Size", fontsize=12, fontweight="bold")
plt.ylabel("Accuracy", fontsize=12, fontweight="bold")
plt.title(
    "Effect of Steering Vector Training Sample Size\n(Evaluation fixed at 100 samples)",
    fontsize=14,
    fontweight="bold",
)

plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(OUT_FIG, dpi=300)

print(f"✅ Saved plot to {OUT_FIG}")
