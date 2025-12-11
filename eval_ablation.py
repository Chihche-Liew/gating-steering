'''#!/usr/bin/env python3
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

print("\n📌 Saved:", f"{ABLATION_DIR}/ablation_plot.png")'''

#!/usr/bin/env python3
"""
Ablation plot for steering, using existing alpha sweep results.

Reads:
  - results/alpha_sweep/alpha_0_0_offset0.json   (α = 0)
  - results/alpha_sweep/alpha_1_0_offset0.json   (α = 1)

and produces:
  - results/ablation/ablation_plot_poster.png
"""
'''
import os
import json
import numpy as np
import matplotlib.pyplot as plt

# ---------- Paths ----------
ALPHA_DIR = "results/alpha_sweep"
OUT_DIR = "results/ablation"
os.makedirs(OUT_DIR, exist_ok=True)

alpha0_path = os.path.join(ALPHA_DIR, "alpha_0_0_offset0.json")
alpha1_path = os.path.join(ALPHA_DIR, "alpha_1_0_offset0.json")

# ---------- Helper: bootstrap CI ----------
def bootstrap_mean_ci(correct_flags, n_boot=3000, alpha=0.05):
    """
    correct_flags: list/array of 0/1 (is_correct)
    returns mean, lower_error, upper_error for errorbar
    """
    arr = np.asarray(correct_flags, dtype=float)
    if len(arr) == 0:
        return None, None, None

    boot_means = []
    n = len(arr)
    for _ in range(n_boot):
        sample = np.random.choice(arr, size=n, replace=True)
        boot_means.append(sample.mean())
    boot_means = np.asarray(boot_means)

    mean = boot_means.mean()
    low, high = np.percentile(boot_means, [100*alpha/2, 100*(1-alpha/2)])
    return mean, mean - low, high - mean


# ---------- Load JSONs ----------
with open(alpha0_path, "r") as f:
    r0 = json.load(f)
with open(alpha1_path, "r") as f:
    r1 = json.load(f)

# Baseline = “without_steering” accuracy at α = 0
baseline_acc = r0["without_steering"]["accuracy"]
# Steering loaded α=0 = “with_steering” accuracy at α = 0
alpha0_acc = r0["with_steering"]["accuracy"]
# Full steering α=1 = “with_steering” accuracy at α = 1
alpha1_acc = r1["with_steering"]["accuracy"]

# ---------- Bootstrap CIs from prediction lists ----------
# baseline (no steering)
base_flags = [int(p["is_correct"]) for p in r0["without_steering"]["predictions"]]
m_base, lo_base, hi_base = bootstrap_mean_ci(base_flags)

# steering loaded α=0
a0_flags = [int(p["is_correct"]) for p in r0["with_steering"]["predictions"]]
m_a0, lo_a0, hi_a0 = bootstrap_mean_ci(a0_flags)

# full steering α=1
a1_flags = [int(p["is_correct"]) for p in r1["with_steering"]["predictions"]]
m_a1, lo_a1, hi_a1 = bootstrap_mean_ci(a1_flags)

means = [m_base, m_a0, m_a1]
lower_err = [lo_base, lo_a0, lo_a1]
upper_err = [hi_base, hi_a0, hi_a1]

labels = [
    "Baseline (no steering)",
    "Steering loaded α = 0",
    "Full steering α = 1",
]

colors = ["#9e9e9e", "#f4a522", "#0f8f2f"]  # grey, orange, green

x = np.arange(len(labels))

# ---------- Plot ----------
plt.figure(figsize=(7, 4))

bars = plt.bar(
    x,
    means,
    yerr=[lower_err, upper_err],
    capsize=6,
    color=colors,
    edgecolor="black",
)

# numeric labels on top
for bar, val in zip(bars, means):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        val + 0.01,
        f"{val*100:.1f}%",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
    )

# horizontal line at baseline mean
plt.axhline(means[0], color="grey", linestyle="--", linewidth=1)

plt.ylabel("Accuracy", fontsize=12, fontweight="bold")
plt.xticks(x, labels, rotation=10, fontsize=11)
plt.title(
    "Ablation Study: Steering Causal Effect",
    fontsize=14,
    fontweight="bold",
)
plt.tight_layout()

out_path = os.path.join(OUT_DIR, "ablation_plot_poster.png")
plt.savefig(out_path, dpi=300)
print(f"✅ Saved ablation plot to: {out_path}")'''

import os
import json
import numpy as np
import matplotlib.pyplot as plt

# ---------- Paths ----------
ALPHA_DIR = "results/alpha_sweep"
OUT_DIR = "results/ablation"
os.makedirs(OUT_DIR, exist_ok=True)

alpha0_path = os.path.join(ALPHA_DIR, "alpha_0_0_offset0.json")  # baseline + α=0 loaded
alpha1_path = os.path.join(ALPHA_DIR, "alpha_1_0_offset0.json")  # α=1 steering

# ---------- Helper: bootstrap CI ----------
def bootstrap_mean_ci(correct_flags, n_boot=3000, alpha=0.05):
    """
    correct_flags: list/array of 0/1 (is_correct)
    returns: observed_mean, lower_error, upper_error
    """
    arr = np.asarray(correct_flags, dtype=float)
    if len(arr) == 0:
        return None, None, None

    boot_means = []
    n = len(arr)
    for _ in range(n_boot):
        sample = np.random.choice(arr, size=n, replace=True)
        boot_means.append(sample.mean())
    boot_means = np.asarray(boot_means)

    observed_mean = arr.mean()   # Use actual accuracy (not bootstrap mean)
    low, high = np.percentile(boot_means, [100*alpha/2, 100*(1-alpha/2)])
    return observed_mean, observed_mean - low, high - observed_mean


# ---------- Load JSONs ----------
with open(alpha0_path, "r") as f:
    r0 = json.load(f)
with open(alpha1_path, "r") as f:
    r1 = json.load(f)

# Baseline (without steering) at α=0
baseline_flags = [int(p["is_correct"]) for p in r0["without_steering"]["predictions"]]
base_m, base_lo, base_hi = bootstrap_mean_ci(baseline_flags)

# Steering loaded α=0 (vector injected but coef=0)
a0_flags = [int(p["is_correct"]) for p in r0["with_steering"]["predictions"]]
a0_m, a0_lo, a0_hi = bootstrap_mean_ci(a0_flags)

# Full steering α=1
a1_flags = [int(p["is_correct"]) for p in r1["with_steering"]["predictions"]]
a1_m, a1_lo, a1_hi = bootstrap_mean_ci(a1_flags)

means = [base_m, a0_m, a1_m]
lower_err = [base_lo, a0_lo, a1_lo]
upper_err = [base_hi, a0_hi, a1_hi]

labels = [
    "Baseline (no steering)",
    "Steering loaded α = 0",
    "Full steering α = 1",
]

# Same color style as sweep plot
colors = ["#9e9e9e", "#ffb74d", "#1976d2"]  # grey, orange, blue

# ---------- Plot ----------
plt.figure(figsize=(8, 4))

x = np.arange(len(labels))
bars = plt.bar(
    x,
    means,
    yerr=[lower_err, upper_err],
    capsize=6,
    color=colors,
    edgecolor="black",
)

# numeric labels on bars
for bar, val in zip(bars, means):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        val + 0.01,
        f"{val*100:.1f}%",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
    )

# horizontal reference dashed line (baseline)
plt.axhline(means[0], color="grey", linestyle="--", linewidth=1)

plt.ylabel("Accuracy", fontsize=12, fontweight="bold")
plt.xticks(x, labels, rotation=10, fontsize=11)
plt.title(
    "Ablation Study: Steering Causal Effect",
    fontsize=15,
    fontweight="bold",
)
plt.tight_layout()

out_path = os.path.join(OUT_DIR, "ablation_plot_poster.png")
plt.savefig(out_path, dpi=300)
print(f"✅ Saved ablation plot to: {out_path}")
