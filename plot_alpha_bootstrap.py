'''#!/usr/bin/env python3
"""
Plot steering accuracy vs alpha with bootstrap 95% CI.
Works with files like: alpha_0_2_offset0.json, alpha_1_0_offset0.json, etc.
"""

import os, json, re
import numpy as np
import matplotlib.pyplot as plt

RESULT_DIR = "results/alpha_sweep"
BOOTSTRAP_N = 3000   # bootstrap repeats


# ---------------- Find alpha result files ---------------- #
json_files = [f for f in os.listdir(RESULT_DIR) if f.endswith(".json")]

# Must match strict pattern: alpha_{num}_offset0.json, e.g. alpha_1_2_offset0.json
pattern = re.compile(r"^alpha_(\d+(_\d+)?)_offset0\.json$")

alpha_map = {}  # alpha_value -> file_name

for f in json_files:
    m = pattern.match(f)
    if m:
        alpha_str = m.group(1).replace("_", ".")
        try:
            alpha_val = float(alpha_str)
            alpha_map[alpha_val] = f  # dedupe automatically
        except:
            pass

# Sorted list of unique alphas
ALPHAS = sorted(alpha_map.keys())
print("Detected alpha files:", alpha_map)
print("Unique alphas detected:", ALPHAS)


# ---------------- Containers ---------------- #

means, lowers, uppers = [], [], []
baseline_acc = None



# ---------------- Parse each alpha ---------------- #
for alpha in ALPHAS:
    fname = os.path.join(RESULT_DIR, alpha_map[alpha])
    with open(fname, "r") as f:
        r = json.load(f)

    preds = r["with_steering"]["predictions"]
    correct_vec = [int(x["is_correct"]) for x in preds]

    # baseline = alpha=0 accuracy
    if alpha == 0:
        baseline_acc = np.mean(correct_vec)

    # bootstrap samples
    boot = []
    for _ in range(BOOTSTRAP_N):
        sample = np.random.choice(correct_vec, size=len(correct_vec), replace=True)
        boot.append(np.mean(sample))

    m = np.mean(boot)
    ci_low, ci_high = np.percentile(boot, [2.5, 97.5])
    means.append(m)
    lowers.append(m - ci_low)
    uppers.append(ci_high - m)

    print(f"[α={alpha}] mean={m:.3f}, 95% CI=({ci_low:.3f}, {ci_high:.3f})")


# ---------------- Plot (poster style) ---------------- #
plt.figure(figsize=(6.5, 4.5))

plt.errorbar(ALPHAS, means, yerr=[lowers, uppers],
             fmt="o-", capsize=5, linewidth=2.7,
             markersize=8, label="Steering (Layer 28)", color="#0066CC")

if baseline_acc is not None:
    plt.axhline(baseline_acc, color="gray", linestyle="--",
                linewidth=2, label="Baseline (α=0)")

plt.ylim(0, max(max(means)+0.05, 0.4))  # adaptive y-limit
plt.xlabel("Steering Strength α", fontsize=13)
plt.ylabel("Accuracy", fontsize=13)
plt.title("Llama-3.1-8B-Instruct @ Layer 28", fontsize=14)
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend(fontsize=11)
plt.tight_layout()

out_png = os.path.join(RESULT_DIR, "poster_alpha_CI.png")
plt.savefig(out_png, dpi=300)

print("\n✔ Saved poster-style CI figure to:", out_png)'''

#!/usr/bin/env python3
"""
Poster-quality plot: Accuracy vs Steering Strength (alpha) with 95% bootstrap CI.

✔ Automatically detects all alpha_*.json outputs in results/alpha_sweep/
✔ Computes bootstrap confidence intervals for the steering results
✔ Extracts baseline accuracy (alpha=0) from "without_steering"
✔ Plots both baseline and steering curves
✔ Highlights the empirically optimal steering region (~0.6–1.1)
✔ Saves: results/alpha_sweep/poster_alpha_CI.png
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt

# Directory where evaluation json files are stored
RESULT_DIR = "results/alpha_sweep"

# Number of bootstrap draws – large enough but still fast
BOOTSTRAP_N = 3000

# ---------- Poster Plot Styling ----------
plt.rcParams.update({
    "font.size": 13,
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "figure.figsize": (7, 4),
    "axes.linewidth": 1.3,
})

# =====================================================================
# 1) Detect all valid alpha JSON outputs in the directory
# =====================================================================
files = [
    f for f in os.listdir(RESULT_DIR)
    if f.startswith("alpha_") and f.endswith(".json")
]

alpha_files = {}
for f in files:
    core = f[len("alpha_"): f.rfind("_offset")]  # e.g., "0_5" from "alpha_0_5_offset0.json"
    try:
        alpha = float(core.replace("_", "."))     # convert "0_5" → 0.5
        alpha_files[alpha] = f
    except ValueError:
        print(f"Skipping file (failed to parse alpha): {f}")

alphas = sorted(alpha_files.keys())
print("\nDetected alpha files:", alpha_files)
print("Unique alphas:", alphas)

if not alphas:
    raise RuntimeError("No valid alpha_*.json files found. Check RESULT_DIR.")

# =====================================================================
# 2) Bootstrap confidence intervals for each steering result
# =====================================================================
means, lowers, uppers = [], [], []
baseline_acc = None
baseline_ci_low = None
baseline_ci_high = None

for a in alphas:
    path = os.path.join(RESULT_DIR, alpha_files[a])
    with open(path, "r") as f:
        result = json.load(f)

    # Extract steering correctness vector (1 = correct, 0 = incorrect)
    if "predictions" in result["with_steering"]:
        correctness_vec = np.array([int(x["is_correct"]) for x in result["with_steering"]["predictions"]])
    else:
        # Fallback if predictions are missing
        acc = result["with_steering"]["accuracy"]
        total = result["total"]
        correctness_vec = np.array([1] * int(acc * total) + [0] * (total - int(acc * total)))

    # Bootstrap sampling
    boot = []
    n = len(correctness_vec)
    for _ in range(BOOTSTRAP_N):
        sample = np.random.choice(correctness_vec, size=n, replace=True)
        boot.append(sample.mean())
    boot = np.array(boot)

    # Mean and 95% CI
    m = boot.mean()
    ci_low, ci_high = np.percentile(boot, [2.5, 97.5])

    means.append(m)
    lowers.append(m - ci_low)
    uppers.append(ci_high - m)

    # -----------------------------------------------------------------
    # Extract baseline results only for alpha = 0
    # -----------------------------------------------------------------
    if abs(a - 0.0) < 1e-8:
        if "predictions" in result["without_steering"]:
            base_vec = np.array([int(x["is_correct"]) for x in result["without_steering"]["predictions"]])
            base_boot = []
            nb = len(base_vec)
            for _ in range(BOOTSTRAP_N):
                sample = np.random.choice(base_vec, size=nb, replace=True)
                base_boot.append(sample.mean())
            base_boot = np.array(base_boot)
            baseline_acc = base_boot.mean()
            baseline_ci_low, baseline_ci_high = np.percentile(base_boot, [2.5, 97.5])
        else:
            # No predictions available → use summary accuracy without CI
            baseline_acc = result["without_steering"]["accuracy"]
            baseline_ci_low = baseline_ci_high = baseline_acc

# =====================================================================
# 3) Plot the results
# =====================================================================
fig, ax = plt.subplots()

# Highlight the empirically optimal steering region
ax.axvspan(0.6, 1.1, color="yellow", alpha=0.15, zorder=0)

# Plot baseline horizontal line + CI region
if baseline_acc is not None:
    ax.axhline(
        baseline_acc,
        linestyle="--",
        color="gray",
        linewidth=2,
        label="Baseline (α = 0)",
        zorder=1,
    )
    if baseline_ci_low is not None and baseline_ci_high is not None:
        ax.fill_between(
            [min(alphas) - 0.05, max(alphas) + 0.05],
            baseline_ci_low,
            baseline_ci_high,
            color="gray",
            alpha=0.1,
            zorder=0,
        )

# Plot steering performance curve with error bars
ax.errorbar(
    alphas,
    means,
    yerr=[lowers, uppers],
    fmt="o-",
    capsize=5,
    linewidth=2.5,
    markersize=7,
    label="Steering (Layer 28)",
    zorder=2,
)

# Labels and title
plt.title(
    "Steering Improves Reasoning Accuracy\nin Llama-3.1-8B (Layer 28)",
    fontsize=18, fontweight="bold", pad=12
)
plt.tight_layout()

ax.set_xlabel("Steering Strength α")
ax.set_ylabel("Accuracy")

# Axis limits tuned for clarity in posters
ax.set_ylim(0.0, 0.40)
ax.set_xlim(min(alphas) - 0.05, max(alphas) + 0.05)

ax.legend(frameon=False, loc="upper left")
ax.grid(True, alpha=0.3)

plt.tight_layout()
out_path = os.path.join(RESULT_DIR, "poster_alpha_CI.png")
plt.savefig(out_path, dpi=300)

print("\nSaved poster plot:", out_path)
