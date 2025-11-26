#!/usr/bin/env python3
"""Plot probe analysis visualizations: PCA, UMAP, CAA projections, and cluster overlays."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import json
import math

import matplotlib.pyplot as plt
import numpy as np

from setup import setup_logging


LOGGER = logging.getLogger("plot_probes")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        required=True,
        help="Directory containing computed probe analysis data (output from compute_probes.py).",
    )
    parser.add_argument(
        "--layer",
        type=int,
        required=True,
        help="Layer number to plot.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/hidden_state_viz"),
        help="Directory for saving plots.",
    )
    parser.add_argument(
        "--dp-alignment",
        action="store_true",
        help="Plot only the points that passed DP alignment filtering (if metadata is available).",
    )
    parser.add_argument(
        "--dp-tail-k",
        type=int,
        default=0,
        help="Annotate outputs for DP tail sampling (match --dp-tail-k from compute_probes).",
    )
    return parser.parse_args()


def _load_dp_tail_mask(analysis_dir: Path, layer: int, expected_length: int) -> np.ndarray | None:
    """Return mask selecting final third of DP-aligned tokens per sample."""
    metrics_path = analysis_dir / f"layer{layer}_metrics.json"
    dp_requested = False
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            dp_info = metrics.get("dp_alignment")
            dp_requested = bool(dp_info and dp_info.get("requested"))
        except Exception as exc:
            LOGGER.warning("Failed to read metrics file %s: %s", metrics_path, exc)
    if not dp_requested:
        return None

    probe_data_path = analysis_dir.parent / "probe_data" / f"layer{layer}_probe_data.npz"
    if not probe_data_path.exists():
        LOGGER.warning("Probe data not found at %s; skipping DP tail filter.", probe_data_path)
        return None

    probe_data = np.load(probe_data_path)
    if "dp_pair_index" not in probe_data or "sample_ids" not in probe_data:
        LOGGER.warning("Probe data missing DP metadata; skipping DP tail filter.")
        return None

    base_mask = np.ones(probe_data["dp_pair_index"].shape[0], dtype=bool)
    if "valid" in probe_data:
        base_mask &= probe_data["valid"].astype(bool)
    if "dp_used_alignment" in probe_data:
        base_mask &= probe_data["dp_used_alignment"].astype(bool)
    if "dp_used_fallback" in probe_data:
        base_mask &= ~probe_data["dp_used_fallback"].astype(bool)

    pair_indices = probe_data["dp_pair_index"][base_mask]
    sample_ids = probe_data["sample_ids"][base_mask]

    if pair_indices.shape[0] != expected_length:
        LOGGER.warning(
            "DP tail filter length mismatch for layer %d (expected %d, found %d); skipping.",
            layer,
            expected_length,
            pair_indices.shape[0],
        )
        return None

    tail_mask = np.zeros(expected_length, dtype=bool)
    unique_samples = np.unique(sample_ids)
    for sample in unique_samples:
        sample_indices = np.where(sample_ids == sample)[0]
        sample_pairs = pair_indices[sample_indices]
        valid_pairs = sample_pairs[sample_pairs >= 0]
        if valid_pairs.size == 0:
            continue

        pair_count = int(valid_pairs.max()) + 1
        tail_len = max(1, int(math.ceil(pair_count / 3.0)))
        tail_start = pair_count - tail_len
        sample_tail_mask = sample_pairs >= tail_start

        if not np.any(sample_tail_mask):
            max_pair = np.max(sample_pairs)
            tail_mask[sample_indices[sample_pairs == max_pair]] = True
        else:
            tail_mask[sample_indices[sample_tail_mask]] = True

    if not tail_mask.any():
        LOGGER.warning("DP tail filtering removed all samples; skipping tail filter.")
        return None

    return tail_mask


def plot_cluster_overlay(
    points: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
    title: str,
    xlabel: str = "Component 1",
    ylabel: str = "Component 2",
) -> None:
    """
    Save a 2D scatter plot with different markers for each class.

    Args:
        points: 2D points (n_samples, 2)
        labels: Binary labels
        output_path: Path to save plot
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 6))
    label_palette = {0: "red", 1: "blue"}

    for label in np.unique(labels):
        mask = labels == label
        plt.scatter(
            points[mask, 0],
            points[mask, 1],
            c=label_palette.get(label, "gray"),
            marker="o" if label == 0 else "x",
            s=30,
            alpha=0.7,
            label="wrong" if label == 0 else "right",
        )

    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.legend(loc="best", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    LOGGER.info("Saved cluster overlay to %s", output_path)


def main() -> None:
    setup_logging()
    args = parse_args()

    layer = args.layer

    # Check if analysis directory exists
    if not args.analysis_dir.exists():
        LOGGER.error("Analysis directory not found: %s", args.analysis_dir)
        return

    suffix_parts = []
    title_parts = []
    if args.dp_alignment:
        suffix_parts.append("dp")
        title_parts.append("DP aligned")
    if args.dp_tail_k > 0:
        suffix_parts.append(f"tail{args.dp_tail_k}")
        title_parts.append(f"DP tail k={args.dp_tail_k}")
    suffix = "_" + "_".join(suffix_parts) if suffix_parts else ""
    title_suffix = f" ({', '.join(title_parts)})" if title_parts else ""

    tail_mask_cache: np.ndarray | None = None

    # Load PCA data
    pca_data_path = args.analysis_dir / f"layer{layer}_pca_data.npz"
    if pca_data_path.exists():
        LOGGER.info("Loading PCA data from %s", pca_data_path)
        pca_data = np.load(pca_data_path)
        pca_transformed = pca_data["transformed"]
        pca_labels = pca_data["labels"]

        if args.dp_alignment:
            if args.dp_tail_k > 0 and "dp_tail_used" in pca_data.files:
                dp_mask = pca_data["dp_tail_used"].astype(bool)
                retained = int(dp_mask.sum())
                LOGGER.info("PCA DP-tail filter retained %d samples", retained)
                if retained == 0:
                    LOGGER.warning("No PCA samples remain after DP filtering; skipping PCA plot.")
                    pca_transformed = None
                else:
                    pca_transformed = pca_transformed[dp_mask]
                    pca_labels = pca_labels[dp_mask]
            elif "dp_used_alignment" in pca_data.files:
                dp_mask = pca_data["dp_used_alignment"].astype(bool)
                if "dp_used_fallback" in pca_data.files:
                    dp_mask &= ~pca_data["dp_used_fallback"].astype(bool)
                retained = int(dp_mask.sum())
                LOGGER.info("PCA DP filter retained %d samples", retained)
                if retained == 0:
                    LOGGER.warning("No PCA samples remain after DP filtering; skipping PCA plot.")
                    pca_transformed = None
                else:
                    pca_transformed = pca_transformed[dp_mask]
                    pca_labels = pca_labels[dp_mask]
            else:
                LOGGER.warning("PCA data missing 'dp_used_alignment'; skipping DP filter.")

        if (
            args.dp_tail_k == 0
            and pca_transformed is not None
            and pca_transformed.shape[0] > 0
        ):
            tail_mask_cache = _load_dp_tail_mask(args.analysis_dir, layer, pca_transformed.shape[0])
            if tail_mask_cache is not None:
                retained_tail = int(tail_mask_cache.sum())
                LOGGER.info("Applying DP tail filter to PCA data (retained %d samples).", retained_tail)
                if retained_tail == 0:
                    LOGGER.warning("DP tail filter removed all PCA samples; skipping PCA plot.")
                    pca_transformed = None
                else:
                    pca_transformed = pca_transformed[tail_mask_cache]
                    pca_labels = pca_labels[tail_mask_cache]

        if pca_transformed is not None and pca_transformed.shape[0] > 0:
            overlay_dir = args.output_dir / f"cluster_overlays{suffix}"
            pca_cluster_path = overlay_dir / f"layer{layer}_pca_clusters{suffix}.png"
            plot_cluster_overlay(
                pca_transformed,
                pca_labels,
                pca_cluster_path,
                f"PCA Separation{title_suffix}",
            )
    else:
        LOGGER.warning("PCA data not found: %s", pca_data_path)

    # Load UMAP data
    umap_data_path = args.analysis_dir / f"layer{layer}_umap_data.npz"
    if umap_data_path.exists():
        LOGGER.info("Loading UMAP data from %s", umap_data_path)
        umap_data = np.load(umap_data_path)
        umap_transformed = umap_data["transformed"]
        umap_labels = umap_data["labels"]

        if args.dp_alignment:
            if args.dp_tail_k > 0 and "dp_tail_used" in umap_data.files:
                dp_mask = umap_data["dp_tail_used"].astype(bool)
                retained = int(dp_mask.sum())
                LOGGER.info("UMAP DP-tail filter retained %d samples", retained)
                if retained == 0:
                    LOGGER.warning("No UMAP samples remain after DP filtering; skipping UMAP plot.")
                    umap_transformed = None
                else:
                    umap_transformed = umap_transformed[dp_mask]
                    umap_labels = umap_labels[dp_mask]
            elif "dp_used_alignment" in umap_data.files:
                dp_mask = umap_data["dp_used_alignment"].astype(bool)
                if "dp_used_fallback" in umap_data.files:
                    dp_mask &= ~umap_data["dp_used_fallback"].astype(bool)
                retained = int(dp_mask.sum())
                LOGGER.info("UMAP DP filter retained %d samples", retained)
                if retained == 0:
                    LOGGER.warning("No UMAP samples remain after DP filtering; skipping UMAP plot.")
                    umap_transformed = None
                else:
                    umap_transformed = umap_transformed[dp_mask]
                    umap_labels = umap_labels[dp_mask]
            else:
                LOGGER.warning("UMAP data missing 'dp_used_alignment'; skipping DP filter.")

        if (
            args.dp_tail_k == 0
            and umap_transformed is not None
            and umap_transformed.shape[0] > 0
        ):
            if tail_mask_cache is None or tail_mask_cache.shape[0] != umap_transformed.shape[0]:
                tail_mask_cache = _load_dp_tail_mask(args.analysis_dir, layer, umap_transformed.shape[0])
            if tail_mask_cache is not None:
                retained_tail = int(tail_mask_cache.sum())
                LOGGER.info("Applying DP tail filter to UMAP data (retained %d samples).", retained_tail)
                if retained_tail == 0:
                    LOGGER.warning("DP tail filter removed all UMAP samples; skipping UMAP plot.")
                    umap_transformed = None
                else:
                    umap_transformed = umap_transformed[tail_mask_cache]
                    umap_labels = umap_labels[tail_mask_cache]

        if umap_transformed is not None and umap_transformed.shape[0] > 0:
            overlay_dir = args.output_dir / f"cluster_overlays{suffix}"
            umap_cluster_path = overlay_dir / f"layer{layer}_umap_clusters{suffix}.png"
            plot_cluster_overlay(
                umap_transformed,
                umap_labels,
                umap_cluster_path,
                f"UMAP Separation{title_suffix}",
            )
    else:
        LOGGER.warning("UMAP data not found: %s", umap_data_path)

    LOGGER.info("All plots saved to %s", args.output_dir)


if __name__ == "__main__":
    main()


'''
def plot_caa_histogram(
    projections: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
) -> None:
    """
    Plot histogram of feature projections onto CAA direction.
    
    Args:
        projections: 1D array of projections onto CAA vector
        labels: Binary labels (0=wrong, 1=right)
        output_path: Path to save plot
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    wrong_proj = projections[labels == 0]
    right_proj = projections[labels == 1]
    
    plt.figure(figsize=(10, 6))
    plt.hist(wrong_proj, bins=50, alpha=0.6, label='Wrong', color='red', density=True)
    plt.hist(right_proj, bins=50, alpha=0.6, label='Right', color='blue', density=True)
    
    plt.xlabel('Projection onto CAA Direction', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title('Feature Projections onto CAA Vector', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    LOGGER.info("Saved CAA projection histogram to %s", output_path)


def plot_caa_box_comparison(
    projections: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
) -> None:
    """
    Plot box plot comparing distributions along CAA direction.
    
    Args:
        projections: 1D array of projections onto CAA vector
        labels: Binary labels
        output_path: Path to save plot
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    wrong_proj = projections[labels == 0]
    right_proj = projections[labels == 1]
    
    plt.figure(figsize=(8, 6))
    plt.boxplot(
        [wrong_proj, right_proj],
        labels=['Wrong', 'Right'],
        patch_artist=True,
        boxprops=dict(facecolor='lightblue', alpha=0.7),
        medianprops=dict(color='red', linewidth=2),
    )
    
    plt.ylabel('Projection onto CAA Direction', fontsize=12)
    plt.title('Distribution Comparison Along CAA Vector', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    LOGGER.info("Saved CAA box plot to %s", output_path)


def plot_caa_density(
    projections: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
) -> None:
    """
    Plot density curves for correct/incorrect along CAA direction.
    
    Args:
        projections: 1D array of projections
        labels: Binary labels
        output_path: Path to save plot
    """
    try:
        from scipy import stats
    except ImportError:
        LOGGER.warning("scipy not available, skipping density plot")
        return
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    wrong_proj = projections[labels == 0]
    right_proj = projections[labels == 1]
    
    # Compute KDEs
    x_range = np.linspace(projections.min(), projections.max(), 200)
    kde_wrong = stats.gaussian_kde(wrong_proj)
    kde_right = stats.gaussian_kde(right_proj)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_range, kde_wrong(x_range), 'r-', linewidth=2, label='Wrong', alpha=0.8)
    plt.plot(x_range, kde_right(x_range), 'b-', linewidth=2, label='Right', alpha=0.8)
    plt.fill_between(x_range, kde_wrong(x_range), alpha=0.3, color='red')
    plt.fill_between(x_range, kde_right(x_range), alpha=0.3, color='blue')
    
    plt.xlabel('Projection onto CAA Direction', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title('Kernel Density Estimate Along CAA Vector', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    LOGGER.info("Saved CAA density plot to %s", output_path)
    '''