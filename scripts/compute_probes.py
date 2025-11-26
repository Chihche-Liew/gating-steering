#!/usr/bin/env python3
"""Compute probe analysis: linear probe and CAA vectors from hidden state features."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

try:
    import umap
except ImportError:
    umap = None

from setup import setup_logging


LOGGER = logging.getLogger("compute_probes")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--probe-data-path",
        type=Path,
        required=True,
        help="Path to .npz file containing probe data (features and labels).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/probe_analysis"),
        help="Directory for saving computed probes, vectors, and metrics.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/test split and dimensionality reduction.",
    )
    parser.add_argument(
        "--dp-alignment",
        action="store_true",
        help="Restrict analysis to tokens that passed DP alignment (requires metadata in probe data).",
    )
    parser.add_argument(
        "--dp-tail-k",
        type=int,
        default=0,
        help="If > 0, rebuild features using the last k DP-aligned token pairs per sample (fallback to last token when unavailable).",
    )
    parser.add_argument(
        "--alignments-dir",
        type=Path,
        default=Path("artifacts/alignments"),
        help="Directory containing alignment JSON files (used for DP tail sampling).",
    )
    parser.add_argument(
        "--hidden-states-dir",
        type=Path,
        default=Path("artifacts/hidden_states"),
        help="Directory containing cached hidden state tensors (used for DP tail sampling).",
    )
    return parser.parse_args()


def load_alignment_payload(sample_id: str, alignments_dir: Path) -> Tuple[List[Tuple[int, int]], Dict]:
    """Load DP alignment matches and metadata for a sample."""
    path = alignments_dir / f"{sample_id}.json"
    if not path.exists():
        return [], {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        LOGGER.warning("Failed to read alignment file for sample %s: %s", sample_id, exc)
        return [], {}

    matches: List[Tuple[int, int]] = []
    for entry in payload.get("matches", []):
        if not isinstance(entry, dict):
            continue
        try:
            wrong_idx = int(entry["wrong_index"])
            right_idx = int(entry["right_index"])
        except (KeyError, TypeError, ValueError):
            continue
        matches.append((wrong_idx, right_idx))

    metadata = payload.get("metadata", {})
    return matches, metadata


def load_hidden_states_for_layer(
    sample_id: str,
    layer: int,
    hidden_states_dir: Path,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Load wrong/right hidden state tensors for a sample and layer."""
    sample_dir = hidden_states_dir / sample_id
    wrong_path = sample_dir / f"wrong_layer{layer}.pt"
    right_path = sample_dir / f"right_layer{layer}.pt"

    if not wrong_path.exists() or not right_path.exists():
        LOGGER.warning(
            "Hidden state files missing for sample %s layer %d (expected %s / %s)",
            sample_id,
            layer,
            wrong_path,
            right_path,
        )
        return None, None

    try:
        wrong_tensor = torch.load(wrong_path, map_location="cpu").to(torch.float32)
        right_tensor = torch.load(right_path, map_location="cpu").to(torch.float32)
    except Exception as exc:
        LOGGER.warning("Failed to load hidden states for sample %s layer %d: %s", sample_id, layer, exc)
        return None, None

    return wrong_tensor, right_tensor


def build_dp_tail_dataset(
    layer: int,
    sample_ids: np.ndarray,
    dp_tail_k: int,
    alignments_dir: Path,
    hidden_states_dir: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
    """Reconstruct features using the last-k DP-aligned token pairs (fallback to last token)."""
    unique_samples: List[str] = []
    seen = set()
    for sid in sample_ids:
        sid_str = str(sid)
        if sid_str not in seen:
            unique_samples.append(sid_str)
            seen.add(sid_str)

    features: List[np.ndarray] = []
    labels: List[int] = []
    token_positions: List[int] = []
    sample_list: List[str] = []
    dp_tail_flags: List[bool] = []
    pair_indices: List[int] = []

    samples_with_dp = 0
    fallback_samples = 0
    pairs_used = 0
    processed_samples = 0

    for sample_id in unique_samples:
        matches, metadata = load_alignment_payload(sample_id, alignments_dir)
        wrong_tensor, right_tensor = load_hidden_states_for_layer(sample_id, layer, hidden_states_dir)
        if wrong_tensor is None or right_tensor is None:
            fallback_samples += 1
            continue

        tail_pairs: List[Tuple[int, int]] = []
        if matches and metadata.get("strategy") == "hidden_dp" and not metadata.get("used_fallback", False):
            matches_sorted = sorted(matches, key=lambda pair: pair[0])
            tail_pairs = matches_sorted[-dp_tail_k:] if dp_tail_k > 0 else matches_sorted

        processed_samples += 1

        if tail_pairs:
            samples_with_dp += 1
        else:
            fallback_samples += 1
            tail_pairs = []

        if tail_pairs:
            for pair_idx, (wrong_idx, right_idx) in enumerate(tail_pairs):
                if wrong_idx >= wrong_tensor.shape[0] or right_idx >= right_tensor.shape[0]:
                    continue
                wrong_vec = wrong_tensor[wrong_idx].cpu().numpy()
                right_vec = right_tensor[right_idx].cpu().numpy()

                features.append(wrong_vec)
                labels.append(0)
                token_positions.append(int(wrong_idx))
                sample_list.append(sample_id)
                dp_tail_flags.append(True)
                pair_indices.append(pair_idx)

                features.append(right_vec)
                labels.append(1)
                token_positions.append(int(right_idx))
                sample_list.append(sample_id)
                dp_tail_flags.append(True)
                pair_indices.append(pair_idx)
                pairs_used += 1
        else:
            wrong_idx = max(0, wrong_tensor.shape[0] - 1)
            right_idx = max(0, min(right_tensor.shape[0] - 1, wrong_idx))

            features.append(wrong_tensor[wrong_idx].cpu().numpy())
            labels.append(0)
            token_positions.append(int(wrong_idx))
            sample_list.append(sample_id)
            dp_tail_flags.append(False)
            pair_indices.append(-1)

            features.append(right_tensor[right_idx].cpu().numpy())
            labels.append(1)
            token_positions.append(int(right_idx))
            sample_list.append(sample_id)
            dp_tail_flags.append(False)
            pair_indices.append(-1)

    stats = {
        "total_samples": len(unique_samples),
        "processed_samples": processed_samples,
        "samples_with_dp": samples_with_dp,
        "fallback_samples": fallback_samples,
        "pairs_used": pairs_used,
    }

    if not features:
        return (
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int32),
            np.empty((0,), dtype=object),
            np.empty((0,), dtype=bool),
            np.empty((0,), dtype=np.int32),
            stats,
        )

    features_np = np.stack(features)
    labels_np = np.array(labels, dtype=np.int64)
    token_positions_np = np.array(token_positions, dtype=np.int32)
    sample_ids_np = np.array(sample_list, dtype=object)
    dp_tail_flags_np = np.array(dp_tail_flags, dtype=bool)
    pair_indices_np = np.array(pair_indices, dtype=np.int32)

    return (
        features_np,
        labels_np,
        token_positions_np,
        sample_ids_np,
        dp_tail_flags_np,
        pair_indices_np,
        stats,
    )


def fit_linear_probe(features: np.ndarray, labels: np.ndarray, seed: int = 0) -> Dict:
    """
    Train a logistic regression probe and evaluate it.

    Args:
        features: Feature matrix (n_samples, n_features)
        labels: Binary labels (0=wrong, 1=right)
        seed: Random seed

    Returns:
        Dictionary with accuracy and classification report
    """
    if len(np.unique(labels)) < 2:
        return {"warning": "Not enough class diversity for probe."}

    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.3, random_state=seed, stratify=labels
    )

    clf = LogisticRegression(max_iter=1000, n_jobs=-1)
    clf.fit(x_train, y_train)

    preds = clf.predict(x_test)
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, output_dict=True)

    return {
        "accuracy": float(acc),
        "report": report,
        "coefficients": clf.coef_.tolist(),
        "intercept": float(clf.intercept_[0]),
    }


def compute_caa_vector(features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    Compute the Contrast Activation Addition (CAA) vector.
    
    Args:
        features: Hidden state features [num_samples, hidden_dim]
        labels: Binary labels (0=incorrect, 1=correct)
    
    Returns:
        CAA vector (difference between correct and incorrect means)
    """
    correct_features = features[labels == 1]
    incorrect_features = features[labels == 0]
    
    if len(correct_features) == 0 or len(incorrect_features) == 0:
        raise ValueError("Need both correct and incorrect samples to compute CAA vector")
    
    correct_mean = correct_features.mean(axis=0)
    incorrect_mean = incorrect_features.mean(axis=0)
    
    caa_vector = correct_mean - incorrect_mean
    return caa_vector


def compute_separation_metrics(
    features: np.ndarray,
    labels: np.ndarray,
    caa_vector: np.ndarray,
) -> Dict[str, float]:
    """
    Compute separation metrics for the CAA vector.
    
    Args:
        features: Hidden state features
        labels: Binary labels
        caa_vector: The computed contrast vector
    
    Returns:
        Dictionary with separation metrics
    """
    # Project features onto CAA direction
    caa_unit = caa_vector / (np.linalg.norm(caa_vector) + 1e-8)
    projections = features @ caa_unit
    
    correct_proj = projections[labels == 1]
    incorrect_proj = projections[labels == 0]
    
    # Compute separation statistics
    correct_mean = correct_proj.mean()
    incorrect_mean = incorrect_proj.mean()
    correct_std = correct_proj.std()
    incorrect_std = incorrect_proj.std()
    
    # Cohen's d effect size
    pooled_std = np.sqrt((correct_std**2 + incorrect_std**2) / 2)
    cohens_d = (correct_mean - incorrect_mean) / (pooled_std + 1e-8)
    
    return {
        "correct_mean": float(correct_mean),
        "incorrect_mean": float(incorrect_mean),
        "correct_std": float(correct_std),
        "incorrect_std": float(incorrect_std),
        "mean_separation": float(correct_mean - incorrect_mean),
        "cohens_d": float(cohens_d),
    }


def compute_pca_transform(features: np.ndarray, seed: int = 0) -> Dict:
    """
    Compute PCA transformation and return components and explained variance.
    
    Args:
        features: Feature matrix
        seed: Random seed
    
    Returns:
        Dictionary with PCA results
    """
    pca = PCA(n_components=2, random_state=seed)
    transformed = pca.fit_transform(features)
    
    return {
        "transformed": transformed,
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "components": pca.components_.tolist(),
    }


def compute_umap_transform(features: np.ndarray, seed: int = 0) -> Dict:
    """
    Compute UMAP transformation.
    
    Args:
        features: Feature matrix
        seed: Random seed
    
    Returns:
        Dictionary with UMAP results
    """
    if umap is None:
        LOGGER.warning("UMAP not installed, skipping UMAP computation")
        return {"warning": "umap-learn not installed"}
    
    reducer = umap.UMAP(n_components=2, random_state=seed)
    embedded = reducer.fit_transform(features)
    
    return {
        "transformed": embedded,
    }


def main() -> None:
    setup_logging()
    args = parse_args()

    # Load probe data
    if not args.probe_data_path.exists():
        LOGGER.error("Probe data file not found: %s", args.probe_data_path)
        return

    LOGGER.info("Loading probe data from %s", args.probe_data_path)
    data = np.load(args.probe_data_path, allow_pickle=True)
    features = data["features"]
    labels = data["labels"]
    layer = int(data["layer"])
    token_positions = data.get("token_positions")
    sample_ids = data.get("sample_ids")
    if token_positions is not None:
        token_positions = np.array(token_positions, dtype=np.int32)
    if sample_ids is not None:
        sample_ids = np.array(sample_ids, dtype=object)

    total_samples = features.shape[0]
    LOGGER.info("Loaded %d samples from layer %d", total_samples, layer)

    dp_tail_flags: Optional[np.ndarray] = None
    dp_pair_indices: Optional[np.ndarray] = None
    dp_tail_stats: Dict[str, int] = {"k": 0}

    if args.dp_tail_k > 0:
        if sample_ids is None:
            LOGGER.error("--dp-tail-k requires 'sample_ids' in probe data; aborting.")
            return
        (
            features,
            labels,
            token_positions,
            sample_ids,
            dp_tail_flags,
            dp_pair_indices,
            stats,
        ) = build_dp_tail_dataset(
            layer=layer,
            sample_ids=sample_ids,
            dp_tail_k=args.dp_tail_k,
            alignments_dir=args.alignments_dir,
            hidden_states_dir=args.hidden_states_dir,
        )
        if features.size == 0:
            LOGGER.error(
                "DP tail mode requested (k=%d) but no usable token pairs were found.",
                args.dp_tail_k,
            )
            return
        dp_tail_stats = {"k": args.dp_tail_k, **stats}
        LOGGER.info(
            "DP tail sampling applied: %d samples processed, %d with DP matches, %d fallbacks, %d pairs used.",
            stats["processed_samples"],
            stats["samples_with_dp"],
            stats["fallback_samples"],
            stats["pairs_used"],
        )

    valid_mask = data["valid"].astype(bool) if "valid" in data and args.dp_tail_k == 0 else None
    dp_used_alignment = (
        data["dp_used_alignment"].astype(bool)
        if "dp_used_alignment" in data and args.dp_tail_k == 0
        else None
    )
    dp_used_fallback = (
        data["dp_used_fallback"].astype(bool)
        if "dp_used_fallback" in data and args.dp_tail_k == 0
        else None
    )

    if args.dp_tail_k == 0:
        mask = np.ones(features.shape[0], dtype=bool)
        if valid_mask is not None:
            LOGGER.info(
                "Validity flag found: %d valid, %d invalid",
                int(valid_mask.sum()),
                int((~valid_mask).sum()),
            )
            mask &= valid_mask
        else:
            LOGGER.info("No validity flag found; treating all samples as valid.")

        if args.dp_alignment:
            if dp_used_alignment is None:
                LOGGER.warning(
                    "--dp-alignment requested, but probe data lacks 'dp_used_alignment'. Skipping DP filtering."
                )
            else:
                mask &= dp_used_alignment
                if dp_used_fallback is not None:
                    mask &= ~dp_used_fallback
                LOGGER.info(
                    "DP alignment filter applied: %d of %d samples retained",
                    int(mask.sum()),
                    total_samples,
                )

        if mask.sum() == 0:
            LOGGER.error("No samples remain after filtering; nothing to analyze.")
            return

        features = features[mask]
        labels = labels[mask]
        if token_positions is not None:
            token_positions = token_positions[mask]
        if sample_ids is not None:
            sample_ids = sample_ids[mask]
        if dp_used_alignment is not None:
            dp_used_alignment = dp_used_alignment[mask]
        if dp_used_fallback is not None:
            dp_used_fallback = dp_used_fallback[mask]
    else:
        if args.dp_alignment:
            if dp_tail_flags is None:
                LOGGER.warning(
                    "--dp-alignment requested, but DP tail flags are unavailable; skipping DP filtering."
                )
            else:
                dp_mask = dp_tail_flags.astype(bool)
                if dp_mask.sum() == 0:
                    LOGGER.error("DP alignment filter removed all samples; aborting.")
                    return
                features = features[dp_mask]
                labels = labels[dp_mask]
                if token_positions is not None:
                    token_positions = token_positions[dp_mask]
                if sample_ids is not None:
                    sample_ids = sample_ids[dp_mask]
                dp_tail_flags = dp_tail_flags[dp_mask]
                if dp_pair_indices is not None:
                    dp_pair_indices = dp_pair_indices[dp_mask]
                LOGGER.info(
                    "DP alignment filter in tail mode retained %d of %d tokens.",
                    int(dp_mask.sum()),
                    dp_mask.size,
                )

    LOGGER.info("Feature dimension: %d", features.shape[1])
    LOGGER.info(
        "Label distribution: %d wrong, %d right",
        int((labels == 0).sum()),
        int((labels == 1).sum()),
    )

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize results
    results = {
        "probe_layer": layer,
        "num_samples": int(features.shape[0]),
        "feature_dim": int(features.shape[1]),
        "num_wrong": int((labels == 0).sum()),
        "num_right": int((labels == 1).sum()),
    }
    if args.dp_tail_k > 0:
        results["dp_tail"] = dp_tail_stats
    else:
        results["dp_tail"] = {"k": 0}
    if args.dp_tail_k == 0:
        if dp_used_alignment is not None:
            results["dp_alignment"] = {
                "requested": bool(args.dp_alignment),
                "available": True,
                "num_aligned_samples": int(dp_used_alignment.sum()),
                "num_after_filter": int(features.shape[0]),
            }
        elif args.dp_alignment:
            results["dp_alignment"] = {
                "requested": True,
                "available": False,
                "num_aligned_samples": 0,
                "num_after_filter": int(features.shape[0]),
            }
    else:
        if args.dp_alignment and dp_tail_flags is not None:
            results["dp_alignment"] = {
                "requested": True,
                "available": True,
                "num_aligned_samples": int(dp_tail_flags.sum()),
                "num_after_filter": int(features.shape[0]),
            }

    # Skip linear probe, just visualize
    LOGGER.info("Skipping linear probe training, computing visualizations only...")

    # 1. Compute PCA
    LOGGER.info("Computing PCA transformation...")
    pca_result = compute_pca_transform(features, seed=args.seed)
    pca_transformed = pca_result.pop("transformed")
    results["pca"] = pca_result
    LOGGER.info("PCA explained variance: %.3f, %.3f", *pca_result["explained_variance_ratio"])
    
    # Save PCA data
    pca_data_path = args.output_dir / f"layer{layer}_pca_data.npz"
    pca_payload = {
        "transformed": pca_transformed,
        "labels": labels,
        "explained_variance_ratio": np.array(pca_result["explained_variance_ratio"]),
    }
    if token_positions is not None:
        pca_payload["token_positions"] = token_positions
    if sample_ids is not None:
        pca_payload["sample_ids"] = sample_ids
    if dp_tail_flags is not None:
        pca_payload["dp_tail_used"] = dp_tail_flags
    if dp_pair_indices is not None:
        pca_payload["dp_tail_pair_index"] = dp_pair_indices
    if args.dp_tail_k == 0 and dp_used_alignment is not None:
        pca_payload["dp_used_alignment"] = dp_used_alignment
    if args.dp_tail_k == 0 and dp_used_fallback is not None:
        pca_payload["dp_used_fallback"] = dp_used_fallback
    np.savez(pca_data_path, **pca_payload)
    LOGGER.info("Saved PCA data to %s", pca_data_path)

    # 5. Compute UMAP
    LOGGER.info("Computing UMAP transformation...")
    umap_result = compute_umap_transform(features, seed=args.seed)
    if "transformed" in umap_result:
        umap_transformed = umap_result.pop("transformed")
        results["umap"] = umap_result
        
        # Save UMAP data
        umap_data_path = args.output_dir / f"layer{layer}_umap_data.npz"
        umap_payload = {"transformed": umap_transformed, "labels": labels}
        if token_positions is not None:
            umap_payload["token_positions"] = token_positions
        if sample_ids is not None:
            umap_payload["sample_ids"] = sample_ids
        if dp_tail_flags is not None:
            umap_payload["dp_tail_used"] = dp_tail_flags
        if dp_pair_indices is not None:
            umap_payload["dp_tail_pair_index"] = dp_pair_indices
        if args.dp_tail_k == 0 and dp_used_alignment is not None:
            umap_payload["dp_used_alignment"] = dp_used_alignment
        if args.dp_tail_k == 0 and dp_used_fallback is not None:
            umap_payload["dp_used_fallback"] = dp_used_fallback
        np.savez(umap_data_path, **umap_payload)
        LOGGER.info("Saved UMAP data to %s", umap_data_path)
    else:
        results["umap"] = umap_result

    # Save summary metrics
    metrics_path = args.output_dir / f"layer{layer}_metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    LOGGER.info("Saved metrics to %s", metrics_path)
    
    LOGGER.info("All probe computations complete!")


if __name__ == "__main__":
    main()
