#!/usr/bin/env python3
"""Analyze which token positions are most critical for separating correct vs incorrect reasoning."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from setup import setup_logging


LOGGER = logging.getLogger("analyze_critical_tokens")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--probe-data-path",
        type=Path,
        required=True,
        help="Path to .npz file containing probe data with token positions.",
    )
    parser.add_argument(
        "--alignments-dir",
        type=Path,
        default=Path("artifacts/alignments"),
        help="Directory containing alignment JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/critical_tokens"),
        help="Directory for saving analysis results.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of top critical tokens to display.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for PCA.",
    )
    parser.add_argument(
        "--dp-alignment",
        action="store_true",
        help="Restrict analysis to tokens that passed DP alignment (requires metadata).",
    )
    return parser.parse_args()


def load_token_text(sample_id: str, token_idx: int, is_wrong: bool, alignments_dir: Path) -> str:
    """Load the actual token text from alignment file."""
    alignment_path = alignments_dir / f"{sample_id}.json"
    if not alignment_path.exists():
        return "<unknown>"
    
    with open(alignment_path) as f:
        alignment = json.load(f)
    
    tokens = alignment['wrong_tokens'] if is_wrong else alignment['right_tokens']
    if token_idx < len(tokens):
        return tokens[token_idx]
    return "<out_of_range>"


def analyze_critical_tokens(
    features: np.ndarray,
    labels: np.ndarray,
    token_positions: np.ndarray,
    sample_ids: List[str],
    alignments_dir: Path,
    seed: int = 42,
) -> Dict:
    """
    Analyze which token positions contribute most to PC2 separation.
    
    Args:
        features: Feature matrix [num_samples, hidden_dim]
        labels: Binary labels [num_samples]
        token_positions: Token position indices [num_samples]
        sample_ids: Sample IDs for each token [num_samples]
        alignments_dir: Directory with alignment files
        seed: Random seed
    
    Returns:
        Dictionary with analysis results
    """
    # Run PCA
    pca = PCA(n_components=2, random_state=seed)
    transformed = pca.fit_transform(features)
    
    pc1_scores = transformed[:, 0]
    pc2_scores = transformed[:, 1]
    
    # Analyze PC2 contributions (since you noticed separation on y-axis/Component 2)
    results = []
    for idx in range(len(features)):
        is_wrong = labels[idx] == 0
        token_pos = token_positions[idx]
        sample_id = sample_ids[idx]
        pc2_score = pc2_scores[idx]
        
        # Load actual token text
        token_text = load_token_text(sample_id, token_pos, is_wrong, alignments_dir)
        
        results.append({
            'sample_id': sample_id,
            'token_position': int(token_pos),
            'label': 'wrong' if is_wrong else 'right',
            'pc2_score': float(pc2_score),
            'pc1_score': float(pc1_scores[idx]),
            'token_text': token_text,
        })
    
    return {
        'explained_variance': pca.explained_variance_ratio_.tolist(),
        'tokens': results,
    }


def plot_pc2_distribution(
    tokens_data: List[Dict],
    output_path: Path,
) -> None:
    """Plot distribution of PC2 scores for wrong vs right tokens."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    wrong_pc2 = [t['pc2_score'] for t in tokens_data if t['label'] == 'wrong']
    right_pc2 = [t['pc2_score'] for t in tokens_data if t['label'] == 'right']
    
    plt.figure(figsize=(10, 6))
    plt.hist(wrong_pc2, bins=30, alpha=0.6, label='Wrong', color='red', density=True)
    plt.hist(right_pc2, bins=30, alpha=0.6, label='Right', color='blue', density=True)
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    plt.xlabel('PC2 Score', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title('PC2 Score Distribution (Component 2)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    LOGGER.info("Saved PC2 distribution plot to %s", output_path)


def main() -> None:
    setup_logging()
    args = parse_args()
    
    # Load probe data
    LOGGER.info("Loading probe data from %s", args.probe_data_path)
    data = np.load(args.probe_data_path, allow_pickle=True)
    features = data["features"]
    labels = data["labels"]
    token_positions = data.get("token_positions", None)
    sample_ids = data.get("sample_ids", None)
    layer = int(data["layer"])

    total_tokens = len(features)
    LOGGER.info("Loaded %d tokens from layer %d", total_tokens, layer)

    mask = np.ones(total_tokens, dtype=bool)

    if "valid" in data:
        valid = data["valid"].astype(bool)
        LOGGER.info(
            "Validity flag found: %d valid, %d invalid",
            int(valid.sum()),
            int((~valid).sum()),
        )
        mask &= valid
    else:
        LOGGER.info("No validity flag found; treating all tokens as valid.")

    dp_used_alignment = data["dp_used_alignment"].astype(bool) if "dp_used_alignment" in data else None
    dp_used_fallback = data["dp_used_fallback"].astype(bool) if "dp_used_fallback" in data else None

    if args.dp_alignment:
        if dp_used_alignment is None:
            LOGGER.warning(
                "--dp-alignment requested, but probe data lacks 'dp_used_alignment'. Skipping DP filter."
            )
        else:
            mask &= dp_used_alignment
            if dp_used_fallback is not None:
                mask &= ~dp_used_fallback
            LOGGER.info(
                "DP alignment filter applied: %d of %d tokens retained",
                int(mask.sum()),
                total_tokens,
            )

    retained_tokens = int(mask.sum())
    if retained_tokens == 0:
        LOGGER.error("No tokens remain after filtering; aborting critical token analysis.")
        return

    features = features[mask]
    labels = labels[mask]
    if token_positions is not None:
        token_positions = token_positions[mask]
    if sample_ids is not None:
        sample_ids = sample_ids[mask]
    LOGGER.info("Using %d tokens after filtering.", retained_tokens)

    if token_positions is None:
        LOGGER.error("Probe data does not contain token positions. Re-run collect_hidden_states.py")
        return
    
    # Analyze critical tokens
    LOGGER.info("Analyzing token contributions to PC2 separation...")
    results = analyze_critical_tokens(
        features, labels, token_positions, sample_ids, args.alignments_dir, seed=args.seed
    )
    results["filter_summary"] = {
        "total_tokens": int(total_tokens),
        "tokens_retained": retained_tokens,
        "dp_alignment_requested": bool(args.dp_alignment),
        "dp_alignment_available": dp_used_alignment is not None,
    }
    
    explained_var = results['explained_variance']
    LOGGER.info("PCA explained variance: PC1=%.3f, PC2=%.3f", *explained_var)
    
    tokens_data = results['tokens']
    
    # Sort by PC2 score to find most separating tokens
    tokens_sorted = sorted(tokens_data, key=lambda x: x['pc2_score'])
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save full results
    results_path = args.output_dir / f"layer{layer}_critical_tokens.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    LOGGER.info("Saved full results to %s", results_path)
    
    # Plot PC2 distribution
    plot_path = args.output_dir / f"layer{layer}_pc2_distribution.png"
    plot_pc2_distribution(tokens_data, plot_path)
    
    # Print top tokens with lowest PC2 (most "wrong-like")
    print("\n" + "="*80)
    print(f"Top {args.top_k} tokens with LOWEST PC2 scores (most wrong-like):")
    print("="*80)
    for i, token_data in enumerate(tokens_sorted[:args.top_k], 1):
        print(f"{i:2d}. [{token_data['label']:>5s}] PC2={token_data['pc2_score']:7.3f} "
              f"pos={token_data['token_position']:3d} "
              f"'{token_data['token_text']}' ({token_data['sample_id']})")
    
    # Print top tokens with highest PC2 (most "right-like")
    print("\n" + "="*80)
    print(f"Top {args.top_k} tokens with HIGHEST PC2 scores (most right-like):")
    print("="*80)
    for i, token_data in enumerate(reversed(tokens_sorted[-args.top_k:]), 1):
        print(f"{i:2d}. [{token_data['label']:>5s}] PC2={token_data['pc2_score']:7.3f} "
              f"pos={token_data['token_position']:3d} "
              f"'{token_data['token_text']}' ({token_data['sample_id']})")
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)


if __name__ == "__main__":
    main()
