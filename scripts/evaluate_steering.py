#!/usr/bin/env python3
"""Evaluate steering vectors on triples to see if they improve predictions."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from model_wrapper import ModelWrapper
from modules.layer_features import build_layer_feature_dict, load_steering_metadata
from modules.layer_selector import LayerSelectorMLP
from setup import (
    configure_hf_caches,
    setup_logging,
    extract_final_answer,
    normalize_answer,
    answers_match,
    reconstruct_prompt,
)


LOGGER = logging.getLogger("evaluate_steering")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--triples-path",
        type=Path,
        required=True,
        help="Path to JSONL file produced by prepare_triples.py",
    )
    parser.add_argument(
        "--steering-vectors-dir",
        type=Path,
        default=Path("artifacts/steering_vectors"),
        help="Directory containing steering vectors (layer{layer_id}_steering_vector.npy files).",
    )
    parser.add_argument(
        "--model-name",
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="Identifier for a causal LM (default mirrors Step 1).",
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        required=True,
        type=int,
        help="List of layer ids to use for steering (e.g., --layers 28).",
    )
    parser.add_argument(
        "--steering-coefficient",
        type=float,
        default=1.0,
        help="Multiplier for steering vectors (default: 1.0).",
    )
    parser.add_argument(
        "--layer-selection-method",
        choices=["fixed", "mlp"],
        default="fixed",
        help="How to choose layers during steering evaluation.",
    )
    parser.add_argument(
        "--layer-selection-mlp-path",
        type=Path,
        default=None,
        help="Checkpoint for the layer selector (required when method=mlp).",
    )
    parser.add_argument(
        "--layer-selection-topk",
        type=int,
        default=1,
        help="Number of top layers to keep when using layer selector (default: 1).",
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip baseline (without steering) evaluation to save computation time.",
    )
    parser.add_argument(
        "--baseline-correct",
        type=int,
        default=None,
        help="Number of correct baseline predictions (required if --skip-baseline is used).",
    )
    parser.add_argument(
        "--baseline-incorrect",
        type=int,
        default=None,
        help="Number of incorrect baseline predictions (required if --skip-baseline is used).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/steered_embeddings"),
        help="Directory for saving evaluation results and steered embeddings.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on number of triples to evaluate (for debugging).",
    )
    parser.add_argument(
        "--sample-offset",
        type=int,
        default=0,
        help="Number of triples to skip before evaluation (useful for out-of-sample splits).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="Maximum tokens to generate.",
    )
    parser.add_argument(
        "--system-prompt",
        default="You are a careful reasoning assistant. Think step by step and end with 'Final answer: <choice>'.",
        help="System message used when the tokenizer supports chat templates.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional torch device override (defaults to cuda if available else cpu).",
    )
    parser.add_argument(
        "--hidden-states-dir",
        type=Path,
        default=Path("artifacts/hidden_states"),
        help="Directory containing stored embeddings (sample_id/wrong_layer{layer}.pt).",
    )
    parser.add_argument(
        "--save-steered-embeddings",
        action="store_true",
        help="Save steered embeddings by applying steering vectors to wrong embeddings.",
    )
    parser.add_argument(
        "--pooling-method",
        type=str,
        choices=["last_token", "mean"],
        default="last_token",
        help="Method to pool hidden states when saving steered embeddings: last_token or mean.",
    )
    parser.add_argument(
        "--steering-metadata-path",
        type=Path,
        default=Path("artifacts/steering_vectors/steering_vectors_metadata.json"),
        help="Metadata file produced during steering vector computation.",
    )
    parser.add_argument(
        "--probe-metrics-dir",
        type=Path,
        default=Path("artifacts/probe_analysis"),
        help="Directory with per-layer probe metrics (used for layer selection features).",
    )
    return parser.parse_args()


def load_triples(path: Path) -> List[Dict]:
    """Load triples from JSONL or JSON file."""
    triples: List[Dict] = []
    with path.open("r", encoding="utf-8") as src:
        first_chunk = src.read(1)

    if not first_chunk:
        LOGGER.error("Triples file %s is empty.", path)
        return triples

    is_json_array = first_chunk in ("[", "{")

    if is_json_array:
        with path.open("r", encoding="utf-8") as src:
            try:
                payloads = json.load(src)
            except json.JSONDecodeError as exc:
                LOGGER.error("Failed to parse JSON file %s: %s", path, exc)
                return triples

        if isinstance(payloads, dict):
            payloads = payloads.get("records") or payloads.get("data") or []
            if not isinstance(payloads, list):
                LOGGER.error("JSON file %s does not contain a list of records.", path)
                return triples
    else:
        payloads = []
        with path.open("r", encoding="utf-8") as src:
            for line in src:
                line = line.strip()
                if not line:
                    continue
                try:
                    payloads.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    LOGGER.warning("Skipping malformed JSON line: %s (error: %s)", line[:200], exc)

    for payload in payloads:
        if isinstance(payload, dict) and payload.get("correct_chain"):
            triples.append(payload)
    return triples


def load_steering_vectors(steering_dir: Path, layers: List[int]) -> Dict[int, np.ndarray]:
    """Load steering vectors for specified layers."""
    steering_vectors = {}
    for layer_id in layers:
        steering_path = steering_dir / f"layer{layer_id}_steering_vector.npy"
        if not steering_path.exists():
            LOGGER.warning("Steering vector not found for layer %d at %s", layer_id, steering_path)
            continue
        steering_vector = np.load(steering_path)
        steering_vectors[layer_id] = steering_vector
        LOGGER.info(
            "Loaded steering vector for layer %d (shape: %s, norm: %.4f)",
            layer_id,
            steering_vector.shape,
            np.linalg.norm(steering_vector),
        )
    return steering_vectors


def select_layers_via_mlp(
    args: argparse.Namespace,
    steering_vectors: Dict[int, np.ndarray],
) -> List[int]:
    model, norm_stats = LayerSelectorMLP.load(args.layer_selection_mlp_path, map_location="cpu")
    feature_keys = (norm_stats.get("feature_keys") if norm_stats else None) or model.config.feature_keys
    if not feature_keys:
        raise ValueError("Layer selector checkpoint is missing feature key metadata.")
    steering_metadata = load_steering_metadata(args.steering_metadata_path)

    mean = torch.tensor(norm_stats["mean"], dtype=torch.float32) if norm_stats else None
    std = torch.tensor(norm_stats["std"], dtype=torch.float32) if norm_stats else None

    scores: List[Tuple[int, float]] = []
    for layer_id in sorted(steering_vectors.keys()):
        feature_dict = build_layer_feature_dict(layer_id, steering_metadata, args.probe_metrics_dir)
        feature_vector = torch.tensor(
            [float(feature_dict.get(key, 0.0)) for key in feature_keys],
            dtype=torch.float32,
        )
        if mean is not None and std is not None:
            feature_vector = (feature_vector - mean) / std
        with torch.no_grad():
            score = model.score(feature_vector.unsqueeze(0)).item()
        scores.append((layer_id, score))

    scores.sort(key=lambda item: item[1], reverse=True)
    topk = max(1, args.layer_selection_topk)
    selected_layers = [layer for layer, _ in scores[:topk]]
    LOGGER.info("Layer selector scores: %s", scores)
    LOGGER.info("Selected top-%d layers via MLP: %s", topk, selected_layers)
    return selected_layers


def get_sample_ids_from_triples(triples: List[Dict]) -> List[str]:
    """Extract sample IDs from triples."""
    return [triple.get("sample_id", "") for triple in triples if triple.get("sample_id")]


def save_steered_embeddings(
    hidden_states_dir: Path,
    output_dir: Path,
    sample_ids: List[str],
    layers: List[int],
    steering_vectors: Dict[int, np.ndarray],
    steering_coefficient: float,
    pooling_method: str = "last_token",
) -> None:
    """
    Apply steering vectors to wrong embeddings and save as steered embeddings.
    
    Args:
        hidden_states_dir: Directory containing sample_id subdirectories with wrong embeddings
        output_dir: Directory where steered embeddings will be saved
        sample_ids: List of sample IDs to process
        layers: List of layer IDs to process
        steering_vectors: Dictionary mapping layer_id -> steering vector [hidden_dim]
        steering_coefficient: Multiplier for steering vector
        pooling_method: Method to pool hidden states (for logging, but we save full tensors)
    """
    LOGGER.info("=" * 80)
    LOGGER.info("Saving steered embeddings")
    LOGGER.info("=" * 80)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for layer in layers:
        if layer not in steering_vectors:
            LOGGER.warning("No steering vector for layer %d, skipping", layer)
            continue
        
        steering_vector = steering_vectors[layer]
        steering_tensor = torch.from_numpy(steering_vector * steering_coefficient)
        
        saved_count = 0
        skipped_count = 0
        
        for sample_id in tqdm(sample_ids, desc=f"Saving steered embeddings for layer {layer}"):
            # Load wrong embedding from hidden_states_dir
            wrong_sample_dir = hidden_states_dir / sample_id
            wrong_path = wrong_sample_dir / f"wrong_layer{layer}.pt"
            
            # Save steered embedding to output_dir
            steered_sample_dir = output_dir / sample_id
            steered_path = steered_sample_dir / f"steered_layer{layer}.pt"
            
            if not wrong_path.exists():
                LOGGER.debug("Wrong embedding not found: %s", wrong_path)
                skipped_count += 1
                continue
            
            try:
                # Load wrong embedding tensor [seq_len, hidden_dim]
                wrong_tensor = torch.load(wrong_path, map_location="cpu")
                
                # Apply steering to all tokens and save full tensor
                steered_tensor_full = wrong_tensor.clone()
                for i in range(steered_tensor_full.shape[0]):
                    steered_tensor_full[i] = steered_tensor_full[i] + steering_tensor
                
                # Save steered embedding
                steered_sample_dir.mkdir(parents=True, exist_ok=True)
                torch.save(steered_tensor_full, steered_path)
                saved_count += 1
                
            except Exception as e:
                LOGGER.warning("Error processing sample %s, layer %d: %s", sample_id, layer, e)
                skipped_count += 1
                continue
        
        LOGGER.info(
            "Layer %d: Saved %d steered embeddings, skipped %d samples",
            layer,
            saved_count,
            skipped_count,
        )
    
    LOGGER.info("=" * 80)
    LOGGER.info("Finished saving steered embeddings")
    LOGGER.info("=" * 80)


def main() -> None:
    setup_logging()
    configure_hf_caches()
    args = parse_args()

    # Validate baseline arguments
    if args.skip_baseline:
        if args.baseline_correct is None or args.baseline_incorrect is None:
            LOGGER.error("--baseline-correct and --baseline-incorrect are required when --skip-baseline is used.")
            return
        LOGGER.info("Skipping baseline evaluation. Using provided baseline: %d correct, %d incorrect", 
                   args.baseline_correct, args.baseline_incorrect)

    # Load triples
    triples = load_triples(args.triples_path)
    if not triples:
        LOGGER.error("No usable triples found. Exiting.")
        return

    if args.sample_offset:
        triples = triples[args.sample_offset:]

    if args.max_samples:
        triples = triples[:args.max_samples]

    LOGGER.info("Loaded %d triples for evaluation", len(triples))

    # Validate baseline matches number of triples if provided
    if args.skip_baseline:
        if args.baseline_correct + args.baseline_incorrect != len(triples):
            LOGGER.warning("Baseline totals (%d) don't match number of triples (%d). Continuing anyway.",
                          args.baseline_correct + args.baseline_incorrect, len(triples))

    # Load steering vectors
    steering_vectors = load_steering_vectors(args.steering_vectors_dir, args.layers)
    if not steering_vectors:
        LOGGER.error("No steering vectors loaded. Exiting.")
        return

    selected_layers = list(args.layers)
    if args.layer_selection_method == "mlp":
        if not args.layer_selection_mlp_path:
            LOGGER.error("--layer-selection-mlp-path is required when using layer-selection-method=mlp.")
            return
        try:
            selected_layers = select_layers_via_mlp(args, steering_vectors)
        except Exception as exc:
            LOGGER.error("Layer selection failed: %s", exc)
            return
        steering_vectors = {layer: steering_vectors[layer] for layer in selected_layers if layer in steering_vectors}
        if not steering_vectors:
            LOGGER.error("Layer selector chose layers with no available steering vectors.")
            return

    args.layers = selected_layers

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Save steered embeddings if requested
    if args.save_steered_embeddings:
        sample_ids = get_sample_ids_from_triples(triples)
        save_steered_embeddings(
            args.hidden_states_dir,
            args.output_dir,
            sample_ids,
            args.layers,
            steering_vectors,
            args.steering_coefficient,
            args.pooling_method,
        )

    # Load model
    model = ModelWrapper(args.model_name, device=args.device)

    # Evaluation results
    results = {
        "model_name": args.model_name,
        "layers": args.layers,
        "steering_coefficient": args.steering_coefficient,
        "num_triples": len(triples),
        "sample_offset": args.sample_offset,
        "without_steering": {
            "correct": args.baseline_correct if args.skip_baseline else 0,
            "incorrect": args.baseline_incorrect if args.skip_baseline else 0,
            "accuracy": 0.0,
            "predictions": [],
        },
        "with_steering": {
            "correct": 0,
            "incorrect": 0,
            "accuracy": 0.0,
            "predictions": [],
        },
        "improvements": 0,  # Cases where steering fixed wrong predictions
        "degradations": 0,  # Cases where steering broke correct predictions
        "skip_baseline": args.skip_baseline,
    }

    # Evaluate each triple
    for triple in tqdm(triples, desc="Evaluating steering"):
        sample_id = triple.get("sample_id", "unknown")
        prompt_text = triple.get("prompt", "")
        correct_answer = triple.get("correct_answer", "")
        metadata = triple.get("metadata", {})
        
        # Reconstruct prompt using template from metadata
        formatted_prompt = reconstruct_prompt(metadata, prompt_text)    # this contains the {questions} and then the {prompt_template}

        # Generate without steering (skip if baseline is skipped)
        predicted_no_steering = ""
        correct_no_steering = False
        response_no_steering = ""
        if not args.skip_baseline:
            try:
                response_no_steering = model.generate(
                    formatted_prompt,
                    max_new_tokens=args.max_new_tokens,
                    temperature=0.0,
                    strip_prompt=True,
                    system_prompt=args.system_prompt,
                )
                predicted_no_steering = extract_final_answer(response_no_steering)
                correct_no_steering = answers_match(predicted_no_steering, correct_answer)
            except Exception as e:
                LOGGER.warning("Error generating without steering for sample %s: %s", sample_id, e)
                response_no_steering = ""
                predicted_no_steering = ""
                correct_no_steering = False

        # Generate with steering
        try:
            # Pass numpy arrays directly - generate_with_steering will handle conversion and device placement
            response_with_steering = model.generate_with_steering(
                formatted_prompt,
                steering_vectors=steering_vectors,
                steering_coefficient=args.steering_coefficient,
                max_new_tokens=args.max_new_tokens,
                temperature=0.0,
                strip_prompt=True,
                system_prompt=args.system_prompt,
            )
            predicted_with_steering = extract_final_answer(response_with_steering)
            print(f"With steering: {predicted_with_steering}")
            correct_with_steering = answers_match(predicted_with_steering, correct_answer)
        except Exception as e:
            LOGGER.warning("Error generating with steering for sample %s: %s", sample_id, e)
            response_with_steering = ""
            predicted_with_steering = ""
            correct_with_steering = False

        # Update results
        if not args.skip_baseline:
            if correct_no_steering:
                results["without_steering"]["correct"] += 1
            else:
                results["without_steering"]["incorrect"] += 1

            # Track improvements and degradations (only if baseline was computed)
            if not correct_no_steering and correct_with_steering:
                results["improvements"] += 1
            elif correct_no_steering and not correct_with_steering:
                results["degradations"] += 1

        if correct_with_steering:
            results["with_steering"]["correct"] += 1
        else:
            results["with_steering"]["incorrect"] += 1

        print(f"results['with_steering']['correct']: {results['with_steering']['correct']}")

        # Store predictions
        if not args.skip_baseline:
            results["without_steering"]["predictions"].append({
                "sample_id": sample_id,
                "predicted": predicted_no_steering,
                "correct": correct_answer,
                "is_correct": correct_no_steering,
                "response": response_no_steering[:500],  # Truncate for storage
            })
        results["with_steering"]["predictions"].append({
            "sample_id": sample_id,
            "predicted": predicted_with_steering,
            "correct": correct_answer,
            "is_correct": correct_with_steering,
            "response": response_with_steering[:500],  # Truncate for storage
        })

    # Calculate accuracies
    total = len(triples)
    results["without_steering"]["accuracy"] = results["without_steering"]["correct"] / total if total > 0 else 0.0
    results["with_steering"]["accuracy"] = results["with_steering"]["correct"] / total if total > 0 else 0.0

    # Save results
    split_suffix = f"offset{args.sample_offset}" if args.sample_offset else "offset0"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / f"steering_evaluation_results_{split_suffix}.json"
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    LOGGER.info("Saved evaluation results to %s", results_path)

    # Print summary
    LOGGER.info("=" * 60)
    LOGGER.info("Steering Evaluation Results")
    LOGGER.info("=" * 60)
    LOGGER.info("Total triples evaluated: %d", total)
    LOGGER.info("")
    if args.skip_baseline:
        LOGGER.info("Without steering (from provided baseline):")
        LOGGER.info("  Correct: %d", results["without_steering"]["correct"])
        LOGGER.info("  Incorrect: %d", results["without_steering"]["incorrect"])
        LOGGER.info("  Accuracy: %.2f%%", results["without_steering"]["accuracy"] * 100)
    else:
        LOGGER.info("Without steering:")
        LOGGER.info("  Correct: %d", results["without_steering"]["correct"])
        LOGGER.info("  Incorrect: %d", results["without_steering"]["incorrect"])
        LOGGER.info("  Accuracy: %.2f%%", results["without_steering"]["accuracy"] * 100)
    LOGGER.info("")
    LOGGER.info("With steering (coefficient=%.2f):", args.steering_coefficient)
    LOGGER.info("  Correct: %d", results["with_steering"]["correct"])
    LOGGER.info("  Incorrect: %d", results["with_steering"]["incorrect"])
    LOGGER.info("  Accuracy: %.2f%%", results["with_steering"]["accuracy"] * 100)
    LOGGER.info("")
    if not args.skip_baseline:
        LOGGER.info("Improvements (wrong -> correct): %d", results["improvements"])
        LOGGER.info("Degradations (correct -> wrong): %d", results["degradations"])
        LOGGER.info("")
        accuracy_diff = results["with_steering"]["accuracy"] - results["without_steering"]["accuracy"]
        LOGGER.info("Accuracy change: %+.2f%%", accuracy_diff * 100)
    else:
        accuracy_diff = results["with_steering"]["accuracy"] - results["without_steering"]["accuracy"]
        LOGGER.info("Accuracy change (vs. baseline): %+.2f%%", accuracy_diff * 100)
        LOGGER.info("(Improvements/degradations not tracked when baseline is skipped)")
    LOGGER.info("=" * 60)


if __name__ == "__main__":
    main()

