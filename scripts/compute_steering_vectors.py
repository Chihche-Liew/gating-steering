#!/usr/bin/env python3
"""Compute steering vectors from last_token hidden states (right - wrong, averaged)."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from model_wrapper import ModelWrapper
from modules.token_selector import TokenSelectorMLP
from setup import (
    configure_hf_caches,
    setup_logging,
    build_teacher_forcing_text,
    reconstruct_prompt,
)


LOGGER = logging.getLogger("compute_steering_vectors")


@dataclass
class Triple:
    sample_id: str
    prompt: str
    wrong_chain: str
    correct_chain: str
    wrong_answer: str
    correct_answer: str
    metadata: Dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--triples-path",
        type=Path,
        required=True,
        help="Path to JSONL file produced by prepare_triples.py",
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
        help="List of 0-indexed transformer layer ids to compute steering vectors for (e.g., --layers 28 30 31).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/steering_vectors"),
        help="Directory for saving steering vectors.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on number of triples to process (for debugging).",
    )
    parser.add_argument(
        "--token-selection-method",
        choices=["last_token", "gradient", "dp_gradient", "dp_average", "token_mlp"],
        default="last_token",
        help="How to choose the token position for computing steering vectors.",
    )
    parser.add_argument(
        "--token-selection-mlp-path",
        type=Path,
        default=None,
        help="Checkpoint for token selector MLP (required when method=token_mlp).",
    )
    parser.add_argument(
        "--alignments-dir",
        type=Path,
        default=Path("artifacts/alignments"),
        help="Directory containing DP alignment JSON files (required for dp_* token selection).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional torch device override (defaults to cuda if available else cpu).",
    )
    parser.add_argument(
        "--system-prompt",
        default="You are a careful reasoning assistant. Think step by step and end with 'Final answer: <choice>'.",
        help="System message used when the tokenizer supports chat templates.",
    )
    return parser.parse_args()


def load_triples(path: Path) -> List[Triple]:
    """Load triples from JSONL or JSON file."""
    triples: List[Triple] = []
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
        if not isinstance(payload, dict):
            continue
        correct_chain = payload.get("correct_chain")
        if not correct_chain:
            continue
        triples.append(
            Triple(
                sample_id=str(payload.get("sample_id")),
                prompt=str(payload.get("prompt", "")),
                wrong_chain=str(payload.get("wrong_chain", "")),
                correct_chain=str(correct_chain),
                wrong_answer=str(payload.get("wrong_answer", "")),
                correct_answer=str(payload.get("correct_answer", "")),
                metadata=dict(payload.get("metadata", {})),
            )
        )
    return triples


def encode_answer_token_id(model: ModelWrapper, answer: str) -> Optional[int]:
    """Encode an answer string and return the final token id (if available)."""
    if not answer:
        return None
    cleaned = answer.strip()
    if not cleaned:
        return None
    token_ids = model.tokenizer.encode(cleaned, add_special_tokens=False)
    if token_ids:
        return int(token_ids[-1])
    # Try using only the first character (useful for single letter answers)
    if len(cleaned) > 1:
        token_ids = model.tokenizer.encode(cleaned[0], add_special_tokens=False)
        if token_ids:
            return int(token_ids[-1])
    return None


def compute_prompt_token_length(model: ModelWrapper, prompt_text: str) -> int:
    """Return the number of tokens corresponding to the prompt (used to mask gradients)."""
    encoded = model.tokenize(prompt_text)
    return int(encoded["input_ids"].shape[1])


def load_alignment_payload(
    sample_id: str,
    alignments_dir: Path,
) -> Tuple[List[Tuple[int, int]], Dict[str, Any]]:
    """Load precomputed alignment matches for a sample."""
    path = alignments_dir / f"{sample_id}.json"
    if not path.exists():
        return [], {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        LOGGER.warning("Failed to read alignment file %s: %s", path, exc)
        return [], {}

    matches: List[Tuple[int, int]] = []
    for item in payload.get("matches", []):
        if not isinstance(item, dict):
            continue
        try:
            wrong_idx = int(item["wrong_index"])
            right_idx = int(item["right_index"])
        except (KeyError, TypeError, ValueError):
            continue
        matches.append((wrong_idx, right_idx))

    metadata = payload.get("metadata", {})
    return matches, metadata


def select_tokens_via_gradient(
    model: ModelWrapper,
    forward_outputs: Dict[str, Any],
    target_layers: List[int],
    prompt_token_count: int,
    correct_token_id: Optional[int],
    wrong_token_id: Optional[int],
) -> Tuple[Dict[int, int], Dict[int, float]]:
    """
    Use gradient of the correct answer logit (optionally contrasted with wrong answer logit)
    to identify the most influential token per layer.
    """
    if correct_token_id is None:
        raise ValueError("Correct answer token id unavailable for gradient-based selection.")

    logits: torch.Tensor = forward_outputs["logits"]
    last_position = logits.shape[0] - 1

    score = logits[last_position, correct_token_id]
    if wrong_token_id is not None and wrong_token_id != correct_token_id:
        score = score - logits[last_position, wrong_token_id]

    model.model.zero_grad(set_to_none=True)
    score.backward(retain_graph=False)

    selected_positions: Dict[int, int] = {}
    selected_norms: Dict[int, float] = {}

    for layer_id in target_layers:
        layer_tensor: torch.Tensor = forward_outputs["hidden_states"][layer_id]
        grad_tensor = layer_tensor.grad
        if grad_tensor is None:
            raise ValueError(f"Gradient not available for layer {layer_id}.")

        grad_tensor = grad_tensor.squeeze(0)
        grad_norm = grad_tensor.norm(p=2, dim=1)
        if grad_norm.numel() == 0:
            raise ValueError(f"No tokens available for layer {layer_id}.")

        mask_upto = min(prompt_token_count, grad_norm.shape[0] - 1)
        if mask_upto > 0:
            grad_norm[:mask_upto] = -float("inf")

        token_idx_tensor = torch.argmax(grad_norm)
        token_idx = int(token_idx_tensor.item())

        if token_idx < 0 or token_idx >= grad_norm.shape[0]:
            raise ValueError(f"Invalid token index {token_idx} for layer {layer_id}.")

        selected_positions[layer_id] = token_idx
        selected_norms[layer_id] = float(grad_norm[token_idx_tensor].item())

    model.model.zero_grad(set_to_none=True)

    return selected_positions, selected_norms


def summarize_numeric(values: List[float]) -> Dict[str, Any]:
    """Compute basic summary statistics for a list of numeric values."""
    if not values:
        return {"count": 0}
    arr = np.array(values, dtype=float)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def main() -> None:
    setup_logging()
    configure_hf_caches()
    args = parse_args()

    # Load triples
    triples = load_triples(args.triples_path)
    if not triples:
        LOGGER.error("No usable triples found (correct_chain missing?). Exiting.")
        return

    if args.max_samples:
        triples = triples[:args.max_samples]

    LOGGER.info("Loaded %d triples", len(triples))

    # Setup layers
    target_layers = sorted(set(args.layers))
    LOGGER.info("Target layers for steering vector computation: %s", target_layers)

    # Load model
    model = ModelWrapper(args.model_name, device=args.device)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Collect difference vectors per layer
    # For each layer, we'll collect: right_last_token - wrong_last_token
    difference_vectors_per_layer: Dict[int, List[np.ndarray]] = {
        layer: [] for layer in target_layers
    }
    token_positions_per_layer: Dict[int, List[int]] = {
        layer: [] for layer in target_layers
    }
    gradient_norms_per_layer: Dict[int, List[float]] = {
        layer: [] for layer in target_layers
    }

    processed = 0
    skipped = 0
    gradient_fallbacks = 0

    method = args.token_selection_method
    use_gradient = method in {"gradient", "dp_gradient"}
    use_dp_alignment = method in {"dp_gradient", "dp_average"}
    use_token_mlp = method == "token_mlp"
    token_selector: TokenSelectorMLP | None = None
    if use_token_mlp:
        if not args.token_selection_mlp_path:
            LOGGER.error("--token-selection-mlp-path is required when method=token_mlp.")
            return
        token_selector = TokenSelectorMLP.load(args.token_selection_mlp_path, map_location="cpu")
        LOGGER.info("Loaded token selector MLP from %s", args.token_selection_mlp_path)
    if use_dp_alignment and not args.alignments_dir.exists():
        LOGGER.warning(
            "Alignments directory %s not found; DP-based token selection will fall back to default behavior.",
            args.alignments_dir,
        )

    dp_alignment_missing_files = 0
    dp_alignment_fallbacks = 0
    dp_alignment_unavailable = 0
    dp_alignment_attempts = 0
    dp_alignment_successes = 0
    dp_lookup_misses = 0
    dp_average_fallbacks = 0
    dp_pairs_used_total = 0
    dp_average_pair_counts_per_layer = (
        {layer: [] for layer in target_layers} if method == "dp_average" else None
    )

    for triple in tqdm(triples, desc="Computing steering vectors"):
        if not triple.correct_chain.strip():
            skipped += 1
            continue

        # Build texts
        wrong_text = build_teacher_forcing_text(
            triple.prompt, triple.wrong_chain, triple.metadata
        )
        right_text = build_teacher_forcing_text(
            triple.prompt, triple.correct_chain, triple.metadata
        )

        formatted_prompt = reconstruct_prompt(triple.metadata, triple.prompt)
        prompt_token_count = compute_prompt_token_length(model, formatted_prompt)

        # Forward passes
        wrong_forward = model.forward_with_hidden_states(
            wrong_text,
            target_layers=target_layers,
            enable_grad=use_gradient,
            system_prompt=args.system_prompt,
        )
        right_forward = model.forward_with_hidden_states(
            right_text,
            target_layers=target_layers,
            enable_grad=False,
            system_prompt=args.system_prompt,
        )

        matches: List[Tuple[int, int]] = []
        alignment_metadata: Dict[str, Any] = {}
        match_lookup: Dict[int, int] = {}
        dp_alignment_available = False

        if use_dp_alignment:
            matches, alignment_metadata = load_alignment_payload(triple.sample_id, args.alignments_dir)
            if not matches and not alignment_metadata:
                dp_alignment_missing_files += 1
            else:
                strategy = alignment_metadata.get("strategy")
                used_fallback = alignment_metadata.get("used_fallback", False)
                if strategy == "hidden_dp" and not used_fallback and len(matches) > 0:
                    dp_alignment_available = True
                    match_lookup = {wrong: right for wrong, right in matches}
                else:
                    if used_fallback:
                        dp_alignment_fallbacks += 1
                    else:
                        dp_alignment_unavailable += 1

        selected_positions: Dict[int, int] = {}
        gradient_norms: Dict[int, float] = {}
        gradient_used = False

        if use_gradient:
            try:
                correct_token_id = encode_answer_token_id(model, triple.correct_answer)
                wrong_token_id = encode_answer_token_id(model, triple.wrong_answer)
                selected_positions, gradient_norms = select_tokens_via_gradient(
                    model=model,
                    forward_outputs=wrong_forward,
                    target_layers=target_layers,
                    prompt_token_count=prompt_token_count,
                    correct_token_id=correct_token_id,
                    wrong_token_id=wrong_token_id,
                )
                gradient_used = True
            except Exception as exc:
                gradient_fallbacks += 1
                LOGGER.warning(
                    "Gradient-based token selection failed for sample %s: %s. Falling back to last token.",
                    triple.sample_id,
                    exc,
                )
                selected_positions = {}
                gradient_used = False
                model.model.zero_grad(set_to_none=True)

        if method == "token_mlp" and token_selector is not None:
            for layer_id in target_layers:
                hidden_states = wrong_forward["hidden_states"][layer_id].detach().squeeze(0).cpu()
                seq_len = hidden_states.shape[0]
                rel_positions = torch.linspace(0, 1, steps=seq_len, dtype=torch.float32).unsqueeze(-1)
                with torch.no_grad():
                    scores = token_selector.score_tokens(hidden_states, rel_positions)
                if prompt_token_count > 0:
                    scores[:prompt_token_count] = -float("inf")
                token_idx = int(torch.argmax(scores).item())
                selected_positions[layer_id] = token_idx

        if method != "dp_average":
            # Fallback or last token selection
            for layer_id in target_layers:
                layer_states = wrong_forward["hidden_states"][layer_id]
                seq_len = layer_states.shape[0]
                default_idx = max(0, seq_len - 1)
                selected_positions.setdefault(layer_id, default_idx)

        if method == "dp_average":
            for layer_id in target_layers:
                wrong_states = wrong_forward["hidden_states"][layer_id].squeeze(0)
                right_states = right_forward["hidden_states"][layer_id].squeeze(0)

                pair_count = 0
                if dp_alignment_available:
                    dp_alignment_attempts += 1
                    pair_diffs: List[np.ndarray] = []
                    for wrong_idx, right_idx in matches:
                        if (
                            0 <= wrong_idx < wrong_states.shape[0]
                            and 0 <= right_idx < right_states.shape[0]
                        ):
                            wrong_vec = wrong_states[wrong_idx].detach().float().cpu().numpy()
                            right_vec = right_states[right_idx].detach().float().cpu().numpy()
                            pair_diffs.append(right_vec - wrong_vec)
                    pair_count = len(pair_diffs)
                    if dp_average_pair_counts_per_layer is not None:
                        dp_average_pair_counts_per_layer[layer_id].append(pair_count)
                    if pair_count > 0:
                        dp_alignment_successes += 1
                        dp_pairs_used_total += pair_count
                        diff = np.mean(pair_diffs, axis=0)
                        difference_vectors_per_layer[layer_id].append(diff)
                        token_positions_per_layer[layer_id].append(-1)
                        continue
                    dp_average_fallbacks += 1

                fallback_idx = max(0, wrong_states.shape[0] - 1)
                fallback_right_idx = max(0, min(fallback_idx, right_states.shape[0] - 1))
                wrong_vec = wrong_states[fallback_idx].detach().float().cpu().numpy()
                right_vec = right_states[fallback_right_idx].detach().float().cpu().numpy()
                difference_vectors_per_layer[layer_id].append(right_vec - wrong_vec)
                token_positions_per_layer[layer_id].append(fallback_idx)

        else:
            # Compute differences using selected tokens (with optional DP mapping)
            for layer_id in target_layers:
                wrong_states = wrong_forward["hidden_states"][layer_id].squeeze(0)
                right_states = right_forward["hidden_states"][layer_id].squeeze(0)

                token_idx = selected_positions[layer_id]
                token_idx = max(0, min(token_idx, wrong_states.shape[0] - 1))
                right_idx = max(0, min(token_idx, right_states.shape[0] - 1))

                if method == "dp_gradient" and dp_alignment_available:
                    dp_alignment_attempts += 1
                    mapped_idx = match_lookup.get(token_idx)
                    if mapped_idx is not None and 0 <= mapped_idx < right_states.shape[0]:
                        right_idx = mapped_idx
                        dp_alignment_successes += 1
                        dp_pairs_used_total += 1
                    else:
                        dp_lookup_misses += 1

                wrong_vec = wrong_states[token_idx].detach().float().cpu().numpy()
                right_vec = right_states[right_idx].detach().float().cpu().numpy()

                diff = right_vec - wrong_vec
                difference_vectors_per_layer[layer_id].append(diff)
                token_positions_per_layer[layer_id].append(token_idx)

                if gradient_used and layer_id in gradient_norms:
                    gradient_norms_per_layer[layer_id].append(gradient_norms[layer_id])

        processed += 1

    if use_dp_alignment:
        dp_fallback_count = dp_average_fallbacks if method == "dp_average" else dp_alignment_fallbacks
        LOGGER.info(
            "Processed %d triples (skipped %d, gradient fallbacks %d, DP attempts %d, successes %d, lookup misses %d, DP fallbacks %d, unavailable %d, missing files %d)",
            processed,
            skipped,
            gradient_fallbacks,
            dp_alignment_attempts,
            dp_alignment_successes,
            dp_lookup_misses,
            dp_fallback_count,
            dp_alignment_unavailable,
            dp_alignment_missing_files,
        )
    else:
        LOGGER.info(
            "Processed %d triples (skipped %d, gradient fallbacks %d)",
            processed,
            skipped,
            gradient_fallbacks,
        )

    # Compute steering vectors by averaging differences
    steering_vectors: Dict[int, np.ndarray] = {}
    for layer_id in target_layers:
        if not difference_vectors_per_layer[layer_id]:
            LOGGER.warning("No difference vectors collected for layer %d", layer_id)
            continue

        # Stack all difference vectors and compute mean
        differences = np.stack(difference_vectors_per_layer[layer_id])  # [n_samples, hidden_dim]
        steering_vector = differences.mean(axis=0)  # [hidden_dim]
        
        steering_vectors[layer_id] = steering_vector
        
        LOGGER.info(
            "Layer %d: Computed steering vector from %d samples (shape: %s, norm: %.4f)",
            layer_id,
            len(difference_vectors_per_layer[layer_id]),
            steering_vector.shape,
            np.linalg.norm(steering_vector),
        )

        # Save steering vector
        steering_path = args.output_dir / f"layer{layer_id}_steering_vector.npy"
        np.save(steering_path, steering_vector)
        LOGGER.info("Saved steering vector for layer %d to %s", layer_id, steering_path)

    # Save metadata
    metadata = {
        "model_name": args.model_name,
        "layers": target_layers,
        "num_triples": processed,
        "num_samples_per_layer": {
            layer_id: len(difference_vectors_per_layer[layer_id])
            for layer_id in target_layers
        },
        "steering_vector_norms": {
            layer_id: float(np.linalg.norm(steering_vectors[layer_id]))
            for layer_id in steering_vectors.keys()
        },
        "token_selection_method": args.token_selection_method,
        "token_position_summary": {
            layer_id: summarize_numeric(token_positions_per_layer[layer_id])
            for layer_id in target_layers
        },
        "gradient_fallbacks": gradient_fallbacks,
    }
    if use_gradient:
        metadata["gradient_norm_summary"] = {
            layer_id: summarize_numeric(gradient_norms_per_layer[layer_id])
            for layer_id in target_layers
        }
    if use_token_mlp and args.token_selection_mlp_path:
        metadata["token_selector_checkpoint"] = str(args.token_selection_mlp_path)
    if use_dp_alignment:
        metadata["dp_alignment_stats"] = {
            "missing_files": dp_alignment_missing_files,
            "fallback_samples": dp_alignment_fallbacks,
            "unavailable_samples": dp_alignment_unavailable,
            "attempts": dp_alignment_attempts,
            "successes": dp_alignment_successes,
            "lookup_misses": dp_lookup_misses,
            "pairs_used_total": dp_pairs_used_total,
        }
        if method == "dp_average":
            metadata["dp_alignment_stats"]["average_fallbacks"] = dp_average_fallbacks
            if dp_average_pair_counts_per_layer is not None:
                metadata["dp_average_pair_counts"] = {
                    layer_id: summarize_numeric(dp_average_pair_counts_per_layer[layer_id])
                    for layer_id in target_layers
                }
    metadata_path = args.output_dir / "steering_vectors_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    LOGGER.info("Saved metadata to %s", metadata_path)

    LOGGER.info("Steering vector computation complete!")


if __name__ == "__main__":
    main()

