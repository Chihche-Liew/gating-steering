#!/usr/bin/env python3
"""Collect hidden activations for wrong/right chains."""

from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from Levenshtein import opcodes
from tqdm import tqdm

from model_wrapper import ModelWrapper
from setup import (
    configure_hf_caches,
    setup_logging,
    build_teacher_forcing_text,
)


LOGGER = logging.getLogger("collect_hidden_states")


@dataclass
class Triple:
    sample_id: str
    prompt: str
    wrong_chain: str
    correct_chain: str
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
        "--output-dir",
        type=Path,
        default=Path("artifacts/hidden_states"),
        help="Directory for storing per-sample hidden state tensors.",
    )
    parser.add_argument(
        "--alignment-dir",
        type=Path,
        default=Path("artifacts/alignments"),
        help="Directory for saving token alignment metadata per sample.",
    )
    parser.add_argument(
        "--probe-data-dir",
        type=Path,
        default=Path("artifacts/probe_data"),
        help="Directory for saving probe training data (features and labels).",
    )
    parser.add_argument(
        "--model-name",
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="Identifier for a causal LM to evaluate (default mirrors Step 1).",
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        required=True,
        type=int,
        help="List of 0-indexed transformer layer ids to capture (e.g., --layers 28 30 31).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on number of triples to process (for debugging).",
    )
    parser.add_argument(
        "--probe-max-samples",
        type=int,
        default=4000,
        help="Maximum number of matched token activations to use for probing and visualization per layer.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional torch device override (defaults to cuda if available else cpu).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility when subsampling tokens.",
    )
    parser.add_argument(
        "--pooling-method",
        type=str,
        default="mean",
        choices=["mean", "last_token", "per_token"],
        help="Pooling method: 'mean' (average all tokens), 'last_token' (final token), 'per_token' (each matched token).",
    )
    parser.add_argument(
        "--alignment-method",
        type=str,
        default="text",
        choices=["text", "hidden_dp"],
        help="Token alignment strategy. 'text' uses exact token matches; 'hidden_dp' uses hidden-state dynamic programming with fallback.",
    )
    parser.add_argument(
        "--alignment-layer",
        type=int,
        default=None,
        help="Layer id whose hidden states are used for hidden_dp alignment (defaults to first layer in --layers).",
    )
    parser.add_argument(
        "--dp-max-shift",
        type=int,
        default=None,
        help="Maximum index shift allowed between matched tokens for hidden_dp alignment.",
    )
    parser.add_argument(
        "--dp-gap-penalty",
        type=float,
        default=None,
        help="Gap penalty used in hidden_dp alignment (defaults to mean cosine distance).",
    )
    parser.add_argument(
        "--dp-distance-threshold",
        type=float,
        default=None,
        help="Distance threshold for accepting matches in hidden_dp alignment (defaults to mean + std of distances).",
    )
    parser.add_argument(
        "--dp-alignment",
        action="store_true",
        help="Use hidden-state DP alignments for token feature sampling (affects per_token pooling).",
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
                correct_answer=str(payload.get("correct_answer", "")),
                metadata=dict(payload.get("metadata", {})),
            )
        )
    return triples


def _align_tokens_text(
    wrong_tokens: List[str],
    right_tokens: List[str],
) -> Tuple[List[Tuple[int, int]], Dict[str, Any]]:
    alignment_opcodes = opcodes(wrong_tokens, right_tokens)
    matches: List[Tuple[int, int]] = []
    for tag, i1, i2, j1, j2 in alignment_opcodes:
        if tag == "equal":
            matches.extend(zip(range(i1, i2), range(j1, j2)))
    metadata: Dict[str, Any] = {
        "strategy": "text",
        "num_matches": len(matches),
    }
    return matches, metadata


def _cosine_distance_matrix(
    wrong_hidden: torch.Tensor,
    right_hidden: torch.Tensor,
) -> np.ndarray:
    wrong_norm = F.normalize(wrong_hidden, p=2, dim=1)
    right_norm = F.normalize(right_hidden, p=2, dim=1)
    similarity = torch.matmul(wrong_norm, right_norm.T).clamp(-1.0, 1.0)
    distances = 1.0 - similarity
    return distances.cpu().numpy()


def _hidden_alignment_dp(
    wrong_hidden: torch.Tensor,
    right_hidden: torch.Tensor,
    max_shift: Optional[int] = None,
    gap_penalty: Optional[float] = None,
    distance_threshold: Optional[float] = None,
) -> Tuple[List[Tuple[int, int]], Dict[str, Any]]:
    if wrong_hidden.ndim != 2 or right_hidden.ndim != 2:
        raise ValueError("Hidden states must be 2D tensors [seq_len, hidden_dim].")

    distance_matrix = _cosine_distance_matrix(wrong_hidden, right_hidden)
    if np.isnan(distance_matrix).any():
        return [], {"strategy": "hidden_dp", "error": "NaN distances detected."}

    n, m = distance_matrix.shape
    if n == 0 or m == 0:
        return [], {"strategy": "hidden_dp", "error": "Empty hidden state sequence."}

    if gap_penalty is None:
        gap_penalty = float(max(np.mean(distance_matrix), 1e-4))

    if distance_threshold is None:
        distance_threshold = float(np.mean(distance_matrix) + np.std(distance_matrix))

    if max_shift is None:
        max_shift = max(abs(n - m) + 5, int(0.2 * max(n, m)))

    scores = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    traceback = np.full((n + 1, m + 1), -1, dtype=np.int8)

    scores[0, 0] = 0.0
    for i in range(1, n + 1):
        scores[i, 0] = scores[i - 1, 0] + gap_penalty
        traceback[i, 0] = 1  # up
    for j in range(1, m + 1):
        scores[0, j] = scores[0, j - 1] + gap_penalty
        traceback[0, j] = 2  # left

    for i in range(1, n + 1):
        i_idx = i - 1
        for j in range(1, m + 1):
            j_idx = j - 1
            if abs(i_idx - j_idx) > max_shift:
                match_cost = np.inf
            else:
                match_cost = scores[i - 1, j - 1] + distance_matrix[i_idx, j_idx]
            delete_cost = scores[i - 1, j] + gap_penalty
            insert_cost = scores[i, j - 1] + gap_penalty
            best_cost = min(match_cost, delete_cost, insert_cost)

            scores[i, j] = best_cost
            if best_cost == match_cost:
                traceback[i, j] = 0  # diagonal
            elif best_cost == delete_cost:
                traceback[i, j] = 1  # up
            else:
                traceback[i, j] = 2  # left

    if not np.isfinite(scores[n, m]):
        return [], {
            "strategy": "hidden_dp",
            "error": "Alignment score is infinite (path not found).",
        }

    matches: List[Tuple[int, int]] = []
    matched_distances: List[float] = []
    discarded = 0
    gap_steps = 0

    i, j = n, m
    while i > 0 or j > 0:
        direction = traceback[i, j]
        if direction == 0 and i > 0 and j > 0:
            i -= 1
            j -= 1
            dist_val = float(distance_matrix[i, j])
            if dist_val <= distance_threshold:
                matches.append((i, j))
                matched_distances.append(dist_val)
            else:
                discarded += 1
        elif direction == 1 and i > 0:
            i -= 1
            gap_steps += 1
        elif direction == 2 and j > 0:
            j -= 1
            gap_steps += 1
        else:
            LOGGER.warning("Unexpected traceback direction %s at (%d, %d)", direction, i, j)
            break

    matches.reverse()
    matched_distances.reverse()

    total_steps = len(matches) + discarded + gap_steps
    avg_distance = float(np.mean(matched_distances)) if matched_distances else None

    metrics: Dict[str, Any] = {
        "strategy": "hidden_dp",
        "num_matches": len(matches),
        "avg_distance": avg_distance,
        "distance_threshold": float(distance_threshold),
        "gap_steps": gap_steps,
        "discarded_high_distance": discarded,
        "total_steps": total_steps,
        "gap_ratio": float(gap_steps / total_steps) if total_steps else None,
        "score": float(scores[n, m]),
        "sequence_lengths": {"wrong": n, "right": m},
    }

    return matches, metrics


def _dp_quality_ok(metrics: Dict[str, Any]) -> bool:
    if metrics.get("error"):
        return False
    num_matches = metrics.get("num_matches", 0)
    if num_matches == 0:
        return False
    avg_distance = metrics.get("avg_distance")
    if avg_distance is None:
        return False
    threshold = metrics.get("distance_threshold")
    if threshold is not None and avg_distance > threshold:
        return False
    gap_ratio = metrics.get("gap_ratio")
    if gap_ratio is not None and gap_ratio > 0.6:
        return False
    return True


def align_tokens(
    wrong_tokens: List[str],
    right_tokens: List[str],
    wrong_hidden: Optional[torch.Tensor],
    right_hidden: Optional[torch.Tensor],
    method: str = "text",
    max_shift: Optional[int] = None,
    gap_penalty: Optional[float] = None,
    distance_threshold: Optional[float] = None,
) -> Tuple[List[Tuple[int, int]], Dict[str, Any]]:
    text_matches, text_metadata = _align_tokens_text(wrong_tokens, right_tokens)

    if method != "hidden_dp":
        return text_matches, text_metadata

    if wrong_hidden is None or right_hidden is None:
        fallback_meta = dict(text_metadata)
        fallback_meta.update(
            {
                "strategy": "text_fallback",
                "used_fallback": True,
                "fallback_reason": "hidden_states_unavailable",
            }
        )
        return text_matches, fallback_meta

    try:
        dp_matches, dp_metrics = _hidden_alignment_dp(
            wrong_hidden,
            right_hidden,
            max_shift=max_shift,
            gap_penalty=gap_penalty,
            distance_threshold=distance_threshold,
        )
    except Exception as exc:
        LOGGER.warning("Hidden-state DP alignment failed; falling back to text alignment: %s", exc)
        fallback_meta = dict(text_metadata)
        fallback_meta.update(
            {
                "strategy": "text_fallback",
                "used_fallback": True,
                "fallback_reason": f"exception: {exc}",
            }
        )
        return text_matches, fallback_meta

    if _dp_quality_ok(dp_metrics):
        dp_metrics["used_fallback"] = False
        return dp_matches, dp_metrics

    fallback_meta = dict(text_metadata)
    fallback_meta.update(
        {
            "strategy": "text_fallback",
            "used_fallback": True,
            "fallback_reason": dp_metrics.get("error", "quality_check_failed"),
            "dp_metrics": dp_metrics,
        }
    )
    return text_matches, fallback_meta


def subsample_pairs(pairs: List[Tuple[int, int]], max_samples: int, rng: random.Random) -> List[Tuple[int, int]]:
    if len(pairs) <= max_samples:
        return pairs
    return rng.sample(pairs, max_samples)


def save_alignment(
    alignment_path: Path,
    wrong_tokens: List[str],
    right_tokens: List[str],
    matches: List[Tuple[int, int]],
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Save token alignment to JSON."""
    alignment_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "wrong_tokens": wrong_tokens,
        "right_tokens": right_tokens,
        "matches": [
            {"wrong_index": wrong_idx, "right_index": right_idx}
            for wrong_idx, right_idx in matches
        ],
    }
    if metadata:
        payload["metadata"] = metadata
    alignment_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def find_position_before_final_answer(tokens: List[str]) -> int:
    """
    Find the token position just before "Final answer" or "final answer" appears.

    Args:
        tokens: List of token strings

    Returns:
        Index of the token just before "final answer" starts, or -1 if not found
    """
    # Common patterns for "final answer" in tokenization
    # Could be ["Final", " answer"], ["final", " answer"], ["Final", "answer"], etc.

    for i in range(len(tokens) - 1):
        # Check current and next token
        current = tokens[i].lower().strip()
        next_token = tokens[i + 1].lower().strip() if i + 1 < len(tokens) else ""

        # Pattern 1: "Final" or "final" followed by "answer" or " answer"
        if current in ["final", "▁final"] and "answer" in next_token:
            # Return position before "Final"
            return max(0, i - 1)

        # Pattern 2: Single token "finalanswer" or "final_answer"
        if "final" in current and "answer" in current:
            return max(0, i - 1)

    # Fallback: try searching in reconstructed text
    # Join tokens and search for "final answer" case-insensitively
    text = "".join(tokens).lower()
    final_answer_pos = text.find("finalanswer")
    if final_answer_pos == -1:
        final_answer_pos = text.find("final answer")

    if final_answer_pos != -1:
        # Count tokens up to this position
        char_count = 0
        for i, token in enumerate(tokens):
            char_count += len(token)
            if char_count >= final_answer_pos:
                return max(0, i - 1)

    # If not found, return -1 (will use last token as fallback)
    return -1


def main() -> None:
    setup_logging()
    configure_hf_caches()
    args = parse_args()
    rng = random.Random(args.seed)

    dp_alignment_requested = bool(args.dp_alignment)
    dp_alignment_active = dp_alignment_requested and args.pooling_method == "per_token"
    if dp_alignment_requested and args.pooling_method != "per_token":
        LOGGER.info(
            "Ignoring --dp-alignment because pooling method '%s' does not use token-level features.",
            args.pooling_method,
        )
    alignment_method = "hidden_dp" if dp_alignment_active else "text"

    # Load triples
    triples = load_triples(args.triples_path)
    if not triples:
        LOGGER.error("No usable triples found (correct_chain missing?). Exiting.")
        return

    # Setup layers
    target_layers = sorted(set(args.layers))
    LOGGER.info("Target layers for hidden state capture: %s", target_layers)

    alignment_layer = args.alignment_layer if args.alignment_layer is not None else target_layers[0]
    if alignment_method == "hidden_dp" and alignment_layer not in target_layers:
        LOGGER.info(
            "Alignment layer %s not in --layers; adding for hidden-state capture.",
            alignment_layer,
        )
    capture_layers = sorted(
        set(target_layers + ([alignment_layer] if alignment_method == "hidden_dp" else []))
    )

    if dp_alignment_active:
        LOGGER.info("Dynamic programming alignment enabled for per-token pooling.")

    # Load model
    model = ModelWrapper(args.model_name, device=args.device)

    # Print GPU information
    import os
    num_gpus = torch.cuda.device_count()
    visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', 'All')
    print(f"\n{'='*80}")
    print(f"GPU CONFIGURATION:")
    print(f"  Number of GPUs available: {num_gpus}")
    print(f"  CUDA_VISIBLE_DEVICES: {visible_devices}")
    print(f"  PyTorch CUDA available: {torch.cuda.is_available()}")

    # Print device map (how model is distributed across GPUs)
    if hasattr(model.model, 'hf_device_map'):
        print(f"\n  Model Distribution Across GPUs:")
        device_map = model.model.hf_device_map
        device_counts = {}
        for name, device in device_map.items():
            device_str = str(device)
            device_counts[device_str] = device_counts.get(device_str, 0) + 1

        for device, count in sorted(device_counts.items()):
            print(f"    {device}: {count} layers/components")

    # Print memory usage per GPU
    if torch.cuda.is_available():
        print(f"\n  GPU Memory Usage:")
        for i in range(num_gpus):
            allocated = torch.cuda.memory_allocated(i) / 1024**3  # Convert to GB
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            print(f"    GPU {i}: Allocated={allocated:.2f} GB, Reserved={reserved:.2f} GB")
    print(f"{'='*80}\n")

    # Create output directories
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.alignment_dir.mkdir(parents=True, exist_ok=True)
    args.probe_data_dir.mkdir(parents=True, exist_ok=True)

    # Probe data collection - track per layer
    probe_data_per_layer = {
        layer: {
            "features": [],
            "labels": [],
            "token_positions": [],
            "sample_ids": [],
            "valid": [],
            "dp_used_alignment": [],
            "dp_used_fallback": [],
            "dp_pair_index": [],
            "dp_gap_ratio": [],
            "dp_avg_distance": [],
        }
        for layer in target_layers
    }

    processed = 0
    skipped_missing_chain = 0

    for triple in tqdm(triples, desc="Capturing hidden states"):
        if args.max_samples is not None and processed >= args.max_samples:
            break

        if not triple.correct_chain.strip():
            skipped_missing_chain += 1
            continue

        # Build teacher forcing texts
        wrong_text = build_teacher_forcing_text(
            triple.prompt, triple.wrong_chain, triple.metadata
        )
        right_text = build_teacher_forcing_text(
            triple.prompt, triple.correct_chain, triple.metadata
        )

        # Apply system prompt formatting for tokenization (to match get_hidden_states)
        wrong_text_formatted = wrong_text
        right_text_formatted = right_text
        if hasattr(model.tokenizer, "apply_chat_template") and args.system_prompt:
            wrong_messages = [
                {"role": "system", "content": args.system_prompt},
                {"role": "user", "content": wrong_text},
            ]
            right_messages = [
                {"role": "system", "content": args.system_prompt},
                {"role": "user", "content": right_text},
            ]
            wrong_text_formatted = model.tokenizer.apply_chat_template(
                wrong_messages, tokenize=False, add_generation_prompt=False
            )
            right_text_formatted = model.tokenizer.apply_chat_template(
                right_messages, tokenize=False, add_generation_prompt=False
            )

        # Tokenize
        wrong_inputs = model.tokenize(wrong_text_formatted, return_offsets_mapping=True)
        right_inputs = model.tokenize(right_text_formatted, return_offsets_mapping=True)

        # Get hidden states
        wrong_hidden = model.get_hidden_states(wrong_text, capture_layers, system_prompt=args.system_prompt)
        right_hidden = model.get_hidden_states(right_text, capture_layers, system_prompt=args.system_prompt)

        # Get tokens and align
        wrong_tokens = model.token_strings(wrong_inputs["input_ids"].squeeze(0))
        right_tokens = model.token_strings(right_inputs["input_ids"].squeeze(0))
        alignment_hidden_wrong = (
            wrong_hidden.get(alignment_layer) if alignment_method == "hidden_dp" else None
        )
        alignment_hidden_right = (
            right_hidden.get(alignment_layer) if alignment_method == "hidden_dp" else None
        )
        matches, alignment_metadata = align_tokens(
            wrong_tokens,
            right_tokens,
            alignment_hidden_wrong,
            alignment_hidden_right,
            method=alignment_method,
            max_shift=args.dp_max_shift,
            gap_penalty=args.dp_gap_penalty,
            distance_threshold=args.dp_distance_threshold,
        )

        # Save hidden states per sample
        sample_dir = args.output_dir / triple.sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        for layer_id, tensor in wrong_hidden.items():
            torch.save(tensor, sample_dir / f"wrong_layer{layer_id}.pt")
        for layer_id, tensor in right_hidden.items():
            torch.save(tensor, sample_dir / f"right_layer{layer_id}.pt")

        # Save alignment
        alignment_path = args.alignment_dir / f"{triple.sample_id}.json"
        save_alignment(alignment_path, wrong_tokens, right_tokens, matches, alignment_metadata)

        alignment_metadata = alignment_metadata or {}
        alignment_strategy = alignment_metadata.get("strategy", "text")
        dp_used_alignment_flag = (
            alignment_strategy == "hidden_dp" and not alignment_metadata.get("used_fallback", False)
        )
        dp_used_fallback_flag = alignment_metadata.get("used_fallback", False)
        dp_metrics = alignment_metadata.get("dp_metrics") if isinstance(alignment_metadata, dict) else None
        dp_gap_ratio = alignment_metadata.get("gap_ratio")
        if dp_gap_ratio is None and isinstance(dp_metrics, dict):
            dp_gap_ratio = dp_metrics.get("gap_ratio")
        dp_avg_distance = alignment_metadata.get("avg_distance")
        if dp_avg_distance is None and isinstance(dp_metrics, dict):
            dp_avg_distance = dp_metrics.get("avg_distance")
        dp_gap_ratio_value = float(dp_gap_ratio) if dp_gap_ratio is not None else float("nan")
        dp_avg_distance_value = float(dp_avg_distance) if dp_avg_distance is not None else float("nan")

        # Collect probe data for all layers using specified pooling method
        for layer_id in target_layers:
            layer_data = probe_data_per_layer[layer_id]
            layer_tensor_wrong = wrong_hidden[layer_id]
            layer_tensor_right = right_hidden[layer_id]

            # Track validity for this sample (will be set to False if we need to skip)
            sample_valid = True

            if args.pooling_method == "mean":
                # Mean pooling over entire sequence
                if len(layer_data["features"]) < args.probe_max_samples * 2:  # *2 for wrong+right
                    wrong_pooled = layer_tensor_wrong.mean(dim=0).numpy()
                    right_pooled = layer_tensor_right.mean(dim=0).numpy()
                    
                    layer_data["features"].append(wrong_pooled)
                    layer_data["labels"].append(0)  # 0 = wrong
                    layer_data["token_positions"].append(-1)  # -1 indicates pooled
                    layer_data["sample_ids"].append(triple.sample_id)
                    layer_data["valid"].append(True)
                    layer_data["dp_used_alignment"].append(False)
                    layer_data["dp_used_fallback"].append(False)
                    layer_data["dp_pair_index"].append(-1)
                    layer_data["dp_gap_ratio"].append(float("nan"))
                    layer_data["dp_avg_distance"].append(float("nan"))

                    layer_data["features"].append(right_pooled)
                    layer_data["labels"].append(1)  # 1 = right
                    layer_data["token_positions"].append(-1)
                    layer_data["sample_ids"].append(triple.sample_id)
                    layer_data["valid"].append(True)
                    layer_data["dp_used_alignment"].append(False)
                    layer_data["dp_used_fallback"].append(False)
                    layer_data["dp_pair_index"].append(-1)
                    layer_data["dp_gap_ratio"].append(float("nan"))
                    layer_data["dp_avg_distance"].append(float("nan"))
            
            elif args.pooling_method == "last_token":
                # Last token (typically where the answer is generated)
                if len(layer_data["features"]) < args.probe_max_samples * 2:
                    wrong_last = layer_tensor_wrong[-1].numpy()
                    right_last = layer_tensor_right[-1].numpy()
                    
                    layer_data["features"].append(wrong_last)
                    layer_data["labels"].append(0)
                    layer_data["token_positions"].append(len(layer_tensor_wrong) - 1)
                    layer_data["sample_ids"].append(triple.sample_id)
                    layer_data["valid"].append(True)
                    layer_data["dp_used_alignment"].append(False)
                    layer_data["dp_used_fallback"].append(False)
                    layer_data["dp_pair_index"].append(-1)
                    layer_data["dp_gap_ratio"].append(float("nan"))
                    layer_data["dp_avg_distance"].append(float("nan"))

                    layer_data["features"].append(right_last)
                    layer_data["labels"].append(1)
                    layer_data["token_positions"].append(len(layer_tensor_right) - 1)
                    layer_data["sample_ids"].append(triple.sample_id)
                    layer_data["valid"].append(True)
                    layer_data["dp_used_alignment"].append(False)
                    layer_data["dp_used_fallback"].append(False)
                    layer_data["dp_pair_index"].append(-1)
                    layer_data["dp_gap_ratio"].append(float("nan"))
                    layer_data["dp_avg_distance"].append(float("nan"))
            
            elif args.pooling_method == "per_token":
                # Per-token: use matched token positions
                if len(layer_data["features"]) < args.probe_max_samples:
                    # Subsample matched token positions
                    selected_matches = subsample_pairs(
                        matches, args.probe_max_samples - len(layer_data["features"]), rng
                    )
                    
                    for pair_idx, (wrong_idx, right_idx) in enumerate(selected_matches):
                        # Add wrong token representation
                        layer_data["features"].append(layer_tensor_wrong[wrong_idx].numpy())
                        layer_data["labels"].append(0)  # 0 = wrong
                        layer_data["token_positions"].append(wrong_idx)
                        layer_data["sample_ids"].append(triple.sample_id)
                        layer_data["valid"].append(True)
                        layer_data["dp_used_alignment"].append(dp_used_alignment_flag)
                        layer_data["dp_used_fallback"].append(dp_used_fallback_flag)
                        layer_data["dp_pair_index"].append(pair_idx if dp_used_alignment_flag else -1)
                        layer_data["dp_gap_ratio"].append(dp_gap_ratio_value)
                        layer_data["dp_avg_distance"].append(dp_avg_distance_value)

                        # Add right token representation
                        layer_data["features"].append(layer_tensor_right[right_idx].numpy())
                        layer_data["labels"].append(1)  # 1 = right
                        layer_data["token_positions"].append(right_idx)
                        layer_data["sample_ids"].append(triple.sample_id)
                        layer_data["valid"].append(True)
                        layer_data["dp_used_alignment"].append(dp_used_alignment_flag)
                        layer_data["dp_used_fallback"].append(dp_used_fallback_flag)
                        layer_data["dp_pair_index"].append(pair_idx if dp_used_alignment_flag else -1)
                        layer_data["dp_gap_ratio"].append(dp_gap_ratio_value)
                        layer_data["dp_avg_distance"].append(dp_avg_distance_value)

            elif args.pooling_method == "before_final_answer":
                # Token just before "Final answer" appears
                if len(layer_data["features"]) < args.probe_max_samples * 2:
                    # Find position before "Final answer" in wrong chain
                    wrong_pos = find_position_before_final_answer(wrong_tokens)
                    # Find position before "Final answer" in correct chain
                    right_pos = find_position_before_final_answer(right_tokens)

                    # Mark sample as invalid if "Final answer" not found in either chain
                    if wrong_pos == -1 or right_pos == -1:
                        if wrong_pos == -1:
                            LOGGER.warning("Sample %s (wrong): 'Final answer' not found, marking as invalid", triple.sample_id)
                        if right_pos == -1:
                            LOGGER.warning("Sample %s (right): 'Final answer' not found, marking as invalid", triple.sample_id)
                        sample_valid = False
                        # Use last token as fallback position for invalid samples
                        wrong_pos = len(layer_tensor_wrong) - 1 if wrong_pos == -1 else wrong_pos
                        right_pos = len(layer_tensor_right) - 1 if right_pos == -1 else right_pos

                    # Extract hidden states at these positions
                    wrong_before_answer = layer_tensor_wrong[wrong_pos].numpy()
                    right_before_answer = layer_tensor_right[right_pos].numpy()

                    layer_data["features"].append(wrong_before_answer)
                    layer_data["labels"].append(0)  # 0 = wrong
                    layer_data["token_positions"].append(wrong_pos)
                    layer_data["sample_ids"].append(triple.sample_id)
                    layer_data["valid"].append(sample_valid)
                    layer_data["dp_used_alignment"].append(False)
                    layer_data["dp_used_fallback"].append(False)
                    layer_data["dp_pair_index"].append(-1)
                    layer_data["dp_gap_ratio"].append(float("nan"))
                    layer_data["dp_avg_distance"].append(float("nan"))

                    layer_data["features"].append(right_before_answer)
                    layer_data["labels"].append(1)  # 1 = right
                    layer_data["token_positions"].append(right_pos)
                    layer_data["sample_ids"].append(triple.sample_id)
                    layer_data["valid"].append(sample_valid)
                    layer_data["dp_used_alignment"].append(False)
                    layer_data["dp_used_fallback"].append(False)
                    layer_data["dp_pair_index"].append(-1)
                    layer_data["dp_gap_ratio"].append(float("nan"))
                    layer_data["dp_avg_distance"].append(float("nan"))

        # Clean up memory after processing each sample
        del wrong_hidden
        del right_hidden
        del wrong_inputs
        del right_inputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        processed += 1

    LOGGER.info("Processed %d triples (skipped %d missing correct chains)",
               processed, skipped_missing_chain)

    # Save probe data for each layer
    for layer_id in target_layers:
        layer_data = probe_data_per_layer[layer_id]
        if layer_data["features"]:
            feature_matrix = np.stack(layer_data["features"])
            label_array = np.array(layer_data["labels"])
            position_array = np.array(layer_data["token_positions"])
            sample_id_array = np.array(layer_data["sample_ids"])
            valid_array = np.array(layer_data["valid"])
            dp_used_alignment_array = np.array(layer_data["dp_used_alignment"], dtype=bool)
            dp_used_fallback_array = np.array(layer_data["dp_used_fallback"], dtype=bool)
            dp_pair_index_array = np.array(layer_data["dp_pair_index"], dtype=np.int32)
            dp_gap_ratio_array = np.array(layer_data["dp_gap_ratio"], dtype=np.float32)
            dp_avg_distance_array = np.array(layer_data["dp_avg_distance"], dtype=np.float32)

            probe_data_path = args.probe_data_dir / f"layer{layer_id}_probe_data.npz"
            np.savez(
                probe_data_path,
                features=feature_matrix,
                labels=label_array,
                token_positions=position_array,
                sample_ids=sample_id_array,
                valid=valid_array,
                layer=layer_id,
                dp_used_alignment=dp_used_alignment_array,
                dp_used_fallback=dp_used_fallback_array,
                dp_pair_index=dp_pair_index_array,
                dp_gap_ratio=dp_gap_ratio_array,
                dp_avg_distance=dp_avg_distance_array,
            )

            num_valid = np.sum(valid_array)
            num_invalid = len(valid_array) - num_valid
            LOGGER.info("Saved probe data for layer %d to %s (%d total samples, %d valid, %d invalid)",
                       layer_id, probe_data_path, len(layer_data["features"]), num_valid, num_invalid)
        else:
            LOGGER.warning("No probe samples collected for layer %d.", layer_id)


if __name__ == "__main__":
    main()
