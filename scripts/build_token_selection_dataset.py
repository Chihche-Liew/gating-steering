#!/usr/bin/env python3
"""Construct a supervised dataset for the token-selection MLP."""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

from compute_steering_vectors import (
    compute_prompt_token_length,
    encode_answer_token_id,
    load_triples,
)
from model_wrapper import ModelWrapper
from setup import (
    build_teacher_forcing_text,
    configure_hf_caches,
    reconstruct_prompt,
    setup_logging,
)


LOGGER = logging.getLogger("build_token_selection_dataset")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--triples-path", type=Path, required=True)
    parser.add_argument("--model-name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--layers", nargs="+", type=int, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/mlp_features"),
        help="Directory where layer-specific datasets will be saved.",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--system-prompt", type=str, default="You are a careful reasoning assistant. Think step by step and end with 'Final answer: <choice>'.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--positives-per-sample",
        type=int,
        default=1,
        help="Number of positive tokens (highest gradients) to keep per layer/sample.",
    )
    parser.add_argument(
        "--negatives-per-positive",
        type=int,
        default=3,
        help="Number of negative tokens sampled per positive token.",
    )
    parser.add_argument(
        "--max-tokens-per-sample",
        type=int,
        default=64,
        help="Optional cap on total tokens (pos+neg) stored per layer/sample.",
    )
    parser.add_argument(
        "--save-json-summary",
        action="store_true",
        help="Save a JSON manifest summarizing dataset statistics.",
    )
    return parser.parse_args()


def _prepare_feature_vector(hidden_vec: torch.Tensor, token_idx: int, seq_len: int) -> np.ndarray:
    rel_pos = token_idx / max(seq_len - 1, 1)
    rel_pos_tensor = torch.tensor([rel_pos], dtype=torch.float32)
    feature = torch.cat([hidden_vec, rel_pos_tensor], dim=0)
    return feature.cpu().numpy()


def main() -> None:
    setup_logging()
    configure_hf_caches()
    args = parse_args()
    rng = random.Random(args.seed)

    triples = load_triples(args.triples_path)
    if not triples:
        LOGGER.error("No triples were loaded; aborting.")
        return
    if args.max_samples:
        triples = triples[: args.max_samples]

    layers = sorted(set(args.layers))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model = ModelWrapper(args.model_name, device=args.device)
    datasets: Dict[int, Dict[str, List]] = {
        layer: {
            "features": [],
            "labels": [],
            "grad_norms": [],
            "token_positions": [],
            "sample_ids": [],
        }
        for layer in layers
    }

    total_samples = 0
    skipped = 0

    for triple in tqdm(triples, desc="Building token selection dataset"):
        if not triple.correct_chain.strip():
            skipped += 1
            continue

        wrong_text = build_teacher_forcing_text(triple.prompt, triple.wrong_chain, triple.metadata)
        formatted_prompt = reconstruct_prompt(triple.metadata, triple.prompt)
        prompt_token_count = compute_prompt_token_length(model, formatted_prompt)

        wrong_forward = model.forward_with_hidden_states(
            wrong_text,
            target_layers=layers,
            enable_grad=True,
            system_prompt=args.system_prompt,
        )

        correct_token_id = encode_answer_token_id(model, triple.correct_answer)
        wrong_answer_text = getattr(triple, "wrong_answer", "") or triple.metadata.get("wrong_answer", "")
        if not wrong_answer_text and triple.wrong_chain.strip():
            parts = triple.wrong_chain.strip().split()
            if parts:
                wrong_answer_text = parts[-1]
        wrong_token_id = encode_answer_token_id(model, wrong_answer_text)
        logits: torch.Tensor = wrong_forward["logits"]
        last_position = logits.shape[0] - 1
        if correct_token_id is None:
            LOGGER.warning("Sample %s lacks a decodable correct answer; skipping.", triple.sample_id)
            skipped += 1
            model.model.zero_grad(set_to_none=True)
            continue

        score = logits[last_position, correct_token_id]
        if wrong_token_id is not None and wrong_token_id != correct_token_id:
            score = score - logits[last_position, wrong_token_id]
        model.model.zero_grad(set_to_none=True)
        score.backward(retain_graph=False)

        for layer_id in layers:
            hidden_tensor: torch.Tensor = wrong_forward["hidden_states"][layer_id]
            grad_tensor = hidden_tensor.grad
            if grad_tensor is None:
                LOGGER.warning("No gradients for layer %d in sample %s; skipping layer.", layer_id, triple.sample_id)
                continue

            hidden_seq = hidden_tensor.detach().squeeze(0).cpu()
            grad_seq = grad_tensor.detach().squeeze(0)
            grad_norms = grad_seq.norm(p=2, dim=1).cpu()

            seq_len = hidden_seq.shape[0]
            usable_mask = torch.ones(seq_len, dtype=torch.bool)
            mask_limit = min(prompt_token_count, seq_len)
            if mask_limit > 0:
                usable_mask[:mask_limit] = False

            usable_indices = torch.nonzero(usable_mask, as_tuple=False).squeeze(-1)
            if usable_indices.numel() == 0:
                continue

            candidate_norms = grad_norms.clone()
            candidate_norms[~usable_mask] = -float("inf")

            pos_k = min(args.positives_per_sample, int(usable_indices.numel()))
            if pos_k <= 0:
                continue

            top_values, top_indices = torch.topk(candidate_norms, k=pos_k)
            pos_indices = [int(idx.item()) for idx in top_indices]

            neg_pool = [int(idx.item()) for idx in usable_indices.tolist() if int(idx) not in pos_indices]
            max_negatives = min(len(neg_pool), args.negatives_per_positive * len(pos_indices))
            if args.max_tokens_per_sample:
                remaining_after_pos = max(args.max_tokens_per_sample - len(pos_indices), 0)
                max_negatives = min(max_negatives, remaining_after_pos)
            if max_negatives > 0 and neg_pool:
                neg_indices = rng.sample(neg_pool, max_negatives)
            else:
                neg_indices = []

            target_layer_data = datasets[layer_id]
            added = 0
            sample_cap = args.max_tokens_per_sample or 0

            for idx in pos_indices:
                feature_vec = _prepare_feature_vector(hidden_seq[idx], idx, seq_len)
                target_layer_data["features"].append(feature_vec)
                target_layer_data["labels"].append(1.0)
                target_layer_data["grad_norms"].append(float(grad_norms[idx].item()))
                target_layer_data["token_positions"].append(idx)
                target_layer_data["sample_ids"].append(triple.sample_id)
                added += 1
                if sample_cap and added >= sample_cap:
                    break

            for idx in neg_indices:
                if sample_cap and added >= sample_cap:
                    break
                feature_vec = _prepare_feature_vector(hidden_seq[idx], idx, seq_len)
                target_layer_data["features"].append(feature_vec)
                target_layer_data["labels"].append(0.0)
                target_layer_data["grad_norms"].append(float(grad_norms[idx].item()))
                target_layer_data["token_positions"].append(idx)
                target_layer_data["sample_ids"].append(triple.sample_id)
                added += 1

        model.model.zero_grad(set_to_none=True)
        total_samples += 1

        del wrong_forward
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    manifest: Dict[str, Dict[str, float]] = {}
    for layer_id, payload in datasets.items():
        if not payload["features"]:
            LOGGER.warning("Layer %d produced no training pairs.", layer_id)
            continue
        features = np.stack(payload["features"]).astype(np.float32)
        labels = np.array(payload["labels"], dtype=np.float32)
        grad_norms = np.array(payload["grad_norms"], dtype=np.float32)
        token_positions = np.array(payload["token_positions"], dtype=np.int32)
        sample_ids = np.array(payload["sample_ids"], dtype=object)

        output_path = args.output_dir / f"token_selector_layer{layer_id}.npz"
        np.savez(
            output_path,
            features=features,
            labels=labels,
            grad_norms=grad_norms,
            token_positions=token_positions,
            sample_ids=sample_ids,
            layer=layer_id,
            feature_description=json.dumps({"extra_features": ["relative_position"]}),
        )
        LOGGER.info(
            "Layer %d dataset saved to %s (%d samples, positive ratio %.2f)",
            layer_id,
            output_path,
            features.shape[0],
            float(labels.mean()) if labels.size else 0.0,
        )
        manifest[str(layer_id)] = {
            "num_examples": float(features.shape[0]),
            "positive_ratio": float(labels.mean()) if labels.size else 0.0,
            "feature_dim": float(features.shape[1]),
        }

    if args.save_json_summary:
        summary_path = args.output_dir / "token_selector_manifest.json"
        summary_payload = {
            "model_name": args.model_name,
            "layers": layers,
            "num_triples": total_samples,
            "skipped_triples": skipped,
            "positives_per_sample": args.positives_per_sample,
            "negatives_per_positive": args.negatives_per_positive,
            "max_tokens_per_sample": args.max_tokens_per_sample,
            "layers_summary": manifest,
        }
        summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
        LOGGER.info("Wrote dataset summary to %s", summary_path)

    LOGGER.info("Finished building token selection dataset: %d usable samples (skipped %d).", total_samples, skipped)


if __name__ == "__main__":
    main()

