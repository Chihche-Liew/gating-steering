#!/usr/bin/env python3
"""Collect hidden activations for wrong/right chains and create probe visualizations."""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from Levenshtein import opcodes
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import umap
except ImportError as exc:  # pragma: no cover - runtime guard for optional dependency
    raise SystemExit("umap-learn is required for this script. Install via pip install umap-learn.") from exc

import matplotlib.pyplot as plt


LOGGER = logging.getLogger("collect_hidden_states")


def configure_hf_caches() -> None:
    project_root = Path(__file__).resolve().parents[1]
    cache_root = project_root / ".cache"
    hf_home = cache_root / "huggingface"
    transformers_cache = cache_root / "transformers"

    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HF_DATASETS_CACHE", str(hf_home / "datasets"))
    os.environ.setdefault("HF_HUB_CACHE", str(hf_home / "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(transformers_cache))

    for path in [hf_home, hf_home / "datasets", hf_home / "hub", transformers_cache]:
        path.mkdir(parents=True, exist_ok=True)


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
        "--viz-dir",
        type=Path,
        default=Path("reports/hidden_state_viz"),
        help="Directory for saving probe diagnostics and plots (subfolders created automatically).",
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
        "--probe-layer",
        type=int,
        default=None,
        help="Layer id to use for linear probe (defaults to first entry in --layers).",
    )
    parser.add_argument(
        "--probe-max-samples",
        type=int,
        default=4000,
        help="Maximum number of matched token activations to use for probing and visualization.",
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
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def load_triples(path: Path) -> List[Triple]:
    triples: List[Triple] = []
    with path.open("r", encoding="utf-8") as src:
        for line in src:
            payload = json.loads(line)
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


def load_model(model_name: str, device: Optional[str]) -> Tuple[AutoTokenizer, AutoModelForCausalLM, torch.device]:
    LOGGER.info("Loading model %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else None,
    )
    torch_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(torch_device)
    model.eval()
    return tokenizer, model, torch_device


def reconstruct_prompt(metadata: Dict, prompt: str) -> str:
    template = metadata.get("prompt_template", "{prompt}")
    try:
        return template.format(prompt=prompt)
    except KeyError:
        LOGGER.warning("Prompt template missing {prompt} placeholder, using raw prompt")
        return prompt


def strip_duplicate_prefix(text: str, prefix: str) -> str:
    if text.startswith(prefix):
        return text[len(prefix) :].lstrip()
    return text


def build_teacher_forcing_text(prompt_text: str, chain_text: str, metadata: Dict) -> str:
    formatted_prompt = reconstruct_prompt(metadata, prompt_text)
    cleaned_chain = strip_duplicate_prefix(chain_text, prompt_text)
    cleaned_chain = strip_duplicate_prefix(cleaned_chain, formatted_prompt)
    if cleaned_chain:
        return f"{formatted_prompt}\n\n{cleaned_chain}".strip()
    return formatted_prompt


def tokenize(tokenizer: AutoTokenizer, text: str, device: torch.device) -> Dict[str, torch.Tensor]:
    encoded = tokenizer(text, return_tensors="pt", return_attention_mask=True, return_offsets_mapping=True)
    encoded = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in encoded.items()}
    return encoded


def capture_hidden_states(
    model: AutoModelForCausalLM,
    inputs: Dict[str, torch.Tensor],
    target_layers: Iterable[int],
) -> Dict[int, torch.Tensor]:
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, use_cache=False)
    hidden_states: Tuple[torch.Tensor, ...] = outputs.hidden_states  # type: ignore[attr-defined]
    captures: Dict[int, torch.Tensor] = {}
    for layer_id in target_layers:
        index = layer_id + 1  # +1 because hidden_states[0] is embeddings
        if index >= len(hidden_states):
            raise ValueError(f"Requested layer {layer_id} exceeds available hidden states ({len(hidden_states)-1})")
        captures[layer_id] = hidden_states[index].squeeze(0).detach().cpu()
    return captures


def token_strings(tokenizer: AutoTokenizer, input_ids: torch.Tensor) -> List[str]:
    tokens = tokenizer.convert_ids_to_tokens(input_ids.tolist())
    return tokens


def _align_tokens_text(wrong_tokens: List[str], right_tokens: List[str]) -> List[Tuple[int, int]]:
    alignment_opcodes = opcodes(wrong_tokens, right_tokens)
    matches: List[Tuple[int, int]] = []
    for tag, i1, i2, j1, j2 in alignment_opcodes:
        if tag == "equal":
            matches.extend(zip(range(i1, i2), range(j1, j2)))
    return matches


def _cosine_distance_matrix(wrong_hidden: torch.Tensor, right_hidden: torch.Tensor) -> np.ndarray:
    wrong_norm = F.normalize(wrong_hidden, p=2, dim=1)
    right_norm = F.normalize(right_hidden, p=2, dim=1)
    similarity = torch.matmul(wrong_norm, right_norm.T)
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
            # Safety: if direction is invalid, break out to avoid infinite loop.
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
    distance_threshold = metrics.get("distance_threshold")
    if distance_threshold is not None and avg_distance > distance_threshold:
        return False
    gap_ratio = metrics.get("gap_ratio")
    if gap_ratio is not None and gap_ratio > 0.6:
        return False
    return True


def align_tokens(
    wrong_tokens: List[str],
    right_tokens: List[str],
    wrong_hidden: torch.Tensor,
    right_hidden: torch.Tensor,
) -> Tuple[List[Tuple[int, int]], Dict[str, Any]]:
    text_matches = _align_tokens_text(wrong_tokens, right_tokens)

    try:
        dp_matches, dp_metrics = _hidden_alignment_dp(wrong_hidden, right_hidden)
    except Exception as exc:  # pragma: no cover - defensive guard
        LOGGER.exception("Hidden-state DP alignment failed; falling back to text alignment.")
        return text_matches, {
            "strategy": "text_fallback",
            "used_fallback": True,
            "fallback_reason": f"exception: {exc}",
            "num_matches": len(text_matches),
        }

    if _dp_quality_ok(dp_metrics):
        dp_metrics.update({"strategy": "hidden_dp", "used_fallback": False})
        return dp_matches, dp_metrics

    dp_metrics.update({
        "strategy": "text_fallback",
        "used_fallback": True,
        "fallback_reason": dp_metrics.get("error", "quality_check_failed"),
        "num_matches": len(text_matches),
        "fallback_matches": len(text_matches),
    })
    LOGGER.debug(
        "DP alignment rejected (avg_dist=%s, gap_ratio=%s, matches=%s); using text alignment.",
        dp_metrics.get("avg_distance"),
        dp_metrics.get("gap_ratio"),
        dp_metrics.get("num_matches"),
    )
    return text_matches, dp_metrics


def subsample_pairs(pairs: List[Tuple[int, int]], max_samples: int, rng: random.Random) -> List[Tuple[int, int]]:
    if len(pairs) <= max_samples:
        return pairs
    return rng.sample(pairs, max_samples)


def fit_linear_probe(features: np.ndarray, labels: np.ndarray) -> Dict:
    if len(np.unique(labels)) < 2:
        return {"warning": "Not enough class diversity for probe."}
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=0, stratify=labels)
    clf = LogisticRegression(max_iter=1000, n_jobs=None)
    clf.fit(x_train, y_train)
    preds = clf.predict(x_test)
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, output_dict=True)
    return {"accuracy": float(acc), "report": report}


def run_pca(features: np.ndarray, labels: np.ndarray, output_path: Path) -> Optional[np.ndarray]:
    if features.shape[0] < 2:
        LOGGER.warning("Insufficient samples for PCA plot.")
        return None
    pca = PCA(n_components=2, random_state=0)
    transformed = pca.fit_transform(features)
    save_scatter(transformed, labels, output_path, title="PCA")
    return transformed


def run_umap(features: np.ndarray, labels: np.ndarray, output_path: Path) -> Optional[np.ndarray]:
    if features.shape[0] < 2:
        LOGGER.warning("Insufficient samples for UMAP plot.")
        return None
    reducer = umap.UMAP(n_components=2, random_state=0)
    embedded = reducer.fit_transform(features)
    save_scatter(embedded, labels, output_path, title="UMAP")
    return embedded


def save_scatter(points: np.ndarray, labels: np.ndarray, output_path: Path, title: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 5))
    palette = {0: "red", 1: "blue"}
    for label in np.unique(labels):
        mask = labels == label
        plt.scatter(points[mask, 0], points[mask, 1], label="wrong" if label == 0 else "right", alpha=0.6, s=10,
                    c=palette.get(label, "gray"))
    plt.title(f"{title} projection")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_alignment(
    alignment_path: Path,
    wrong_tokens: List[str],
    right_tokens: List[str],
    matches: List[Tuple[int, int]],
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
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


def save_cluster_overlay(
    points: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    if points is None:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 5))
    label_palette = {0: "red", 1: "blue"}
    handled = set()
    for label in np.unique(labels):
        mask = labels == label
        plt.scatter(
            points[mask, 0],
            points[mask, 1],
            c=label_palette.get(label, "gray"),
            marker="o" if label == 0 else "x",
            s=20,
            alpha=0.7,
            label="wrong" if label == 0 else "right",
        )
    plt.title(f"{title} separation")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend(loc="best", fontsize="small")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main() -> None:
    setup_logging()
    configure_hf_caches()
    args = parse_args()
    rng = random.Random(args.seed)

    triples = load_triples(args.triples_path)
    if not triples:
        LOGGER.error("No usable triples found (correct_chain missing?). Exiting.")
        return

    target_layers = sorted(set(args.layers))
    probe_layer = args.probe_layer if args.probe_layer is not None else target_layers[0]
    if probe_layer not in target_layers:
        LOGGER.warning("Probe layer %s not in --layers, adding automatically.", probe_layer)
        target_layers.append(probe_layer)
        target_layers = sorted(set(target_layers))

    tokenizer, model, device = load_model(args.model_name, args.device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.alignment_dir.mkdir(parents=True, exist_ok=True)
    args.viz_dir.mkdir(parents=True, exist_ok=True)

    probe_features: List[np.ndarray] = []
    probe_labels: List[int] = []

    processed = 0
    skipped_missing_chain = 0

    for triple in tqdm(triples, desc="Capturing hidden states"):
        if args.max_samples is not None and processed >= args.max_samples:
            break

        if not triple.correct_chain.strip():
            skipped_missing_chain += 1
            continue

        wrong_text = build_teacher_forcing_text(triple.prompt, triple.wrong_chain, triple.metadata)
        right_text = build_teacher_forcing_text(triple.prompt, triple.correct_chain, triple.metadata)

        wrong_inputs = tokenize(tokenizer, wrong_text, device)
        right_inputs = tokenize(tokenizer, right_text, device)

        wrong_hidden = capture_hidden_states(model, wrong_inputs, target_layers)
        right_hidden = capture_hidden_states(model, right_inputs, target_layers)

        wrong_tokens = token_strings(tokenizer, wrong_inputs["input_ids"].squeeze(0))
        right_tokens = token_strings(tokenizer, right_inputs["input_ids"].squeeze(0))
        matches, alignment_info = align_tokens(
            wrong_tokens,
            right_tokens,
            wrong_hidden[probe_layer],
            right_hidden[probe_layer],
        )

        sample_dir = args.output_dir / triple.sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        for layer_id, tensor in wrong_hidden.items():
            torch.save(tensor, sample_dir / f"wrong_layer{layer_id}.pt")
        for layer_id, tensor in right_hidden.items():
            torch.save(tensor, sample_dir / f"right_layer{layer_id}.pt")

        alignment_path = args.alignment_dir / f"{triple.sample_id}.json"
        save_alignment(alignment_path, wrong_tokens, right_tokens, matches, alignment_info)

        if len(probe_features) < args.probe_max_samples:
            layer_tensor_wrong = wrong_hidden[probe_layer]
            layer_tensor_right = right_hidden[probe_layer]
            selected_matches = subsample_pairs(matches, args.probe_max_samples - len(probe_features), rng)
            for wrong_idx, right_idx in selected_matches:
                probe_features.append(layer_tensor_wrong[wrong_idx].numpy())
                probe_labels.append(0)
                probe_features.append(layer_tensor_right[right_idx].numpy())
                probe_labels.append(1)

        processed += 1

    LOGGER.info("Processed %d triples (skipped %d missing correct chains)", processed, skipped_missing_chain)

    if not probe_features:
        LOGGER.warning("No probe samples collected; skipping diagnostics.")
        return

    feature_matrix = np.stack(probe_features)
    label_array = np.array(probe_labels)

    diagnostics = {
        "probe_layer": probe_layer,
        "num_samples": int(feature_matrix.shape[0]),
    }

    probe_result = fit_linear_probe(feature_matrix, label_array)
    diagnostics["linear_probe"] = probe_result

    pca_path = args.viz_dir / "linear_separation" / f"layer{probe_layer}_pca.png"
    pca_points = run_pca(feature_matrix, label_array, pca_path)
    diagnostics["pca_plot"] = str(pca_path)

    umap_path = args.viz_dir / "linear_separation" / f"layer{probe_layer}_umap.png"
    umap_points = run_umap(feature_matrix, label_array, umap_path)
    diagnostics["umap_plot"] = str(umap_path)

    overlay_dir = args.viz_dir / "cluster_overlays"
    if pca_points is not None:
        pca_cluster_path = overlay_dir / f"layer{probe_layer}_pca_clusters.png"
        save_cluster_overlay(pca_points, label_array, pca_cluster_path, title="PCA")
        diagnostics["pca_cluster_overlay"] = str(pca_cluster_path)
    if umap_points is not None:
        umap_cluster_path = overlay_dir / f"layer{probe_layer}_umap_clusters.png"
        save_cluster_overlay(umap_points, label_array, umap_cluster_path, title="UMAP")
        diagnostics["umap_cluster_overlay"] = str(umap_cluster_path)

    diagnostics_path = args.viz_dir / f"layer{probe_layer}_diagnostics.json"
    diagnostics_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    LOGGER.info("Diagnostics written to %s", diagnostics_path)


if __name__ == "__main__":
    main()

