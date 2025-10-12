#!/usr/bin/env python3
"""Collect hidden activations for wrong/right chains and create probe visualizations."""

from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
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
    encoded = tokenizer(text, return_tensors="pt")
    return {key: tensor.to(device) for key, tensor in encoded.items()}


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


def align_tokens(tokens_a: List[str], tokens_b: List[str]) -> List[Tuple[int, int]]:
    from difflib import SequenceMatcher

    matcher = SequenceMatcher(a=tokens_a, b=tokens_b)
    matches: List[Tuple[int, int]] = []
    for opcode, a0, a1, b0, b1 in matcher.get_opcodes():
        if opcode == "equal":
            matches.extend((ai, bi) for ai, bi in zip(range(a0, a1), range(b0, b1)))
    return matches


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


def save_alignment(alignment_path: Path, wrong_tokens: List[str], right_tokens: List[str], matches: List[Tuple[int, int]]) -> None:
    alignment_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "wrong_tokens": wrong_tokens,
        "right_tokens": right_tokens,
        "matches": [
            {"wrong_index": wrong_idx, "right_index": right_idx}
            for wrong_idx, right_idx in matches
        ],
    }
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
        matches = align_tokens(wrong_tokens, right_tokens)

        sample_dir = args.output_dir / triple.sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        for layer_id, tensor in wrong_hidden.items():
            torch.save(tensor, sample_dir / f"wrong_layer{layer_id}.pt")
        for layer_id, tensor in right_hidden.items():
            torch.save(tensor, sample_dir / f"right_layer{layer_id}.pt")

        alignment_path = args.alignment_dir / f"{triple.sample_id}.json"
        save_alignment(alignment_path, wrong_tokens, right_tokens, matches)

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

