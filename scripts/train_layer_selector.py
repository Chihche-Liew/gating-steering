#!/usr/bin/env python3
"""Train the global-best layer selection MLP."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import List, Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

from modules.layer_selector import LayerSelectorConfig, LayerSelectorMLP
from setup import setup_logging


LOGGER = logging.getLogger("train_layer_selector")


DEFAULT_FEATURE_KEYS = [
    "vector_norm",
    "num_samples",
    "token_pos_mean",
    "token_pos_std",
    "token_pos_min",
    "token_pos_max",
    "grad_norm_mean",
    "grad_norm_std",
    "probe_num_samples",
    "probe_feature_dim",
]


class LayerFeatureDataset(Dataset):
    def __init__(self, dataset_paths: Sequence[Path], feature_keys: Sequence[str]):
        features: List[torch.Tensor] = []
        labels: List[float] = []
        meta: List[dict] = []

        for path in dataset_paths:
            with path.open("r", encoding="utf-8") as src:
                for line in src:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    if "label" not in record:
                        continue
                    feature_vector = [float(record.get(key, 0.0)) for key in feature_keys]
                    features.append(torch.tensor(feature_vector, dtype=torch.float32))
                    labels.append(float(record["label"]))
                    meta.append({"layer": record.get("layer"), "run_id": record.get("run_id")})

        if not features:
            raise ValueError("No labeled layer feature records were found.")

        self.features = torch.stack(features)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.meta = meta
        self.feature_keys = list(feature_keys)
        self.mean = self.features.mean(dim=0)
        self.std = torch.clamp(self.features.std(dim=0), min=1e-6)
        self.features = (self.features - self.mean) / self.std

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, idx: int):
        return self.features[idx], self.labels[idx]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-paths", nargs="+", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, default=Path("artifacts/mlp_models/layer_selector.pt"))
    parser.add_argument("--feature-keys", nargs="*", default=DEFAULT_FEATURE_KEYS)
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=[128, 64])
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--pos-weight", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    torch.manual_seed(args.seed)

    dataset = LayerFeatureDataset(args.dataset_paths, args.feature_keys)
    num_samples = len(dataset)
    if num_samples < 2:
        raise ValueError("Need at least two labeled records for training.")

    val_size = max(1, int(num_samples * args.val_ratio))
    if val_size >= num_samples:
        val_size = num_samples - 1
    train_size = num_samples - val_size

    generator = torch.Generator().manual_seed(args.seed)
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    config = LayerSelectorConfig(
        input_dim=len(args.feature_keys),
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
        feature_keys=args.feature_keys,
    )
    model = LayerSelectorMLP(config)
    pos_weight_tensor = torch.tensor(args.pos_weight, dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for features, labels in train_loader:
            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits.squeeze(-1), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * features.size(0)
        train_loss /= max(1, train_size)

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for features, labels in val_loader:
                logits = model(features)
                loss = criterion(logits.squeeze(-1), labels)
                val_loss += loss.item() * features.size(0)
                preds = torch.sigmoid(logits.squeeze(-1)) > 0.5
                labels_bin = labels > 0.5
                correct += (preds == labels_bin).sum().item()
                total += labels_bin.numel()
        val_loss /= max(1, val_size)
        val_acc = correct / max(1, total)
        LOGGER.info("Epoch %d/%d - train_loss=%.4f val_loss=%.4f val_acc=%.3f", epoch, args.epochs, train_loss, val_loss, val_acc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()

    if best_state is not None:
        model.load_state_dict(best_state)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    norm_stats = {"mean": dataset.mean.tolist(), "std": dataset.std.tolist(), "feature_keys": dataset.feature_keys}
    model.save(args.output_path, norm_stats=norm_stats)

    metrics = {
        "num_samples": num_samples,
        "val_loss": best_val_loss,
        "feature_keys": args.feature_keys,
    }
    metrics_path = args.output_path.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    LOGGER.info("Layer selector saved to %s", args.output_path)


if __name__ == "__main__":
    main()

