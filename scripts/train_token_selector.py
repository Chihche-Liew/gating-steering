#!/usr/bin/env python3
"""Train the token-selection MLP using the curated dataset."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import List, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

from modules.token_selector import TokenSelectorConfig, TokenSelectorMLP
from setup import setup_logging


LOGGER = logging.getLogger("train_token_selector")


class TokenSelectionDataset(Dataset):
    def __init__(self, feature_paths: Sequence[Path]):
        features_list = []
        labels_list = []
        layer_ids = []

        for path in feature_paths:
            data = np.load(path, allow_pickle=True)
            features_list.append(data["features"])
            labels_list.append(data["labels"])
            layer_ids.extend([int(data["layer"])] * data["features"].shape[0])

        self.features = torch.from_numpy(np.vstack(features_list)).float()
        self.labels = torch.from_numpy(np.concatenate(labels_list)).float()
        self.layer_ids = torch.tensor(layer_ids, dtype=torch.int64)

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, idx: int):
        return self.features[idx], self.labels[idx], self.layer_ids[idx]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-paths", nargs="+", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, default=Path("artifacts/mlp_models/token_selector.pt"))
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=[512, 256])
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pos-weight", type=float, default=1.0)
    return parser.parse_args()


def train() -> None:
    setup_logging()
    args = parse_args()
    torch.manual_seed(args.seed)

    dataset = TokenSelectionDataset(args.dataset_paths)
    num_samples = len(dataset)
    if num_samples < 2:
        raise ValueError("Need at least 2 samples to train/validate the token selector.")
    val_size = max(1, int(num_samples * args.val_ratio))
    if val_size >= num_samples:
        val_size = num_samples - 1
    train_size = num_samples - val_size
    generator = torch.Generator().manual_seed(args.seed)
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    input_dim = dataset.features.shape[1]
    layer_ids = sorted(set(dataset.layer_ids.tolist()))
    config = TokenSelectorConfig(
        input_dim=input_dim,
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
        layer_ids=layer_ids,
        extra_features=["relative_position"],
    )
    model = TokenSelectorMLP(config)

    pos_weight_tensor = torch.tensor(args.pos_weight, dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for features, labels, _ in train_loader:
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
            for features, labels, _ in val_loader:
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
    model.save(args.output_path)
    metrics_path = args.output_path.with_suffix(".metrics.json")
    metrics = {
        "num_samples": num_samples,
        "val_loss": best_val_loss,
        "layers": layer_ids,
        "input_dim": input_dim,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    LOGGER.info("Token selector saved to %s", args.output_path)


if __name__ == "__main__":
    train()

