#!/usr/bin/env python3
"""
Lightweight MLP that picks the best steering layer given summary features.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Sequence

import torch
from torch import nn


@dataclass
class LayerSelectorConfig:
    input_dim: int
    hidden_dims: Sequence[int]
    dropout: float = 0.15
    feature_keys: Sequence[str] | None = None

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["hidden_dims"] = list(payload["hidden_dims"])
        if payload["feature_keys"] is not None:
            payload["feature_keys"] = list(payload["feature_keys"])
        return payload

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "LayerSelectorConfig":
        return cls(
            input_dim=int(payload["input_dim"]),
            hidden_dims=list(payload.get("hidden_dims", [])),
            dropout=float(payload.get("dropout", 0.15)),
            feature_keys=list(payload["feature_keys"]) if payload.get("feature_keys") is not None else None,
        )


class LayerSelectorMLP(nn.Module):
    """
    Binary classifier trained per layer (1 = best layer).
    Scores each layer independently; the highest score wins.
    """

    def __init__(self, config: LayerSelectorConfig):
        super().__init__()
        if config.input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        self.config = config

        layers: list[nn.Module] = []
        in_dim = config.input_dim
        for hidden_dim in config.hidden_dims:
            layers.extend(
                [
                    nn.Linear(in_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(p=config.dropout),
                ]
            )
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)

    def score(self, features: torch.Tensor) -> torch.Tensor:
        logits = self.forward(features)
        return logits.squeeze(-1)

    def save(self, checkpoint_path: Path, norm_stats: Dict[str, Any] | None = None) -> None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": self.config.to_dict(),
            "state_dict": self.state_dict(),
            "norm_stats": norm_stats,
        }
        torch.save(payload, checkpoint_path)

    @classmethod
    def load(
        cls,
        checkpoint_path: Path,
        map_location: str | torch.device = "cpu",
    ) -> tuple["LayerSelectorMLP", Dict[str, Any] | None]:
        payload = torch.load(checkpoint_path, map_location=map_location)
        config = LayerSelectorConfig.from_dict(payload["config"])
        model = cls(config)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return model, payload.get("norm_stats")


__all__ = ["LayerSelectorMLP", "LayerSelectorConfig"]

