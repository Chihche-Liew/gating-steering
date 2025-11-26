#!/usr/bin/env python3
"""
Small feed-forward network that scores token hidden states for steering.

The module is intentionally lightweight so it can run on CPU during steering.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

import torch
from torch import nn


@dataclass
class TokenSelectorConfig:
    input_dim: int
    hidden_dims: Sequence[int]
    dropout: float = 0.1
    layer_ids: Sequence[int] | None = None
    extra_features: Sequence[str] | None = None

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["hidden_dims"] = list(payload["hidden_dims"])
        if payload["layer_ids"] is not None:
            payload["layer_ids"] = list(payload["layer_ids"])
        if payload["extra_features"] is not None:
            payload["extra_features"] = list(payload["extra_features"])
        return payload

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "TokenSelectorConfig":
        return cls(
            input_dim=int(payload["input_dim"]),
            hidden_dims=list(payload.get("hidden_dims", [])),
            dropout=float(payload.get("dropout", 0.1)),
            layer_ids=list(payload["layer_ids"]) if payload.get("layer_ids") is not None else None,
            extra_features=list(payload["extra_features"]) if payload.get("extra_features") is not None else None,
        )


class TokenSelectorMLP(nn.Module):
    """
    Simple MLP that ingests concatenated token features (hidden state + metadata)
    and returns a scalar score per token. Higher score == more salient token.
    """

    def __init__(self, config: TokenSelectorConfig):
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

    def score_tokens(
        self,
        hidden_states: torch.Tensor,
        extra_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [seq_len, hidden_dim]
            extra_features: [seq_len, extra_dim]
        Returns:
            scores: [seq_len]
        """
        if hidden_states.ndim != 2:
            raise ValueError("hidden_states must have shape [seq_len, hidden_dim].")
        if extra_features.ndim != 2:
            raise ValueError("extra_features must have shape [seq_len, extra_dim].")
        if hidden_states.shape[0] != extra_features.shape[0]:
            raise ValueError("hidden_states and extra_features must share seq_len.")

        features = torch.cat([hidden_states, extra_features], dim=1)
        logits = self.forward(features)
        return logits.squeeze(-1)

    def save(self, checkpoint_path: Path) -> None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": self.config.to_dict(),
            "state_dict": self.state_dict(),
        }
        torch.save(payload, checkpoint_path)

    @classmethod
    def load(cls, checkpoint_path: Path, map_location: str | torch.device = "cpu") -> "TokenSelectorMLP":
        payload = torch.load(checkpoint_path, map_location=map_location)
        config = TokenSelectorConfig.from_dict(payload["config"])
        model = cls(config)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return model


__all__ = ["TokenSelectorMLP", "TokenSelectorConfig"]

