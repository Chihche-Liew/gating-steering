#!/usr/bin/env python3
"""
Utility helpers for constructing numeric feature vectors for layer selection.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def _get_layer_key(layer_id: int, payload: Dict[str, Any]) -> str:
    if str(layer_id) in payload:
        return str(layer_id)
    return str(int(layer_id))


def _extract_summary(summary: Dict[str, Any] | None) -> Dict[str, float]:
    default = {"count": 0.0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    if not isinstance(summary, dict):
        return default
    output = {}
    for key in default.keys():
        value = summary.get(key, default[key])
        try:
            output[key] = float(value)
        except (TypeError, ValueError):
            output[key] = default[key]
    return output


def load_steering_metadata(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Steering metadata not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_probe_metrics(path: Path) -> Dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def build_layer_feature_dict(
    layer_id: int,
    steering_metadata: Dict[str, Any],
    probe_metrics_dir: Path | None = None,
) -> Dict[str, float]:
    """
    Build a numeric feature dict for a given layer using steering metadata and
    optional probe metrics.
    """
    features: Dict[str, float] = {
        "layer": float(layer_id),
    }

    vector_norms = steering_metadata.get("steering_vector_norms", {})
    samples_per_layer = steering_metadata.get("num_samples_per_layer", {})
    token_summary_map = steering_metadata.get("token_position_summary", {})
    grad_summary_map = steering_metadata.get("gradient_norm_summary", {})

    norm_key = _get_layer_key(layer_id, vector_norms)
    samples_key = _get_layer_key(layer_id, samples_per_layer)
    token_key = _get_layer_key(layer_id, token_summary_map)
    grad_key = _get_layer_key(layer_id, grad_summary_map)

    features["vector_norm"] = float(vector_norms.get(norm_key, 0.0))
    features["num_samples"] = float(samples_per_layer.get(samples_key, 0.0))

    token_summary = _extract_summary(token_summary_map.get(token_key))
    features["token_pos_count"] = token_summary["count"]
    features["token_pos_mean"] = token_summary["mean"]
    features["token_pos_std"] = token_summary["std"]
    features["token_pos_min"] = token_summary["min"]
    features["token_pos_max"] = token_summary["max"]

    grad_summary = _extract_summary(grad_summary_map.get(grad_key))
    features["grad_norm_count"] = grad_summary["count"]
    features["grad_norm_mean"] = grad_summary["mean"]
    features["grad_norm_std"] = grad_summary["std"]
    features["grad_norm_min"] = grad_summary["min"]
    features["grad_norm_max"] = grad_summary["max"]

    probe_metrics: Dict[str, Any] | None = None
    if probe_metrics_dir is not None:
        probe_metrics = load_probe_metrics(probe_metrics_dir / f"layer{layer_id}_metrics.json")

    if probe_metrics:
        features["probe_num_samples"] = float(probe_metrics.get("num_samples", 0.0))
        features["probe_feature_dim"] = float(probe_metrics.get("feature_dim", 0.0))
    else:
        features.setdefault("probe_num_samples", 0.0)
        features.setdefault("probe_feature_dim", 0.0)

    return features


__all__ = ["build_layer_feature_dict", "load_steering_metadata"]

