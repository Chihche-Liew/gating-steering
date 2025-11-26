#!/usr/bin/env python3
"""Aggregate numeric features per layer for the layer-selection MLP."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import List

from modules.layer_features import build_layer_feature_dict, load_steering_metadata
from setup import setup_logging


LOGGER = logging.getLogger("extract_layer_features")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", type=str, required=True, help="Identifier for this experiment/run.")
    parser.add_argument("--layers", nargs="*", type=int, help="Optional list of layer ids to include.")
    parser.add_argument(
        "--steering-metadata-path",
        type=Path,
        default=Path("artifacts/steering_vectors/steering_vectors_metadata.json"),
    )
    parser.add_argument(
        "--probe-metrics-dir",
        type=Path,
        default=Path("artifacts/probe_analysis"),
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("artifacts/mlp_features/layer_selection_features.jsonl"),
    )
    parser.add_argument("--best-layer", type=int, default=None, help="Layer id labeled as optimal for this run.")
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to existing JSONL instead of overwriting.",
    )
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()

    steering_metadata = load_steering_metadata(args.steering_metadata_path)
    candidate_layers: List[int] = args.layers or steering_metadata.get("layers", [])
    if not candidate_layers:
        LOGGER.error("No layers specified or found in metadata.")
        return

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.append and args.output_path.exists() else "w"
    with args.output_path.open(mode, encoding="utf-8") as sink:
        for layer_id in candidate_layers:
            feature_dict = build_layer_feature_dict(
                layer_id=layer_id,
                steering_metadata=steering_metadata,
                probe_metrics_dir=args.probe_metrics_dir,
            )
            record = {
                "run_id": args.run_id,
                **feature_dict,
            }
            if args.best_layer is not None:
                record["label"] = 1 if layer_id == args.best_layer else 0
                record["best_layer"] = args.best_layer
            sink.write(json.dumps(record))
            sink.write("\n")
            LOGGER.info("Recorded layer %d features (label=%s).", layer_id, record.get("label"))

    LOGGER.info("Layer feature records written to %s", args.output_path)


if __name__ == "__main__":
    main()

