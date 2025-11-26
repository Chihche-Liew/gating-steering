## 11232025 MLP Updates

### 1. Token Selecting MLP
- **Dataset**: `scripts/build_token_selection_dataset.py` runs a gradient sweep over the wrong chains, saves per-token hidden states (+ relative position) with labels derived from the highest gradient norms. Positives default to the top-`k` tokens (gradient ranks) per layer/sample; negatives are randomly drawn from the remaining tokens. Adjust `--positives-per-sample`, `--negatives-per-positive`, and `--max-tokens-per-sample` to control coverage. Output lives under `artifacts/mlp_features/token_selector_layer{L}.npz`.
- **Training**: `scripts/train_token_selector.py --dataset-paths artifacts/mlp_features/token_selector_layer*.npz --output-path artifacts/mlp_models/token_selector.pt`. The model is a two-layer MLP (configurable) trained with BCE + optional `--pos-weight`. Features are normalized implicitly by the MLP; no extra scaler file is needed.
- **Inference**: Load happens automatically when `run.bash` sets `TOKEN_SELECTION_METHOD=token_mlp` and `TOKEN_SELECTOR_MLP_PATH`. `compute_steering_vectors.py` keeps the selector on CPU, concatenates hidden states with the relative position feature, and masks prompt tokens before argmax.

**Labeling note**: Gradient magnitudes are already computed for steering; treating top-ranked tokens as positives (and optionally weighting by norm for `--pos-weight`) keeps the signal simple and avoids adding new supervision sources. If you need softer labels later, the NPZ files also store `grad_norms`.

### 2. Layer Selecting MLP (Global mode)
- **Feature extraction**: `scripts/extract_layer_features.py --run-id <tag> --steering-metadata-path artifacts/steering_vectors/steering_vectors_metadata.json --best-layer <L>` writes one JSONL row per layer with stats such as steering vector norm, sample counts, token/gradient summaries, and (optionally) a binary label marking the global best layer for that run.
- **Training**: `scripts/train_layer_selector.py --dataset-paths artifacts/mlp_features/layer_selection_features.jsonl --output-path artifacts/mlp_models/layer_selector.pt`. Features are z-scored internally; the checkpoint stores the means/stds and `feature_keys`.
- **Evaluation hook**: Set `LAYER_SELECTION_METHOD=mlp` (and `LAYER_SELECTOR_MLP_PATH`, `LAYER_SELECTOR_TOPK`) in `run.bash`. `evaluate_steering.py` will score every requested layer on the CPU, normalize with the stored stats, and keep the top-`k` layers before calling `generate_with_steering`.

### 3. Workflow quick start
1. `python scripts/build_token_selection_dataset.py --triples-path ... --layers 26 29 30`
2. `python scripts/train_token_selector.py --dataset-paths artifacts/mlp_features/token_selector_layer*.npz`
3. `python scripts/extract_layer_features.py --run-id smoke --best-layer 29`
4. `python scripts/train_layer_selector.py --dataset-paths artifacts/mlp_features/layer_selection_features.jsonl`
5. Update `run.bash` knobs: `TOKEN_SELECTION_METHOD`, `TOKEN_SELECTOR_MLP_PATH`, `LAYER_SELECTION_METHOD`, `LAYER_SELECTOR_MLP_PATH`, `LAYER_SELECTOR_TOPK`.

Both MLPs default to CPU inference so they can run alongside the main model without extra GPU memory pressure.

