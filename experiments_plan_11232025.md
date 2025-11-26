## Experiment Preparation Checklist

1. **Data and Splits**

   - Clean 200 samples from `artifacts/manual_review/10232025_human_review.json`, documenting filters (dedup, missing labels, category coverage).
   - Set `TRAIN_OFFSET` / `EVAL_OFFSET`, and verify `MAX_SAMPLES` / `MAX_NEW_TOKENS` align with each experiment’s sample size.
   - If you need category-specific runs, materialize both `mixed` and `per-category` triple files under `artifacts/triples/<split>.json`.
2. **Environment and Model Assets**

   - Create/activate the Conda env (Python 3.10 + PyTorch CUDA 11.8) and `pip install -r requirements.txt`.
   - Run `python scripts/collect_hidden_states.py ...` to capture all required layers; ensure `artifacts/hidden_states` and `artifacts/alignments` are complete.
   - Execute `python scripts/compute_probes.py` followed by `python scripts/plot_probes.py` to refresh `artifacts/probe_analysis/` and related plots.
3. **Token Selecting MLP**

   - `python scripts/build_token_selection_dataset.py --triples-path ... --layers 26 29 30 31 --positives-per-sample 2 --negatives-per-positive 3 --save-json-summary`
   - `python scripts/train_token_selector.py --dataset-paths artifacts/mlp_features/token_selector_layer*.npz --output-path artifacts/mlp_models/token_selector.pt --epochs 10 --pos-weight 2.0`
   - In `run.bash`, set `TOKEN_SELECTION_METHOD=token_mlp` and point `TOKEN_SELECTOR_MLP_PATH`; keep gradient/last_token configs for baselines as needed.
4. **Layer Selecting MLP**

   - For each baseline run, execute `python scripts/extract_layer_features.py --run-id <tag> --best-layer <layer>` to append rows to `artifacts/mlp_features/layer_selection_features.jsonl`.
   - `python scripts/train_layer_selector.py --dataset-paths artifacts/mlp_features/layer_selection_features.jsonl --output-path artifacts/mlp_models/layer_selector.pt --epochs 50 --pos-weight 3.0`
   - Configure `LAYER_SELECTION_METHOD=mlp` and `LAYER_SELECTOR_TOPK` (usually 1–2) in `run.bash`; keep `fixed` mode for comparison.
5. **Evaluation and Logging**

   - For every run, persist outputs from `scripts/compute_steering_vectors.py`, `scripts/evaluate_steering.py`, and `scripts/compare_steering_embeddings.py` (`artifacts/steering_vectors`, `artifacts/steering_evaluation_*`, `reports/steering_embedding_comparison_*`).
   - Capture the exact CLI arguments (e.g., `logs/experiment_<id>.txt`) to simplify reproducibility and layer-MLP labeling.

---

## Experiment Matrix (suggested order)

| Phase | Dataset Split                      | Token Selector                          | Layer Selector                                                        | Sample Size      | Eval Scope | Notes                                                              |
| ----- | ---------------------------------- | --------------------------------------- | --------------------------------------------------------------------- | ---------------- | ---------- | ------------------------------------------------------------------ |
| P0    | Mixed (baseline)                   | last_token                              | fixed (layer 29)                                                      | 100              | in/out     | Sanity-check pipeline, produce layer-selector labels               |
| P1    | Mixed                              | gradient                                | fixed (29)                                                            | 100              | in/out     | Compare gradient vs last-token                                     |
| P2    | Mixed                              | token_mlp                               | fixed (29)                                                            | 100              | in/out     | Check if token MLP beats gradient                                  |
| P3    | Mixed                              | **best of (gradient, token_mlp)** | layer_mlp (top1)                                                      | 100              | in/out     | Use whichever token selector wins P1 vs P2, then turn on layer MLP |
| P5    | Mixed                              | best from P1/P2                         | **layer strategy = better of (P3, best single-layer baseline)** | 50               | in/out     | Sample-size sensitivity (small) using the best selectors           |
| P6    | Mixed                              | best from P1/P2                         | same as P5                                                            | 200              | in/out     | Sample-size sensitivity (large) using the best selectors           |
| P7    | Per-category (run per group)       | follow best selectors (from P1–P3)     | follow best selectors (from P1–P3)                                   | 100 per category | in/out     | Category-level analysis with global best configuration             |
| P8    | Alternate dataset (generalization) | follow best selectors (from P1–P3)     | follow best selectors (from P1–P3)                                   | 100              | in/out     | Generalization test with the optimal selector combo                |

### Run Notes

- After finishing P1 & P2, lock in the stronger token selector; reuse that choice for P3 onward.
- Compare P3 against the best single-layer baseline (P1 or P2) to decide whether to keep layer MLP or fixed-layer steering for P5+.
- For P7, loop through categories (math, logical, commonsense, …) and aggregate for ANOVA/significance testing.
- For P8, validate dataset compatibility with `build_teacher_forcing_text` before running.
- For P5/P6, track accuracy vs. sample-size curves to include learning-curve analysis in the final report.
