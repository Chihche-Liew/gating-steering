#!/usr/bin/env bash
set -euo pipefail

# ---- Initialize conda ----
# Source conda.sh to enable conda commands in this script
# Try common conda installation locations
CONDA_SH=""
for conda_path in "$HOME/anaconda3" "$HOME/miniconda3" "$CONDA_PREFIX/../.." "$(dirname $(dirname $(which conda 2>/dev/null) 2>/dev/null) 2>/dev/null)"; do
    if [ -f "$conda_path/etc/profile.d/conda.sh" ]; then
        CONDA_SH="$conda_path/etc/profile.d/conda.sh"
        break
    fi
done
if [ -n "$CONDA_SH" ]; then
    source "$CONDA_SH"
else
    echo "Error: conda.sh not found. Please ensure Anaconda/Miniconda is installed."
    exit 1
fi

# ---- config for the smoke test ----
MODEL_ID="meta-llama/Llama-3.1-8B-Instruct"
MAX_SAMPLES=100
MAX_NEW_TOKENS=1024

# Pooling method: mean, last_token, or per_token
POOLING_METHOD="per_token"

# Layers to analyze (space-separated list)
# For Llama-3.1-8B: layers 0-31 available
# Focusing on layer 28 which shows good separation with mean pooling
LAYERS="26 27 28 29 30 31"

PROBE_MAX=1000  # Max samples per layer for probe data
DATASET_PATH="UW-Madison-Lee-Lab/MMLU-Pro-CoT-Eval"
DATASET_CONFIG="ALL"

# Derived artifact locations
ANALYSIS_OUTPUT="artifacts/probe_analysis"
PLOT_OUTPUT="reports/hidden_state_viz_${POOLING_METHOD}"

TOKEN_SELECTION_METHOD="last_token"  # last_token | gradient | dp_gradient | dp_average | token_mlp
TOKEN_SELECTOR_MLP_PATH="artifacts/mlp_models/token_selector.pt"
LAYER_SELECTION_METHOD="fixed"  # fixed | mlp
LAYER_SELECTOR_MLP_PATH="artifacts/mlp_models/layer_selector.pt"
LAYER_SELECTOR_TOPK=1

DP_ALIGNMENT_ARGS=(--dp-alignment)

# ---- setup ----

# Create environment if it doesn't exist
# if ! conda info --envs | grep -q '^steering'; then
#   conda create -y -n steering python=3.10
# fi
# conda activate steering
export HF_HOME="$(pwd)/.cache/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="$(pwd)/.cache/transformers"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE"

# echo "Installing requirements..."
# pip install -q --upgrade pip
# pip3 install -q torch torchvision --index-url https://download.pytorch.org/whl/cu118
# pip install -q -r requirements.txt

# If first time using HF Hub, uncomment and run once (will prompt for token)
#hf auth login

# ---- step 1: generate triple set ----
TRIPLES_PATH="./artifacts/manual_review/10232025_human_review.json"
TRAIN_OFFSET=0
EVAL_OFFSET=100

if [ -f "${TRIPLES_PATH}" ]; then
  echo "=== Step 1: Using manually reviewed triples: ${TRIPLES_PATH} ==="
else
  echo "ERROR: Manual triples file not found at ${TRIPLES_PATH}"
  exit 1
fi

if [ -f "${TRIPLES_OUT}" ]; then
  echo "=== Step 1: Skipping triple generation (file exists: ${TRIPLES_OUT}) ==="
  echo "To regenerate, delete the file and rerun."
else
  echo "=== Step 1: Generating triples ==="
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  python scripts/prepare_triples.py \
    --dataset-name "${DATASET_PATH}" \
    --split test \
    --model-name "${MODEL_ID}" \
    --max-samples "${MAX_SAMPLES}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --only-wrong \
    --output-path "${TRIPLES_OUT}" \
    --system-prompt "You are a careful reasoning assistant. Think step by step and end with 'Final answer: <choice>'."
fi

# # ---- step 2: capture hidden states ----
# echo "=== Step 2: Capturing hidden states with ${POOLING_METHOD} pooling ==="
# # Note: collect_hidden_states.py will generate probe data for all layers specified
# python scripts/collect_hidden_states.py \
#   --triples-path "${TRIPLES_PATH}" \
#   --model-name "${MODEL_ID}" \
#   --layers ${LAYERS} \
#   --probe-max-samples "${PROBE_MAX}" \
#   --max-samples "${MAX_SAMPLES}" \
#   --pooling-method "${POOLING_METHOD}" \
#   --alignment-layer 28 \
#   --dp-max-shift 40 \
#   --system-prompt "You are a careful reasoning assistant. Think step by step and end with 'Final answer: <choice>'." \
#   "${DP_ALIGNMENT_ARGS[@]}"

# ---- step 3: compute probes and vectors for multiple layers ----
echo "=== Step 3: Computing probes and vectors ==="
ANALYSIS_OUTPUT="artifacts/probe_analysis"

for layer in ${LAYERS}; do
  PROBE_DATA="artifacts/probe_data/layer${layer}_probe_data.npz"

  if [ -f "${PROBE_DATA}" ]; then
    echo "  Computing probes for layer ${layer}..."
    cmd=(
      python scripts/compute_probes.py
      --probe-data-path "${PROBE_DATA}"
      --output-dir "${ANALYSIS_OUTPUT}"
      --seed 42
    )
    cmd+=("${DP_ALIGNMENT_ARGS[@]}")
    "${cmd[@]}"
  else
    echo "  Warning: Probe data not found for layer ${layer} at ${PROBE_DATA}"
  fi
done

# ---- step 4: generate visualizations for multiple layers ----
if [ -d "${ANALYSIS_OUTPUT}" ]; then
  echo "=== Step 4: Generating visualizations ==="
  for layer in ${LAYERS}; do
    METRICS_FILE="${ANALYSIS_OUTPUT}/layer${layer}_metrics.json"
    if [ -f "${METRICS_FILE}" ]; then
      echo "  Plotting layer ${layer}..."
      cmd=(
        python scripts/plot_probes.py
        --analysis-dir "${ANALYSIS_OUTPUT}"
        --layer ${layer}
        --output-dir "${PLOT_OUTPUT}"
      )
      cmd+=("${DP_ALIGNMENT_ARGS[@]}")
      "${cmd[@]}"
    else
      echo "  Warning: Metrics not found for layer ${layer}, skipping plots"
    fi
  done
else
  echo "=== Step 4: Skipping visualizations (analysis directory missing) ==="
fi

# ---- step 5: analyze critical tokens (only for per_token pooling) ----
if [ "${POOLING_METHOD}" = "per_token" ]; then
  echo "=== Step 5: Analyzing critical token positions ==="
  CRITICAL_TOKENS_OUTPUT="reports/critical_tokens_${POOLING_METHOD}"

  for layer in ${LAYERS}; do
    PROBE_DATA="artifacts/probe_data/layer${layer}_probe_data.npz"

    if [ -f "${PROBE_DATA}" ]; then
      echo "  Analyzing critical tokens for layer ${layer}..."
      cmd=(
        python scripts/analyze_critical_tokens.py
        --probe-data-path "${PROBE_DATA}"
        --alignments-dir "artifacts/alignments"
        --output-dir "${CRITICAL_TOKENS_OUTPUT}"
        --top-k 20
      )
      cmd+=("${DP_ALIGNMENT_ARGS[@]}")
      "${cmd[@]}"
    else
      echo "  Warning: Probe data not found for layer ${layer}"
    fi
  done
else
  echo "=== Step 5: Skipping critical token analysis (only applicable for per_token pooling) ==="
fi

# ---- step 6: compute steering vectors ----
#echo "=== Step 6: Computing steering vectors from last_token representations ==="
STEERING_VECTORS_DIR="artifacts/steering_vectors"

TOKEN_SELECTOR_ARGS=()
if [ "${TOKEN_SELECTION_METHOD}" = "token_mlp" ]; then
  TOKEN_SELECTOR_ARGS+=(--token-selection-mlp-path "${TOKEN_SELECTOR_MLP_PATH}")
fi

python scripts/compute_steering_vectors.py \
  --triples-path "${TRIPLES_PATH}" \
  --model-name "${MODEL_ID}" \
  --layers ${LAYERS} \
  --output-dir "${STEERING_VECTORS_DIR}" \
  --max-samples "${MAX_SAMPLES}" \
  --token-selection-method "${TOKEN_SELECTION_METHOD}" \
  --alignments-dir "artifacts/alignments" \
  --system-prompt "${SYSTEM_PROMPT}" \
  "${TOKEN_SELECTOR_ARGS[@]}"

# ---- step 7: evaluate steering ----
echo "=== Step 7: Evaluating steering vectors ==="
STEERING_EVAL_DIR_IN="artifacts/steering_evaluation_in_sample"
STEERING_EVAL_DIR_OUT="artifacts/steering_evaluation_out_of_sample"

# Only evaluate steering if vectors were computed
if [ "${POOLING_METHOD}" = "last_token" ] && [ -d "${STEERING_VECTORS_DIR}" ]; then
  # Evaluate each layer separately (or all together - you can modify this)
  # For now, evaluate with the best layer (typically layer 28)
  # To test a specific layer (e.g., layer 26), change BEST_LAYER="26"
  BEST_LAYER="29"  # Can be changed based on visualization results  # can add a list of layers here
  EVAL_LAYERS="${BEST_LAYER}"
  LAYER_SELECTOR_ARGS=()
  if [ "${LAYER_SELECTION_METHOD}" = "mlp" ]; then
    EVAL_LAYERS="${LAYERS}"
    LAYER_SELECTOR_ARGS+=(--layer-selection-mlp-path "${LAYER_SELECTOR_MLP_PATH}")
  fi
  
  # Baseline results (known: 8 correct, 92 incorrect for 100 samples)
  # Set SKIP_BASELINE=1 to skip baseline computation and use provided values
  SKIP_BASELINE=0
  BASELINE_CORRECT=0
  BASELINE_INCORRECT=0
  
  # Check if steering vector exists for the best layer
  if [ -f "${STEERING_VECTORS_DIR}/layer${BEST_LAYER}_steering_vector.npy" ]; then
    echo "Evaluating steering for layer ${BEST_LAYER}..."
    
    if [ "${SKIP_BASELINE}" = "1" ]; then
      echo "Skipping baseline evaluation (using provided: ${BASELINE_CORRECT} correct, ${BASELINE_INCORRECT} incorrect)"
      SKIP_ARGS=(--skip-baseline --baseline-correct "${BASELINE_CORRECT}" --baseline-incorrect "${BASELINE_INCORRECT}")
    else
      SKIP_ARGS=()
    fi

    echo "Evaluating in-sample steering performance..."
    python scripts/evaluate_steering.py \
      --triples-path "${TRIPLES_PATH}" \
      --steering-vectors-dir "${STEERING_VECTORS_DIR}" \
      --model-name "${MODEL_ID}" \
      --layers ${EVAL_LAYERS} \
      --steering-coefficient 1.0 \
      --output-dir "${STEERING_EVAL_DIR_IN}" \
      --max-samples "${MAX_SAMPLES}" \
      --max-new-tokens "${MAX_NEW_TOKENS}" \
      --system-prompt "${SYSTEM_PROMPT}" \
      --sample-offset "${TRAIN_OFFSET}" \
      --layer-selection-method "${LAYER_SELECTION_METHOD}" \
      --layer-selection-topk "${LAYER_SELECTOR_TOPK}" \
      "${SKIP_ARGS[@]}" \
      "${LAYER_SELECTOR_ARGS[@]}"

    echo "Evaluating out-of-sample steering performance..."
    python scripts/evaluate_steering.py \
      --triples-path "${TRIPLES_PATH}" \
      --steering-vectors-dir "${STEERING_VECTORS_DIR}" \
      --model-name "${MODEL_ID}" \
      --layers ${EVAL_LAYERS} \
      --steering-coefficient 1.0 \
      --output-dir "${STEERING_EVAL_DIR_OUT}" \
      --max-samples "${MAX_SAMPLES}" \
      --max-new-tokens "${MAX_NEW_TOKENS}" \
      --system-prompt "${SYSTEM_PROMPT}" \
      --sample-offset "${EVAL_OFFSET}" \
      --layer-selection-method "${LAYER_SELECTION_METHOD}" \
      --layer-selection-topk "${LAYER_SELECTOR_TOPK}" \
      "${SKIP_ARGS[@]}" \
      "${LAYER_SELECTOR_ARGS[@]}"
    
    echo "Steering evaluation complete. Results saved to ${STEERING_EVAL_DIR_IN} and ${STEERING_EVAL_DIR_OUT}"
  else
    echo "  Warning: Steering vector not found for layer ${BEST_LAYER}, skipping evaluation"
  fi
else
  echo "=== Step 7: Skipping steering evaluation (steering vectors not computed) ==="
fi

# ---- step 8: compare embeddings with and without steering ----
echo "=== Step 8: Comparing embeddings with and without steering ==="
EMBEDDING_COMPARISON_DIR_IN="reports/steering_embedding_comparison_in_sample"
EMBEDDING_COMPARISON_DIR_OUT="reports/steering_embedding_comparison_out_of_sample"

# Only compare embeddings if steering vectors exist
if [ "${POOLING_METHOD}" = "last_token" ] && [ -d "${STEERING_VECTORS_DIR}" ]; then
  if [ -f "${STEERING_VECTORS_DIR}/layer${BEST_LAYER}_steering_vector.npy" ]; then
    echo "Comparing embeddings for layer ${BEST_LAYER}..."
    python scripts/compare_steering_embeddings.py \
      --triples-path "${TRIPLES_PATH}" \
      --steering-vectors-dir "${STEERING_VECTORS_DIR}" \
      --model-name "${MODEL_ID}" \
      --layer ${BEST_LAYER} \
      --steering-coefficient 1.0 \
      --output-dir "${EMBEDDING_COMPARISON_DIR_IN}" \
      --max-samples "${MAX_SAMPLES}" \
      --system-prompt "You are a careful reasoning assistant. Think step by step and end with 'Final answer: <choice>'." \
      --pooling-method "${POOLING_METHOD}" \
      --sample-offset "${TRAIN_OFFSET}" \
      --seed 42
    
    python scripts/compare_steering_embeddings.py \
      --triples-path "${TRIPLES_PATH}" \
      --steering-vectors-dir "${STEERING_VECTORS_DIR}" \
      --model-name "${MODEL_ID}" \
      --layer ${BEST_LAYER} \
      --steering-coefficient 1.0 \
      --output-dir "${EMBEDDING_COMPARISON_DIR_OUT}" \
      --max-samples "${MAX_SAMPLES}" \
      --system-prompt "You are a careful reasoning assistant. Think step by step and end with 'Final answer: <choice>'." \
      --pooling-method "${POOLING_METHOD}" \
      --sample-offset "${EVAL_OFFSET}" \
      --seed 42
    
    echo "Embedding comparison complete. Results saved to ${EMBEDDING_COMPARISON_DIR_IN} and ${EMBEDDING_COMPARISON_DIR_OUT}"
  else
    echo "  Warning: Steering vector not found for layer ${BEST_LAYER}, skipping embedding comparison"
  fi
else
  echo "=== Step 8: Skipping embedding comparison (steering vectors not computed) ==="
fi

echo ""
echo "=== Done! ==="
echo "Results:"
echo "  - Triples source: ${TRIPLES_PATH}"
echo "  - Train offset: ${TRAIN_OFFSET}, Eval offset: ${EVAL_OFFSET}"
echo "  - Hidden states: artifacts/hidden_states/"
echo "  - Alignments: artifacts/alignments/"
echo "  - Probe data: artifacts/probe_data/"
echo "  - Computed probes: ${ANALYSIS_OUTPUT}/"
echo "  - Visualizations: ${PLOT_OUTPUT}/"
if [ "${POOLING_METHOD}" = "last_token" ]; then
  echo "  - Steering vectors: ${STEERING_VECTORS_DIR}/"
  echo "  - Steering evaluation (in-sample): ${STEERING_EVAL_DIR_IN}/"
  echo "  - Steering evaluation (out-of-sample): ${STEERING_EVAL_DIR_OUT}/"
  echo "  - Embedding comparison (in-sample): ${EMBEDDING_COMPARISON_DIR_IN}/"
  echo "  - Embedding comparison (out-of-sample): ${EMBEDDING_COMPARISON_DIR_OUT}/"
fi
echo ""
echo "Processed layers: ${LAYERS}"