#!/usr/bin/env bash
set -euo pipefail

# ---- config for the smoke test ----
MODEL_ID="meta-llama/Meta-Llama-3.1-8B"
TRIPLES_OUT="artifacts/triples/triples_small.jsonl"
MAX_SAMPLES=5
MAX_NEW_TOKENS=1024
LAYERS="28 30"
PROBE_LAYER=30
PROBE_MAX=1000

# ---- setup ----
conda create -n steering python=3.10
conda activate steering
export HF_HOME="$(pwd)/.cache/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="$(pwd)/.cache/transformers"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE"
pip install --upgrade pip
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt


# If first time using HF Hub, uncomment and run once (will prompt for token)
# hf auth login

# ---- step 1: generate small triple set ----
python scripts/prepare_triples.py \
  --dataset-name "UW-Madison-Lee-Lab/MMLU-Pro-CoT-Eval" \
  --split test \
  --model-name "${MODEL_ID}" \
  --max-samples "${MAX_SAMPLES}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --only-wrong \
  --output-path "${TRIPLES_OUT}"

python scripts/prepare_triples.py --dataset-name "UW-Madison-Lee-Lab/MMLU-Pro-CoT-Eval" --split test --model-name "meta-llama/Meta-Llama-3.1-8B" --max-samples "5" --max-new-tokens "64" --only-wrong --output-path "artifacts/triples/triples_small.jsonl"

# ---- step 2: capture hidden states + viz ----
python scripts/collect_hidden_states.py \
  --triples-path "${TRIPLES_OUT}" \
  --model-name "${MODEL_ID}" \
  --layers ${LAYERS} \
  --probe-layer "${PROBE_LAYER}" \
  --probe-max-samples "${PROBE_MAX}" \
  --max-samples "${MAX_SAMPLES}"

echo "Done. Check artifacts/hidden_states/, artifacts/alignments/, and reports/hidden_state_viz/ for outputs."