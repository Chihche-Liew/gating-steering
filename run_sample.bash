CUDA_VISIBLE_DEVICES="" python scripts/compute_steering_vectors.py \
  --triples-path artifacts/manual_review/10232025_human_review.json \
  --model-name meta-llama/Llama-3.1-8B-Instruct \
  --layers 28 \
  --output-dir artifacts/steering_vectors_ss10 \
  --max-samples 10 \
  --token-selection-method last_token \
  --system-prompt "You are a careful reasoning assistant. Think step by step and end with 'Final answer: <choice>'."

CUDA_VISIBLE_DEVICES="" python scripts/compute_steering_vectors.py \
  --triples-path artifacts/manual_review/10232025_human_review.json \
  --model-name meta-llama/Llama-3.1-8B-Instruct \
  --layers 28 \
  --output-dir artifacts/steering_vectors_ss50 \
  --max-samples 50 \
  --token-selection-method last_token \
  --system-prompt "You are a careful reasoning assistant. Think step by step and end with 'Final answer: <choice>'."
