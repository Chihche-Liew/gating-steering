#!/usr/bin/env python3
"""Generate prompt/response triples with incorrect and correct reasoning chains."""

from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

from model_wrapper import ModelWrapper
from setup import (
    configure_hf_caches,
    setup_logging,
    extract_final_answer,
    normalize_answer,
    should_include
)


LOGGER = logging.getLogger("prepare_triples")


@dataclass
class TripleRecord:
    """Serializable container for a single prompt/response triple."""

    sample_id: str
    prompt: str
    wrong_chain: str
    wrong_answer: str
    correct_chain: Optional[str]
    correct_answer: str
    model_name: str
    metadata: Dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-name",
        default="UW-Madison-Lee-Lab/MMLU-Pro-CoT-Eval",
        help="Name of the dataset on the Hugging Face Hub (default: MMLU-Pro-CoT-Eval).",
    )
    parser.add_argument("--dataset-config", default=None, help="Optional dataset configuration name.")
    parser.add_argument("--split", default="test", help="Dataset split to load (default: test).")
    parser.add_argument("--prompt-field", default="question", help="Field containing the model prompt.")
    parser.add_argument(
        "--gold-answer-field",
        default="answer",
        help="Field containing the gold final answer used for correctness checks.",
    )
    parser.add_argument(
        "--gold-chain-field",
        default="chain_of_thoughts",
        help="Optional field containing gold chain-of-thought text (if available).",
    )
    parser.add_argument(
        "--model-name",
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="Identifier for a causal LM on the Hugging Face Hub (default: meta-llama/Meta-Llama-3-8B-Instruct).",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("artifacts/triples/triples.jsonl"),
        help="Destination JSONL file for collected triples.",
    )
    parser.add_argument(
        "--stats-path",
        type=Path,
        default=None,
        help="Optional path to write collection statistics (defaults to output_path with .stats.json).",
    )
    parser.add_argument("--max-samples", type=int, default=None, help="Stop after processing this many samples.")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum number of tokens to generate for each prompt.",
    )
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (default: greedy).")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p nucleus sampling parameter.")
    parser.add_argument(
        "--do-sample",
        action="store_true",
        help="Enable stochastic sampling during generation (default: disabled).",
    )
    parser.add_argument(
        "--only-wrong",
        action="store_true",
        help="If set, store only cases where the model answer mismatches the gold answer.",
    )
    parser.add_argument(
        "--prompt-template",
        default="{prompt}\n\nProvide your reasoning and final answer.",
        help="Python format string used to build the model prompt (must include {prompt}).",
    )
    parser.add_argument(
        "--final-answer-regex",
        default=r"final answer\s*[:\-]?\s*(.*)",
        help="Case-insensitive regex to extract the final answer from the model response.",
    )
    parser.add_argument(
        "--system-prompt",
        default="You are a careful reasoning assistant. Think step by step and end with 'Final answer: <choice>'.",
        help="System message used when the tokenizer supports chat templates.",
    )
    parser.add_argument(
        "--keep-prompt-in-response",
        action="store_false",
        dest="strip_prompt_from_response",
        help="Keep the original prompt inside model outputs instead of stripping it.",
    )
    parser.set_defaults(strip_prompt_from_response=True)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling and sampling.")
    parser.add_argument(
        "--device",
        default=None,
        help="Optional torch device override (defaults to cuda if available else cpu).",
    )
    return parser.parse_args()



def main() -> None:
    setup_logging()
    configure_hf_caches()
    args = parse_args()

    # Load dataset
    LOGGER.info("Loading dataset %s (split: %s)", args.dataset_name, args.split)
    raw_dataset = load_dataset(args.dataset_name, args.dataset_config, split=args.split)
    dataset_frame = raw_dataset.to_pandas()
    
    # Filter by correctness if available
    if "parsed_answer_correctness" in dataset_frame.columns:
        original_len = len(dataset_frame)
        dataset_frame = dataset_frame[dataset_frame["parsed_answer_correctness"] == True]
        LOGGER.info("Filtered %d -> %d samples with parsed_answer_correctness=True", 
                   original_len, len(dataset_frame))
    
    # Remove duplicates
    dataset_frame = dataset_frame.drop_duplicates(subset="id", keep="first")
    
    examples = dataset_frame.to_dict(orient="records")
    
    # Shuffle
    rng = random.Random(args.seed)
    rng.shuffle(examples)
    
    if not examples:
        LOGGER.error("No examples remain after filtering.")
        return

    # Load model
    model = ModelWrapper(args.model_name, device=args.device)

    # Prepare output
    output_path: Path = args.output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stats_path: Path = args.stats_path or output_path.with_suffix(".stats.json")
    stats: Dict[str, Any] = {
        "dataset_name": args.dataset_name,
        "dataset_config": args.dataset_config,
        "split": args.split,
        "model_name": args.model_name,
        "total_samples": 0,
        "evaluated_samples": 0,
        "collected_triples": 0,
        "skipped_no_prompt": 0,
        "skipped_no_gold_answer": 0,
    }

    with output_path.open("w", encoding="utf-8") as sink:
        progress_total = args.max_samples if args.max_samples is not None else len(examples)
        progress_total = min(progress_total, len(examples))
        
        for example in tqdm(examples, desc="Collecting triples", total=progress_total):
            stats["total_samples"] += 1

            # Extract fields
            prompt_text = example.get(args.prompt_field)
            gold_answer = example.get(args.gold_answer_field)
            gold_chain_raw = example.get(args.gold_chain_field)
            
            # Process gold chain
            gold_chain: Optional[str] = None
            if gold_chain_raw:
                if isinstance(gold_chain_raw, (list, tuple)):
                    gold_chain = "\n".join(str(item) for item in gold_chain_raw if item is not None)
                else:
                    gold_chain = str(gold_chain_raw)

            # Validate required fields
            if prompt_text is None or not str(prompt_text).strip():
                stats["skipped_no_prompt"] += 1
                continue
            if gold_answer is None or not str(gold_answer).strip():
                stats["skipped_no_gold_answer"] += 1
                continue

            stats["evaluated_samples"] += 1

            # Format prompt
            formatted_prompt = args.prompt_template.format(prompt=str(prompt_text))

            # Generate response
            decoded_response = model.generate(
                formatted_prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                do_sample=args.do_sample,
                strip_prompt=args.strip_prompt_from_response,
                system_prompt=args.system_prompt,
            )

            # Extract answer
            predicted_answer = extract_final_answer(decoded_response, args.final_answer_regex)

            # Filter if needed
            if not should_include(predicted_answer, str(gold_answer), args.only_wrong):
                if args.max_samples and stats["collected_triples"] >= args.max_samples:
                    break
                continue

            # Create triple
            triple = TripleRecord(
                sample_id=str(example.get("id", stats["evaluated_samples"])),
                prompt=str(prompt_text),
                wrong_chain=decoded_response,
                wrong_answer=predicted_answer,
                correct_chain=str(gold_chain) if gold_chain is not None else None,
                correct_answer=str(gold_answer),
                model_name=args.model_name,
                metadata={
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "max_new_tokens": args.max_new_tokens,
                    "prompt_template": args.prompt_template,
                },
            )

            sink.write(json.dumps(asdict(triple), ensure_ascii=False) + "\n")
            stats["collected_triples"] += 1

            if args.max_samples and stats["collected_triples"] >= args.max_samples:
                break

    # Write stats
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    
    LOGGER.info("Wrote %d triples to %s", stats["collected_triples"], output_path)
    LOGGER.info("Stats logged to %s", stats_path)


if __name__ == "__main__":
    main()
