#!/usr/bin/env python3
"""Generate prompt/response triples with incorrect and correct reasoning chains."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


LOGGER = logging.getLogger("prepare_triples")


def configure_hf_caches() -> None:
    project_root = Path(__file__).resolve().parents[1]
    cache_root = project_root / ".cache"
    hf_home = cache_root / "huggingface"
    transformers_cache = cache_root / "transformers"

    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HF_DATASETS_CACHE", str(hf_home / "datasets"))
    os.environ.setdefault("HF_HUB_CACHE", str(hf_home / "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(transformers_cache))

    for path in [hf_home, hf_home / "datasets", hf_home / "hub", transformers_cache]:
        path.mkdir(parents=True, exist_ok=True)


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
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


ANSWER_PATTERNS = [
    r"final\s+answer\s*[:\-]?\s*([A-J]|\w+)",
    r"answer\s*[:\-]?\s*([A-J]|\w+)",
    r"conclusion\s*[:\-]?\s*([A-J]|\w+)",
    r"choice\s*[:\-]?\s*([A-J]|\w+)",
]


def extract_final_answer(text: str, pattern: str) -> str:
    candidates: List[str] = []
    compiled = re.compile(pattern, flags=re.IGNORECASE)
    for matcher in [compiled] + [re.compile(pat, flags=re.IGNORECASE) for pat in ANSWER_PATTERNS]:
        match = matcher.search(text)
        if match and match.group(1):
            candidates.append(match.group(1).strip().strip("."))
    if candidates:
        return candidates[0]

    fallback = text.strip().splitlines()
    if fallback:
        last_line = fallback[-1].strip()
        single_choice = re.fullmatch(r"([A-J])\.?", last_line, flags=re.IGNORECASE)
        if single_choice:
            return single_choice.group(1)
        return last_line
    return ""


def normalise_answer(answer: str) -> str:
    return re.sub(r"\s+", " ", answer.strip().lower())


def load_model_and_tokenizer(model_name: str):
    LOGGER.info("Loading model %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device


def should_include(predicted: str, gold: str, only_wrong: bool) -> bool:
    if not only_wrong:
        return True
    return normalise_answer(predicted) != normalise_answer(gold)


def main() -> None:
    setup_logging()
    configure_hf_caches()
    args = parse_args()

    dataset = load_dataset(args.dataset_name, args.dataset_config, split=args.split)
    rng = random.Random(args.seed)
    tokenizer, model, device = load_model_and_tokenizer(args.model_name)

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

    final_answer_pattern = args.final_answer_regex

    with output_path.open("w", encoding="utf-8") as sink:
        seen_ids: set[str] = set()
        progress_total = args.max_samples if args.max_samples is not None else len(dataset)
        for example in tqdm(dataset.shuffle(seed=args.seed), desc="Collecting triples", total=progress_total):
            stats["total_samples"] += 1

            sample_identifier = str(example.get("id", stats["total_samples"]))
            if sample_identifier in seen_ids:
                continue
            seen_ids.add(sample_identifier)

            prompt_text = example.get(args.prompt_field)
            gold_answer = example.get(args.gold_answer_field)
            gold_chain_raw = example.get(args.gold_chain_field)
            gold_chain: Optional[str] = None
            if gold_chain_raw:
                if isinstance(gold_chain_raw, (list, tuple)):
                    gold_chain = "\n".join(str(item) for item in gold_chain_raw if item is not None)
                else:
                    gold_chain = str(gold_chain_raw)

            if prompt_text is None or not str(prompt_text).strip():
                stats["skipped_no_prompt"] += 1
                continue
            if gold_answer is None or not str(gold_answer).strip():
                stats["skipped_no_gold_answer"] += 1
                continue

            stats["evaluated_samples"] += 1

            formatted_prompt = args.prompt_template.format(prompt=str(prompt_text))
            if hasattr(tokenizer, "apply_chat_template"):
                messages = [
                    {"role": "system", "content": args.system_prompt},
                    {"role": "user", "content": formatted_prompt},
                ]
                formatted_prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )

            encoded = tokenizer(
                formatted_prompt,
                return_tensors="pt",
                return_attention_mask=True,
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)

            generation_kwargs = {
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "do_sample": args.do_sample,
                "pad_token_id": tokenizer.eos_token_id,
            }

            with torch.no_grad():
                generated_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, **generation_kwargs)

            response_ids = generated_ids[0]
            if args.strip_prompt_from_response:
                response_ids = response_ids[input_ids.shape[1] :]

            decoded_response = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
            predicted_answer = extract_final_answer(decoded_response, final_answer_pattern)

            if not should_include(predicted_answer, str(gold_answer), args.only_wrong):
                if args.max_samples and stats["collected_triples"] >= args.max_samples:
                    break
                continue

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

    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    LOGGER.info("Wrote %d triples to %s", stats["collected_triples"], output_path)
    LOGGER.info("Stats logged to %s", stats_path)


if __name__ == "__main__":
    main()

