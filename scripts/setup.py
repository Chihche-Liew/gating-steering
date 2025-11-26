#!/usr/bin/env python3
"""Shared setup utilities for data processing, caching, and answer extraction."""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import List, Optional


LOGGER = logging.getLogger("setup")


def configure_hf_caches() -> None:
    """Configure HuggingFace cache directories relative to project root."""
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

    LOGGER.debug("HF_HOME: %s", os.environ["HF_HOME"])
    LOGGER.debug("TRANSFORMERS_CACHE: %s", os.environ["TRANSFORMERS_CACHE"])


def setup_logging(level: int = logging.INFO) -> None:
    """Configure basic logging format."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )


# Answer extraction patterns (ordered by priority)
ANSWER_PATTERNS = [
    r"final\s+answer\s*[:\-]?\s*([A-J]|\w+)",
    r"answer\s*[:\-]?\s*([A-J]|\w+)",
    r"conclusion\s*[:\-]?\s*([A-J]|\w+)",
    r"choice\s*[:\-]?\s*([A-J]|\w+)",
]


def extract_final_answer(text: str, pattern: Optional[str] = None) -> str:
    """
    Extract the final answer from model-generated text.

    Args:
        text: Generated text to extract answer from
        pattern: Optional custom regex pattern (case-insensitive)

    Returns:
        Extracted answer string, or empty string if not found
    """
    candidates: List[str] = []
    
    # Try custom pattern first
    if pattern:
        compiled = re.compile(pattern, flags=re.IGNORECASE)
        match = compiled.search(text)
        if match and match.group(1):
            candidates.append(match.group(1).strip().strip("."))
    
    # Try default patterns
    for pat in ANSWER_PATTERNS:
        matcher = re.compile(pat, flags=re.IGNORECASE)
        match = matcher.search(text)
        if match and match.group(1):
            candidates.append(match.group(1).strip().strip("."))
    
    if candidates:
        return candidates[0]

    # Fallback: try last line for single letter choice
    fallback = text.strip().splitlines()
    if fallback:
        last_line = fallback[-1].strip()
        single_choice = re.fullmatch(r"([A-J])\.?", last_line, flags=re.IGNORECASE)
        if single_choice:
            return single_choice.group(1)
        return last_line
    
    return ""


def normalize_answer(answer: str) -> str:
    return re.sub(r"\s+", " ", answer.strip().lower())


def answers_match(answer1: str, answer2: str) -> bool:
    return normalize_answer(answer1) == normalize_answer(answer2)


def reconstruct_prompt(metadata: dict, prompt: str) -> str:
    """
    Reconstruct full prompt using template from metadata.

    Args:
        metadata: Metadata dict containing optional 'prompt_template'
        prompt: Raw prompt text

    Returns:
        Formatted prompt
    """
    template = metadata.get("prompt_template", "{prompt}")
    try:
        return template.format(prompt=prompt)
    except KeyError:
        LOGGER.warning("Prompt template missing {prompt} placeholder, using raw prompt")
        return prompt


def strip_duplicate_prefix(text: str, prefix: str) -> str:
    """
    Remove duplicate prefix from text if present.

    Args:
        text: Text to strip
        prefix: Prefix to remove

    Returns:
        Text with prefix removed (if it was present)
    """
    if text.startswith(prefix):
        return text[len(prefix):].lstrip()
    return text


def build_teacher_forcing_text(prompt_text: str, chain_text: str, metadata: dict) -> str:
    """
    Build text for teacher forcing (prompt + chain).

    Args:
        prompt_text: Raw prompt
        chain_text: Chain of thought
        metadata: Metadata with optional prompt_template

    Returns:
        Combined text for teacher forcing
    """
    formatted_prompt = reconstruct_prompt(metadata, prompt_text)
    cleaned_chain = strip_duplicate_prefix(chain_text, prompt_text)
    cleaned_chain = strip_duplicate_prefix(cleaned_chain, formatted_prompt)
    
    if cleaned_chain:
        return f"{formatted_prompt}\n\n{cleaned_chain}".strip()
    return formatted_prompt

def should_include(predicted: str, gold: str, only_wrong: bool) -> bool:
    """
    Determine if a triple should be included in the output.

    Args:
        predicted: Model's predicted answer
        gold: Gold standard answer
        only_wrong: If True, only include mismatches

    Returns:
        True if the triple should be included
    """
    if not only_wrong:
        return True
    return normalize_answer(predicted) != normalize_answer(gold)