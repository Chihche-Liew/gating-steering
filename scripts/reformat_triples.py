#!/usr/bin/env python3
"""Reformat triples to use 'Final answer: X' format instead of 'The answer is (X)'."""

import json
import re
from pathlib import Path
import argparse
import shutil


def reformat_correct_chain(correct_chain, correct_answer):
    """
    Reformat the correct_chain to end with 'Final answer: X' instead of 'The answer is (X)'.

    Args:
        correct_chain: Can be a string or a list
        correct_answer: The correct answer letter/value

    Returns:
        Reformatted correct_chain (same type as input)
    """
    # Handle list format
    if isinstance(correct_chain, list):
        if not correct_chain:
            return correct_chain

        # Get the last element
        last_element = correct_chain[-1]

        # Pattern to match "The answer is (X)" or "The answer is X"
        pattern = r"The answer is \(?([A-J])\)?\.?"

        # Check if last element matches the pattern
        match = re.search(pattern, last_element, re.IGNORECASE)

        if match:
            # Replace with "Final answer: X" format
            new_last_element = f"Final answer: {correct_answer}"
            correct_chain[-1] = new_last_element
        else:
            # If pattern doesn't match, append "Final answer: X"
            correct_chain.append(f"Final answer: {correct_answer}")

        return correct_chain

    # Handle string format
    elif isinstance(correct_chain, str):
        # Pattern to match "The answer is (X)" or "The answer is X"
        pattern = r"The answer is \(?([A-J])\)?\.?"

        # Replace with "Final answer: X"
        new_chain = re.sub(
            pattern,
            f"Final answer: {correct_answer}",
            correct_chain,
            flags=re.IGNORECASE
        )

        # If no match was found, append "Final answer: X"
        if new_chain == correct_chain:
            new_chain = f"{correct_chain}\nFinal answer: {correct_answer}"

        return new_chain

    return correct_chain


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-path",
        type=Path,
        default=Path("artifacts/triples/triples_small.jsonl"),
        help="Path to input JSONL file",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Path to output JSONL file (defaults to input path with .reformatted.jsonl)",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Create a backup of the original file",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Modify the file in-place (overwrites original)",
    )
    args = parser.parse_args()

    input_path = args.input_path

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return

    # Determine output path
    if args.in_place:
        output_path = input_path
        if args.backup:
            backup_path = input_path.with_suffix(input_path.suffix + ".backup")
            shutil.copy2(input_path, backup_path)
            print(f"Created backup: {backup_path}")
    else:
        output_path = args.output_path or input_path.with_suffix(".reformatted.jsonl")

    # Read, reformat, and write
    reformatted_count = 0
    total_count = 0

    # If in-place, write to temp file first
    temp_path = output_path.with_suffix(".tmp")

    with input_path.open("r", encoding="utf-8") as infile, \
         temp_path.open("w", encoding="utf-8") as outfile:

        for line in infile:
            total_count += 1

            # Parse JSON
            triple = json.loads(line)

            # Get correct_chain and correct_answer
            correct_chain = triple.get("correct_chain")
            correct_answer = triple.get("correct_answer")

            if correct_chain and correct_answer:
                # Reformat
                original_chain = json.dumps(correct_chain)
                triple["correct_chain"] = reformat_correct_chain(correct_chain, correct_answer)
                new_chain = json.dumps(triple["correct_chain"])

                if original_chain != new_chain:
                    reformatted_count += 1

            # Write reformatted triple
            outfile.write(json.dumps(triple, ensure_ascii=False) + "\n")

    # Replace original with temp if in-place
    if args.in_place or output_path == input_path:
        temp_path.replace(output_path)
    else:
        temp_path.rename(output_path)

    print(f"Processed {total_count} triples")
    print(f"Reformatted {reformatted_count} correct_chains")
    print(f"Output written to: {output_path}")


if __name__ == "__main__":
    main()
