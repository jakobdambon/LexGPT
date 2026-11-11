"""Prepare the Swiss Criminal Code corpus and write reproducible UTF-8 splits."""

import argparse
import json
from pathlib import Path

from .utils.paths import PROJECT_ROOT
from .utils.text_preprocessing import preprocess_text, split_at_line_boundary


def prepare_corpus(
    input_path: Path,
    output_directory: Path,
    train_fraction: float,
) -> dict[str, int | float | str]:
    """Clean the raw source and write full, training, and test text files."""
    with input_path.open(encoding="utf-8", newline="") as source_file:
        raw_text = source_file.read()
    clean_text = preprocess_text(raw_text)
    if not clean_text:
        raise ValueError(f"Preprocessing produced empty text from {input_path}.")

    train_text, test_text = split_at_line_boundary(clean_text, train_fraction)
    output_directory.mkdir(parents=True, exist_ok=True)
    full_path = output_directory / "scc-eng.txt"
    train_path = output_directory / "scc-eng-train.txt"
    test_path = output_directory / "scc-eng-test.txt"

    full_path.write_text(clean_text, encoding="utf-8")
    train_path.write_text(train_text, encoding="utf-8")
    test_path.write_text(test_text, encoding="utf-8")

    return {
        "source_characters": len(raw_text),
        "processed_characters": len(clean_text),
        "train_characters": len(train_text),
        "test_characters": len(test_text),
        "train_fraction_requested": train_fraction,
        "train_fraction_actual": len(train_text) / len(clean_text),
        "processed_path": str(full_path),
        "train_path": str(train_path),
        "test_path": str(test_path),
    }


def main() -> None:
    """Parse paths and split fraction, then regenerate the prepared corpus."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data/raw/scc-eng-raw.txt",
        help="Raw UTF-8 source text (default: data/raw/scc-eng-raw.txt).",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=PROJECT_ROOT / "data/staged",
        help="Directory for the processed corpus and splits.",
    )
    parser.add_argument("--train-fraction", type=float, default=0.9)
    args = parser.parse_args()

    result = prepare_corpus(args.input, args.output_directory, args.train_fraction)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
