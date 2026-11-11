"""Print an exploratory profile of the raw and cleaned SCC corpus."""

import json
import re
import unicodedata
from collections import Counter

from ..utils.paths import PROJECT_ROOT
from ..utils.text_preprocessing import preprocess_text


def profile_text(text: str) -> dict:
    """Summarize text size, whitespace, Unicode, repetition, and legal structure."""
    crlf_line_endings = text.count("\r\n")
    lone_cr_line_endings = text.count("\r") - crlf_line_endings
    lf_line_endings = text.count("\n") - crlf_line_endings
    line_normalized_text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = line_normalized_text.splitlines()
    line_frequencies = Counter(line for line in lines if line.strip())
    unicode_characters = {
        f"U+{ord(character):04X} {unicodedata.name(character, 'unnamed')}": count
        for character, count in sorted(Counter(text).items())
        if ord(character) > 127
    }
    return {
        "characters": len(text),
        "utf8_bytes": len(text.encode("utf-8")),
        "unique_characters": len(set(text)),
        "line_endings": {
            "lf": lf_line_endings,
            "crlf": crlf_line_endings,
            "cr": lone_cr_line_endings,
        },
        "lines": len(lines),
        "nonempty_lines": sum(bool(line.strip()) for line in lines),
        "blank_lines": sum(not line.strip() for line in lines),
        "lines_with_trailing_whitespace": sum(
            line.rstrip() != line for line in lines
        ),
        "duplicate_nonempty_line_occurrences": sum(
            count - 1 for count in line_frequencies.values() if count > 1
        ),
        "article_heading_lines": len(
            re.findall(
                r"(?m)^\s*Art\.\s*\d+[a-z]?(?:-\d+[a-z]?)?",
                line_normalized_text,
            )
        ),
        "book_heading_lines": len(
            re.findall(
                r"(?im)^\s*Book\s+(?:One|Two|Three|Four)\b",
                line_normalized_text,
            )
        ),
        "page_number_lines": len(
            re.findall(r"(?m)^\s*\d+\s*/\s*\d+\s*$", line_normalized_text)
        ),
        "non_ascii_characters": unicode_characters,
    }


def create_report(raw_text: str, processed_text: str) -> dict:
    """Compare raw and processed corpus profiles and report cleaning effects."""
    raw_lines = raw_text.splitlines()
    processed_lines = processed_text.splitlines()
    raw_profile = profile_text(raw_text)
    processed_profile = profile_text(processed_text)
    return {
        "raw": raw_profile,
        "processed": processed_profile,
        "preprocessing_effects": {
            "leading_bom_removed": int(raw_text.startswith("\ufeff")),
            "private_use_hyphen_glyphs": raw_text.count("\uf02d"),
            "tabs_normalized": raw_text.count("\t"),
            "unicode_space_characters_normalized": sum(
                count
                for character, count in Counter(raw_text).items()
                if unicodedata.category(character) == "Zs" and character != " "
            ),
            "trailing_whitespace_lines_removed": sum(
                line.rstrip() != line for line in raw_lines
            ),
            "source_lines_removed": len(raw_lines) - len(processed_lines),
            "blank_lines_removed_or_collapsed": (
                raw_profile["blank_lines"] - processed_profile["blank_lines"]
            ),
            "page_number_lines_removed": (
                raw_profile["page_number_lines"] - processed_profile["page_number_lines"]
            ),
            "article_heading_lines_preserved": (
                raw_profile["article_heading_lines"]
                == processed_profile["article_heading_lines"]
            ),
            "unicode_accents_preserved": all(
                character in processed_text
                for character in ("ä", "ö", "ü", "Ä", "Ö", "Ü")
                if character in raw_text
            ),
        },
    }


def main() -> None:
    """Read the raw source, preprocess it in memory, and print the EDA report."""
    raw_path = PROJECT_ROOT / "data/raw/scc-eng-raw.txt"
    with raw_path.open(encoding="utf-8", newline="") as source_file:
        raw_text = source_file.read()
    report = create_report(raw_text, preprocess_text(raw_text))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
