"""Unicode-safe cleanup and deterministic splitting for the SCC text corpus."""

import re
import unicodedata


_PRIVATE_HYPHEN = "\uf02d"
_PAGE_NUMBER = re.compile(r"\d+\s*/\s*\d+")
_MAX_BLANK_LINES = 1


def preprocess_text(raw_text: str) -> str:
    """Clean extracted source text while preserving its wording and Unicode.

    The transformation removes a leading UTF-8 BOM, normalizes line endings
    and canonically equivalent Unicode, maps the source's private-use hyphen
    glyph to an ordinary hyphen, and maps Unicode space separators to ASCII
    spaces. It trims line-end whitespace, limits long blank-line runs, and
    removes the verified title/page-number footer at the very end.

    Other control, format, private-use, surrogate, or unassigned characters
    raise ``ValueError`` so new source anomalies are reviewed rather than
    silently discarded.
    """
    text = raw_text
    if text.startswith("\ufeff"):
        text = text[1:]

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = unicodedata.normalize("NFC", text)
    text = text.replace(_PRIVATE_HYPHEN, "-")
    text = text.replace("\t", " ")

    normalized_characters = []
    for character in text:
        category = unicodedata.category(character)
        if category == "Zs":
            normalized_characters.append(" ")
        elif character == "\n":
            normalized_characters.append(character)
        elif category.startswith("C"):
            raise ValueError(
                f"Unsupported Unicode character U+{ord(character):04X} "
                f"({unicodedata.name(character, 'unnamed')})."
            )
        else:
            normalized_characters.append(character)

    lines = [line.rstrip(" \t") for line in "".join(normalized_characters).split("\n")]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()

    footer_start = len(lines)
    saw_title = False
    saw_page_number = False
    while footer_start > 0:
        footer_line = lines[footer_start - 1].strip()
        if footer_line == "Swiss Criminal Code":
            saw_title = True
        elif _PAGE_NUMBER.fullmatch(footer_line):
            saw_page_number = True
        else:
            break
        footer_start -= 1

    if saw_title and saw_page_number:
        lines = lines[:footer_start]

    while lines and not lines[-1].strip():
        lines.pop()

    cleaned_lines = []
    consecutive_blank_lines = 0
    for line in lines:
        if line:
            consecutive_blank_lines = 0
            cleaned_lines.append(line)
        else:
            consecutive_blank_lines += 1
            if consecutive_blank_lines <= _MAX_BLANK_LINES:
                cleaned_lines.append(line)

    if not cleaned_lines:
        return ""
    return "\n".join(cleaned_lines) + "\n"


def split_at_line_boundary(
    text: str,
    train_fraction: float = 0.9,
) -> tuple[str, str]:
    """Split text near a requested fraction without cutting through a line.

    Among the starts of non-empty lines, choose the one closest to the target
    character offset. This keeps the split deterministic and prevents a test
    split from beginning with the tail end of a word or paragraph line.
    """
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1.")

    target_offset = int(len(text) * train_fraction)
    line_starts = []
    offset = 0
    for line in text.splitlines(keepends=True):
        if line.strip():
            line_starts.append(offset)
        offset += len(line)

    valid_starts = [start for start in line_starts if 0 < start < len(text)]
    if not valid_starts:
        raise ValueError("Text must contain at least two non-empty lines to split.")

    split_offset = min(valid_starts, key=lambda start: abs(start - target_offset))
    return text[:split_offset], text[split_offset:]
