from pathlib import Path

import pytest

from ..analysis.corpus_profile import create_report, profile_text
from ..preprocess import prepare_corpus
from ..utils.baseline_support import PROJECT_ROOT
from ..utils.text_preprocessing import preprocess_text, split_at_line_boundary


def test_preprocessing_preserves_accents_and_maps_source_punctuation() -> None:
    raw_text = "\ufeffA\u0308rt. 1\u2002\uf02d 2\r\n"

    assert preprocess_text(raw_text) == "Ärt. 1 - 2\n"


def test_preprocessing_removes_only_verified_terminal_footer_and_extra_blanks() -> None:
    raw_text = "Article text.   \n\n\n\nSwiss Criminal Code\n1 / 1\n1 / 1\n"

    assert preprocess_text(raw_text) == "Article text.\n"


def test_preprocessing_rejects_unreviewed_format_characters() -> None:
    with pytest.raises(ValueError, match="Unsupported Unicode character"):
        preprocess_text("text\u200b")


def test_split_uses_a_nonempty_line_start_and_reconstructs_text() -> None:
    text = "one\nsecond line\nthird\nfourth\n"

    train_text, test_text = split_at_line_boundary(text, 0.5)

    assert train_text + test_text == text
    assert test_text.startswith("third\n")


def test_split_rejects_invalid_fraction() -> None:
    with pytest.raises(ValueError, match="train_fraction"):
        split_at_line_boundary("one\ntwo\n", 1.0)


def test_corpus_profile_counts_source_anomalies() -> None:
    profile = profile_text("Ä\nArt. 1  \nSwiss Criminal Code\n1 / 1\n")

    assert profile["unique_characters"] > 0
    assert profile["lines_with_trailing_whitespace"] == 1
    assert profile["article_heading_lines"] == 1
    assert profile["page_number_lines"] == 1
    assert profile["non_ascii_characters"] == {"U+00C4 LATIN CAPITAL LETTER A WITH DIAERESIS": 1}


def test_report_confirms_meaningful_content_is_preserved() -> None:
    report = create_report("Art. 1\nä\n", "Art. 1\nä\n")

    assert report["preprocessing_effects"]["article_heading_lines_preserved"]
    assert report["preprocessing_effects"]["unicode_accents_preserved"]


def test_preparation_writes_utf8_full_text_and_splits(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    source.write_text("\ufeffFirst line\nSecond line\nThird line\nFourth line\n", encoding="utf-8")
    output_directory = tmp_path / "processed"

    result = prepare_corpus(source, output_directory, 0.5)

    full_text = (output_directory / "scc-eng.txt").read_text(encoding="utf-8")
    train_text = (output_directory / "scc-eng-train.txt").read_text(encoding="utf-8")
    test_text = (output_directory / "scc-eng-test.txt").read_text(encoding="utf-8")
    assert train_text + test_text == full_text
    assert result["source_characters"] > result["processed_characters"]


def test_actual_raw_corpus_has_no_unsupported_characters() -> None:
    raw_text = (PROJECT_ROOT / "data/raw/scc-eng-raw.txt").read_text(encoding="utf-8")

    processed = preprocess_text(raw_text)

    assert "\uf02d" in raw_text
    assert "\uf02d" not in processed
    assert "Swiss Criminal Code\n1 / 1" not in processed[-100:]
