from pathlib import Path

import pytest

from ..baseline import run_baseline
from ..utils.baseline_support import (
    PROJECT_ROOT,
    load_corpus,
    loss_metrics,
    select_device,
)
from ..utils.tokenizer import CharacterTokenizer, char_token


def test_character_tokenizer_handles_unseen_characters() -> None:
    tokenizer = CharacterTokenizer("aab")

    assert tokenizer.vocab_size == 3
    assert tokenizer.decode(tokenizer.encode("aba")) == "aba"
    assert tokenizer.decode(tokenizer.encode("abc")) == "ab\ufffd"
    with pytest.raises(ValueError):
        tokenizer.decode([-1])


def test_legacy_char_token_helper_keeps_round_trip_api() -> None:
    tokens, encode, decode = char_token("aba")

    assert tokens[-1] == "<unk>"
    assert decode(encode("aba")) == "aba"


def test_corpus_splits_reconstruct_staged_text() -> None:
    fit_text, validation_text, test_text = load_corpus()
    staged_text = Path(
        PROJECT_ROOT / "data/staged/scc-eng.txt"
    ).read_text(encoding="utf-8")

    assert fit_text + validation_text + test_text == staged_text


def test_loss_metrics_are_finite_and_byte_normalized() -> None:
    metrics = loss_metrics(1.0, 2, "abc")

    assert metrics["target_tokens"] == 2
    assert metrics["loss_nats_per_token"] == 1.0
    assert metrics["perplexity_per_token"] > 1.0
    assert metrics["bits_per_byte"] > 0

    with pytest.raises(ValueError):
        loss_metrics(1.0, 0, "abc")


def test_auto_device_is_available() -> None:
    assert select_device("auto").type in {"cpu", "cuda", "mps"}


def test_short_fit_reports_validation_selected_checkpoint() -> None:
    report = run_baseline(
        steps=2,
        batch_size=8,
        learning_rate=0.005,
        seed=1337,
        requested_device="cpu",
    )

    assert 0 <= report["configuration"]["best_validation_step"] <= 2
    assert report["test"]["bits_per_byte"] > 0
