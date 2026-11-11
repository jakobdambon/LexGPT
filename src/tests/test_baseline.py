from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from ..baseline import CharacterContextLanguageModel, run_baseline
from ..compare_fitting import run_comparison
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
    batch_schedule = torch.zeros((2, 8), dtype=torch.long)
    report = run_baseline(
        steps=2,
        batch_size=8,
        learning_rate=0.005,
        seed=1337,
        requested_device="cpu",
        batch_index_schedule=batch_schedule,
    )

    assert 0 <= report["configuration"]["best_validation_step"] <= 2
    assert report["configuration"]["shared_batch_schedule"]
    assert report["test"]["bits_per_byte"] > 0
    assert report["training"]["sampled_examples"] == 16
    assert report["training"]["approximate_epochs"] == pytest.approx(
        16 / (report["configuration"]["fit_characters"] - 1)
    )


def test_short_fit_rejects_mismatched_batch_schedule() -> None:
    with pytest.raises(ValueError, match="batch_index_schedule must have shape"):
        run_baseline(
            steps=2,
            batch_size=8,
            learning_rate=0.005,
            seed=1337,
            requested_device="cpu",
            batch_index_schedule=torch.zeros((1, 8), dtype=torch.long),
        )


def test_context_four_model_uses_window_and_generates() -> None:
    model = CharacterContextLanguageModel(vocab_size=10, context_size=4)
    contexts = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])

    logits = model(contexts)
    generated = model.generate(torch.tensor([1, 2, 3, 4]), new_tokens=2)

    assert logits.shape == (2, 10)
    assert generated.shape == (6,)


def test_context_four_self_attention_forward_and_generation() -> None:
    model = CharacterContextLanguageModel(
        vocab_size=10,
        context_size=4,
        use_self_attention=True,
    )
    contexts = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])

    logits = model(contexts)
    generated = model.generate(torch.tensor([1, 2, 3, 4]), new_tokens=2)

    assert logits.shape == (2, 10)
    assert generated.shape == (6,)
    assert model.causal_mask[0, 1]
    assert not model.causal_mask[1, 0]


def test_self_attention_requires_multi_character_context() -> None:
    with pytest.raises(ValueError, match="context_size of at least 2"):
        CharacterContextLanguageModel(
            vocab_size=10,
            context_size=1,
            use_self_attention=True,
        )


def test_fit_rejects_nonpositive_validation_interval() -> None:
    with pytest.raises(ValueError, match="validation_interval must be positive"):
        run_baseline(
            steps=2,
            batch_size=8,
            learning_rate=0.005,
            seed=1337,
            requested_device="cpu",
            validation_interval=0,
        )


def test_context_four_fit_reports_correct_targets() -> None:
    schedule = torch.full((2, 8), 3, dtype=torch.long)
    report = run_baseline(
        steps=2,
        batch_size=8,
        learning_rate=0.005,
        seed=1337,
        requested_device="cpu",
        batch_index_schedule=schedule,
        context_size=4,
    )

    assert report["configuration"]["context_size"] == 4
    assert report["configuration"]["architecture"] == "character-embedding-mlp"
    assert report["validation"]["target_tokens"] == (
        report["configuration"]["validation_characters"] - 4
    )
    assert report["test"]["target_tokens"] == (
        report["configuration"]["test_characters"] - 4
    )


def test_context_four_self_attention_fit_reports_architecture() -> None:
    schedule = torch.full((2, 8), 3, dtype=torch.long)
    report = run_baseline(
        steps=2,
        batch_size=8,
        learning_rate=0.005,
        seed=1337,
        requested_device="cpu",
        batch_index_schedule=schedule,
        context_size=4,
        use_self_attention=True,
    )

    assert report["configuration"]["self_attention"]
    assert report["configuration"]["architecture"] == (
        "character-causal-self-attention"
    )
    assert report["configuration"]["attention_heads"] == 4


def test_comparison_supports_variant_specific_steps_and_learning_rates() -> None:
    configurations = [{"name": "cpu-single-thread", "device": "cpu", "threads": 1}]
    with patch(
        "src.compare_fitting.build_run_configurations",
        return_value=configurations,
    ):
        comparison = run_comparison(
            steps=2,
            batch_size=8,
            learning_rate=0.001,
            seed=1337,
            cpu_parallel_threads=1,
            context_sizes=(4,),
            attention_learning_rate=0.0005,
            attention_steps=3,
        )

    plain_run, attention_run = comparison["runs"]
    assert plain_run["report"]["configuration"]["steps"] == 2
    assert plain_run["report"]["configuration"]["learning_rate"] == 0.001
    assert attention_run["report"]["configuration"]["steps"] == 3
    assert attention_run["report"]["configuration"]["learning_rate"] == 0.0005
