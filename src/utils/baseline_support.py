"""Reusable data, device, evaluation, and reporting helpers for experiments."""

import math
import os
import platform
import subprocess
import sys

import torch
from torch import nn
from torch.nn import functional as F

from .paths import PROJECT_ROOT
from .tokenizer import CharacterTokenizer


def select_device(requested: str) -> torch.device:
    """Choose a supported PyTorch device, or validate an explicit request.

    Automatic selection prefers CUDA, then Apple MPS, and finally CPU. An
    explicitly requested accelerator raises an error if it is unavailable,
    rather than silently running on a different device.
    """
    if requested == "auto":
        if torch.cuda.is_available():
            requested = "cuda"
        elif torch.backends.mps.is_available():
            requested = "mps"
        else:
            requested = "cpu"

    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available.")
    if device.type not in {"cpu", "cuda", "mps"}:
        raise ValueError(f"Unsupported device: {requested}")
    return device


def load_corpus() -> tuple[str, str, str]:
    """Load the staged corpus and split it into fit, validation, and test text.

    The checked-in test text must be the exact suffix of the staged corpus.
    The remaining prefix is divided in order: the first 90% is used to fit the
    model and tokenizer, and the final 10% is used for validation. Keeping the
    test suffix separate prevents it from influencing training or model choice.

    Returns:
        A tuple of ``(fit_text, validation_text, test_text)``.
    """
    staged_text = (PROJECT_ROOT / "data/staged/scc-eng.txt").read_text(
        encoding="utf-8"
    )
    test_text = (PROJECT_ROOT / "data/staged/scc-eng-test.txt").read_text(
        encoding="utf-8"
    )
    if not test_text or not staged_text.endswith(test_text):
        raise ValueError("The held-out test file must match the staged corpus suffix.")

    # Remove the held-out suffix before creating the fit/validation split.
    training_text = staged_text[: -len(test_text)]
    validation_start = int(len(training_text) * 0.9)
    fit_text = training_text[:validation_start]
    validation_text = training_text[validation_start:]
    if not fit_text or not validation_text:
        raise ValueError("The corpus must contain non-empty training and validation text.")
    return fit_text, validation_text, test_text


def encode_tensor(
    text: str,
    tokenizer: CharacterTokenizer,
    device: torch.device,
) -> torch.Tensor:
    """Tokenize text and place the integer IDs in a tensor on the selected device.

    ``torch.long`` is required because embedding layers use token IDs as integer
    indices. Moving the tensor once up front avoids copying the full corpus
    during each training step.
    """
    return torch.tensor(tokenizer.encode(text), dtype=torch.long, device=device)


@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    token_ids: torch.Tensor,
    batch_size: int = 8192,
    context_size: int = 1,
) -> float:
    """Calculate average next-token cross-entropy over an entire text split.

    Evaluation is chunked to bound temporary memory use. Each token is the
    target of one context window, and the first ``context_size`` characters
    cannot be scored because this split contains no preceding context for them.

    Args:
        model: A model that returns next-token logits for input token IDs.
        token_ids: One-dimensional IDs for the validation or test text.
        batch_size: Maximum number of input tokens evaluated in one chunk.

    Returns:
        Mean cross-entropy in nats per target token.
    """
    total_loss = 0.0
    if context_size < 1:
        raise ValueError("context_size must be positive.")
    target_count = len(token_ids) - context_size
    if target_count <= 0:
        raise ValueError("Evaluation text must be longer than context_size.")

    model.eval()
    context_windows = token_ids.unfold(0, context_size, 1)
    for start in range(0, target_count, batch_size):
        end = min(start + batch_size, target_count)
        # Context windows and targets are aligned without exposing target text.
        logits = model(context_windows[start:end])
        targets = token_ids[context_size + start : context_size + end]
        # Sum chunk losses so the final average weights every target equally.
        total_loss += F.cross_entropy(logits, targets, reduction="sum").item()
    model.train()
    return total_loss / target_count


def loss_metrics(
    loss_nats_per_token: float,
    target_token_count: int,
    text: str,
) -> dict[str, float]:
    """Convert average token loss into metrics useful for model comparisons.

    Perplexity is interpretable for runs with the same tokenizer. Bits per byte
    normalizes the total negative log-likelihood by the UTF-8 byte length, so
    later runs with different tokenizers can be compared on the same text.
    """
    if target_token_count < 1 or not text:
        raise ValueError("Metrics require at least one target token and non-empty text.")
    byte_count = len(text.encode("utf-8"))
    return {
        "target_tokens": target_token_count,
        "loss_nats_per_token": loss_nats_per_token,
        "perplexity_per_token": math.exp(loss_nats_per_token),
        "bits_per_byte": loss_nats_per_token
        * target_token_count
        / (byte_count * math.log(2)),
    }


def system_metadata() -> dict:
    """Collect hardware and software details to make a run reproducible."""
    available_accelerators = ["cpu"]
    if torch.cuda.is_available():
        available_accelerators.append("cuda")
    if torch.backends.mps.is_available():
        available_accelerators.append("mps")

    total_memory_bytes = None
    if hasattr(os, "sysconf"):
        total_memory_bytes = (
            os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        )

    # macOS often reports only "arm" for Apple Silicon; sysctl provides the
    # specific chip name useful when comparing local performance results.
    processor = platform.processor() or platform.machine()
    if sys.platform == "darwin":
        processor = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": processor,
        "ram_bytes": total_memory_bytes,
        "available_accelerators": available_accelerators,
        "pytorch_threads": torch.get_num_threads(),
        "python": platform.python_version(),
        "pytorch": torch.__version__,
    }
