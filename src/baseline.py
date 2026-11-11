"""Reproducible character-context model for the Swiss Criminal Code."""

import argparse
import copy
import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .utils.baseline_support import (
    PROJECT_ROOT,
    encode_tensor,
    evaluate_loss,
    load_corpus,
    loss_metrics,
    select_device,
    system_metadata,
)
from .utils.tokenizer import CharacterTokenizer


class CharacterContextLanguageModel(nn.Module):
    """Predict the next character from a fixed window of preceding characters.

    A context size of one retains the original bigram lookup table. Larger
    windows embed each context character, concatenate those embeddings, and
    use a small feed-forward network to produce next-character logits. This
    avoids a direct context-to-output table with
    ``vocab_size ** (context_size + 1)`` scores.
    """

    def __init__(
        self,
        vocab_size: int,
        context_size: int = 1,
        embedding_size: int = 32,
        hidden_size: int = 128,
        use_self_attention: bool = False,
        attention_heads: int = 4,
    ) -> None:
        """Create the model layers for the requested context window.

        Args:
            vocab_size: Number of possible token IDs, including the unknown ID.
            context_size: Number of preceding characters available for prediction.
            embedding_size: Character embedding width for context sizes above one.
            hidden_size: Hidden-layer width for context sizes above one.
            use_self_attention: Add a causal self-attention block for contexts
                larger than one.
            attention_heads: Number of attention heads when attention is enabled.
        """
        super().__init__()
        if vocab_size < 1 or context_size < 1:
            raise ValueError("vocab_size and context_size must be positive.")
        if embedding_size < 1 or hidden_size < 1:
            raise ValueError("embedding_size and hidden_size must be positive.")
        if use_self_attention and context_size < 2:
            raise ValueError("Self-attention requires a context_size of at least 2.")
        if use_self_attention and (
            attention_heads < 1 or embedding_size % attention_heads != 0
        ):
            raise ValueError(
                "attention_heads must be positive and divide embedding_size."
            )
        self.context_size = context_size
        self.embedding_size = embedding_size if context_size > 1 else 0
        self.hidden_size = hidden_size if context_size > 1 else 0
        self.use_self_attention = use_self_attention
        self.attention_heads = attention_heads if use_self_attention else 0

        if context_size == 1:
            # Shape: (input character, candidate next character).
            self.next_token_logits = nn.Embedding(vocab_size, vocab_size)
        else:
            self.context_embedding = nn.Embedding(vocab_size, embedding_size)
            if use_self_attention:
                self.position_embedding = nn.Embedding(context_size, embedding_size)
                self.self_attention = nn.MultiheadAttention(
                    embed_dim=embedding_size,
                    num_heads=attention_heads,
                    batch_first=True,
                )
                self.attention_norm = nn.LayerNorm(embedding_size)
                # True entries prevent each position from reading future characters.
                causal_mask = torch.triu(
                    torch.ones(context_size, context_size, dtype=torch.bool),
                    diagonal=1,
                )
                self.register_buffer("causal_mask", causal_mask, persistent=False)
                self.context_network = nn.Sequential(
                    nn.Linear(context_size * embedding_size, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, vocab_size),
                )
            else:
                self.context_network = nn.Sequential(
                    nn.Linear(context_size * embedding_size, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, vocab_size),
                )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Calculate next-character logits from one or more context windows.

        Args:
            token_ids: Integer token IDs whose final dimension is context size.

        Returns:
            Unnormalized next-character scores. All dimensions except the
            context dimension are preserved.
        """
        if token_ids.shape[-1] != self.context_size:
            raise ValueError(
                f"Expected context windows of {self.context_size} characters."
            )
        if self.context_size == 1:
            return self.next_token_logits(token_ids).squeeze(-2)

        embeddings = self.context_embedding(token_ids)
        if self.use_self_attention:
            positions = torch.arange(self.context_size, device=token_ids.device)
            embeddings = embeddings + self.position_embedding(positions)
            attended, _ = self.self_attention(
                embeddings,
                embeddings,
                embeddings,
                attn_mask=self.causal_mask,
                need_weights=False,
            )
            contextualized = self.attention_norm(embeddings + attended)
            flattened = contextualized.flatten(start_dim=-2)
            return self.context_network(flattened)

        flattened = embeddings.flatten(start_dim=-2)
        return self.context_network(flattened)

    @torch.no_grad()
    def generate(self, prompt_ids: torch.Tensor, new_tokens: int) -> torch.Tensor:
        """Extend a prompt by repeatedly sampling one next token.

        At each step, the model uses at most its configured number of most
        recent characters. Sampling from softmax probabilities turns its logits
        into a next-character distribution.

        Args:
            prompt_ids: One-dimensional tensor with at least one token.
            new_tokens: Number of sampled IDs to append to the prompt.

        Returns:
            The prompt followed by the sampled IDs.
        """
        if prompt_ids.ndim != 1 or prompt_ids.numel() < self.context_size:
            raise ValueError(
                "The prompt must be one-dimensional and at least context_size long."
            )
        token_ids = prompt_ids
        for _ in range(new_tokens):
            # The model sees only the latest context window, not the full history.
            context = token_ids[-self.context_size :]
            probabilities = F.softmax(self(context), dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            # Append the sample so it becomes the context for the next step.
            token_ids = torch.cat((token_ids, next_token))
        return token_ids


def run_baseline(
    steps: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    requested_device: str,
    batch_index_schedule: torch.Tensor | None = None,
    context_size: int = 1,
    use_self_attention: bool = False,
    validation_interval: int | None = None,
) -> dict:
    """Train the character-context model and return its metrics and samples.

    The tokenizer is built only from the fitting text. Training samples
    character windows and their following targets; an optional shared batch
    schedule lets hardware comparisons use the exact same target positions.
    Validation loss is checked during training, while held-out test loss is
    calculated once at the end.
    Generation gets a separate deterministic random seed so adding or changing
    generation does not alter the training run.
    """
    if steps < 1 or batch_size < 1 or learning_rate <= 0 or context_size < 1:
        raise ValueError(
            "steps, batch_size, learning_rate, and context_size must be positive."
        )
    if validation_interval is not None and validation_interval < 1:
        raise ValueError("validation_interval must be positive when provided.")
    if use_self_attention and context_size < 2:
        raise ValueError("Self-attention requires a context_size of at least 2.")
    if batch_index_schedule is not None:
        expected_shape = (steps, batch_size)
        if tuple(batch_index_schedule.shape) != expected_shape:
            raise ValueError(
                f"batch_index_schedule must have shape {expected_shape}."
            )
        if batch_index_schedule.dtype != torch.long:
            raise ValueError("batch_index_schedule must use torch.long indices.")

    # Initialize randomness before model construction and sampling.
    device = select_device(requested_device)
    torch.manual_seed(seed)
    fit_text, validation_text, test_text = load_corpus()
    tokenizer = CharacterTokenizer(fit_text)
    fit_ids = encode_tensor(fit_text, tokenizer, device)
    validation_ids = encode_tensor(validation_text, tokenizer, device)
    test_ids = encode_tensor(test_text, tokenizer, device)
    if batch_index_schedule is not None:
        if torch.any(batch_index_schedule < context_size - 1) or torch.any(
            batch_index_schedule >= len(fit_ids) - 1
        ):
            raise ValueError(
                "batch_index_schedule contains a position without a complete context."
            )
        # Transfer once before timing so every backend reads the same batches
        # without measuring repeated host-to-device schedule transfers.
        batch_index_schedule = batch_index_schedule.to(device)

    model = CharacterContextLanguageModel(
        tokenizer.vocab_size,
        context_size=context_size,
        use_self_attention=use_self_attention,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    validation_loss = evaluate_loss(model, validation_ids, context_size=context_size)
    best_validation_loss = validation_loss
    best_validation_step = 0
    best_model_state = copy.deepcopy(model.state_dict())
    context_offsets = torch.arange(context_size, device=device)
    if validation_interval is None:
        validation_interval = max(steps // 20, 1)

    # Time fitting and scheduled validation, excluding setup and final testing.
    start_time = time.perf_counter()

    for step in range(steps):
        # Choose a target position and collect its immediately preceding context.
        if batch_index_schedule is None:
            indices = torch.randint(
                context_size - 1,
                len(fit_ids) - 1,
                (batch_size,),
                device=device,
            )
        else:
            indices = batch_index_schedule[step]
        inputs = fit_ids[indices[:, None] - context_size + 1 + context_offsets]
        targets = fit_ids[indices + 1]
        logits = model(inputs)
        loss = F.cross_entropy(logits, targets)

        # Compute prediction gradients and update the context-to-character model.
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # Periodic validation shows learning progress without updating weights.
        if (step + 1) % validation_interval == 0 or step + 1 == steps:
            validation_loss = evaluate_loss(
                model,
                validation_ids,
                context_size=context_size,
            )
            print(f"step {step + 1}/{steps}: validation loss {validation_loss:.4f} nats/token")
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                best_validation_step = step + 1
                best_model_state = copy.deepcopy(model.state_dict())

    elapsed_seconds = time.perf_counter() - start_time
    # Use the checkpoint selected by validation rather than assuming the last
    # optimization step is best. The test data does not influence this choice.
    model.load_state_dict(best_model_state)
    test_loss = evaluate_loss(model, test_ids, context_size=context_size)

    # Use a separate seed so sample text is repeatable without affecting metrics.
    torch.manual_seed(seed + 1)
    prompt = encode_tensor("Swiss Criminal Code\n", tokenizer, device)
    generated_ids = model.generate(prompt, new_tokens=300).tolist()

    # Store metrics and run conditions together, so results remain interpretable.
    return {
        "experiment": (
            "character-bigram"
            if context_size == 1
            else (
                "character-causal-self-attention"
                if use_self_attention
                else "character-context-mlp"
            )
        ),
        "seed": seed,
        "device": str(device),
        "system": system_metadata(),
        "configuration": {
            "steps": steps,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "validation_interval": validation_interval,
            "context_size": context_size,
            "architecture": (
                "bigram-lookup"
                if context_size == 1
                else (
                    "character-causal-self-attention"
                    if use_self_attention
                    else "character-embedding-mlp"
                )
            ),
            "embedding_size": model.embedding_size,
            "hidden_size": model.hidden_size,
            "self_attention": use_self_attention,
            "attention_heads": model.attention_heads,
            "shared_batch_schedule": batch_index_schedule is not None,
            "best_validation_step": best_validation_step,
            "vocab_size": tokenizer.vocab_size,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "fit_characters": len(fit_text),
            "validation_characters": len(validation_text),
            "test_characters": len(test_text),
        },
        "training": {
            "sampled_examples": steps * batch_size,
            "approximate_epochs": (
                steps * batch_size / (len(fit_ids) - context_size)
            ),
            "elapsed_seconds": elapsed_seconds,
            "tokens_per_second": steps * batch_size / elapsed_seconds,
        },
        "validation": loss_metrics(
            best_validation_loss,
            len(validation_ids) - context_size,
            validation_text[context_size:],
        ),
        "test": loss_metrics(
            test_loss,
            len(test_ids) - context_size,
            test_text[context_size:],
        ),
        "example_generation": tokenizer.decode(generated_ids),
    }


def main() -> None:
    """Parse command-line options, run the experiment, and save its JSON report."""
    parser = argparse.ArgumentParser(description=__doc__)
    # Thirty thousand mini-batch updates match the tuned local fit; validation
    # checkpointing protects against later updates that stop improving.
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--context-size",
        type=int,
        default=1,
        help="Number of preceding characters used to predict each next character.",
    )
    parser.add_argument(
        "--self-attention",
        action="store_true",
        help="Use causal self-attention (requires --context-size >= 2).",
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--validation-interval",
        type=int,
        default=None,
        help="Updates between validation checks (default: about 20 checks per fit).",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results/baseline.json",
    )
    args = parser.parse_args()

    results = run_baseline(
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        requested_device=args.device,
        context_size=args.context_size,
        use_self_attention=args.self_attention,
        validation_interval=args.validation_interval,
    )
    # Create the results directory on first run, then save and print the report.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
