"""Reproducible character-tokenized bigram baseline for the Swiss Criminal Code."""

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


class BigramLanguageModel(nn.Module):
    """Predict the next token using only the immediately previous token.

    This is the simplest language-model baseline: there is no attention or
    longer context. The embedding table is used as a lookup table whose row
    for each input token contains scores (logits) for all possible next tokens.
    """

    def __init__(self, vocab_size: int) -> None:
        """Create a learnable next-token score for every vocabulary pair.

        Args:
            vocab_size: Number of possible token IDs, including the unknown ID.
        """
        super().__init__()
        # Shape: (input token, candidate next token).
        self.next_token_logits = nn.Embedding(vocab_size, vocab_size)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Look up next-token logits for each input token.

        Args:
            token_ids: Integer token IDs, with any leading batch/time dimensions.

        Returns:
            A tensor with the same leading dimensions and a final dimension of
            ``vocab_size`` containing unnormalized next-token scores.
        """
        return self.next_token_logits(token_ids)

    @torch.no_grad()
    def generate(self, prompt_ids: torch.Tensor, new_tokens: int) -> torch.Tensor:
        """Extend a prompt by repeatedly sampling one next token.

        At each step only the prompt's most recently generated token is used,
        which is the defining limitation of a bigram model. Sampling from
        softmax probabilities turns the model's logits into a distribution.

        Args:
            prompt_ids: One-dimensional, non-empty tensor of token IDs.
            new_tokens: Number of sampled IDs to append to the prompt.

        Returns:
            The prompt followed by the sampled IDs.
        """
        token_ids = prompt_ids
        for _ in range(new_tokens):
            # Convert the newest token's scores into probabilities.
            probabilities = F.softmax(self(token_ids[-1]), dim=-1)
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
) -> dict:
    """Train the character bigram and return its configuration and results.

    The tokenizer is built only from the fitting text. Training randomly
    samples adjacent character pairs; validation loss is checked during
    training, while the held-out test loss is calculated once at the end.
    Generation gets a separate deterministic random seed so adding or changing
    generation does not alter the training run.
    """
    if steps < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError("steps, batch_size, and learning_rate must be positive.")

    # Initialize randomness before model construction and sampling.
    device = select_device(requested_device)
    torch.manual_seed(seed)
    fit_text, validation_text, test_text = load_corpus()
    tokenizer = CharacterTokenizer(fit_text)
    fit_ids = encode_tensor(fit_text, tokenizer, device)
    validation_ids = encode_tensor(validation_text, tokenizer, device)
    test_ids = encode_tensor(test_text, tokenizer, device)

    model = BigramLanguageModel(tokenizer.vocab_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    validation_loss = evaluate_loss(model, validation_ids)
    best_validation_loss = validation_loss
    best_validation_step = 0
    best_model_state = copy.deepcopy(model.state_dict())

    # Time optimization steps separately from corpus preparation and evaluation.
    start_time = time.perf_counter()

    for step in range(steps):
        # Sample random positions, then form one-character input/target pairs.
        indices = torch.randint(len(fit_ids) - 1, (batch_size,), device=device)
        inputs = fit_ids[indices]
        targets = fit_ids[indices + 1]
        logits = model(inputs)
        loss = F.cross_entropy(logits, targets)

        # Compute gradients of prediction error and update the token-pair scores.
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # Periodic validation shows learning progress without updating weights.
        if (step + 1) % max(steps // 10, 1) == 0 or step + 1 == steps:
            validation_loss = evaluate_loss(model, validation_ids)
            print(f"step {step + 1}/{steps}: validation loss {validation_loss:.4f} nats/token")
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                best_validation_step = step + 1
                best_model_state = copy.deepcopy(model.state_dict())

    elapsed_seconds = time.perf_counter() - start_time
    # Use the checkpoint selected by validation rather than assuming the last
    # optimization step is best. The test data does not influence this choice.
    model.load_state_dict(best_model_state)
    test_loss = evaluate_loss(model, test_ids)

    # Use a separate seed so sample text is repeatable without affecting metrics.
    torch.manual_seed(seed + 1)
    prompt = encode_tensor("Swiss Criminal Code\n", tokenizer, device)
    generated_ids = model.generate(prompt, new_tokens=300).tolist()

    # Store metrics and run conditions together, so results remain interpretable.
    return {
        "experiment": "character-bigram",
        "seed": seed,
        "device": str(device),
        "system": system_metadata(),
        "configuration": {
            "steps": steps,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "best_validation_step": best_validation_step,
            "vocab_size": tokenizer.vocab_size,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "fit_characters": len(fit_text),
            "validation_characters": len(validation_text),
            "test_characters": len(test_text),
        },
        "training": {
            "elapsed_seconds": elapsed_seconds,
            "tokens_per_second": steps * batch_size / elapsed_seconds,
        },
        "validation": loss_metrics(
            best_validation_loss,
            len(validation_ids) - 1,
            validation_text,
        ),
        "test": loss_metrics(test_loss, len(test_ids) - 1, test_text),
        "example_generation": tokenizer.decode(generated_ids),
    }


def main() -> None:
    """Parse command-line options, run the experiment, and save its JSON report."""
    parser = argparse.ArgumentParser(description=__doc__)
    # Ten thousand updates give the bigram time to converge; the best
    # validation checkpoint is retained if later updates stop helping.
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=5e-3)
    parser.add_argument("--seed", type=int, default=1337)
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
    )
    # Create the results directory on first run, then save and print the report.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
