"""Compare hardware performance with and without causal self-attention."""

import argparse
import json
import os
from pathlib import Path

import torch

from .baseline import run_baseline
from .utils.baseline_support import PROJECT_ROOT, load_corpus


def build_run_configurations(cpu_parallel_threads: int) -> list[dict[str, int | str]]:
    """Describe comparable CPU and available accelerator runs."""
    configurations: list[dict[str, int | str]] = [
        {"name": "cpu-single-thread", "device": "cpu", "threads": 1},
        {
            "name": f"cpu-{cpu_parallel_threads}-threads",
            "device": "cpu",
            "threads": cpu_parallel_threads,
        },
    ]
    if torch.cuda.is_available():
        configurations.append(
            {"name": "cuda-accelerated", "device": "cuda", "threads": 1}
        )
    if torch.backends.mps.is_available():
        configurations.append(
            {"name": "mps-accelerated", "device": "mps", "threads": 1}
        )
    return configurations


def run_comparison(
    steps: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    cpu_parallel_threads: int,
    context_sizes: tuple[int, ...] = (4,),
    attention_learning_rate: float | None = None,
    attention_steps: int | None = None,
) -> dict:
    """Compare attention variants on each backend with matched training batches."""
    if steps < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError("steps, batch_size, and learning_rate must be positive.")
    if attention_learning_rate is not None and attention_learning_rate <= 0:
        raise ValueError("attention_learning_rate must be positive when provided.")
    if attention_steps is not None and attention_steps < 1:
        raise ValueError("attention_steps must be positive when provided.")
    if cpu_parallel_threads < 1:
        raise ValueError("cpu_parallel_threads must be positive.")
    if not context_sizes or any(context_size < 2 for context_size in context_sizes):
        raise ValueError(
            "Self-attention comparisons require context_sizes of at least 2."
        )
    if len(set(context_sizes)) != len(context_sizes):
        raise ValueError("context_sizes must not contain duplicates.")

    fit_text, _, _ = load_corpus()
    configurations = build_run_configurations(cpu_parallel_threads)
    previous_thread_count = torch.get_num_threads()
    runs = []
    try:
        for context_size in context_sizes:
            schedule_steps = max(steps, attention_steps or steps)
            validation_interval = max(schedule_steps // 30, 1)
            batch_generator = torch.Generator(device="cpu")
            batch_generator.manual_seed(seed + 2)
            batch_index_schedule = torch.randint(
                context_size - 1,
                len(fit_text) - 1,
                (schedule_steps, batch_size),
                generator=batch_generator,
            )
            for use_self_attention in (False, True):
                run_steps = (
                    attention_steps
                    if use_self_attention and attention_steps is not None
                    else steps
                )
                for configuration in configurations:
                    thread_count = int(configuration["threads"])
                    torch.set_num_threads(thread_count)
                    attention_label = (
                        "with-attention" if use_self_attention else "without-attention"
                    )
                    run_name = (
                        f"{configuration['name']}-context-{context_size}-"
                        f"{attention_label}"
                    )
                    print(
                        f"\n=== {run_name}: "
                        f"{run_steps} steps, {thread_count} PyTorch thread(s) ==="
                    )
                    result = run_baseline(
                        steps=run_steps,
                        batch_size=batch_size,
                        learning_rate=(
                            attention_learning_rate
                            if use_self_attention
                            and attention_learning_rate is not None
                            else learning_rate
                        ),
                        seed=seed,
                        requested_device=str(configuration["device"]),
                        batch_index_schedule=batch_index_schedule[:run_steps],
                        context_size=context_size,
                        use_self_attention=use_self_attention,
                        validation_interval=validation_interval,
                    )
                    runs.append(
                        {
                            "name": run_name,
                            "context_size": context_size,
                            "self_attention": use_self_attention,
                            "report": result,
                        }
                    )
    finally:
        torch.set_num_threads(previous_thread_count)

    reference_seconds_by_variant = {
        (run["context_size"], run["self_attention"]): run["report"]["training"][
            "elapsed_seconds"
        ]
        for run in runs
        if run["name"].startswith("cpu-single-thread")
    }
    summary = []
    for run in runs:
        report = run["report"]
        summary.append(
            {
                "name": run["name"],
                "context_size": run["context_size"],
                "self_attention": run["self_attention"],
                "learning_rate": report["configuration"]["learning_rate"],
                "steps": report["configuration"]["steps"],
                "validation_interval": report["configuration"][
                    "validation_interval"
                ],
                "approximate_epochs": report["training"]["approximate_epochs"],
                "device": report["device"],
                "pytorch_threads": report["system"]["pytorch_threads"],
                "fit_seconds": report["training"]["elapsed_seconds"],
                "fit_tokens_per_second": report["training"]["tokens_per_second"],
                "speedup_vs_cpu_single_thread": (
                    reference_seconds_by_variant[
                        (run["context_size"], run["self_attention"])
                    ]
                    / report["training"]["elapsed_seconds"]
                ),
                "validation_loss_nats_per_token": report["validation"][
                    "loss_nats_per_token"
                ],
                "validation_bits_per_byte": report["validation"]["bits_per_byte"],
                "test_loss_nats_per_token": report["test"]["loss_nats_per_token"],
                "test_bits_per_byte": report["test"]["bits_per_byte"],
                "test_perplexity_per_token": report["test"]["perplexity_per_token"],
                "parameter_count": report["configuration"]["parameter_count"],
                "architecture": report["configuration"]["architecture"],
                "best_validation_step": report["configuration"][
                    "best_validation_step"
                ],
            }
        )

    return {
        "experiment": "character-self-attention-hardware-comparison",
        "comparison_controls": {
            "same_architecture_across_hardware_per_model_variant": True,
            "same_initialization_seed": seed,
            "same_precomputed_batches_across_hardware_per_context_size": True,
            "batch_schedule_seed": seed + 2,
            "validation_checkpoint_selection": True,
            "test_set_used_for_selection": False,
        },
        "configuration": {
            "steps": steps,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "attention_learning_rate": (
                attention_learning_rate
                if attention_learning_rate is not None
                else learning_rate
            ),
            "seed": seed,
            "cpu_parallel_threads": cpu_parallel_threads,
            "context_sizes": list(context_sizes),
            "steps_without_attention": steps,
            "steps_with_attention": attention_steps or steps,
            "validation_interval": max(
                max(steps, attention_steps or steps) // 30,
                1,
            ),
            "self_attention_variants": [False, True],
        },
        "summary": summary,
        "runs": runs,
    }


def main() -> None:
    """Compare attention on/off and save metrics plus generated examples."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=45000)
    parser.add_argument(
        "--attention-steps",
        type=int,
        default=30000,
        help="Override the update count for attention runs.",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--attention-learning-rate",
        type=float,
        default=1e-3,
        help="Override learning rate for attention runs (otherwise --learning-rate).",
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--context-sizes",
        type=int,
        nargs="+",
        default=(4,),
        help="Context window sizes to compare (must be at least 2; default: 4).",
    )
    parser.add_argument(
        "--cpu-parallel-threads",
        type=int,
        default=min(5, os.cpu_count() or 1),
        help="CPU worker-thread count for the parallel run (default: at most 5).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results/attention_comparison.json",
    )
    args = parser.parse_args()

    comparison = run_comparison(
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        cpu_parallel_threads=args.cpu_parallel_threads,
        context_sizes=tuple(args.context_sizes),
        attention_learning_rate=args.attention_learning_rate,
        attention_steps=args.attention_steps,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    print("\nComparison summary:")
    print(json.dumps(comparison["summary"], indent=2))
    for run in comparison["runs"]:
        print(f"\n--- {run['name']} generated sample ---")
        print(run["report"]["example_generation"])
    print(f"\nFull comparison saved to {args.output}")


if __name__ == "__main__":
    main()
