# LexGPT

An educational project for building a small language model locally from the
Swiss Criminal Code.

## Reproducible baseline

The first experiment is a character-tokenized language model. Its default
context size is one character (a bigram); a context size of four is also
supported. The staged UTF-8 corpus is
split into a fitting portion and validation portion; the existing test file
must match the staged corpus suffix and is evaluated only after training.
Tokenizer vocabulary is learned from fitting text only, with unseen characters
mapped to an unknown token.

Run the baseline from the repository root:

```sh
python -m src.baseline
```

By default, the script trains for up to 30,000 mini-batch updates with a
learning rate of `0.001`, selects CUDA, Apple MPS, or CPU automatically, and
writes the configuration, hardware/software details, throughput, validation
metrics, and held-out test metrics to
`results/baseline.json`. Use `--device cpu`, `--steps 100`, or
`--context-size 4` to override defaults. Runs use a fixed seed (`1337` by
default).

To run the four-character model directly:

```sh
python -m src.baseline --context-size 4
```

Add a causal four-head self-attention block with:

```sh
python -m src.baseline --context-size 4 --self-attention
```

The attention mask prevents a character position from using later positions
within its context window. The model still predicts from only the fixed
`context_size` characters; this is a small causal-attention example, not a
full GPT transformer.

For context size one, a lookup table maps the previous character directly to
next-character scores. For larger windows, the model embeds each character
with 32 features, concatenates the embeddings, and uses a 128-unit ReLU layer
to predict the next character. This avoids an exponentially large table of
all possible character sequences. A four-character context is a modest,
MacBook-friendly step up that can learn short sequences while remaining a
small model. Generation uses only the last `context_size` characters at each
step.

With `--self-attention`, the four-character embeddings also receive positional
embeddings and pass through a causal four-head attention layer before the same
prediction head. This adds parameters and computation; the attention mask
ensures each character representation only uses itself and earlier positions.

Validation is checked about twenty times during fitting. The model checkpoint with the
lowest validation loss is used for the final held-out test evaluation, so a
longer run does not assume that its final update is its best one. The selected
step is recorded as `configuration.best_validation_step`.

The report records the OS, processor, RAM, available PyTorch accelerators, and
software versions. It includes token-level perplexity and bits per byte.
Perplexity is useful within one tokenizer; bits per byte supports later
comparisons across tokenizers on the same text. It also includes a short
generated sample to illustrate the limitations of a bigram model.

Each fitting step processes one random mini-batch, so steps are not literal
full-corpus epochs. At batch size 256, 30,000 updates sample 7.68 million
examples, or about 30 passes over this fitting split, with positions sampled
with replacement. Reports give the number of sampled examples and the
approximate number of corpus passes (`steps * batch size / fitting characters`).
Lower validation/test loss, perplexity, and bits per byte indicate better
next-character predictions. Perplexity is `exp(loss)` and can be read as the
model's effective number of character choices; bits per byte measures
compression-style prediction quality. Fit time measures speed, not quality.

## Longer-fit tuning

A validation-only sweep compared learning rates `0.001`, `0.003`, and `0.005`
at 30,000 updates. The lower rate (`0.001`) was best for both context-four
variants. Extending the MLP run to 45,000 updates improved validation loss
slightly; extending to 60,000 did not improve on its best checkpoint at
45,000. For attention, the 30,000-update run at `0.001` beat a 45,000-update
trial at `0.0005`, with the best checkpoint at update 25,500. These settings
were selected using validation only, not test results. Full sweep values are
in `results/fitting_tuning.json`.

Run the selected fit configurations across hardware with:

```sh
python -m src.compare_fitting \
  --steps 45000 \
  --attention-steps 30000 \
  --learning-rate 0.001 \
  --attention-learning-rate 0.001
```

The longer MLP run samples about 11.5 million fitting examples (roughly 45
passes with replacement); attention uses 7.68 million (roughly 30 passes).

## Fitting hardware comparison

Compare the tuned context-size-four model with and without self-attention on
single-thread CPU, multi-thread CPU, and each available accelerator using:

```sh
python -m src.compare_fitting
```

Within each model variant, hardware runs use the same initialization seed and
precomputed training batches. CPU thread counts and the chosen device are
recorded with each run.
The report includes fit time, throughput, validation/test metrics, parameter
count, and a generated sample for each configuration, and is saved to
`results/attention_comparison.json`. Test results are reported only after
choosing a checkpoint by validation loss. The example text may still vary
slightly between devices due to backend-specific floating-point and sampling
behavior. Fit time excludes corpus setup and final testing but includes the
periodic validation performed during training.

The default comparison uses 45,000 updates for the MLP and 30,000 for
self-attention, both at learning rate `0.001`. Validation is checked every
1,500 updates. The number of updates differs because the MLP continued to
improve up to 45,000, while the attention model's best validation checkpoint
was earlier. Model selection uses validation loss; the test set is evaluated
after selection.

### Tuned context-four results

The table summarizes the held-out test results from the comparison report.
Lower loss, bits per byte, and perplexity mean better predictions. Approximate
epochs are random sampled passes over the fitting data, not ordered full-dataset
epochs.

| Model | Hardware | Updates (approx. passes) | Best checkpoint | Fit time | Validation loss (nats/char) | Test bits/byte | Test perplexity |
|---|---|---:|---:|---:|---:|---:|---:|
| MLP, no attention | CPU, 1 thread | 45,000 (~45.5) | 45,000 | 19.42 s | 1.3757 | 2.4091 | 5.311 |
| MLP, no attention | CPU, 5 threads | 45,000 (~45.5) | 45,000 | 19.79 s | 1.3757 | 2.4091 | 5.311 |
| MLP, no attention | MPS | 45,000 (~45.5) | 40,500 | 66.38 s | 1.3885 | 2.3857 | 5.226 |
| Causal self-attention | CPU, 1 thread | 30,000 (~30.3) | 25,500 | 48.85 s | 1.3636 | 2.3912 | 5.246 |
| Causal self-attention | CPU, 5 threads | 30,000 (~30.3) | 25,500 | 41.76 s | 1.3501 | 2.3667 | 5.158 |
| Causal self-attention | MPS | 30,000 (~30.3) | 25,500 | 120.24 s | 1.3579 | 2.3758 | 5.190 |

Attention improved validation loss relative to the MLP on these selected CPU
runs, and the five-thread CPU attention run also had the lowest test bits per
byte (2.3667). The effect on held-out quality is modest: CPU single-thread
test bits per byte changed from 2.4091 without attention to 2.3912 with it.
Attention took about 2.5 times as long on single-thread CPU and 1.8 times as
long on MPS, so the accuracy gain comes at a real fitting cost. For this
small model, MPS was not the fastest option. These are individual runs, not
repeated statistical benchmarks; small metric differences across hardware
can result from floating-point variation.

Perplexity `5.2` means the model's next-character uncertainty is comparable to
roughly five equally likely choices; it is not an accuracy percentage. Bits
per byte is a normalized negative log-likelihood, useful for comparing these
character models on the same corpus. Neither metric makes the generated text
fully grammatical.

### Generated text examples

These are samples from the single-thread CPU runs, prompted with
`Swiss Criminal Code\n`. They illustrate the difference in local character
patterns; neither model reliably produces legally accurate text.

**Context 4, MLP without attention:**

```text
Swiss Criminal Code
least dissence
not to circing the offences or has been produced agraph 1; santial sentence The prohibit, the minors the outsid womal or be-
cen-
ments in or a
c. reduce; or monection which
could its entacts assistanceristion
1. Gails to lentention or,
markets trainy him.  commissions is the
probatio
```

**Context 4, causal self-attention:**

```text
Swiss Criminal Code
month the endanger, impose recordance not mark twied in who out and 4is a facto custody to a thry hire offing gerulable to are woman one or this shall inst para.
3 An or oo where in commonly doising entative of the faling on
the act to the of their anothere is raning the officio montance attence, th
```

## Corpus analysis and preprocessing

Inspect the raw source and the proposed cleaned text before regenerating the
dataset:

```sh
python -m src.analysis.corpus_profile
```

The report includes Unicode characters and counts, line and whitespace
statistics, repeated lines, article-heading counts, and page-number artifacts.
The current source has a UTF-8 BOM, accented German names, two Unicode space
characters, one private-use hyphen glyph, substantial trailing line whitespace,
and a repeated footer at the end. The previous ASCII conversion silently
discarded the accents and private-use punctuation.

Regenerate the UTF-8 corpus and its splits with:

```sh
python -m src.preprocess
```

Preprocessing preserves accented letters, maps the source's private-use
hyphen to `-`, normalizes tabs and Unicode space separators, strips line-end
whitespace, collapses long blank-line runs, and removes only the verified
trailing title and page-number footer. It deliberately keeps line breaks
instead of guessing how to rejoin wrapped legal text. The 90/10 train/test
split is placed at the nearest non-empty line boundary. Outputs are written to
`data/staged/`.
