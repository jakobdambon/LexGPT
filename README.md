# LexGPT

An educational project for building a small language model locally from the
Swiss Criminal Code.

## Reproducible baseline

The first experiment is a character-tokenized bigram model: it predicts the
next character using only the current character. The staged UTF-8 corpus is
split into a fitting portion and validation portion; the existing test file
must match the staged corpus suffix and is evaluated only after training.
Tokenizer vocabulary is learned from fitting text only, with unseen characters
mapped to an unknown token.

Run the baseline from the repository root:

```sh
python -m src.baseline
```

By default, the script trains for up to 10,000 steps with a learning rate of
`0.005`, selects CUDA, Apple MPS, or CPU automatically, and writes the
configuration, hardware/software details, throughput, validation metrics, and
held-out test metrics to
`results/baseline.json`. Use `--device cpu` or `--steps 100` to override the
defaults. Runs use a fixed seed (`1337` by default).

Validation is checked ten times during fitting. The model checkpoint with the
lowest validation loss is used for the final held-out test evaluation, so a
longer run does not assume that its final update is its best one. The selected
step is recorded as `configuration.best_validation_step`.

The report records the OS, processor, RAM, available PyTorch accelerators, and
software versions. It includes token-level perplexity and bits per byte.
Perplexity is useful within one tokenizer; bits per byte supports later
comparisons across tokenizers on the same text. It also includes a short
generated sample to illustrate the limitations of a bigram model.

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
