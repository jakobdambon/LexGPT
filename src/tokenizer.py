import sentencepiece as spm
import sys


# Train SentencePiece model
spm.SentencePieceTrainer.train(
    input='./data/staged/scc-eng.txt',  # Path to your dataset
    model_prefix='tokenizer',  # Output files: tokenizer.model and tokenizer.vocab
    vocab_size=1000,  # Vocabulary size
    model_type='bpe'  # Tokenization algorithm (bpe, unigram, char, or word)
)


print(sys.path)
sp = spm.SentencePieceProcessor()
sp.load('./tokenizer.model')

text = "This is the Swiss Criminal Code."

# Tokenize text into IDs
token_ids = sp.encode(text, out_type=int)
print("Token IDs:", token_ids)

# Tokenize text into pieces
tokens = sp.encode(text, out_type=str)
print("Tokens:", tokens)
