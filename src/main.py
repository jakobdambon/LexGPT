
from utils.train import gpt_generate
from utils.tokenizer import char_token
import sentencepiece as spm

import time
import torch

import pandas as pd


tr_te_split = 0.7
test_text = 'This is the Swiss Criminal Code.'
# ------------

with open('data/staged/scc-eng.txt', 'r', encoding='utf-8') as f:
    text = f.read()


def run_model(tokenizer = 'char', vocab_size = None, spm_model_type = 'bpe'):

    t0 = time.time()

    # tokenizer
    if tokenizer == 'char':
        tokens, encode, decode = char_token(text)
        encode_str = lambda x: decode(encode(x))
    elif tokenizer == 'spm':
        # Train SentencePiece model
        vocab_size = 500 if vocab_size is None else vocab_size
        spm.SentencePieceTrainer.train(
            input='data/staged/scc-eng.txt',  # Path to your dataset
            model_prefix='tokenizer',  # Output files: tokenizer.model and tokenizer.vocab
            vocab_size=vocab_size,  # Vocabulary size
            model_type=spm_model_type # Tokenization algorithm (bpe, unigram, char, or word)
        )
        sp = spm.SentencePieceProcessor()
        sp.load('./tokenizer.model')
        tokens = [sp.id_to_piece(id) for id in range(sp.get_piece_size())]
        encode = lambda x: sp.encode(x, out_type=int)
        decode = lambda x: sp.decode(x, out_type=str)
        encode_str = lambda x: sp.encode(x, out_type=str)
    
    vocab_size = len(tokens)
    
    t1 = time.time()

    # create example text
    test_encode_num = encode(test_text)
    test_encode_str = encode_str(test_text)

    # Train Model
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9*len(data)) # first 90% will be train, rest val
    train_data = data[:n]
    val_data = data[n:]

    i_out = gpt_generate(train_data, val_data, vocab_size)

    t2 = time.time()

    # store data
    df = pd.DataFrame({
        'time_tok': [t1-t0],
        'time_model': [t2-t1],
        'vocab_size': vocab_size,
        'text_encoding_num': str(test_encode_num),
        'text_decoding_str': str(test_encode_str),
        'text_out': str(decode(i_out))
    })

    return df

tripples = [
    ('char', None, None), 
    ('spm', None, 'char'),
    ('spm', 500, 'bpe'),
    ('spm', 3000, 'bpe')
]

df = pd.DataFrame(columns=['time_tok','time_model','vocab_size','text_encoding_num','text_decoding_str','text_out'])


for tripple in tripples:
    print(tripple)
    df_new = run_model(*tripple)
    df = pd.concat([df, df_new], ignore_index=True)


df.to_csv('run_examples.csv', index=False)