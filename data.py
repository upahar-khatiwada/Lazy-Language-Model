import numpy as np
import pandas as pd
from tokenizer import Tokenizer

def load_corpus(path, column, max_rows) -> str:
    df = pd.read_csv(path, nrows=max_rows)
 
    if column not in df.columns:
        raise ValueError(f"no column {column!r}, the file has {list(df.columns)}")
 
    texts = df[column].dropna().astype(str).str.strip()
    return "\n".join(texts) + "\n"

# BPE tokenizer is called to turn this into a 1D array of integers
def encode_corpus(text: str, tok: Tokenizer) -> np.ndarray:
    return np.array(tok.encode(text), dtype=np.int64)

# this cuts id stream so 0.1 keeps first 90% for training and last 10% for testing
def split_ids(ids, val_fraction):
    cut = int((1 - val_fraction) * len(ids))
    return ids[:cut], ids[cut:]

# A language model learns to predict the next token. We slide a window of length T along the ids. The input is the window, and the target is the same window shifted one token to the left.
def make_windows(ids, seq_len, stride = 1):
    if len(ids) <= seq_len:
        raise ValueError(f"need more than seq_len={seq_len} tokens, got {len(ids)}")
 
    starts = np.arange(0, len(ids) - seq_len, stride)
    rows = starts[:, None] + np.arange(seq_len)
 
    inputs = pd.DataFrame(ids[rows])
    targets = pd.DataFrame(ids[rows + 1])
    return inputs, targets
 
# Training doesn’t use all windows at once, because that would make each update slow. It also doesn’t use one window at a time, because that would make updates noisy. This function hands out batch_size windows at a time.
def iterate_batches(inputs, targets, batch_size, rng, shuffle = True):
    order = inputs.index
    if shuffle:
        order = inputs.sample(frac=1, random_state=rng).index
 
    for i in range(0, len(order), batch_size):
        rows = order[i : i + batch_size]
        yield inputs.loc[rows].to_numpy(), targets.loc[rows].to_numpy()