# saves and loads everything needed to use a trained model later:
# weights, model shape, and the tokenizer (talk.py cannot guess any of these)

import json
import os
from dataclasses import asdict
import numpy as np
from tokenizer import Tokenizer
from transformer import ModelConfig

# one checkpoint is three files that share a prefix, e.g. "checkpoints/model"
#   prefix + ".npz"       every weight array, stored under the same keys as the params dict
#   prefix + ".json"      the ModelConfig (vocab_size, emb_dim, ...), needed to rebuild the shapes
#   prefix + ".tok.json"  the tokenizer: merges + itos

def save_checkpoint(prefix, params, cfg: ModelConfig, tok: Tokenizer):
    folder = os.path.dirname(prefix)
    if folder:
        os.makedirs(folder, exist_ok=True)

    np.savez(prefix + ".npz", **params) # keys like "b0.wq" are kept as they are

    with open(prefix + ".json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f)

    # stoi is not saved, it is just itos turned around, so load rebuilds it
    # json has no tuples, so each merge's "pair" is stored as a list, merge_tokens only
    # uses pair[0] and pair[1], so a list works exactly the same
    with open(prefix + ".tok.json", "w", encoding="utf-8") as f:
        json.dump({"merges": tok.merges, "itos": tok.itos}, f, ensure_ascii=False)

def load_checkpoint(prefix):
    with np.load(prefix + ".npz") as z:
        params = {k: z[k] for k in z.files} # copy out, the file closes after this block

    with open(prefix + ".json", encoding="utf-8") as f:
        cfg = ModelConfig(**json.load(f))

    with open(prefix + ".tok.json", encoding="utf-8") as f:
        d = json.load(f)

    stoi = {}
    for i in range(len(d["itos"])):
        stoi[d["itos"][i]] = i

    return params, cfg, Tokenizer(d["merges"], stoi, d["itos"])