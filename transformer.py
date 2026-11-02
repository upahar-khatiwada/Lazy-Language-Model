from dataclasses import dataclass
import numpy as np
from ops import (cross_entropy_forward, cross_entropy_backward, layer_norm_backward,
                 layer_norm_forward, linear_backward, relu_backward, relu_forward,
                 softmax_forward, softmax_backward, xavier_uniform)

@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    context_len: int
    emb_dim: int
    num_heads: int
    ff_dim: int
    num_layers: int
    ln_eps: float = 1e-5

def init_params(cfg: ModelConfig, rng):
    d, F, V = cfg.emb_dim, cfg.ff_dim, cfg.vocab_size

    p = {
        "tok_emb": xavier_uniform(V, d, rng), # lookup table with one row per vocabulary token
        "pos_emb": xavier_uniform(cfg.context_len, d, rng) # second table with one row per position, weighted average of values
        # input to first block is tok_emb + pos_emb
        # this makes cat at first position and fifth position different in a sentence
    }

    for l in range(cfg.num_layers):
        b = f"b{l}." # key
        p[b + "ln1_g"], p[b + "ln1_b"] = np.ones(d), np.zeros(d) # layer norm

        for n in "qkvo":
            p[b + "w" + n] = xavier_uniform(d, d, rng)
            p[b + "b" + n] = np.zeros(d)

        p[b + "ln2_g"], p[b + "ln2_b"] = np.ones(d), np.zeros(d)
        p[b + "ff1_w"], p[b + "ff1_b"] = xavier_uniform(d, F, rng), np.zeros(F)
        p[b + "ff2_w"], p[b + "ff2_b"] = xavier_uniform(F, d, rng), np.zeros(d)

    p["lnf_g"], p["lnf_b"] = np.ones(d), np.zeros(d)
    p["head_w"], p["head_b"] = xavier_uniform(d, V, rng), np.zeros(V)

    return p

def count_params(params):
    return int(sum(v.size for v in params.values()))

# X0[t] = E[i_t] + P[t]
# pos_emb has one row for every position the model could ever see (context_len)
# training window has 16 tokens, suppose 3 tokens are given for generation
# tok_emb[ids]  ->  (T, d)        e.g. (3, 32)
# pos_emb       ->  (Tmax, d)     (32, 32)    # shapes don't match
def embed_forward(params, ids):
    return params["tok_emb"][ids] + params["pos_emb"][: len(ids)]

def embed_backward(dX0, ids, grads):
    np.add.at(grads["tok_emb"], ids, dX0) # grads["tok_emb"][ids[t]] += dX0[t]
    grads["pos_emb"][: len(ids)] += dX0