# the probabilities are transformed into words

from ops import softmax_forward
from transformer import ModelConfig, forward
import numpy as np

def sample_next(logits, temperature, top_k, top_p, rng):
    if temperature < 1e-6:
        return int(np.argmax(logits))

    p = softmax_forward(logits / temperature)

    if 0 < top_k < len(p):
        cutoff = np.sort(p)[-top_k] # # k-th largest probabilit
        p = np.where(p >= cutoff, p, 0.0) # zero everything below it
        p = p / p.sum() # renormalise so it sums to 1 again

    if top_p < 1.0:
        order = np.argsort(-p)
        cum = np.cumsum(p[order])
        keep = order[: np.searchsorted(cum, top_p) + 1]
        q = np.zeros_like(p)
        q[keep] = p[keep]
        p = q / q.sum()

    return int(rng.choice(len(p), p=p))

def generate(params, cfg: ModelConfig, seed_ids, max_new_tokens, window, temperature, top_k, top_p, rng):
    ids = list(seed_ids)

    for _ in range(max_new_tokens):
        logits, _ = forward(params, np.array(ids[-window:]), cfg)
        ids.append(sample_next(logits[-1], temperature, top_k, top_p, rng)) # logits[-1] is used because the last position is the only one that has seen the whole context

    return ids