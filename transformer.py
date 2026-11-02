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

# (T,d)->(H,T,dh)
def split_heads(M, H):
    T, d = M.shape
    return M.reshape(T, H, d // H).transpose(1, 0, 2)

# (H,T,dh)->(T,d)
def merge_heads(Mh):
    H, T, dh = Mh.shape
    return Mh.transpose(1, 0, 2).reshape(T, H * dh)

# takes one sequence of normalized token vectors and returns a new vector built from itself and earlier positions
def attention_forward(Xn, params, p, cfg: ModelConfig):
    T, d = Xn.shape
    H = cfg.num_heads
    dh = d // H # width of one head

    Q = Xn @ params[p+"wq"] + params[p+"bq"]    # (T,d), QUERY
    K = Xn @ params[p+"wk"] + params[p+"bk"]    # KEY
    V = Xn @ params[p+"wv"] + params[p+"bv"]    # VALUE

    Qh, Kh, Vh = split_heads(Q, H), split_heads(K, H), split_heads(V, H)

    S = (Qh @ Kh.transpose(0, 2, 1)) / np.sqrt(dh)
    S = np.where(np.triu(np.ones((T, T), dtype=bool), 1), -np.inf, S)
    A = softmax_forward(S)
    O = merge_heads(A @ Vh)
    out = O @ params[p + "wo"] + params[p + "bo"]
    return out, dict(Xn=Xn, Qh=Qh, Kh=Kh, Vh=Vh, A=A, O=O)

# receives the gradient on attention output and returns gradient on its input
# given   dOut = ∂L/∂out         (T,d)   what arrives from the residual branch
# fills   grads[p+"wo","bo","wq","bq","wk","bk","wv","bv"]
# returns dXn  = ∂L/∂Xn          (T,d)   goes on to layer-norm 1's backward
def attention_backward(dOut, c, params, p, cfg: ModelConfig, grads):
    H = cfg.num_heads
    dh = cfg.emb_dim // H
    grads[p + "wo"] += c["O"].T @ dOut
    grads[p + "bo"] += dOut.sum(axis=0)
    dOh = split_heads(dOut @ params[p + "wo"].T, H)
    dVh = c["A"].transpose(0, 2, 1) @ dOh 
    dA = dOh @ c["Vh"].transpose(0, 2, 1)
    dS = softmax_backward(c["A"], dA)
    dQh = dS @ c["Kh"] / np.sqrt(dh)
    dKh = dS.transpose(0, 2, 1) @ c["Qh"] / np.sqrt(dh)
    dQ, dK, dV = merge_heads(dQh), merge_heads(dKh), merge_heads(dVh)
    dXn = np.zeros_like(c["Xn"])
    for n, dM in (("q", dQ), ("k", dK), ("v", dV)):
        grads[p + "w" + n] += c["Xn"].T @ dM
        grads[p + "b" + n] += dM.sum(axis=0)
        dXn += dM @ params[p + "w" + n].T
    return dXn

# attention moves information between positions
def ffn_forward(Xn, params, p):
    Z = Xn @ params[p + "ff1_w"] + params[p + "ff1_b"]
    A = relu_forward(Z)

    return A @ params[p + "ff2_w"] + params[p + "ff2_b"], dict(Xn=Xn, Z=Z, A=A)

def ffn_backward(dOut, c, params, p, grads):
    dA, dW2, db2 = linear_backward(c["A"], params[p + "ff2_w"], dOut)
    grads[p + "ff2_w"] += dW2
    grads[p + "ff2_b"] += db2
    dZ = relu_backward(c["Z"], dA)
    dXn, dW1, db1 = linear_backward(c["Xn"], params[p + "ff1_w"], dZ)

    grads[p + "ff1_w"] += dW1
    grads[p + "ff1_b"] += db1

    return dXn

# one transformer block in pre-LN form
def block_forward(X, params, l: int, cfg: ModelConfig):
    p = f"b{l}."
    Xn, c_ln1 = layer_norm_forward(X, params[p + "ln1_g"], params[p + "ln1_b"], cfg.ln_eps) # make a normalized copy for attention to read
    att, c_att = attention_forward(Xn, params, p, cfg) # each position gathers info from earlier positions
    X1 = X + att
    Xn2, c_ln2 = layer_norm_forward(X1, params[p + "ln2_g"], params[p + "ln2_b"], cfg.ln_eps) # normalized copy again, for the feed-forward
    ff, c_ff = ffn_forward(Xn2, params, p) # ach position thinks on its own

    return X1 + ff, dict(ln1=c_ln1, att=c_att, ln2=c_ln2, ff=c_ff)

def block_backward(dY, c, params, l, cfg: ModelConfig, grads):
    p = f"b{l}."
    dXn2 = ffn_backward(dY, c["ff"], params, p, grads)
    dX1_branch, dg, db = layer_norm_backward(dXn2, c["ln2"], params[p + "ln2_g"])
    grads[p + "ln2_g"] += dg
    grads[p + "ln2_b"] += db
    dX1 = dY + dX1_branch
    dXn = attention_backward(dX1, c["att"], params, p, cfg, grads)
    dX_branch, dg, db = layer_norm_backward(dXn, c["ln1"], params[p + "ln1_g"])
    grads[p + "ln1_g"] += dg
    grads[p + "ln1_b"] += db
    return dX1 + dX_branch

# whole model in the forward direction
def forward(params, ids, cfg: ModelConfig):
    assert len(ids) <= cfg.context_len, "sequence longer than the position table"

    X = embed_forward(params, ids)
    block_caches = []

    for l in range(cfg.num_layers):
        X, c = block_forward(X, params, l, cfg)
        block_caches.append(c)

    Xf, c_lnf = layer_norm_forward(X, params["lnf_g"], params["lnf_b"], cfg.ln_eps)
    logits = Xf @ params["head_w"] + params["head_b"]

    return logits, dict(blocks=block_caches, lnf=c_lnf, Xf=Xf)

# runs forward, computes the loss, then the whole backward pass
def loss_and_grads(params, ids, targets, cfg: ModelConfig):
    logits, cache = forward(params, ids, cfg)
    loss, probs = cross_entropy_forward(logits, targets)
    grads = {k: np.zeros_like(v) for k, v in params.items()}
    dZ = cross_entropy_backward(probs, targets)
    dXf, dW, db = linear_backward(cache["Xf"], params["head_w"], dZ)

    grads["head_w"] += dW
    grads["head_b"] += db

    dX, dg, db = layer_norm_backward(dXf, cache["lnf"], params["lnf_g"])

    grads["lnf_g"] += dg
    grads["lnf_b"] += db

    for l in reversed(range(cfg.num_layers)):
        dX = block_backward(dX, cache["blocks"][l], params, l, cfg, grads)
    embed_backward(dX, ids, grads)

    return loss, grads