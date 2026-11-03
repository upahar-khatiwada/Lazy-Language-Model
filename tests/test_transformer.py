import numpy as np

from optim import Adam
from transformer import (ModelConfig, init_params, count_params, forward, loss_and_grads,
                         embed_forward, embed_backward, ffn_forward, split_heads,
                         merge_heads, attention_forward)


def tiny_cfg(**kw):
    base = dict(vocab_size=7, context_len=8, emb_dim=4, num_heads=2, ff_dim=8, num_layers=2)
    base.update(kw)
    return ModelConfig(**base)


def test_param_count_matches_formula():
    # #params = (2d+1)V + (Tmax+2)d + L(4d^2 + 2dF + F + 9d); d=32, F=128, L=2, Tmax=32, V=300
    cfg = ModelConfig(300, 32, 32, 2, 128, 2)
    params = init_params(cfg, np.random.default_rng(0))
    assert count_params(params) == 45_996


def test_initial_loss_is_close_to_ln_vocab():
    cfg = ModelConfig(300, 32, 32, 2, 128, 2)
    rng = np.random.default_rng(0)
    params = init_params(cfg, rng)
    ids, targets = rng.integers(300, size=16), rng.integers(300, size=16)
    loss, _ = loss_and_grads(params, ids, targets, cfg)
    assert abs(loss - np.log(300)) < 0.5


def test_embedding_worked_example():
    params = {"tok_emb": np.array([[.1, .2], [.3, -.1], [.5, .4], [-.2, .6]]),
              "pos_emb": np.array([[0, .1], [.1, 0], [.2, .1]])}
    X0 = embed_forward(params, np.array([2, 0, 2]))
    assert np.allclose(X0, [[.5, .5], [.2, .2], [.7, .5]])


def test_embedding_backward_sums_repeated_ids():
    grads = {"tok_emb": np.zeros((4, 2)), "pos_emb": np.zeros((3, 2))}
    dX0 = np.array([[1., 0.], [0., 1.], [2., 2.]])
    embed_backward(dX0, np.array([2, 0, 2]), grads)
    assert np.allclose(grads["tok_emb"][2], [3, 2])       # rows 0 and 2 both landed here
    assert np.allclose(grads["tok_emb"][0], [0, 1])
    assert np.allclose(grads["pos_emb"], dX0)


def test_ffn_worked_example():
    p = "b0."
    params = {p + "ff1_w": np.array([[1., 0., -1.], [0., 1., 1.]]),
              p + "ff1_b": np.array([0., .5, 1.]),
              p + "ff2_w": np.array([[1., 2.], [0., 1.], [1., -1.]]),
              p + "ff2_b": np.zeros(2)}
    y, cache = ffn_forward(np.array([[1., -1.]]), params, p)
    assert np.allclose(cache["Z"], [[1, -.5, -1]])
    assert np.allclose(cache["A"], [[1, 0, 0]])           # ReLU zeroes the negatives
    assert np.allclose(y, [[1, 2]])


def test_split_merge_heads_round_trip():
    M = np.arange(24, dtype=float).reshape(3, 8)
    Mh = split_heads(M, 2)
    assert Mh.shape == (2, 3, 4)
    assert np.array_equal(merge_heads(Mh), M)


def test_attention_weights_are_causal_and_normalised():
    cfg = tiny_cfg(num_layers=1)
    params = init_params(cfg, np.random.default_rng(0))
    Xn = np.random.default_rng(1).normal(size=(5, 4))
    _, c = attention_forward(Xn, params, "b0.", cfg)
    A = c["A"]                                            # (H, T, T)
    assert np.allclose(np.triu(A[0], 1), 0.0)             # nothing flows from the future
    assert np.allclose(A.sum(axis=-1), 1.0)
    assert np.allclose(A[:, 0, 0], 1.0)                   # first token only sees itself


def test_future_tokens_do_not_change_earlier_logits():
    cfg = tiny_cfg()
    params = init_params(cfg, np.random.default_rng(0))
    a = np.array([1, 2, 3, 4, 5])
    b = a.copy()
    b[4] = 0                                              # change only the last token
    la, _ = forward(params, a, cfg)
    lb, _ = forward(params, b, cfg)
    assert np.allclose(la[:4], lb[:4])
    assert not np.allclose(la[4], lb[4])


def test_grads_have_the_shape_of_params():
    cfg = tiny_cfg()
    rng = np.random.default_rng(0)
    params = init_params(cfg, rng)
    _, grads = loss_and_grads(params, rng.integers(7, size=5), rng.integers(7, size=5), cfg)
    assert grads.keys() == params.keys()
    assert all(grads[k].shape == params[k].shape for k in params)


def test_overfit_one_window():
    # a correct forward + backward + optimiser must be able to memorise 8 tokens
    cfg = tiny_cfg(emb_dim=16, ff_dim=32)
    rng = np.random.default_rng(0)
    params = init_params(cfg, rng)
    ids, targets = rng.integers(7, size=8), rng.integers(7, size=8)
    opt = Adam(params)
    for _ in range(200):
        loss, grads = loss_and_grads(params, ids, targets, cfg)
        opt.step(params, grads, lr=1e-2)
    # (no "starts near ln V" check here: with V = 7 the initial logit variance
    #  ~ 2d/(d+V) is too large for that rule; the V = 300 test covers it)
    assert loss < 0.05