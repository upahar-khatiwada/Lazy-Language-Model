import numpy as np
from gradcheck import numerical_grad, check_params
from transformer import ModelConfig, init_params, loss_and_grads


def tiny_setup():
    cfg = ModelConfig(vocab_size=7, context_len=8, emb_dim=4,
                      num_heads=2, ff_dim=8, num_layers=2)
    rng = np.random.default_rng(0)
    params = init_params(cfg, rng)              # float64
    ids = rng.integers(7, size=5)
    targets = rng.integers(7, size=5)
    return cfg, params, ids, targets, rng


def test_numerical_grad_on_quadratic():
    x = np.array([1., -2., 3.])
    g = numerical_grad(lambda v: float((v ** 2).sum()), x)   # d/dx sum(x^2) = 2x
    assert np.allclose(g, 2 * x, atol=1e-8)


def test_every_tensor_passes():
    cfg, params, ids, targets, rng = tiny_setup()
    loss_fn = lambda p: loss_and_grads(p, ids, targets, cfg)[0]
    _, grads = loss_and_grads(params, ids, targets, cfg)
    worst = check_params(loss_fn, params, grads, rng)
    for name, err in worst.items():
        if name.endswith(".bk"):
            continue                       # true gradient is 0, checked below
        assert err < 1e-4, (name, err)


def test_key_bias_gradient_is_zero():
    # adding the same vector to every key shifts each score row by a constant,
    # softmax ignores constants, so dL/dbk must be 0
    cfg, params, ids, targets, rng = tiny_setup()
    _, grads = loss_and_grads(params, ids, targets, cfg)
    for l in range(cfg.num_layers):
        assert np.abs(grads[f"b{l}.bk"]).max() < 1e-12