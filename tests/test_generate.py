import numpy as np

from generate import sample_next, generate
from transformer import ModelConfig, init_params


class SpyRng:
    """Stands in for the rng: records the distribution handed to choice()."""
    def choice(self, n, p):
        self.p = np.asarray(p)
        return 0


def probs_used(logits, temperature=1.0, top_k=0, top_p=1.0):
    rng = SpyRng()
    sample_next(np.asarray(logits, dtype=float), temperature, top_k, top_p, rng)
    return rng.p


def test_temperature_examples():
    z = [2.0, 1.0, 0.0]
    assert np.allclose(probs_used(z, 0.5), [0.867, 0.117, 0.016], atol=1e-3)
    assert np.allclose(probs_used(z, 1.0), [0.665, 0.245, 0.090], atol=1e-3)
    assert np.allclose(probs_used(z, 2.0), [0.506, 0.307, 0.186], atol=1e-3)


def test_temperature_zero_is_greedy():
    rng = np.random.default_rng(0)
    assert all(sample_next(np.array([0.1, 2.0, 1.0]), 0.0, 0, 1.0, rng) == 1
               for _ in range(20))


def test_top_k_example():
    z = np.log([0.50, 0.30, 0.15, 0.05])
    assert np.allclose(probs_used(z, top_k=2), [0.625, 0.375, 0, 0])


def test_top_k_one_is_greedy():
    rng = np.random.default_rng(0)
    z = np.array([0.1, 2.0, 1.9, 0.0])
    assert all(sample_next(z, 1.0, 1, 1.0, rng) == 1 for _ in range(30))


def test_top_p_example():
    # cumulative sums (.50, .80, .95, 1): the token that crosses 0.9 is kept
    z = np.log([0.50, 0.30, 0.15, 0.05])
    assert np.allclose(probs_used(z, top_p=0.9), [0.526, 0.316, 0.158, 0], atol=1e-3)


def test_top_p_adapts_to_confidence():
    confident = np.log([0.95, 0.03, 0.01, 0.01])
    unsure = np.log([0.30, 0.30, 0.20, 0.20])
    assert (probs_used(confident, top_p=0.9) > 0).sum() == 1
    assert (probs_used(unsure, top_p=0.9) > 0).sum() == 4


def tiny():
    cfg = ModelConfig(vocab_size=7, context_len=8, emb_dim=4,
                      num_heads=2, ff_dim=8, num_layers=1)
    return cfg, init_params(cfg, np.random.default_rng(0))


def test_generate_length_and_range():
    cfg, params = tiny()
    out = generate(params, cfg, [1, 2, 3], 10, 8, 0.8, 0, 0.9, np.random.default_rng(1))
    assert len(out) == 13 and out[:3] == [1, 2, 3]
    assert all(0 <= i < cfg.vocab_size for i in out)


def test_generate_window_keeps_long_sequences_legal():
    # 20 tokens exceed context_len = 8; the window cuts what the model sees
    cfg, params = tiny()
    out = generate(params, cfg, list(range(7)) * 3, 5, 8, 0.0, 0, 1.0,
                   np.random.default_rng(0))
    assert len(out) == 26


def test_greedy_generation_is_deterministic():
    cfg, params = tiny()
    a = generate(params, cfg, [1, 2], 8, 8, 0.0, 0, 1.0, np.random.default_rng(0))
    b = generate(params, cfg, [1, 2], 8, 8, 0.0, 0, 1.0, np.random.default_rng(99))
    assert a == b