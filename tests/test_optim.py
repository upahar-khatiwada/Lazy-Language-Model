import numpy as np

from optim import Adam, lr_schedule, clip_grad_norm


def test_adam_two_step_hand_trace():
    # theta0 = 1, lr = 0.1, g = 0.5 then 0.25
    params = {"w": np.array([1.0])}
    opt = Adam(params)
    opt.step(params, {"w": np.array([0.5])}, lr=0.1)
    assert np.isclose(params["w"][0], 0.9000, atol=1e-4)
    opt.step(params, {"w": np.array([0.25])}, lr=0.1)
    assert np.isclose(params["w"][0], 0.8068, atol=1e-4)


def test_adam_first_step_is_lr_times_sign_of_gradient():
    # bias correction makes mhat/sqrt(vhat) = g/|g| at t = 1, whatever the size of g
    for g in (1e-4, 3.0, -250.0):
        params = {"w": np.array([0.0])}
        Adam(params).step(params, {"w": np.array([g])}, lr=0.01)
        assert np.isclose(params["w"][0], -0.01 * np.sign(g), atol=1e-6)


def test_weight_decay_only_touches_matrices():
    params = {"W": np.ones((2, 2)), "b": np.ones(2)}
    zeros = {k: np.zeros_like(v) for k, v in params.items()}
    Adam(params, weight_decay=0.1).step(params, zeros, lr=0.5)
    assert np.allclose(params["W"], 1 - 0.5 * 0.1)        # shrunk
    assert np.allclose(params["b"], 1.0)                  # untouched


def test_lr_schedule_values():
    # W = 10, Ttot = 100, lr_max = 1e-3, lr_min = 1e-4, step is 1-based
    got = [lr_schedule(t, 10, 100, 1e-3, 1e-4) for t in (1, 5, 10, 55, 100)]
    assert np.allclose(got, [1e-4, 5e-4, 1e-3, 5.5e-4, 1e-4])


def test_clip_rescales_all_tensors_together():
    grads = {"a": np.array([3.0, 4.0]), "b": np.array([0.0])}
    norm = clip_grad_norm(grads, 1.0)
    assert np.isclose(norm, 5.0)                          # norm BEFORE clipping
    assert np.allclose(grads["a"], [0.6, 0.8])
    assert np.isclose(np.sqrt(sum((g ** 2).sum() for g in grads.values())), 1.0)


def test_clip_leaves_small_gradients_alone():
    grads = {"a": np.array([0.3, 0.4])}
    clip_grad_norm(grads, 1.0)
    assert np.allclose(grads["a"], [0.3, 0.4])