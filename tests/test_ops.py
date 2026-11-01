import numpy as np

from ops import (
    xavier_uniform, linear_forward, linear_backward, relu_forward, relu_backward,
    softmax_forward, softmax_backward, cross_entropy_forward, cross_entropy_backward,
    layer_norm_forward, layer_norm_backward, perplexity,
)


def test_linear_worked_example():
    X = np.array([[1., 2.], [0., 1.]])
    W = np.array([[1., 0., 2.], [-1., 1., 0.]])
    b = np.array([0., 1., -1.])
    dY = np.array([[1., 0., 1.], [0., 2., -1.]])
    assert np.allclose(linear_forward(X, W, b), [[-1, 3, 1], [-1, 2, -1]])
    dX, dW, db = linear_backward(X, W, dY)
    assert np.allclose(dW, [[1, 0, 1], [2, 2, 1]])
    assert np.allclose(db, [1, 2, 0])
    assert np.allclose(dX, [[3, -1], [-2, 2]])


def test_relu():
    Z = np.array([-1., 2., 0., 3.])
    assert np.allclose(relu_forward(Z), [0, 2, 0, 3])
    assert np.allclose(relu_backward(Z, np.ones(4)), [0, 1, 0, 1])


def test_softmax_values_and_stability():
    p = softmax_forward(np.array([[2., 1., 0.]]))
    assert np.allclose(p, [[0.6652, 0.2447, 0.0900]], atol=1e-3)
    big = softmax_forward(np.array([[1000., 1001., 1002.]]))
    assert not np.isnan(big).any()
    assert np.allclose(big.sum(axis=-1), 1.0)


def test_softmax_backward_rows_sum_to_zero():
    z = np.array([[2., 1., 0.], [0., 3., 1.]])
    p = softmax_forward(z)
    dp = np.array([[1., 0., 0.], [0., 2., -1.]])
    dz = softmax_backward(p, dp)
    assert np.allclose(dz.sum(axis=-1), 0.0)
    assert np.allclose(dz[0], [0.2227, -0.1628, -0.0599], atol=1e-3)


def test_cross_entropy_worked_example():
    logits = np.array([[2., 1., 0.]])
    targets = np.array([0])
    loss, probs = cross_entropy_forward(logits, targets)
    assert np.isclose(loss, 0.4076, atol=1e-3)
    assert np.allclose(cross_entropy_backward(probs, targets),
                       [[-0.3348, 0.2447, 0.0900]], atol=1e-3)


def test_cross_entropy_is_mean_over_positions():
    logits = np.zeros((4, 5))                      # uniform -> loss = ln 5
    targets = np.array([0, 1, 2, 3])
    loss, probs = cross_entropy_forward(logits, targets)
    assert np.isclose(loss, np.log(5))
    dZ = cross_entropy_backward(probs, targets)
    assert np.isclose(dZ[0, 0], (0.2 - 1) / 4)     # the 1/T factor


def test_cross_entropy_extreme_logits_no_nan():
    logits = np.array([[1000., 0., -1000.]])
    loss, probs = cross_entropy_forward(logits, np.array([2]))
    assert np.isfinite(loss)


def test_layer_norm_worked_example():
    x = np.array([[1., 2., 3., 6.]])
    gamma = np.array([1., 1., 2., 2.])
    beta = np.array([0., 0., 0., 1.])
    y, cache = layer_norm_forward(x, gamma, beta)
    assert np.allclose(y, [[-1.069, -0.535, 0, 4.207]], atol=1e-3)
    dx, dgamma, dbeta = layer_norm_backward(np.array([[1., 0., 0., 1.]]), cache, gamma)
    assert np.allclose(dx, [[0.4391, -0.2482, -0.4009, 0.2100]], atol=1e-3)
    assert np.isclose(dx.sum(), 0.0)


def test_xavier_bound():
    rng = np.random.default_rng(0)
    W = xavier_uniform(32, 32, rng)
    assert W.shape == (32, 32)
    assert np.abs(W).max() <= np.sqrt(6 / 64)


def test_perplexity_of_uniform_is_vocab_size():
    assert np.isclose(perplexity(np.log(300)), 300)