import numpy as np

def numerical_grad(f, x, h = 1e-5):
    g = np.zeros_like(x)
    for idx in np.ndindex(*x.shape):
        old = x[idx]
        x[idx] = old + h
        fp = f(x)
        x[idx] = old - h
        fm = f(x)
        x[idx] = old
        g[idx] = (fp - fm) / (2 * h)

    return g

def relative_error(a, b, eps = 1e-12):
    return abs(a - b) / max(abs(a), abs(b), eps)

def check_params(loss_fn, params, grads, rng, samples_per_param = 5, h = 1e-5):
    worst: dict[str, float] = {}
    for name, W in params.items():
        w = 0.0
        for _ in range(samples_per_param):
            idx = tuple(int(rng.integers(s)) for s in W.shape)
            old = W[idx]
            W[idx] = old + h
            fp = loss_fn(params)
            W[idx] = old - h
            fm = loss_fn(params)
            W[idx] = old
            w = max(w, relative_error(grads[name][idx], (fp - fm) / (2 * h)))
        worst[name] = w
    return worst