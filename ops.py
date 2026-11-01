import numpy as np

def xavier_uniform(fan_in, fan_out, rng):
    a = np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-a, a, size=(fan_in, fan_out))

def linear_forward(X, W, b):
    return X @ W + b

def linear_backward(X, W, dY):
    dW = X.T @ dY
    db = dY.sum(axis=0)
    dX = dY @ W.T
    return dX, dW, db

def relu_forward(X):
    return np.maximum(0.0, X)

def relu_backward(Z, dA):
    return dA * (Z > 0)

def softmax_forward(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

def softmax_backward(y, dY):
    return y * (dY - np.sum(y * dY, axis=-1, keepdims=True))

def cross_entropy_forward(logits, targets):
    z = logits - logits.max(axis=-1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(axis=-1, keepdims=True))
    T = len(targets)
    loss = -logp[np.arange(T), targets].mean()
    return float(loss), np.exp(logp)
 
def cross_entropy_backward(probs, targets):
    T = len(targets)
    dZ = probs.copy()
    dZ[np.arange(T), targets] -= 1.0
    return dZ / T

def layer_norm_forward(X, gamma, beta, epsilon = 1e-5):
    mu = X.mean(axis=-1, keepdims=True)
    var = ((X - mu) ** 2).mean(axis=-1, keepdims=True)
    s = 1.0 / np.sqrt(var + epsilon)
    Xhat = (X - mu) * s
    return gamma * Xhat + beta, (Xhat, s)

def layer_norm_backward(dY, cache, gamma):
    Xhat, s = cache
    dXhat = dY * gamma
    dX = s * (dXhat - dXhat.mean(axis=-1, keepdims=True)
                - Xhat * (dXhat * Xhat).mean(axis=-1, keepdims=True))
    
    dgamma = (dY * Xhat).sum(axis=0)
    dbeta = dY.sum(axis=0)
    return dX, dgamma, dbeta

def perplexity(loss):
    return float(np.exp(loss))