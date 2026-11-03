# this file takes the gradients calculated so far and updates them

import math
import numpy as np

class Adam:
    def __init__(self, params, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0):
        self.m = {k: np.zeros_like(v) for k, v in params.items()} # running average of the gradient (which direction to go)
        self.v = {k: np.zeros_like(v) for k, v in params.items()} # running average of gradient squared (how big the gradients usually are)
        self.t = 0 # step counter, needed for bias correction
        self.b1, self.b2, self.eps, self.wd = beta1, beta2, eps, weight_decay

    def step(self, params, grads, lr):
        self.t += 1 # first step is t=1, so 1 - b1**t is never zero

        for k in params:
            g = grads[k]

            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g # mostly old direction, a little new gradient
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g # same idea, but for the squared gradient

            mhat = self.m[k] / (1 - self.b1 ** self.t) # m starts at 0 so it is too small early on, this scales it back up
            vhat = self.v[k] / (1 - self.b2 ** self.t)

            update = mhat / (np.sqrt(vhat) + self.eps) # direction divided by typical size, so every number moves about lr per step

            if params[k].ndim >= 2: # weight decay only for matrices, not biases or layer norm gamma/beta
                update = update + self.wd * params[k] # shrink the weight a little, outside the m/v part (this is the "W" in AdamW)

            params[k] -= lr * update # changes the array in place, nothing to return

def lr_schedule(step, warmup, total, lr_max, lr_min):
    if step <= warmup:
        return lr_max * step / warmup # warm-up: ramp from near 0 up to lr_max, a fresh model is fragile

    frac = (step - warmup) / (total - warmup) # how far through the decay we are, goes from 0 to 1
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * frac)) # cosine slide from lr_max down to lr_min


def clip_grad_norm(grads, max_norm):
    total = 0.0
    for k in grads:
        total += (grads[k] ** 2).sum() # add up the squares of every gradient number in every tensor
    norm = np.sqrt(total) # length of the whole gradient as one big vector

    if norm > max_norm:
        for k in grads:
            grads[k] *= max_norm / norm # shrink every tensor by the same factor, direction stays, length becomes max_norm

    return float(norm) # the length before clipping, train.py prints it as |g|