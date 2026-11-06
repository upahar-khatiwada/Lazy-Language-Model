#!/usr/bin/env python3
"""Run the repository's real functions on fixed small inputs and store the results.

  bun run sync:golden     writes data/golden/{ops,attention,optim,sampling,bpe,params,data,gradcheck,embeddings,block,model,train,checkpoint}.json
  bun run verify:golden   recomputes and fails if the stored numbers differ

tests/math.golden.test.ts compares lib/math with these files. Inputs are hand-picked, so every case
is an Illustrative Example. Repository modules are imported read-only, nothing is written outside
data/golden/, and training is not run: train.main is never called. One real step on a hand-picked
text, with a tiny model, is the most that happens here. data.py needs pandas.

Each golden file records the Python, NumPy and pandas versions that wrote it. verify:golden compares
floats with a tolerance and whole numbers and text exactly. A different version is reported, not failed.

The one-step case's shuffled batch and sampled text depend on the pandas and NumPy versions, so they are
stored in data/golden/learning-step.fixture.json. Neither command regenerates them. The file is checked
against the real windows and the real tok.decode, and edited by hand only if LEARNING_TEXT changes.
"""

import ast
import copy
import io
import json
import math
import os
import platform
import sys
from pathlib import Path
from types import SimpleNamespace

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = Path(__file__).resolve().parents[1] / "data" / "golden"

import numpy as np  # noqa: E402

import pandas as pd  # noqa: E402

import checkpoint  # noqa: E402
import config  # noqa: E402
import data  # noqa: E402
import generate  # noqa: E402
import gradcheck  # noqa: E402
import ops  # noqa: E402
import optim  # noqa: E402
import tokenizer  # noqa: E402
import train  # noqa: E402
import transformer  # noqa: E402
from tests.test_gradcheck import tiny_setup  # noqa: E402

# Relative and absolute, the same 1e-9 that tests/math.golden.test.ts allows the engine.
FLOAT_TOLERANCE = 1e-9
FIXTURE = OUT / "learning-step.fixture.json"


def arr(value):
    return np.array(value, dtype=float)


def encode(value):
    """To JSON-safe values. JSON has no infinity, so -inf is stored as the string "-Infinity"."""
    if isinstance(value, np.ndarray):
        return encode(value.tolist())
    if isinstance(value, np.generic):
        return encode(value.item())
    if isinstance(value, dict):
        return {str(key): encode(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode(item) for item in value]
    if isinstance(value, float):
        if value == -np.inf:
            return "-Infinity"
        if not np.isfinite(value):
            raise ValueError(f"unexpected non-finite value: {value}")
    return value


def shapes(value):
    """The shape of each array inside value, in the same nesting. None if it holds no array."""
    if isinstance(value, np.ndarray):
        return list(value.shape)
    if isinstance(value, dict):
        found = {key: shapes(item) for key, item in value.items()}
        found = {key: item for key, item in found.items() if item is not None}
        return found or None
    if isinstance(value, list):
        found = [shapes(item) for item in value]
        return found if any(item is not None for item in found) else None
    return None


def case(case_id, mirrors, inputs, outputs):
    return {
        "id": case_id,
        "mirrors": mirrors,
        "kind": "illustrative",
        "inputs": encode(inputs),
        "outputs": encode(outputs),
        "shapes": {"inputs": shapes(inputs) or {}, "outputs": shapes(outputs) or {}},
    }


def traced(function, names, *args, **kwargs):
    """Call function. Returns (result, {name: [every value that local held, in order]}).

    A version is copied when the local is rebound or the function returns, so a dict that is filled
    in place, such as pair_freq, is stored complete. No formula is repeated here.
    """
    history = {name: [] for name in names}
    current = {}

    def update(frame):
        for name in names:
            if name not in frame.f_locals:
                continue
            value = frame.f_locals[name]
            if name in current and current[name] is value:
                continue
            if name in current:
                history[name].append(copy.deepcopy(current[name]))
            current[name] = value

    def tracer(frame, _event, _arg):
        if frame.f_code is not function.__code__:
            return None

        def on_event(inner, event, _inner_arg):
            update(inner)
            if event == "return":
                for name, value in current.items():
                    history[name].append(copy.deepcopy(value))
            return on_event

        return on_event

    previous = sys.gettrace()
    sys.settrace(tracer)
    try:
        result = function(*args, **kwargs)
    finally:
        sys.settrace(previous)
    return result, history


# ops.py -------------------------------------------------------------------------------------------

# The starting values of the matrix example on the Notation page. The page edits and slices these.
MATMUL_X = arr([[1.0, 2.0, -1.0], [0.0, 0.5, 3.0]])
MATMUL_W = arr([[0.5, -1.0, 2.0], [1.5, 0.25, -0.5], [2.0, 0.0, 1.0]])
MATMUL_B = arr([0.1, -0.2, 0.3])

X23 = arr([[1.0, 2.0], [-1.0, 0.5], [0.0, 3.0]])
W23 = arr([[0.5, -1.0, 2.0], [1.5, 0.25, -0.5]])
B3 = arr([0.1, -0.2, 0.3])
LOGITS = arr([[2.0, 1.0, 0.0], [0.5, 0.5, 3.0]])
LN_X = arr([[1.0, 2.0, 3.0, 6.0], [-2.0, 0.0, 0.5, 1.5], [4.0, 4.0, 4.0, 4.0]])
LN_G = arr([1.0, 0.5, 2.0, -1.0])
LN_B = arr([0.0, 0.25, -0.5, 1.0])
LN_DY = arr([[0.1, -0.2, 0.3, 0.4], [1.0, 0.5, -0.5, 0.0], [0.2, 0.2, -0.1, 0.3]])

# tests/test_ops.py uses these values. They start the examples on the operations pages.
TEST_X = arr([[1.0, 2.0], [0.0, 1.0]])
TEST_W = arr([[1.0, 0.0, 2.0], [-1.0, 1.0, 0.0]])
TEST_B = arr([0.0, 1.0, -1.0])
TEST_DY = arr([[1.0, 0.0, 1.0], [0.0, 2.0, -1.0]])
TEST_Z = arr([[-1.0, 2.0, 0.0, 3.0]])
TEST_LOGITS = arr([[2.0, 1.0, 0.0]])
TEST_LN_X = arr([[1.0, 2.0, 3.0, 6.0]])
TEST_LN_G = arr([1.0, 1.0, 2.0, 2.0])
TEST_LN_B = arr([0.0, 0.0, 0.0, 1.0])
TEST_LN_DY = arr([[1.0, 0.0, 0.0, 1.0]])


def softmax_case(case_id, x):
    x = arr(x)
    return case(case_id, "ops.softmax_forward", {"x": x}, {"y": ops.softmax_forward(x)})


def cross_entropy_cases(case_id, logits, targets):
    targets = np.array(targets)
    (loss, probs), history = traced(ops.cross_entropy_forward, ["z", "logp"], logits, targets)
    return [
        case(
            case_id,
            "ops.cross_entropy_forward",
            {"logits": logits, "targets": targets},
            {"loss": loss, "probs": probs, "z": history["z"][-1], "logp": history["logp"][-1]},
        ),
        case(
            f"{case_id}.backward",
            "ops.cross_entropy_backward",
            {"probs": probs, "targets": targets},
            {"dZ": ops.cross_entropy_backward(probs, targets)},
        ),
    ]


def layer_norm_cases(case_id, epsilon=None, X=LN_X, gamma=LN_G, beta=LN_B, dY=LN_DY):
    keyword = {} if epsilon is None else {"epsilon": epsilon}
    (Y, (Xhat, s)), history = traced(ops.layer_norm_forward, ["mu", "var"], X, gamma, beta, **keyword)
    dX, dgamma, dbeta = ops.layer_norm_backward(dY, (Xhat, s), gamma)
    inputs = {"X": X, "gamma": gamma, "beta": beta}
    if epsilon is not None:
        inputs["epsilon"] = epsilon
    return [
        case(
            case_id,
            "ops.layer_norm_forward",
            inputs,
            {"Y": Y, "Xhat": Xhat, "s": s, "mu": history["mu"][-1], "var": history["var"][-1]},
        ),
        case(
            f"{case_id}.backward",
            "ops.layer_norm_backward",
            {"dY": dY, "Xhat": Xhat, "s": s, "gamma": gamma},
            {"dX": dX, "dgamma": dgamma, "dbeta": dbeta},
        ),
    ]


def matmul_cases(case_id, X, W, b):
    """X @ W is the same function with b = 0, so the page's two results both have a golden value."""
    return [
        case(case_id, "ops.linear_forward", {"X": X, "W": W, "b": b}, {"Y": ops.linear_forward(X, W, b)}),
        case(
            f"{case_id}.no_bias",
            "ops.linear_forward",
            {"X": X, "W": W, "b": np.zeros_like(b)},
            {"Y": ops.linear_forward(X, W, np.zeros_like(b))},
        ),
    ]


def linear_cases(case_id, X, W, b, dY):
    """Everything the playground shows: X @ W, Y and the backward results for one dY."""
    dX, dW, db = ops.linear_backward(X, W, dY)
    return [
        *matmul_cases(case_id, X, W, b),
        case(f"{case_id}.backward", "ops.linear_backward", {"X": X, "W": W, "dY": dY}, {"dX": dX, "dW": dW, "db": db}),
    ]


def perplexity_case(case_id, loss):
    return case(case_id, "ops.perplexity", {"loss": loss}, {"perplexity": ops.perplexity(loss)})


def xavier_case(case_id, fan_in, fan_out, keep_sample=False):
    W, history = traced(ops.xavier_uniform, ["a"], fan_in, fan_out, np.random.default_rng(config.seed))
    a = history["a"][-1]
    outputs = {
        "a": a,
        "shape": list(W.shape),
        "low": W.min(),
        "high": W.max(),
        "within": bool(np.abs(W).max() <= a),
    }
    if keep_sample:
        outputs["sample"] = W
    return case(
        case_id,
        "ops.xavier_uniform",
        {"fan_in": fan_in, "fan_out": fan_out, "seed": config.seed},
        outputs,
    )


def init_params_case(case_id, V, Tmax, d, H, F, L):
    cfg = transformer.ModelConfig(V, Tmax, d, H, F, L)
    params = transformer.init_params(cfg, np.random.default_rng(config.seed))
    return case(
        case_id,
        "transformer.init_params",
        {"V": V, "Tmax": Tmax, "d": d, "H": H, "F": F, "L": L, "seed": config.seed},
        {
            "rows": [
                {"name": name, "shape": list(value.shape), "low": value.min(), "high": value.max()}
                for name, value in params.items()
            ]
        },
    )


def numeric(f, x):
    """The real numerical_grad, which works in place on x, on a copy."""
    return gradcheck.numerical_grad(f, np.array(x, dtype=float))


def gradcheck_cases(linear, cross_entropy, layer_norm):
    """Central differences from the real gradcheck.numerical_grad, on the real forward functions.

    A backward function takes dY, the gradient on its output, and no loss. So the scalar here is
    (Y * dY).sum(), whose gradient on Y is exactly dY.
    """
    X, W, b, dY = linear
    logits, targets = cross_entropy
    Xn, gamma, beta, dYn = layer_norm
    targets = np.array(targets)
    row = Xn[0]
    d = len(row)
    unit_gamma, zero_beta = np.ones(d), np.zeros(d)

    def xhat(v, i):
        return ops.layer_norm_forward(v, unit_gamma, zero_beta)[1][0][i]

    return [
        case(
            "gradcheck.linear",
            "gradcheck.numerical_grad",
            {"problem": "linear", "X": X, "W": W, "b": b, "dY": dY},
            {
                "dX": numeric(lambda v: float((ops.linear_forward(v, W, b) * dY).sum()), X),
                "dW": numeric(lambda v: float((ops.linear_forward(X, v, b) * dY).sum()), W),
                "db": numeric(lambda v: float((ops.linear_forward(X, W, v) * dY).sum()), b),
            },
        ),
        case(
            "gradcheck.cross_entropy",
            "gradcheck.numerical_grad",
            {"problem": "cross_entropy", "logits": logits, "targets": targets},
            {"dZ": numeric(lambda v: ops.cross_entropy_forward(v, targets)[0], logits)},
        ),
        case(
            "gradcheck.layer_norm",
            "gradcheck.numerical_grad",
            {"problem": "layer_norm", "X": Xn, "gamma": gamma, "beta": beta, "dY": dYn},
            {
                "dX": numeric(lambda v: float((ops.layer_norm_forward(v, gamma, beta)[0] * dYn).sum()), Xn),
                "dgamma": numeric(lambda v: float((ops.layer_norm_forward(Xn, v, beta)[0] * dYn).sum()), gamma),
                "dbeta": numeric(lambda v: float((ops.layer_norm_forward(Xn, gamma, v)[0] * dYn).sum()), beta),
            },
        ),
        case(
            "gradcheck.layer_norm_jacobian",
            "gradcheck.numerical_grad",
            {"problem": "layer_norm_jacobian", "x": row},
            {"J": [numeric(lambda v: float(xhat(v, i)), row) for i in range(d)]},
        ),
    ]


def ops_cases():
    edited_x = MATMUL_X.copy()
    edited_x[0, 0] = 2.0
    dY = arr([[1.0, 0.0, -1.0], [0.5, 2.0, 0.25], [-0.5, 1.0, 1.5]])
    dX, dW, db = ops.linear_backward(X23, W23, dY)
    Z = arr([[-1.0, 0.0, 2.0], [3.0, -0.5, 0.0]])
    dA = arr([[0.5, 0.25, -1.0], [2.0, 4.0, 8.0]])
    y = ops.softmax_forward(arr([2.0, 1.0, 0.0]))
    y_rows = ops.softmax_forward(LOGITS)
    dY_vec = arr([0.5, -1.0, 2.0])
    dY_rows = arr([[0.5, -1.0, 2.0], [1.0, 1.0, -3.0]])
    test_dp = arr([[1.0, 0.0, 0.0], [0.0, 2.0, -1.0]])
    test_y = ops.softmax_forward(arr([[2.0, 1.0, 0.0], [0.0, 3.0, 1.0]]))
    edited_lin_x = TEST_X.copy()
    edited_lin_x[0, 0] = 3.0
    edited_lin_dy = TEST_DY.copy()
    edited_lin_dy[1, 2] = 2.0
    d, F = config.emb_dim, config.ff_dim
    return [
        case(
            "linear_forward.basic",
            "ops.linear_forward",
            {"X": X23, "W": W23, "b": B3},
            {"Y": ops.linear_forward(X23, W23, B3)},
        ),
        case(
            "linear_backward.basic",
            "ops.linear_backward",
            {"X": X23, "W": W23, "dY": dY},
            {"dX": dX, "dW": dW, "db": db},
        ),
        case("relu_forward.basic", "ops.relu_forward", {"X": Z}, {"Y": ops.relu_forward(Z)}),
        case(
            "relu_backward.zero_gets_no_gradient",
            "ops.relu_backward",
            {"Z": Z, "dA": dA},
            {"dZ": ops.relu_backward(Z, dA)},
        ),
        softmax_case("softmax_forward.basic", [2.0, 1.0, 0.0]),
        softmax_case("softmax_forward.equal", [0.5, 0.5, 0.5, 0.5]),
        # Without the max shift these overflow or underflow to NaN. Small inputs would not show it.
        softmax_case("softmax_forward.large_positive", [1000.0, 1001.0, 1002.0]),
        softmax_case("softmax_forward.large_negative", [-1000.0, -1001.0, -1002.0]),
        case(
            "softmax_forward.rows_with_mask",
            "ops.softmax_forward",
            {"x": arr([[1.0, -np.inf, -np.inf], [0.5, 1.5, -np.inf], [1.0, 2.0, 3.0]])},
            {"y": ops.softmax_forward(arr([[1.0, -np.inf, -np.inf], [0.5, 1.5, -np.inf], [1.0, 2.0, 3.0]]))},
        ),
        case(
            "softmax_backward.vector",
            "ops.softmax_backward",
            {"y": y, "dY": dY_vec},
            {"dx": ops.softmax_backward(y, dY_vec)},
        ),
        case(
            "softmax_backward.rows",
            "ops.softmax_backward",
            {"y": y_rows, "dY": dY_rows},
            {"dx": ops.softmax_backward(y_rows, dY_rows)},
        ),
        *cross_entropy_cases("cross_entropy.basic", LOGITS, [0, 2]),
        *cross_entropy_cases("cross_entropy.large_logits", arr([[1000.0, 0.0, -1000.0], [5.0, 5.0, 5.0]]), [1, 2]),
        *layer_norm_cases("layer_norm.explicit_epsilon", 1e-2),
        *layer_norm_cases("layer_norm.default_epsilon", None),
        *matmul_cases("matmul.two_columns", MATMUL_X, MATMUL_W[:, :2], MATMUL_B[:2]),
        *matmul_cases("matmul.three_columns", MATMUL_X, MATMUL_W, MATMUL_B),
        *matmul_cases("matmul.edited_cell", edited_x, MATMUL_W[:, :2], MATMUL_B[:2]),
        # The playground on the Linear page: its start, an edit, and the two-column choice.
        *linear_cases("linear.test_file", TEST_X, TEST_W, TEST_B, TEST_DY),
        *linear_cases("linear.edited", edited_lin_x, TEST_W, TEST_B, edited_lin_dy),
        *linear_cases("linear.two_columns", TEST_X, TEST_W[:, :2], TEST_B[:2], TEST_DY[:, :2]),
        case("relu_forward.test_file", "ops.relu_forward", {"X": TEST_Z}, {"Y": ops.relu_forward(TEST_Z)}),
        case(
            "relu_backward.test_file",
            "ops.relu_backward",
            {"Z": TEST_Z, "dA": np.ones_like(TEST_Z)},
            {"dZ": ops.relu_backward(TEST_Z, np.ones_like(TEST_Z))},
        ),
        # Sizes from config.py. V depends on the corpus, so tok_emb and the head have no row here.
        xavier_case("xavier.tiny", 2, 3, keep_sample=True),
        xavier_case("xavier.attention", d, d),
        xavier_case("xavier.feed_forward_in", d, F),
        xavier_case("xavier.feed_forward_out", F, d),
        xavier_case("xavier.position_table", config.context_len, d),
        init_params_case("init_params.tiny", 5, 3, 4, 2, 6, 1),
        # Equal gaps give equal probabilities, wherever the numbers sit.
        softmax_case("softmax_forward.basic_moved", [-3.0, -4.0, -5.0]),
        softmax_case("softmax_forward.small_same_gaps", [0.0, 1.0, 2.0]),
        case(
            "softmax_backward.test_file",
            "ops.softmax_backward",
            {"y": test_y, "dY": test_dp},
            {"dx": ops.softmax_backward(test_y, test_dp)},
        ),
        *cross_entropy_cases("cross_entropy.test_file", TEST_LOGITS, [0]),
        *cross_entropy_cases("cross_entropy.lab_target_1", TEST_LOGITS, [1]),
        *cross_entropy_cases("cross_entropy.lab_target_2", TEST_LOGITS, [2]),
        *cross_entropy_cases("cross_entropy.lab_edited", arr([[3.0, 1.0, 0.0]]), [1]),
        *cross_entropy_cases("cross_entropy.mean_over_positions", np.zeros((4, 5)), [0, 1, 2, 3]),
        *cross_entropy_cases("cross_entropy.extreme", arr([[1000.0, 0.0, -1000.0]]), [2]),
        perplexity_case("perplexity.test_file", float(np.log(300))),
        perplexity_case("perplexity.uniform_five", float(np.log(5))),
        perplexity_case("perplexity.basic_loss", ops.cross_entropy_forward(LOGITS, np.array([0, 2]))[0]),
        *layer_norm_cases("layer_norm.test_file", None, TEST_LN_X, TEST_LN_G, TEST_LN_B, TEST_LN_DY),
        *layer_norm_cases(
            "layer_norm.edited",
            None,
            arr([[2.0, 2.0, 3.0, 6.0]]),
            arr([1.0, 1.0, 1.0, 2.0]),
            arr([0.0, 1.0, 0.0, 1.0]),
            TEST_LN_DY,
        ),
        # With no spread in the row, var is 0 and only epsilon keeps s finite.
        *layer_norm_cases(
            "layer_norm.constant_row",
            None,
            arr([[4.0, 4.0, 4.0, 4.0]]),
            TEST_LN_G,
            TEST_LN_B,
            TEST_LN_DY,
        ),
        *gradcheck_cases(
            (TEST_X, TEST_W, TEST_B, TEST_DY),
            (LOGITS, [0, 2]),
            (TEST_LN_X, TEST_LN_G, TEST_LN_B, TEST_LN_DY),
        ),
    ]


# transformer.py -----------------------------------------------------------------------------------

ATTENTION_XN = arr([[0.5, -1.0, 1.5, 0.25], [1.0, 0.0, -0.5, 2.0], [-1.5, 0.75, 0.5, -0.25]])
ATTENTION_PARAMS = {
    "wq": arr([[0.2, -0.1, 0.4, 0.3], [0.5, 0.6, -0.2, 0.1], [-0.3, 0.2, 0.7, -0.4], [0.1, -0.5, 0.3, 0.2]]),
    "bq": arr([0.01, -0.02, 0.03, 0.04]),
    "wk": arr([[0.3, 0.2, -0.6, 0.1], [-0.4, 0.5, 0.1, 0.3], [0.6, -0.1, 0.2, -0.2], [0.2, 0.4, -0.3, 0.5]]),
    "bk": arr([-0.03, 0.02, 0.01, -0.04]),
    "wv": arr([[0.1, 0.5, 0.2, -0.3], [0.4, -0.2, 0.3, 0.6], [-0.5, 0.1, 0.4, 0.2], [0.3, 0.3, -0.1, 0.1]]),
    "bv": arr([0.02, 0.04, -0.01, 0.03]),
    "wo": arr([[0.6, -0.2, 0.1, 0.3], [0.2, 0.4, -0.5, 0.1], [-0.1, 0.3, 0.5, -0.2], [0.4, 0.1, 0.2, 0.6]]),
    "bo": arr([-0.01, 0.03, 0.02, -0.02]),
}


def attention_case(case_id, Xn, num_heads, params=None):
    params = params or ATTENTION_PARAMS
    cfg = transformer.ModelConfig(
        vocab_size=1,
        context_len=len(Xn),
        emb_dim=Xn.shape[1],
        num_heads=num_heads,
        ff_dim=1,
        num_layers=1,
    )
    prefixed = {f"b0.{name}": value for name, value in params.items()}
    (out, cache), history = traced(
        transformer.attention_forward,
        ["Q", "K", "V", "S"],
        Xn,
        prefixed,
        "b0.",
        cfg,
    )
    scaled, masked = history["S"]
    return case(
        case_id,
        "transformer.attention_forward",
        {"Xn": Xn, "H": num_heads, **params},
        {
            "Q": history["Q"][-1],
            "K": history["K"][-1],
            "V": history["V"][-1],
            "Qh": cache["Qh"],
            "Kh": cache["Kh"],
            "Vh": cache["Vh"],
            "scaled": scaled,
            "masked": masked,
            "A": cache["A"],
            "O": cache["O"],
            "out": out,
        },
    )


# The Attention page's running example: one row per word of "The cat sat on the mat". Hand-picked. These
# numbers do not encode the words, and the real model reads sub-word tokens at much larger sizes.
CAT_SAT_XN = arr(
    [
        [0.5, -1.0, 1.5, 0.25],
        [1.0, 0.0, -0.5, 2.0],
        [-1.5, 0.75, 0.5, -0.25],
        [0.25, 1.25, -1.0, 0.5],
        [0.75, -0.5, 1.0, -1.0],
        [-0.5, 1.5, 0.25, 0.75],
    ]
)
CAT_SAT_DOUT = arr(
    [
        [0.1, -0.2, 0.3, 0.05],
        [-0.1, 0.25, 0.15, -0.3],
        [0.2, 0.1, -0.25, 0.15],
        [-0.05, 0.3, 0.1, -0.2],
        [0.15, -0.1, 0.2, 0.25],
        [0.3, -0.15, -0.1, 0.1],
    ]
)


def attention_cfg(T, d, num_heads):
    return transformer.ModelConfig(vocab_size=1, context_len=T, emb_dim=d, num_heads=num_heads, ff_dim=1, num_layers=1)


def attention_backward_run(Xn, dOut, params, cfg):
    """One real forward and one real backward. Returns the cache, grads without their prefix, and the
    last value of each local of attention_backward."""
    prefixed = {f"b0.{name}": value for name, value in params.items()}
    out, cache = transformer.attention_forward(Xn, prefixed, "b0.", cfg)
    grads = {name: np.zeros_like(value) for name, value in prefixed.items()}
    names = ["dOh", "dVh", "dA", "dS", "dQh", "dKh", "dQ", "dK", "dV", "dXn"]
    dXn, history = traced(transformer.attention_backward, names, dOut, cache, prefixed, "b0.", cfg, grads)
    local = {name: history[name][-1] for name in names}
    assert np.array_equal(local["dXn"], dXn)
    return out, cache, {name[3:]: value for name, value in grads.items()}, local


def attention_backward_case(case_id, Xn, dOut, num_heads):
    cfg = attention_cfg(len(Xn), Xn.shape[1], num_heads)
    _, _, grads, local = attention_backward_run(Xn, dOut, ATTENTION_PARAMS, cfg)
    return case(
        case_id,
        "transformer.attention_backward",
        {"Xn": Xn, "dOut": dOut, "H": num_heads, **ATTENTION_PARAMS},
        {**local, "grads": grads},
    )


def attention_config_shapes_case():
    """The real attention_forward and attention_backward at the sizes in config.py. Only shapes are
    stored: the inputs are random draws, not model values."""
    rng = np.random.default_rng(config.seed)
    T, d, H = config.seq_len, config.emb_dim, config.num_heads
    params = {}
    for name in "qkvo":
        params[f"w{name}"] = ops.xavier_uniform(d, d, rng)
        params[f"b{name}"] = np.zeros(d)
    Xn, dOut = rng.normal(size=(T, d)), rng.normal(size=(T, d))
    cfg = attention_cfg(T, d, H)
    prefixed = {f"b0.{name}": value for name, value in params.items()}
    (_, _), history = traced(transformer.attention_forward, ["Q", "K", "V", "S"], Xn, prefixed, "b0.", cfg)
    out, cache, grads, local = attention_backward_run(Xn, dOut, params, cfg)
    size = lambda value: list(value.shape)  # noqa: E731
    return case(
        "attention_forward.config_shapes",
        "transformer.attention_forward",
        {"problem": "config_shapes", "seq_len": T, "emb_dim": d, "num_heads": H, "head_dim": config.head_dim},
        {
            "forward": {
                "Xn": size(Xn),
                "Q": size(history["Q"][-1]),
                "K": size(history["K"][-1]),
                "V": size(history["V"][-1]),
                "Qh": size(cache["Qh"]),
                "Kh": size(cache["Kh"]),
                "Vh": size(cache["Vh"]),
                "S": size(history["S"][-1]),
                "A": size(cache["A"]),
                "AVh": size(cache["A"] @ cache["Vh"]),
                "O": size(cache["O"]),
                "out": size(out),
            },
            "backward": {name: size(value) for name, value in {"dOut": dOut, **local}.items()},
            "grads": {name: size(value) for name, value in grads.items()},
        },
    )


def attention_cases():
    M = arr([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]])
    Mh = transformer.split_heads(M, 2)
    # Scores near 1000 make unshifted softmax overflow inside the real attention function.
    large = {**ATTENTION_PARAMS, "wq": ATTENTION_PARAMS["wq"] * 100, "wk": ATTENTION_PARAMS["wk"] * 100}
    return [
        case("split_heads.two", "transformer.split_heads", {"M": M, "H": 2}, {"Mh": Mh}),
        case("split_heads.one", "transformer.split_heads", {"M": M, "H": 1}, {"Mh": transformer.split_heads(M, 1)}),
        case("merge_heads.two", "transformer.merge_heads", {"Mh": Mh}, {"M": transformer.merge_heads(Mh)}),
        attention_case("attention_forward.t3_d4_h2", ATTENTION_XN, 2),
        attention_case("attention_forward.t3_d4_h1", ATTENTION_XN, 1),
        attention_case("attention_forward.t1_one_token", ATTENTION_XN[:1], 2),
        attention_case("attention_forward.large_scores", ATTENTION_XN, 2, large),
        attention_case("attention_forward.cat_sat", CAT_SAT_XN, 2),
        attention_config_shapes_case(),
        attention_backward_case("attention_backward.cat_sat", CAT_SAT_XN, CAT_SAT_DOUT, 2),
        attention_backward_case("attention_backward.t3_d4_h2", ATTENTION_XN, CAT_SAT_DOUT[:3], 2),
        attention_backward_case("attention_backward.t3_d4_h1", ATTENTION_XN, CAT_SAT_DOUT[:3], 1),
    ]


# optim.py -----------------------------------------------------------------------------------------


def adam_case(case_id, params, grads_per_step, lrs, settings):
    live = {name: value.copy() for name, value in params.items()}
    optimizer = optim.Adam(live, **settings)
    steps = []
    for grads, lr in zip(grads_per_step, lrs):
        optimizer.step(live, grads, lr)
        steps.append(
            {
                "params": copy.deepcopy(live),
                "m": copy.deepcopy(optimizer.m),
                "v": copy.deepcopy(optimizer.v),
                "t": optimizer.t,
            }
        )
    return case(
        case_id,
        "optim.Adam.step",
        {"params": params, "grads": grads_per_step, "lrs": lrs, "settings": settings},
        {"steps": steps},
    )


def clip_case(case_id, grads, max_norm):
    live = {name: value.copy() for name, value in grads.items()}
    norm = optim.clip_grad_norm(live, max_norm)
    return case(
        case_id,
        "optim.clip_grad_norm",
        {"grads": grads, "max_norm": max_norm},
        {"grads": live, "norm": norm},
    )


def lr_case(case_id, steps, warmup, total, lr_max, lr_min):
    return case(
        case_id,
        "optim.lr_schedule",
        {"steps": steps, "warmup": warmup, "total": total, "lr_max": lr_max, "lr_min": lr_min},
        {"lrs": [optim.lr_schedule(step, warmup, total, lr_max, lr_min) for step in steps]},
    )


def optim_cases():
    params = {"w": arr([[1.0, -2.0], [0.5, 3.0]]), "b": arr([0.5, -0.5])}
    grads = [
        {"w": arr([[0.1, -0.2], [0.0, 0.4]]), "b": arr([0.3, 0.0])},
        {"w": arr([[-0.3, 0.1], [0.2, 0.0]]), "b": arr([-0.1, 0.2])},
        {"w": arr([[0.5, 0.5], [-0.5, 0.25]]), "b": arr([0.05, -0.4])},
    ]
    three_four = {"a": arr([3.0, 0.0]), "b": arr([[0.0, 4.0]])}
    # warmup, lr_max and lr_min are the config.py values. total depends on the corpus in the real
    # pipeline, so here it is a hand-picked number.
    warmup, lr_max, lr_min = config.warmup_steps, config.lr_max, config.lr_min
    total = 200
    return [
        adam_case("adam.config_values", params, grads, [1e-3, 1e-3, 5e-4], {
            "beta1": config.beta1,
            "beta2": config.beta2,
            "eps": config.adam_eps,
            "weight_decay": config.weight_decay,
        }),
        adam_case("adam.class_defaults", params, grads[:2], [1e-2, 1e-2], {}),
        adam_case("adam.large_decay", params, grads[:2], [1e-1, 1e-1], {"weight_decay": 0.5}),
        lr_case(
            "lr_schedule.boundaries",
            [1, warmup - 1, warmup, warmup + 1, (warmup + total) // 2, total, total + 10],
            warmup, total, lr_max, lr_min,
        ),
        lr_case("lr_schedule.curve", list(range(1, total + 1)), warmup, total, lr_max, lr_min),
        lr_case("lr_schedule.short_warmup", [1, 2, 3, 5, 9], 2, 10, 0.5, 0.1),
        clip_case("clip_grad_norm.above", three_four, 1.0),
        clip_case("clip_grad_norm.exactly_at_limit", three_four, 5.0),
        clip_case("clip_grad_norm.below", three_four, 10.0),
        # The values the schedule section of the AdamW page shows. train.main never calls step 0.
        lr_case("lr_schedule.key_points", [0, 1, warmup, warmup + 1, total], warmup, total, lr_max, lr_min),
        adam_two_step_case(),
        adam_decay_rule_case(),
        one_weight_case(warmup, total, lr_max, lr_min),
    ]


def config_adam_settings():
    return {
        "beta1": config.beta1,
        "beta2": config.beta2,
        "eps": config.adam_eps,
        "weight_decay": config.weight_decay,
    }


def adam_locals(optimizer, params, grads, lr, names):
    """One real step. The values of mhat, vhat and update, per parameter. update is bound twice for a
    parameter with decay: before and after the decay is added."""
    _, history = traced(optimizer.step, ["mhat", "vhat", "update"], params, grads, lr)
    updates = iter(history["update"])
    raw, final = {}, {}
    for name in names:
        raw[name] = next(updates)
        final[name] = next(updates) if params[name].ndim >= 2 else raw[name]
    return {
        "mhat": dict(zip(names, history["mhat"])),
        "vhat": dict(zip(names, history["vhat"])),
        "raw": raw,
        "update": final,
    }


def adam_two_step_case():
    """The numbers of test_adam_two_step_hand_trace (theta0 = 1, g = 0.5 then 0.25, lr = 0.1), with the
    betas, eps and weight_decay of config.py. w has two dimensions and b has one."""
    params = {"w": arr([[1.0]]), "b": arr([1.0])}
    grads = [{"w": arr([[0.5]]), "b": arr([0.5])}, {"w": arr([[0.25]]), "b": arr([0.25])}]
    lrs = [0.1, 0.1]
    settings = config_adam_settings()
    live = copy.deepcopy(params)
    optimizer = optim.Adam(live, **settings)
    steps = []
    for step_grads, lr in zip(grads, lrs):
        found = adam_locals(optimizer, live, step_grads, lr, list(params))
        steps.append(
            {
                "params": copy.deepcopy(live),
                "m": copy.deepcopy(optimizer.m),
                "v": copy.deepcopy(optimizer.v),
                "t": optimizer.t,
                **found,
            }
        )
    return case(
        "adam.two_step",
        "optim.Adam.step",
        {"problem": "trace", "params": params, "grads": grads, "lrs": lrs, "settings": settings},
        {"steps": steps},
    )


def adam_decay_rule_case():
    """Which keys of the real init_params does the real Adam.step shrink? Every parameter starts at ones
    and every gradient is zero, so only decay can change a value. Biases start at zero in the model,
    where decay could not be seen."""
    params = transformer.init_params(TINY_CFG, np.random.default_rng(config.seed))
    live = {name: np.ones_like(value) for name, value in params.items()}
    zeros = {name: np.zeros_like(value) for name, value in params.items()}
    settings = config_adam_settings()
    optim.Adam(live, **settings).step(live, zeros, config.lr_max)
    return case(
        "adam.decay_rule",
        "optim.Adam.step",
        {
            "problem": "decay_rule",
            "sizes": {name: list(value.shape) for name, value in params.items()},
            "settings": settings,
            "lr": config.lr_max,
        },
        {
            "rows": [
                {"name": name, "ndim": params[name].ndim, "changed": bool((live[name] != 1.0).any())}
                for name in params
            ],
        },
    )


def one_weight_case(warmup, total, lr_max, lr_min):
    """One weight over two steps, in the order of train.main: clip, count the step, schedule, Adam. The
    settings are config.py's. The weight, its gradients and total are hand-picked."""
    params = {"w": arr([[0.5]])}
    grads = [{"w": arr([[3.0]])}, {"w": arr([[0.5]])}]
    live = copy.deepcopy(params)
    settings = config_adam_settings()
    optimizer = optim.Adam(live, **settings)
    rows = []
    step = 0
    for original in grads:
        step_grads = copy.deepcopy(original)
        norm = optim.clip_grad_norm(step_grads, config.grad_clip)
        step += 1
        lr = optim.lr_schedule(step, warmup, total, lr_max, lr_min)
        found = adam_locals(optimizer, live, step_grads, lr, ["w"])
        rows.append(
            {
                "step": step,
                "norm": norm,
                "clipped": step_grads["w"],
                "lr": lr,
                "m": copy.deepcopy(optimizer.m["w"]),
                "v": copy.deepcopy(optimizer.v["w"]),
                "t": optimizer.t,
                "mhat": found["mhat"]["w"],
                "vhat": found["vhat"]["w"],
                "raw": found["raw"]["w"],
                "update": found["update"]["w"],
                "w": copy.deepcopy(live["w"]),
            }
        )
    return case(
        "adam.one_weight",
        "optim.Adam.step",
        {
            "problem": "one_weight",
            "params": params,
            "grads": grads,
            "settings": settings,
            "max_norm": config.grad_clip,
            "warmup": warmup,
            "total": total,
            "lr_max": lr_max,
            "lr_min": lr_min,
        },
        {"rows": rows},
    )


# generate.py --------------------------------------------------------------------------------------


class SpyRng:
    """Stands in for the rng and records the distribution handed to choice(), as the tests do."""

    p = None

    def choice(self, _n, p):
        self.p = np.array(p)
        return 0


def sample_case(case_id, logits, temperature, top_k, top_p):
    logits = arr(logits)
    rng = SpyRng()
    result, history = traced(
        generate.sample_next,
        ["p", "cutoff", "order", "cum", "keep"],
        logits,
        temperature,
        top_k,
        top_p,
        rng,
    )
    inputs = {"logits": logits, "temperature": temperature, "top_k": top_k, "top_p": top_p}
    if rng.p is None:
        return case(case_id, "generate.sample_next", inputs, {"kind": "greedy", "index": result})

    versions = history["p"]
    outputs = {"kind": "distribution", "p": rng.p, "after_temperature": versions[0]}
    # p is rebound twice in the top-k branch: zeroed, then renormalised.
    top_k_ran = bool(history["cutoff"])
    outputs["after_top_k"] = versions[2] if top_k_ran else versions[0]
    if top_k_ran:
        outputs["cutoff"] = history["cutoff"][-1]
    if history["order"]:
        order, keep = history["order"][-1], history["keep"][-1]
        before = outputs["after_top_k"]
        last = before[order[len(keep) - 1]]
        tie_split = bool(len(keep) < len(order) and last == before[order[len(keep)]] and last > 0)
        outputs.update(order=order, cum=history["cum"][-1], keep_count=len(keep), tie_split=tie_split)
    return case(case_id, "generate.sample_next", inputs, outputs)


# The probability lab's presets. Hand-picked logits; the settings of the first are those in config.py.
LAB_LOGITS = [2.0, 1.0, 0.5, 0.0, -1.0]


def lab_cases():
    return [
        sample_case("lab.config_defaults", LAB_LOGITS, config.temperature, config.top_k, config.top_p),
        sample_case("lab.greedy", LAB_LOGITS, 0.0, config.top_k, config.top_p),
        sample_case("lab.no_truncation", LAB_LOGITS, 1.0, 0, 1.0),
        sample_case("lab.top_k", LAB_LOGITS, 1.0, 2, 1.0),
        sample_case("lab.top_k_tie", [2.0, 2.0, 2.0, 1.0, 0.0], 1.0, 2, 1.0),
        sample_case("lab.top_p_tie", [1.0, 1.0, 1.0, 1.0, 0.0], 1.0, 0, 0.5),
        sample_case("lab.top_k_then_top_p", LAB_LOGITS, config.temperature, 3, 0.7),
    ]


class RecordingRng:
    """Hands every call to a real generator and records the distribution and the result."""

    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)
        self.distributions = []
        self.draws = []

    def choice(self, n, p):
        drawn = self.rng.choice(n, p=p)
        self.distributions.append(np.array(p))
        self.draws.append(int(drawn))
        return drawn


def choice_case():
    """The real sample_next, called repeatedly with the real generator at config.seed."""
    logits = arr(LAB_LOGITS)
    settings = (config.temperature, config.top_k, config.top_p)
    rng = np.random.default_rng(config.seed)
    draws = [generate.sample_next(logits, *settings, rng) for _ in range(1000)]
    again = np.random.default_rng(config.seed)
    return case(
        "choice.repeated_draws",
        "generate.sample_next.draws",
        {"logits": logits, "temperature": settings[0], "top_k": settings[1], "top_p": settings[2], "seed": config.seed},
        {
            "first_draws": draws[:12],
            "counts": np.bincount(draws, minlength=len(logits)).tolist(),
            "total": len(draws),
            "same_seed_same_draws": draws == [generate.sample_next(logits, *settings, again) for _ in range(1000)],
        },
    )


# The stepper's model stand-in. Row i holds hand-picked logits for the token after token i, so the
# logits depend on one token only. A real model's logits depend on the whole window.
STEPPER_TOKENS = ["the", "cat", "sat", "on", "mat"]
STEPPER_TABLE = [
    [-1.0, 2.0, 0.5, 0.0, 1.5],
    [0.0, -1.0, 2.0, 1.0, -1.0],
    [0.5, -1.0, -1.0, 2.0, 0.0],
    [2.5, 0.0, -1.0, -1.0, 0.5],
    [1.0, 0.0, 1.5, 0.5, -1.0],
]
STEPPER_SEED = [0, 1]
STEPPER_NEW = 6
STEPPER_WINDOW = 4


def stepper_case(case_id, temperature, top_k, top_p):
    """The real generate.generate, with only `forward` replaced by the table above."""
    table = arr(STEPPER_TABLE)
    windows = []
    rows = []

    def table_forward(_params, ids, _cfg):
        windows.append(ids.tolist())
        logits = table[ids]
        rows.append(logits)
        return logits, {}

    rng = RecordingRng(config.seed)
    real_forward = generate.forward
    generate.forward = table_forward
    try:
        ids = generate.generate(None, None, STEPPER_SEED, STEPPER_NEW, STEPPER_WINDOW, temperature, top_k, top_p, rng)
    finally:
        generate.forward = real_forward

    return case(
        case_id,
        "generate.generate",
        {
            "tokens": STEPPER_TOKENS,
            "table": table,
            "seed_ids": STEPPER_SEED,
            "max_new_tokens": STEPPER_NEW,
            "window": STEPPER_WINDOW,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "seed": config.seed,
        },
        {
            "ids": ids,
            "windows": windows,
            "logits_shapes": [list(row.shape) for row in rows],
            "distributions": rng.distributions,
            "draws": rng.draws,
        },
    )


def sampling_cases():
    z = [2.0, 1.0, 0.0]
    z4 = np.log([0.50, 0.30, 0.15, 0.05]).tolist()
    ties = np.log([0.1, 0.2, 0.2, 0.2, 0.3]).tolist()
    many_ties = np.log([0.05, 0.1, 0.1, 0.05, 0.1, 0.05, 0.1, 0.05, 0.1, 0.05, 0.1, 0.05, 0.05, 0.05, 0.05, 0.1, 0.05]).tolist()
    return [
        # Settings in config.py, with hand-picked logits.
        sample_case("sample_next.config_defaults", z4, config.temperature, config.top_k, config.top_p),
        sample_case("temperature.half", z, 0.5, 0, 1.0),
        sample_case("temperature.one", z, 1.0, 0, 1.0),
        sample_case("temperature.two", z, 2.0, 0, 1.0),
        sample_case("temperature.large_logits", [1000.0, 1001.0, 999.0], 1.0, 0, 1.0),
        # The guard is temperature < 1e-6, so exactly 1e-6 is not greedy.
        sample_case("greedy.zero", [0.1, 2.0, 1.0], 0.0, 0, 1.0),
        sample_case("greedy.below_guard", [0.1, 2.0, 1.0], 5e-7, 0, 1.0),
        sample_case("greedy.at_guard_is_not_greedy", [0.1, 2.0, 1.0], 1e-6, 0, 1.0),
        sample_case("greedy.tie_goes_to_first", [1.0, 3.0, 3.0, 2.0], 0.0, 0, 1.0),
        sample_case("top_k.two", z4, 1.0, 2, 1.0),
        sample_case("top_k.one", z4, 1.0, 1, 1.0),
        sample_case("top_k.equal_to_v_is_skipped", z4, 1.0, 4, 1.0),
        sample_case("top_k.above_v_is_skipped", z4, 1.0, 9, 1.0),
        sample_case("top_k.negative_is_skipped", z4, 1.0, -1, 1.0),
        sample_case("top_k.ties_are_all_kept", [2.0, 2.0, 2.0, 1.0, 0.0], 1.0, 2, 1.0),
        sample_case("top_p.keeps_crossing_token", z4, 1.0, 0, 0.9),
        sample_case("top_p.one_is_skipped", z4, 1.0, 0, 1.0),
        sample_case("top_p.zero_keeps_one", z4, 1.0, 0, 0.0),
        # Cumulative sums are exactly .25, .5, .75, 1, so these show searchsorted's left side.
        sample_case("top_p.exact_boundary", [0.0, 0.0, 0.0, 0.0], 1.0, 0, 0.5),
        sample_case("top_p.just_past_boundary", [0.0, 0.0, 0.0, 0.0], 1.0, 0, 0.5000001),
        sample_case("top_p.tie_split_by_cutoff", ties, 1.0, 0, 0.6),
        sample_case("top_p.tie_fully_kept", ties, 1.0, 0, 0.85),
        sample_case("top_p.tie_fully_dropped", np.log([0.7, 0.1, 0.1, 0.1]).tolist(), 1.0, 0, 0.6),
        sample_case("top_p.many_ties_split", many_ties, 1.0, 0, 0.3),
        sample_case("top_k_then_top_p", z4, 1.0, 3, 0.5),
        sample_case("top_k_then_top_p_with_temperature", z4, 0.8, 3, 0.9),
        *lab_cases(),
        choice_case(),
        stepper_case("stepper.config_defaults", config.temperature, config.top_k, config.top_p),
        stepper_case("stepper.greedy", 0.0, config.top_k, config.top_p),
    ]


# tokenizer.py -------------------------------------------------------------------------------------


def bpe_case(case_id, word_freqs, max_merges=None):
    keyword = {} if max_merges is None else {"max_merges": max_merges}
    merges, history = traced(tokenizer.train_bpe, ["pair_freq"], dict(word_freqs), **keyword)
    inputs = {"word_freqs": word_freqs}
    if max_merges is not None:
        inputs["max_merges"] = max_merges
    return case(
        case_id,
        "tokenizer.train_bpe",
        inputs,
        {
            "merges": merges,
            "pair_counts": [
                [{"pair": list(pair), "count": count} for pair, count in pair_freq.items()]
                for pair_freq in history["pair_freq"]
            ],
        },
    )


def merge_case(case_id, tokens, pair, merged):
    return case(
        case_id,
        "tokenizer.merge_tokens",
        {"tokens": tokens, "pair": pair, "merged": merged},
        {"tokens": tokenizer.merge_tokens(tokens, tuple(pair), merged)},
    )


def count_words_case(case_id, text):
    return case(
        case_id,
        "tokenizer.count_words",
        {"text": text},
        {"word_freqs": list(tokenizer.count_words(text).items())},
    )


def bpe_text_case(case_id, text, max_merges=None):
    """What the stepper shows. Splits come from running train_bpe with 0, 1, 2 ... merges and reading
    its own word_splits, so no merge is applied here."""
    word_freqs = tokenizer.count_words(text)
    keyword = {} if max_merges is None else {"max_merges": max_merges}
    merges = tokenizer.train_bpe_on_text(text, **keyword)
    _, history = traced(tokenizer.train_bpe, ["pair_freq"], word_freqs, **keyword)
    splits = []
    for count in range(len(merges) + 1):
        _, seen = traced(tokenizer.train_bpe, ["word_splits"], word_freqs, max_merges=count)
        splits.append(list(seen["word_splits"][-1].items()))
    inputs = {"text": text}
    if max_merges is not None:
        inputs["max_merges"] = max_merges
    return case(
        case_id,
        "tokenizer.train_bpe_on_text",
        inputs,
        {
            "word_freqs": list(word_freqs.items()),
            "merges": merges,
            "pair_counts": [
                [{"pair": list(pair), "count": count} for pair, count in pair_freq.items()]
                for pair_freq in history["pair_freq"]
            ],
            "splits": splits,
        },
    )


def apply_merges_case(case_id, train_text, text):
    merges = tokenizer.train_bpe_on_text(train_text)
    return case(
        case_id,
        "tokenizer.apply_merges",
        {"text": text, "merges": merges},
        {"tokens": tokenizer.apply_merges(text, merges)},
    )


def build_tokenizer_case(case_id, text, max_merges=None):
    keyword = {} if max_merges is None else {"max_merges": max_merges}
    tok = tokenizer.build_tokenizer(text, **keyword)
    inputs = {"text": text}
    if max_merges is not None:
        inputs["max_merges"] = max_merges
    return case(
        case_id,
        "tokenizer.build_tokenizer",
        inputs,
        {
            "merges": tok.merges,
            "itos": tok.itos,
            "stoi": list(tok.stoi.items()),
            "vocab_size": tok.vocab_size,
        },
    )


def encode_case(case_id, train_text, text):
    tok = tokenizer.build_tokenizer(train_text)
    ids = tok.encode(text)
    return case(
        case_id,
        "tokenizer.Tokenizer.encode",
        {"train_text": train_text, "text": text},
        {"tokens": tokenizer.apply_merges(text, tok.merges), "ids": ids, "decoded": tok.decode(ids)},
    )


def decode_case(case_id, train_text, ids):
    tok = tokenizer.build_tokenizer(train_text)
    return case(
        case_id,
        "tokenizer.Tokenizer.decode",
        {"train_text": train_text, "ids": ids},
        {"text": tok.decode(ids)},
    )


STEPPER_TEXT = "low lower lowest"
# tests/test_tokenizer.py uses this corpus.
TEST_TEXT = "low low low lower newest newest"


def bpe_cases():
    words = [["low", 5], ["lower", 2], ["newest", 6], ["widest", 3]]
    return [
        count_words_case("count_words.stepper", STEPPER_TEXT),
        count_words_case("count_words.test_file", "a a b!"),
        # An apostrophe splits a word. An underscore and digits do not. Each whitespace stands alone.
        count_words_case("count_words.classes", "Don't stop_1  now,\nok\t?"),
        bpe_text_case("train_bpe_on_text.stepper", STEPPER_TEXT),
        bpe_text_case("train_bpe_on_text.test_file", TEST_TEXT),
        bpe_text_case("train_bpe_on_text.limit", TEST_TEXT, 3),
        bpe_text_case("train_bpe_on_text.punctuation_never_merges", "a,a,a b.b.b"),
        apply_merges_case("apply_merges.new_word", STEPPER_TEXT, "slow lowest"),
        apply_merges_case("apply_merges.keeps_every_character", TEST_TEXT, "lot  z\n"),
        build_tokenizer_case("build_tokenizer.stepper", STEPPER_TEXT),
        build_tokenizer_case("build_tokenizer.test_file", TEST_TEXT, 5),
        build_tokenizer_case("build_tokenizer.no_merges", "ab ba", 0),
        encode_case("encode.training_text_has_no_unknown", STEPPER_TEXT, STEPPER_TEXT),
        encode_case("encode.known_word", STEPPER_TEXT, "lowest low"),
        encode_case("encode.unseen_punctuation", STEPPER_TEXT, "low!"),
        encode_case("encode.unseen_inside_word", STEPPER_TEXT, "lowz"),
        encode_case("encode.each_unseen_character", STEPPER_TEXT, "zzz"),
        # The string "<unk>" in text is five characters. It only maps to id 0 if they were unseen.
        encode_case("encode.unk_text_unseen", STEPPER_TEXT, "<unk>"),
        encode_case("encode.unk_text_seen", "<unk> <unk>", "<unk>"),
        decode_case("decode.joins_strings", STEPPER_TEXT, [1, 2, 3]),
        decode_case("decode.unknown_id", STEPPER_TEXT, [0, 1, 0]),
        merge_case("merge_tokens.simple", ["a", "b", "c", "a", "b"], ["a", "b"], "ab"),
        merge_case("merge_tokens.overlap_left_to_right", ["a", "a", "a"], ["a", "a"], "aa"),
        merge_case("merge_tokens.no_match", ["a", "b"], ["b", "a"], "ba"),
        bpe_case("train_bpe.words", words),
        bpe_case("train_bpe.merge_limit", words, 2),
        bpe_case("train_bpe.tie_goes_to_first_seen", [["ab", 1], ["cd", 1]]),
        # A plain JavaScript object would move "10" ahead of "ab" and change the tie.
        bpe_case("train_bpe.numeric_word_keeps_its_place", [["ab", 2], ["10", 2]]),
        bpe_case("train_bpe.counts_add_across_words", [["ab", 1], ["cd", 2], ["xab", 2]]),
        bpe_case("train_bpe.code_points", [["\U0001F600\U0001F600a", 2], ["été", 1]]),
        bpe_case("train_bpe.single_characters_stop_at_once", [["a", 3], ["b", 1]]),
        bpe_case("train_bpe.zero_weight_is_never_chosen", [["ab", 0]]),
    ]


# transformer.init_params and count_params ---------------------------------------------------------


def count_case(case_id, V, Tmax, d, H, F, L):
    def count(layers):
        cfg = transformer.ModelConfig(V, Tmax, d, H, F, layers)
        return transformer.count_params(transformer.init_params(cfg, np.random.default_rng(0)))

    return case(
        case_id,
        "transformer.count_params",
        {"V": V, "Tmax": Tmax, "d": d, "H": H, "F": F, "L": L},
        {"count": count(L), "block": count(1) - count(0)},
    )


def params_cases():
    return [
        count_case("count_params.tiny", 11, 7, 4, 2, 6, 3),
        count_case("count_params.one_layer", 13, 5, 6, 3, 10, 1),
        count_case("count_params.no_layers", 9, 3, 2, 1, 4, 0),
        # Shape values from config.py. V = 300 is the value tests/test_transformer.py uses, not this
        # model's vocabulary size, which depends on the corpus.
        count_case(
            "count_params.config_shape_with_test_vocab",
            300, config.context_len, config.emb_dim, config.num_heads, config.ff_dim, config.num_layers,
        ),
    ]


# data.py ------------------------------------------------------------------------------------------


def load_corpus_case(case_id, rows, max_rows):
    """pandas reads the CSV from memory, so nothing is written to disk."""
    buffer = io.StringIO()
    pd.DataFrame({"text": rows}).to_csv(buffer, index=False)
    buffer.seek(0)
    return case(
        case_id,
        "data.load_corpus",
        {"rows": rows, "max_rows": max_rows},
        {"text": data.load_corpus(buffer, "text", max_rows)},
    )


def windows_case(case_id, ids, seq_len, stride=1):
    ids = np.array(ids)
    inputs = {"ids": ids, "seq_len": seq_len, "stride": stride}
    try:
        (x, y), history = traced(data.make_windows, ["starts"], ids, seq_len, stride)
    except ValueError as error:
        return case(case_id, "data.make_windows", inputs, {"raises": "ValueError", "message": str(error)})
    return case(
        case_id,
        "data.make_windows",
        inputs,
        {
            "starts": history["starts"][-1],
            "inputs": x.to_numpy(),
            "targets": y.to_numpy(),
            "count": len(x),
            "stored_as": [type(x).__name__, type(y).__name__],
        },
    )


def split_case(case_id, n, val_fraction, seq_len):
    """ids are their own positions, so a window's values say which side of the cut it touches."""
    ids = np.arange(n)
    train, val = data.split_ids(ids, val_fraction)
    whole_x, whole_y = data.make_windows(ids, seq_len)
    cut = len(train)
    crossing = int(((whole_x.to_numpy()[:, 0] < cut) & (whole_y.to_numpy()[:, -1] >= cut)).sum())
    return case(
        case_id,
        "data.split_ids",
        {"n": n, "val_fraction": val_fraction, "seq_len": seq_len},
        {
            "cut": cut,
            "train_length": len(train),
            "val_length": len(val),
            "train_last": train[-1],
            "val_first": val[0],
            "train_windows": len(data.make_windows(train, seq_len)[0]),
            "val_windows": len(data.make_windows(val, seq_len)[0]),
            "whole_windows": len(whole_x),
            "crossing": crossing,
        },
    )


def batches_case(case_id, n, seq_len, batch_size, seed, shuffle=True):
    x, y = data.make_windows(np.arange(n), seq_len)

    def epochs(count):
        rng = np.random.default_rng(seed)
        return [list(data.iterate_batches(x, y, batch_size, rng, shuffle)) for _ in range(count)]

    runs = epochs(2)
    again = epochs(2)
    first = runs[0]
    return case(
        case_id,
        "data.iterate_batches",
        {"n": n, "seq_len": seq_len, "batch_size": batch_size, "seed": seed, "shuffle": shuffle},
        {
            "epoch_starts": [[xb[:, 0] for xb, _ in epoch] for epoch in runs],
            "sizes": [len(xb) for xb, _ in first],
            "batch_shape": list(first[0][0].shape),
            "stored_as": type(first[0][0]).__name__,
            "pairs_aligned": all((yb[:, :-1] == xb[:, 1:]).all() for epoch in runs for xb, yb in epoch),
            "same_seed_repeats": all(
                (a[0] == b[0]).all() for run_a, run_b in zip(runs, again) for a, b in zip(run_a, run_b)
            ),
        },
    )


def data_cases():
    # tests/test_data.py uses these ids.
    ids = [3, 1, 4, 1, 5, 2, 6, 5, 3, 5]
    return [
        load_corpus_case("load_corpus.skips_missing", [" hello ", None, "world"], 10),
        # max_rows counts the missing row too, because pandas reads rows before dropna runs.
        load_corpus_case("load_corpus.limit_before_dropping", [None, "a", "b"], 2),
        windows_case("make_windows.test_file", ids, 4),
        windows_case("make_windows.stride_two", ids, 4, 2),
        windows_case("make_windows.stride_three", ids, 4, 3),
        windows_case("make_windows.stride_equal_to_gap", ids, 4, 6),
        windows_case("make_windows.stride_above_gap", ids, 4, 7),
        windows_case("make_windows.one_window", ids[:5], 4),
        windows_case("make_windows.length_one", ids, 1),
        # seq_len from config.py, with hand-picked ids.
        windows_case("make_windows.config_seq_len", list(range(40)), config.seq_len, config.stride),
        windows_case("make_windows.too_few", ids[:4], 4),
        split_case("split_ids.small", 20, 0.25, 3),
        split_case("split_ids.fifth", 40, 0.2, 4),
        split_case("split_ids.config_settings", 3000, config.val_fraction, config.seq_len),
        batches_case("iterate_batches.shuffled", 14, 4, 4, config.seed),
        batches_case("iterate_batches.not_shuffled", 14, 4, 4, config.seed, False),
        batches_case("iterate_batches.exact_division", 12, 4, 4, 0),
        # The sizes tests/test_data.py asserts.
        batches_case("iterate_batches.test_file", 50, 4, 16, 0),
    ]


# Embeddings (transformer.py) ----------------------------------------------------------------------

# tests/test_transformer.py uses these tables and this gradient.
EMBED_TOK = arr([[0.1, 0.2], [0.3, -0.1], [0.5, 0.4], [-0.2, 0.6]])
EMBED_POS = arr([[0.0, 0.1], [0.1, 0.0], [0.2, 0.1]])
EMBED_DX0 = arr([[1.0, 0.0], [0.0, 1.0], [2.0, 2.0]])


def embed_forward_case(case_id, ids):
    ids = np.array(ids)
    params = {"tok_emb": EMBED_TOK, "pos_emb": EMBED_POS}
    return case(
        case_id,
        "transformer.embed_forward",
        {"tok_emb": EMBED_TOK, "pos_emb": EMBED_POS, "ids": ids},
        {
            "tokens": EMBED_TOK[ids],
            "positions": EMBED_POS[: len(ids)],
            "X0": transformer.embed_forward(params, ids),
        },
    )


def embed_backward_case(case_id, ids):
    ids = np.array(ids)
    dX0 = EMBED_DX0[: len(ids)]
    grads = {"tok_emb": np.zeros_like(EMBED_TOK), "pos_emb": np.zeros_like(EMBED_POS)}
    transformer.embed_backward(dX0, ids, grads)
    # What a plain fancy-index add would do. The repository does not do this; the page contrasts it.
    overwritten = np.zeros_like(EMBED_TOK)
    overwritten[ids] += dX0
    return case(
        case_id,
        "transformer.embed_backward",
        {"dX0": dX0, "ids": ids, "V": len(EMBED_TOK), "Tmax": len(EMBED_POS)},
        {"tok": grads["tok_emb"], "pos": grads["pos_emb"], "overwritten": overwritten},
    )


def embed_gradcheck_case(case_id, ids):
    """Central differences from the real numerical_grad, on the real embed_forward. The scalar is
    (X0 * dX0).sum(), whose gradient on X0 is exactly dX0."""
    ids = np.array(ids)
    dX0 = EMBED_DX0[: len(ids)]

    def scalar(tok, pos):
        return float((transformer.embed_forward({"tok_emb": tok, "pos_emb": pos}, ids) * dX0).sum())

    return case(
        case_id,
        "gradcheck.numerical_grad",
        {"problem": "embedding", "tok_emb": EMBED_TOK, "pos_emb": EMBED_POS, "ids": ids, "dX0": dX0},
        {
            "tok": numeric(lambda v: scalar(v, EMBED_POS), EMBED_TOK),
            "pos": numeric(lambda v: scalar(EMBED_TOK, v), EMBED_POS),
        },
    )


def embed_config_sizes_case():
    """The real init_params and embed_forward at the sizes in config.py. V has no config value, so it
    is hand-picked. Only shapes are stored: the values are random draws, not learned numbers."""
    V = 7
    cfg = transformer.ModelConfig(
        V, config.context_len, config.emb_dim, config.num_heads, config.ff_dim, config.num_layers
    )
    params = transformer.init_params(cfg, np.random.default_rng(config.seed))
    X0 = transformer.embed_forward(params, np.arange(config.seq_len) % V)
    at_limit = transformer.embed_forward(params, np.arange(config.context_len) % V)
    transformer.forward(params, np.arange(config.context_len) % V, cfg)
    try:
        transformer.forward(params, np.arange(config.context_len + 1) % V, cfg)
        too_long = None
    except AssertionError as error:
        too_long = str(error)
    return case(
        "embed_forward.config_sizes",
        "transformer.embed_forward",
        {
            "problem": "config_sizes",
            "V": V,
            "seq_len": config.seq_len,
            "context_len": config.context_len,
            "emb_dim": config.emb_dim,
        },
        {
            "tok_emb_shape": list(params["tok_emb"].shape),
            "pos_emb_shape": list(params["pos_emb"].shape),
            "X0_shape": list(X0.shape),
            "at_limit_shape": list(at_limit.shape),
            "too_long_message": too_long,
        },
    )


def position_update_case(case_id, weight_decay):
    """One real backward and Adam step on the setup of tests/test_gradcheck.py: which rows of pos_emb
    change? The setup has T = 5 and a table of 8 rows."""
    cfg, params, ids, targets, _rng = tiny_setup()
    before = params["pos_emb"].copy()
    _, grads = transformer.loss_and_grads(params, ids, targets, cfg)
    opt = optim.Adam(params, weight_decay=weight_decay)
    opt.step(params, grads, 1e-3)
    return case(
        case_id,
        "transformer.embed_backward",
        {"problem": "position_update", "T": len(ids), "Tmax": cfg.context_len, "weight_decay": weight_decay},
        {
            "gradient_nonzero": (grads["pos_emb"] != 0).any(axis=1).tolist(),
            "row_changed": (params["pos_emb"] != before).any(axis=1).tolist(),
        },
    )


def embedding_cases():
    return [
        embed_forward_case("embed_forward.test_file", [2, 0, 2]),
        embed_forward_case("embed_forward.two_positions", [3, 3]),
        embed_forward_case("embed_forward.no_repeat", [0, 1, 3]),
        embed_config_sizes_case(),
        embed_backward_case("embed_backward.test_file", [2, 0, 2]),
        embed_backward_case("embed_backward.all_same", [1, 1, 1]),
        embed_backward_case("embed_backward.no_repeat", [0, 1, 3]),
        embed_backward_case("embed_backward.short", [2, 2]),
        embed_gradcheck_case("gradcheck.embedding", [2, 0, 2]),
        position_update_case("position_update.no_decay", 0.0),
        position_update_case("position_update.with_decay", config.weight_decay),
    ]


# gradcheck.py -------------------------------------------------------------------------------------


def relative_error_case(case_id, a, b):
    return case(
        case_id,
        "gradcheck.relative_error",
        {"a": a, "b": b},
        {"error": gradcheck.relative_error(a, b)},
    )


def power_case(case_id, power, x, h):
    """One number, f(x) = x * x or x * x * x. Only multiplication, so Python and JavaScript agree exactly."""

    def f(v):
        return float((v * v if power == 2 else v * v * v).sum())

    g = gradcheck.numerical_grad(f, np.array([x], dtype=float), h)
    analytic = power * x ** (power - 1)
    return case(
        case_id,
        "gradcheck.numerical_grad",
        {"problem": "power", "power": power, "x": x, "h": h},
        {"g": g, "analytic": float(analytic), "error": gradcheck.relative_error(analytic, g[0])},
    )


def check_params_case(case_id, tamper=None, factor=1.01):
    """The real check_params on the real setup of tests/test_gradcheck.py.

    tamper "tensor" scales one finished gradient. tamper "backward" scales what block 1 hands back to
    block 0, for this call only. Neither touches the repository: the patch is undone before returning.
    """
    cfg, params, ids, targets, rng = tiny_setup()
    calls = [0]

    def loss_fn(p):
        calls[0] += 1
        return transformer.loss_and_grads(p, ids, targets, cfg)[0]

    real_block_backward = transformer.block_backward

    def scaled_block_backward(dY, c, p, l, config_, grads_):
        result = real_block_backward(dY, c, p, l, config_, grads_)
        return result * factor if l == 1 else result

    if tamper == "backward":
        transformer.block_backward = scaled_block_backward
    try:
        _, grads = transformer.loss_and_grads(params, ids, targets, cfg)
    finally:
        transformer.block_backward = real_block_backward
    if tamper == "tensor":
        grads["b1.wq"] = grads["b1.wq"] * factor

    worst = gradcheck.check_params(loss_fn, params, grads, rng)
    return case(
        case_id,
        "gradcheck.check_params",
        {"tamper": tamper or "none", "factor": factor if tamper else 1.0},
        {"worst": worst, "names": list(worst), "loss_evaluations": calls[0]},
    )


def gradcheck_file_cases():
    x = arr([1.0, -2.0, 3.0])
    return [
        # The test's own function and input.
        case(
            "numerical_grad.test_quadratic",
            "gradcheck.numerical_grad",
            {"problem": "quadratic", "x": x},
            {"g": gradcheck.numerical_grad(lambda v: float((v ** 2).sum()), x.copy())},
        ),
        power_case("numerical_grad.square_default_h", 2, 1.5, 1e-5),
        power_case("numerical_grad.square_large_h", 2, 1.5, 1e-1),
        power_case("numerical_grad.cube_default_h", 3, 1.5, 1e-5),
        power_case("numerical_grad.cube_large_h", 3, 1.5, 1e-1),
        power_case("numerical_grad.cube_tiny_h", 3, 1.5, 1e-13),
        relative_error_case("relative_error.close", 1.0, 1.0001),
        relative_error_case("relative_error.equal", 2.5, 2.5),
        relative_error_case("relative_error.opposite_signs", 1.0, -1.0),
        relative_error_case("relative_error.both_zero", 0.0, 0.0),
        relative_error_case("relative_error.guard_used", 1e-13, 0.0),
        relative_error_case("relative_error.tiny_against_small", 1e-17, 4e-12),
        check_params_case("check_params.correct"),
        check_params_case("check_params.one_tensor_wrong", "tensor"),
        check_params_case("check_params.block_output_wrong", "backward"),
    ]


# Writing and checking -----------------------------------------------------------------------------


# transformer.py: feed-forward, block and whole model ----------------------------------------------

# tests/test_transformer.py uses these values in test_ffn_worked_example. dOut is hand-picked.
FFN_PARAMS = {
    "b0.ff1_w": arr([[1.0, 0.0, -1.0], [0.0, 1.0, 1.0]]),
    "b0.ff1_b": arr([0.0, 0.5, 1.0]),
    "b0.ff2_w": arr([[1.0, 2.0], [0.0, 1.0], [1.0, -1.0]]),
    "b0.ff2_b": arr([0.0, 0.0]),
}
FFN_XN = arr([[1.0, -1.0]])
FFN_DOUT = arr([[1.0, 0.5]])

# The tiny model of the Transformer Block and Full Model pages. Its parameters are random draws from
# the real init_params, not learned values. X, dY, ids and targets are hand-picked.
TINY_CFG = transformer.ModelConfig(vocab_size=5, context_len=4, emb_dim=4, num_heads=2, ff_dim=6, num_layers=2)
TINY_X = arr([[0.5, -1.0, 1.5, 0.25], [1.0, 0.0, -0.5, 2.0], [-1.5, 0.75, 0.5, -0.25]])
TINY_DY = arr([[0.1, -0.2, 0.3, 0.05], [-0.1, 0.25, 0.15, -0.3], [0.2, 0.1, -0.25, 0.15]])
TINY_IDS = np.array([2, 0, 2])
TINY_TARGETS = np.array([0, 2, 1])
BLOCK_CHECKED = ["b0.ln1_g", "b0.wq", "b0.wo", "b0.ln2_b", "b0.ff1_w", "b0.ff2_w"]
MODEL_CHECKED = ["tok_emb", "pos_emb", "b0.wk", "b0.ff1_w", "b1.ln2_g", "b1.wq", "lnf_g", "head_w", "head_b"]


def tiny_params():
    return transformer.init_params(TINY_CFG, np.random.default_rng(config.seed))


def block_params(params, l):
    return {name: value for name, value in params.items() if name.startswith(f"b{l}.")}


def cfg_inputs(cfg):
    return {
        "vocab_size": cfg.vocab_size,
        "context_len": cfg.context_len,
        "emb_dim": cfg.emb_dim,
        "num_heads": cfg.num_heads,
        "ff_dim": cfg.ff_dim,
        "num_layers": cfg.num_layers,
        "ln_eps": cfg.ln_eps,
    }


def ffn_cases():
    out, cache = transformer.ffn_forward(FFN_XN, FFN_PARAMS, "b0.")
    grads = {name: np.zeros_like(value) for name, value in FFN_PARAMS.items()}
    dXn, history = traced(transformer.ffn_backward, ["dA", "dZ"], FFN_DOUT, cache, FFN_PARAMS, "b0.", grads)
    numeric = {}
    for name in FFN_PARAMS:
        trial = {key: value.copy() for key, value in FFN_PARAMS.items()}
        numeric[name] = gradcheck.numerical_grad(
            lambda v: float((transformer.ffn_forward(FFN_XN, {**trial, name: v}, "b0.")[0] * FFN_DOUT).sum()),
            trial[name],
        )
    numeric["Xn"] = gradcheck.numerical_grad(
        lambda v: float((transformer.ffn_forward(v, FFN_PARAMS, "b0.")[0] * FFN_DOUT).sum()),
        FFN_XN.copy(),
    )
    inputs = {"Xn": FFN_XN, "params": FFN_PARAMS, "dOut": FFN_DOUT}
    return [
        case(
            "ffn_forward.test_file",
            "transformer.ffn_forward",
            {"Xn": FFN_XN, "params": FFN_PARAMS},
            {"out": out, "Z": cache["Z"], "A": cache["A"], "cache_keys": list(cache)},
        ),
        case(
            "ffn_backward.test_file",
            "transformer.ffn_backward",
            inputs,
            {"dA": history["dA"][-1], "dZ": history["dZ"][-1], "dXn": dXn, "grads": grads},
        ),
        case("ffn.numeric", "gradcheck.numerical_grad", {"problem": "ffn", **inputs}, {"numeric": numeric}),
    ]


def block_forward_case():
    params = tiny_params()
    (Y, cache), history = traced(
        transformer.block_forward, ["Xn", "att", "X1", "Xn2", "ff"], TINY_X, params, 0, TINY_CFG
    )
    return case(
        "block_forward.tiny",
        "transformer.block_forward",
        {"X": TINY_X, "params": block_params(params, 0), "l": 0, "cfg": cfg_inputs(TINY_CFG)},
        {
            **{name: history[name][-1] for name in history},
            "Y": Y,
            "cache_keys": {"block": list(cache), "att": list(cache["att"]), "ff": list(cache["ff"])},
        },
    )


def block_backward_run(params, l=0):
    _, cache = transformer.block_forward(TINY_X, params, l, TINY_CFG)
    grads = {name: np.zeros_like(value) for name, value in params.items()}
    names = ["dXn2", "dX1_branch", "dX1", "dXn", "dX_branch"]
    dX, history = traced(transformer.block_backward, names, TINY_DY, cache, params, l, TINY_CFG, grads)
    return {
        **{name: history[name][-1] for name in names},
        "dX": dX,
        "grads": block_params(grads, l),
    }


def block_backward_cases():
    params = tiny_params()
    # With both LayerNorm scales at 0, each branch sends back exactly 0: only the identity path is left.
    off = {**params, "b0.ln1_g": np.zeros(TINY_CFG.emb_dim), "b0.ln2_g": np.zeros(TINY_CFG.emb_dim)}
    numeric = {}
    for name in BLOCK_CHECKED:
        trial = {key: value.copy() for key, value in params.items()}
        numeric[name] = gradcheck.numerical_grad(
            lambda v: float((transformer.block_forward(TINY_X, {**trial, name: v}, 0, TINY_CFG)[0] * TINY_DY).sum()),
            trial[name],
        )
    numeric["X"] = gradcheck.numerical_grad(
        lambda v: float((transformer.block_forward(v, params, 0, TINY_CFG)[0] * TINY_DY).sum()),
        TINY_X.copy(),
    )
    inputs = {"X": TINY_X, "dY": TINY_DY, "l": 0, "cfg": cfg_inputs(TINY_CFG)}
    return [
        case(
            "block_backward.tiny",
            "transformer.block_backward",
            {**inputs, "params": block_params(params, 0)},
            block_backward_run(params),
        ),
        case(
            "block_backward.branches_off",
            "transformer.block_backward",
            {**inputs, "params": block_params(off, 0)},
            block_backward_run(off),
        ),
        case(
            "block.numeric",
            "gradcheck.numerical_grad",
            {"problem": "block", **inputs, "params": block_params(params, 0)},
            {"numeric": numeric},
        ),
    ]


def block_config_shapes_case():
    """The real block functions at the sizes in config.py. Only shapes are stored: the inputs are
    random draws, and V is hand-picked because it has no config value."""
    V = 7
    rng = np.random.default_rng(config.seed)
    T, d = config.seq_len, config.emb_dim
    cfg = transformer.ModelConfig(V, config.context_len, d, config.num_heads, config.ff_dim, config.num_layers)
    params = transformer.init_params(cfg, rng)
    X, dY = rng.normal(size=(T, d)), rng.normal(size=(T, d))
    (Y, cache), history = traced(transformer.block_forward, ["Xn", "att", "X1", "Xn2", "ff"], X, params, 0, cfg)
    grads = {name: np.zeros_like(value) for name, value in params.items()}
    names = ["dXn2", "dX1_branch", "dX1", "dXn", "dX_branch"]
    dX, back = traced(transformer.block_backward, names, dY, cache, params, 0, cfg, grads)
    ffn_names = ["dA", "dZ", "dW2", "db2", "dW1", "db1"]
    spare = {name: np.zeros_like(value) for name, value in params.items()}
    _, ffn_back = traced(transformer.ffn_backward, ffn_names, dY, cache["ff"], params, "b0.", spare)
    size = lambda value: list(value.shape)  # noqa: E731
    return case(
        "block.config_shapes",
        "transformer.block_forward",
        {
            "problem": "config_shapes",
            "seq_len": T,
            "emb_dim": d,
            "num_heads": config.num_heads,
            "ff_dim": config.ff_dim,
        },
        {
            "forward": {
                "X": size(X),
                **{name: size(history[name][-1]) for name in history},
                "Y": size(Y),
                "Z": size(cache["ff"]["Z"]),
                "A": size(cache["ff"]["A"]),
            },
            "backward": {
                "dY": size(dY),
                **{name: size(back[name][-1]) for name in names},
                "dX": size(dX),
                **{name: size(ffn_back[name][-1]) for name in ffn_names},
            },
        },
    )


def block_file_cases():
    return [*ffn_cases(), block_forward_case(), *block_backward_cases(), block_config_shapes_case()]


def forward_case():
    params = tiny_params()
    (logits, cache), history = traced(transformer.forward, ["X", "Xf"], params, TINY_IDS, TINY_CFG)
    return case(
        "forward.tiny",
        "transformer.forward",
        {"params": params, "ids": TINY_IDS, "cfg": cfg_inputs(TINY_CFG)},
        {
            "stages": history["X"],
            "Xf": history["Xf"][-1],
            "logits": logits,
            "cache_keys": list(cache),
            "block_caches": len(cache["blocks"]),
        },
    )


def loss_and_grads_case():
    params = tiny_params()
    (loss, grads), history = traced(
        transformer.loss_and_grads,
        ["probs", "dZ", "dXf", "dX"],
        params,
        TINY_IDS,
        TINY_TARGETS,
        TINY_CFG,
    )
    return case(
        "loss_and_grads.tiny",
        "transformer.loss_and_grads",
        {"params": params, "ids": TINY_IDS, "targets": TINY_TARGETS, "cfg": cfg_inputs(TINY_CFG)},
        {
            "loss": loss,
            "probs": history["probs"][-1],
            "dZ": history["dZ"][-1],
            "dXf": history["dXf"][-1],
            "dX": history["dX"],
            "grads": grads,
        },
    )


def call_log(function, *args):
    """The functions of transformer.py and ops.py that run inside function, in the order they start,
    with their depth. l is recorded for the block functions."""
    log = []
    depth = [0]
    files = {"transformer.py", "ops.py"}

    def profile(frame, event, _arg):
        code = frame.f_code
        if event not in ("call", "return") or Path(code.co_filename).name not in files:
            return
        if code.co_name.startswith("<"):
            return
        if event == "return":
            depth[0] -= 1
            return
        entry = {"name": code.co_name, "depth": depth[0]}
        if code.co_name in ("block_forward", "block_backward"):
            entry["l"] = frame.f_locals["l"]
        log.append(entry)
        depth[0] += 1

    sys.setprofile(profile)
    try:
        function(*args)
    finally:
        sys.setprofile(None)
    return log


def fill_log(params, ids, targets, cfg):
    """Which line of loss_and_grads or block_backward first made each entry of grads non-zero. A call
    into another function counts as the line that made it."""
    watched = {transformer.loss_and_grads.__code__, transformer.block_backward.__code__}
    filled = set()
    log = []

    def tracer(frame, _event, _arg):
        if frame.f_code not in watched:
            return None
        state = {"line": frame.f_lineno}

        def local(inner, _inner_event, _inner_arg):
            grads = inner.f_locals.get("grads")
            for key, value in (grads or {}).items():
                if key not in filled and np.any(value != 0):
                    filled.add(key)
                    log.append({"key": key, "function": inner.f_code.co_name, "line": state["line"]})
            state["line"] = inner.f_lineno
            return local

        return local

    previous = sys.gettrace()
    sys.settrace(tracer)
    try:
        transformer.loss_and_grads(params, ids, targets, cfg)
    finally:
        sys.settrace(previous)
    return log


def loss_and_grads_order_case():
    params = tiny_params()
    return case(
        "loss_and_grads.order",
        "transformer.loss_and_grads",
        {"problem": "order", "ids": TINY_IDS, "targets": TINY_TARGETS, "cfg": cfg_inputs(TINY_CFG)},
        {
            "calls": call_log(transformer.loss_and_grads, params, TINY_IDS, TINY_TARGETS, TINY_CFG),
            "fills": fill_log(params, TINY_IDS, TINY_TARGETS, TINY_CFG),
        },
    )


def model_numeric_case():
    params = tiny_params()
    numeric = {}
    for name in MODEL_CHECKED:
        trial = {key: value.copy() for key, value in params.items()}
        numeric[name] = gradcheck.numerical_grad(
            lambda v: transformer.loss_and_grads({**trial, name: v}, TINY_IDS, TINY_TARGETS, TINY_CFG)[0],
            trial[name],
        )
    return case(
        "model.numeric",
        "gradcheck.numerical_grad",
        {
            "problem": "model",
            "params": params,
            "ids": TINY_IDS,
            "targets": TINY_TARGETS,
            "cfg": cfg_inputs(TINY_CFG),
        },
        {"numeric": numeric},
    )


def model_config_shapes_case():
    """The real init_params, forward and loss_and_grads at the sizes in config.py. Only shapes are
    stored. V has no config value, so it is hand-picked."""
    V = 7
    cfg = transformer.ModelConfig(
        V, config.context_len, config.emb_dim, config.num_heads, config.ff_dim, config.num_layers
    )
    rng = np.random.default_rng(config.seed)
    params = transformer.init_params(cfg, rng)
    ids, targets = rng.integers(V, size=config.seq_len), rng.integers(V, size=config.seq_len)
    (_, grads), history = traced(
        transformer.loss_and_grads, ["logits", "probs", "dZ", "dXf", "dX"], params, ids, targets, cfg
    )
    size = lambda value: list(value.shape)  # noqa: E731
    return case(
        "model.config_shapes",
        "transformer.loss_and_grads",
        {
            "problem": "config_shapes",
            "V": V,
            "seq_len": config.seq_len,
            "context_len": config.context_len,
            "emb_dim": config.emb_dim,
            "ff_dim": config.ff_dim,
            "num_layers": config.num_layers,
        },
        {
            "params": {name: size(value) for name, value in params.items()},
            "grads": {name: size(value) for name, value in grads.items()},
            "logits": size(history["logits"][-1]),
            "probs": size(history["probs"][-1]),
            "dZ": size(history["dZ"][-1]),
            "dXf": size(history["dXf"][-1]),
            "dX": [size(value) for value in history["dX"]],
        },
    )


def model_file_cases():
    return [
        forward_case(),
        loss_and_grads_case(),
        loss_and_grads_order_case(),
        model_numeric_case(),
        model_config_shapes_case(),
    ]


# train.py -----------------------------------------------------------------------------------------

BATCH_X = np.array([[2, 0, 2], [1, 3, 4], [0, 0, 1]])
BATCH_Y = np.array([[0, 2, 1], [3, 4, 0], [0, 1, 2]])

# A hand-picked text for one real step, long enough to leave validation windows at T = 4.
LEARNING_TEXT = "the cat sat on the mat\nthe dog sat on the log\nthe cat and the dog sat on the mat\nthe dog sat\n"
LEARNING_MERGES = 5
LEARNING_T = 4
LEARNING_B = 2


def learning_fixture(tok, tr_x, tr_y, seed_ids):
    """The stored batch and sample. They are not regenerated, so they are checked instead: each row must
    be a real window with its real targets, and the sample must be the seed, 30 ids, and their decoding."""
    fixture = json.loads(FIXTURE.read_text(encoding="ascii"))
    xb, yb = fixture["xb"], fixture["yb"]
    windows = set(zip(map(tuple, tr_x.to_numpy().tolist()), map(tuple, tr_y.to_numpy().tolist())))
    ids = fixture["sample_ids"]
    problems = [
        *([] if len(xb) == len(yb) == LEARNING_B else ["the batch has the wrong size"]),
        *([] if all((tuple(x), tuple(y)) in windows for x, y in zip(xb, yb)) else ["a row is not a real window"]),
        *([] if ids[: len(seed_ids)] == seed_ids else ["the sample does not start with the seed"]),
        *([] if len(ids) == len(seed_ids) + 30 else ["the sample does not have 30 new ids"]),
        *([] if tok.decode(ids) == fixture["sample_text"] else ["sample_text is not the decoding of sample_ids"]),
    ]
    if problems:
        raise ValueError(f"{FIXTURE.name}: {'; '.join(problems)}")
    return np.array(xb), np.array(yb), ids


def main_function():
    tree = ast.parse((ROOT / "train.py").read_text(encoding="utf-8"))
    return next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")


def real_print_lines(namespace):
    """Evaluate each print call of train.main with these values. The formats are the file's own."""
    printed = []
    calls = [
        node
        for node in ast.walk(main_function())
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "print"
    ]
    names = {**namespace, "print": lambda *args: printed.append(" ".join(str(arg) for arg in args))}
    for call in sorted(calls, key=lambda node: node.lineno):
        eval(compile(ast.Expression(call), "train.py", "eval"), names)  # noqa: S307
    return printed


def real_step_counts(windows, batch_size, epochs):
    """Run the two real assignments of train.main, on a stand-in for tr_x and for config."""
    wanted = {"steps_per_epoch", "total_steps"}
    statements = [
        node
        for node in main_function().body
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) in wanted
    ]
    assert len(statements) == 2
    names = {
        "math": math,
        "config": SimpleNamespace(batch_size=batch_size, epochs=epochs),
        "tr_x": range(windows),
    }
    exec(compile(ast.Module(statements, []), "train.py", "exec"), names)  # noqa: S102
    return names["steps_per_epoch"], names["total_steps"]


def step_counts_case():
    rows = []
    for windows, batch_size, epochs in [(1000, 16, 20), (1600, 16, 20), (1601, 16, 20), (15, 16, 2), (1, 16, 1)]:
        steps_per_epoch, total_steps = real_step_counts(windows, batch_size, epochs)
        frame = pd.DataFrame(np.arange(windows))
        sizes = [len(xb) for xb, _ in data.iterate_batches(frame, frame, batch_size, np.random.default_rng(config.seed))]
        rows.append(
            {
                "windows": windows,
                "batch_size": batch_size,
                "epochs": epochs,
                "steps_per_epoch": steps_per_epoch,
                "total_steps": total_steps,
                "yielded": len(sizes),
                "last_size": sizes[-1],
            }
        )
    return case("steps.counts", "train.main", {"problem": "step_counts"}, {"rows": rows})


def loop_order_case():
    """The statements of the batch loop in train.main, read from the file without running it."""
    loops = [
        node
        for node in ast.walk(main_function())
        if isinstance(node, ast.For) and "iterate_batches" in ast.unparse(node.iter)
    ]
    assert len(loops) == 1
    return case(
        "loop.order",
        "train.main",
        {"problem": "loop_order"},
        {"statements": [ast.unparse(node).splitlines()[0] for node in loops[0].body]},
    )


def batch_case(case_id, count):
    params = tiny_params()
    xb, yb = BATCH_X[:count], BATCH_Y[:count]
    total_loss, grads = train.batch_loss_and_grads(params, xb, yb, TINY_CFG)
    each = [transformer.loss_and_grads(params, x, y, TINY_CFG) for x, y in zip(xb, yb)]
    return case(
        case_id,
        "train.batch_loss_and_grads",
        {"params": params, "xb": xb, "yb": yb, "cfg": cfg_inputs(TINY_CFG)},
        {
            "loss": total_loss,
            "grads": grads,
            "losses": [loss for loss, _ in each],
            "head_b_each": [g["head_b"] for _, g in each],
        },
    )


def validation_case():
    params = tiny_params()
    inputs, targets = pd.DataFrame(BATCH_X), pd.DataFrame(BATCH_Y)
    value = train.validation_loss(params, TINY_CFG, inputs, targets)
    losses = [
        ops.cross_entropy_forward(transformer.forward(params, x, TINY_CFG)[0], y)[0]
        for x, y in zip(BATCH_X, BATCH_Y)
    ]
    return case(
        "validation.tiny",
        "train.validation_loss",
        {"params": params, "inputs": BATCH_X, "targets": BATCH_Y, "cfg": cfg_inputs(TINY_CFG)},
        {"loss": value, "losses": losses, "perplexity": ops.perplexity(value)},
    )


def learning_step_case():
    """One real step from text to updated parameters, calling what train.main calls in the same order,
    on a hand-picked text and a tiny model. train.main is not called, and nothing is saved. The batch and
    the sampled ids come from the fixture, so this case does not depend on the shuffle or on rng.choice."""
    rng = np.random.default_rng(config.seed)
    tok = tokenizer.build_tokenizer(LEARNING_TEXT, LEARNING_MERGES)
    ids = data.encode_corpus(LEARNING_TEXT, tok)
    train_ids, val_ids = data.split_ids(ids, config.val_fraction)
    tr_x, tr_y = data.make_windows(train_ids, LEARNING_T, config.stride)
    va_x, va_y = data.make_windows(val_ids, LEARNING_T, config.stride)
    cfg = transformer.ModelConfig(tok.vocab_size, LEARNING_T, 4, 2, 6, 1)
    params = transformer.init_params(cfg, rng)
    before = copy.deepcopy(params)
    opt = optim.Adam(params, **config_adam_settings())
    steps_per_epoch, total_steps = real_step_counts(len(tr_x), LEARNING_B, config.epochs)

    seed_ids = [int(i) for i in va_x.iloc[0][:3]]
    xb, yb, out = learning_fixture(tok, tr_x, tr_y, seed_ids)
    logits, _ = transformer.forward(params, xb[0], cfg)
    _, probs = ops.cross_entropy_forward(logits, yb[0])
    loss, grads = train.batch_loss_and_grads(params, xb, yb, cfg)
    averaged = copy.deepcopy(grads)
    norm = optim.clip_grad_norm(grads, config.grad_clip)
    step = 1
    lr = optim.lr_schedule(step, config.warmup_steps, total_steps, config.lr_max, config.lr_min)
    opt.step(params, grads, lr)

    v = train.validation_loss(params, cfg, va_x, va_y)
    printed = real_print_lines(
        {
            "math": math, "cfg": cfg, "step": step, "loss": loss, "norm": norm, "lr": lr, "epoch": 0, "v": v,
            "perplexity": ops.perplexity, "tok": tok, "out": out, "prefix": "checkpoints/model", "repr": repr,
        }
    )
    return case(
        "learning.one_step",
        "train.batch_loss_and_grads",
        {
            "problem": "learning_step",
            "text": LEARNING_TEXT,
            "num_merges": LEARNING_MERGES,
            "seq_len": LEARNING_T,
            "batch_size": LEARNING_B,
            "val_fraction": config.val_fraction,
            "stride": config.stride,
            "epochs": config.epochs,
            "params": before,
            "settings": config_adam_settings(),
            "max_norm": config.grad_clip,
            "warmup": config.warmup_steps,
            "lr_max": config.lr_max,
            "lr_min": config.lr_min,
            "cfg": cfg_inputs(cfg),
        },
        {
            "tokens": tokenizer.apply_merges(LEARNING_TEXT, tok.merges),
            "ids": ids,
            "vocab_size": tok.vocab_size,
            "train_windows": len(tr_x),
            "val_windows": len(va_x),
            "xb": xb,
            "yb": yb,
            "logits": logits,
            "probs": probs,
            "loss": loss,
            "grads": averaged,
            "norm": norm,
            "clipped": grads,
            "steps_per_epoch": steps_per_epoch,
            "total_steps": total_steps,
            "lr": lr,
            "params": params,
            "val_loss": v,
            "seed_ids": seed_ids,
            "printed": printed,
        },
    )


def train_file_cases():
    return [
        batch_case("batch.three_windows", 3),
        batch_case("batch.one_window", 1),
        validation_case(),
        step_counts_case(),
        loop_order_case(),
        learning_step_case(),
    ]


CHECKPOINT_TEXT = "low low low lower"


class CapturedFile(io.StringIO):
    def close(self):
        self.text = self.getvalue()
        super().close()


def checkpoint_case():
    """The real save_checkpoint, with np.savez, os.makedirs and open replaced so that nothing is
    written to disk. Everything the function would write is captured here instead."""
    tok = tokenizer.build_tokenizer(CHECKPOINT_TEXT, 2)
    cfg = transformer.ModelConfig(tok.vocab_size, 8, 4, 2, 8, 1)
    params = transformer.init_params(cfg, np.random.default_rng(0))

    writes = []
    folders = []
    arrays = {}

    def savez(path, **saved):
        writes.append(path)
        arrays.update(saved)

    def fake_open(path, mode, encoding):
        assert mode == "w" and encoding == "utf-8"
        writes.append(path)
        captured[path] = CapturedFile()
        return captured[path]

    captured = {}
    replaced = {
        "np": SimpleNamespace(savez=savez),
        "os": SimpleNamespace(path=os.path, makedirs=lambda folder, exist_ok: folders.append(folder)),
        "open": fake_open,
    }
    saved_names = {name: vars(checkpoint).get(name) for name in replaced}
    vars(checkpoint).update(replaced)
    try:
        checkpoint.save_checkpoint("checkpoints/model", params, cfg, tok)
    finally:
        for name, value in saved_names.items():
            if value is None:
                del vars(checkpoint)[name]
            else:
                vars(checkpoint)[name] = value

    return case(
        "checkpoint.tiny_model",
        "checkpoint.save_checkpoint",
        {"text": CHECKPOINT_TEXT, "max_merges": 2, "prefix": "checkpoints/model"},
        {
            "folders": folders,
            "writes": writes,
            "arrays": [{"key": key, "shape": list(array.shape)} for key, array in arrays.items()],
            "config_json": captured["checkpoints/model.json"].text,
            "tokenizer_json": captured["checkpoints/model.tok.json"].text,
        },
    )


def build():
    return {
        "checkpoint": [checkpoint_case()],
        "ops": ops_cases(),
        "attention": attention_cases(),
        "optim": optim_cases(),
        "sampling": sampling_cases(),
        "bpe": bpe_cases(),
        "params": params_cases(),
        "data": data_cases(),
        "gradcheck": gradcheck_file_cases(),
        "embeddings": embedding_cases(),
        "block": block_file_cases(),
        "model": model_file_cases(),
        "train": train_file_cases(),
    }


def comparable(case_data):
    """NumPy's default sort is not stable, and its tie order may differ between machines, so `order`
    is not compared. Where the cutoff splits a tie, the kept tokens are compared as sorted values.

    check_params reports differences of round-off, which depend on the machine's arithmetic library.
    Values below the test's 1e-4 are compared only as "below 1e-4", larger ones to 4 decimals, and the
    key biases, whose true gradient is 0, not at all: their value is pure round-off."""
    result = copy.deepcopy(case_data)
    outputs = result["outputs"]
    outputs.pop("order", None)
    if result["mirrors"] == "gradcheck.check_params":
        outputs["worst"] = {
            name: 0 if value < 1e-4 else round(value, 4)
            for name, value in outputs["worst"].items()
            if not name.endswith(".bk")
        }
    # Central differences on a whole block or model carry round-off that depends on the machine.
    if result["inputs"].get("problem") in ("ffn", "block", "model"):
        outputs["numeric"] = {
            name: np.round(np.array(value, dtype=float), 6).tolist() for name, value in outputs["numeric"].items()
        }
    if outputs.get("tie_split"):
        outputs["p"] = sorted(outputs["p"])
    return result


def same_number(a, b):
    if isinstance(a, int) and isinstance(b, int):
        return a == b
    return math.isclose(a, b, rel_tol=FLOAT_TOLERANCE, abs_tol=FLOAT_TOLERANCE)


def differences(path, a, b):
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return [] if same_number(a, b) else [f"{path}: stored {a}, computed {b}"]
    if isinstance(a, dict) and isinstance(b, dict):
        return [item for key in sorted({*a, *b}) for item in differences(f"{path}.{key}", a.get(key), b.get(key))]
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path}: stored length {len(a)}, computed {len(b)}"]
        return [item for i, (x, y) in enumerate(zip(a, b)) for item in differences(f"{path}[{i}]", x, y)]
    return [] if a == b else [f"{path}: stored {a}, computed {b}"]


def environment():
    return {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__}


def serialise(cases):
    document = {"environment": environment(), "cases": cases}
    return json.dumps(document, indent=1, ensure_ascii=True, allow_nan=False) + "\n"


def main():
    built = build()
    count = sum(len(cases) for cases in built.values())

    if "--check" not in sys.argv[1:]:
        for name, cases in built.items():
            (OUT / f"{name}.json").write_bytes(serialise(cases).encode("ascii"))
        print(f"Wrote {count} cases to data/golden/ in {len(built)} files.")
        return 0

    problems = []
    for name, cases in built.items():
        path = OUT / f"{name}.json"
        if not path.exists():
            problems.append(f"missing: {path.name}")
            continue
        document = json.loads(path.read_text(encoding="ascii"))
        if document.get("environment") != environment():
            print(f"Note: {path.name} was written with {document.get('environment')}, this run is {environment()}.")
        problems += differences(name, [comparable(c) for c in document["cases"]], [comparable(c) for c in cases])
    if problems:
        print("FAILED: stored golden data differs from a fresh run. Run: bun run sync:golden", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1
    print(f"OK: {count} golden cases match a fresh run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
