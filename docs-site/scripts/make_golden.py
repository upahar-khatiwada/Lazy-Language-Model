#!/usr/bin/env python3
"""Run the repository's real functions on fixed small inputs and store the results.

  bun run sync:golden     writes data/golden/{ops,attention,optim,sampling,bpe,params,data,gradcheck,embeddings}.json
  bun run verify:golden   recomputes and fails if the stored numbers differ

tests/math.golden.test.ts compares lib/math with these files. Inputs are hand-picked, so every case
is an Illustrative Example. Repository modules are imported read-only, nothing is written outside
data/golden/, and training is not run: train is not imported. data.py needs pandas.
"""

import copy
import io
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = Path(__file__).resolve().parents[1] / "data" / "golden"

import numpy as np  # noqa: E402

import pandas as pd  # noqa: E402

import config  # noqa: E402
import data  # noqa: E402
import generate  # noqa: E402
import gradcheck  # noqa: E402
import ops  # noqa: E402
import optim  # noqa: E402
import tokenizer  # noqa: E402
import transformer  # noqa: E402
from tests.test_gradcheck import tiny_setup  # noqa: E402

TOLERANCE = 1e-12


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
    ]


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
    if history["order"]:
        order, keep = history["order"][-1], history["keep"][-1]
        before = outputs["after_top_k"]
        last = before[order[len(keep) - 1]]
        tie_split = bool(len(keep) < len(order) and last == before[order[len(keep)]] and last > 0)
        outputs.update(order=order, cum=history["cum"][-1], keep_count=len(keep), tie_split=tie_split)
    return case(case_id, "generate.sample_next", inputs, outputs)


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


def build():
    return {
        "ops": ops_cases(),
        "attention": attention_cases(),
        "optim": optim_cases(),
        "sampling": sampling_cases(),
        "bpe": bpe_cases(),
        "params": params_cases(),
        "data": data_cases(),
        "gradcheck": gradcheck_file_cases(),
        "embeddings": embedding_cases(),
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
    if outputs.get("tie_split"):
        outputs["p"] = sorted(outputs["p"])
    return result


def differences(path, a, b):
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return [] if abs(a - b) <= TOLERANCE else [f"{path}: stored {a}, computed {b}"]
    if isinstance(a, dict) and isinstance(b, dict):
        return [item for key in sorted({*a, *b}) for item in differences(f"{path}.{key}", a.get(key), b.get(key))]
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path}: stored length {len(a)}, computed {len(b)}"]
        return [item for i, (x, y) in enumerate(zip(a, b)) for item in differences(f"{path}[{i}]", x, y)]
    return [] if a == b else [f"{path}: stored {a}, computed {b}"]


def serialise(cases):
    return json.dumps({"cases": cases}, indent=1, ensure_ascii=True, allow_nan=False) + "\n"


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
        stored = json.loads(path.read_text(encoding="ascii"))["cases"]
        problems += differences(name, [comparable(c) for c in stored], [comparable(c) for c in cases])
    if problems:
        print("FAILED: stored golden data differs from a fresh run. Run: bun run sync:golden", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1
    print(f"OK: {count} golden cases match a fresh run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
