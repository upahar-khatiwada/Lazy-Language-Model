#!/usr/bin/env python3
"""Run the repository's real functions on fixed small inputs and store the results.

  bun run sync:golden     writes data/golden/{ops,attention,optim,sampling,bpe}.json
  bun run verify:golden   recomputes and fails if the stored numbers differ

tests/math.golden.test.ts compares lib/math with these files. Inputs are hand-picked, so every case
is an Illustrative Example. Repository modules are imported read-only, nothing is written outside
data/golden/, and training is not run: train and data are not imported.
"""

import copy
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = Path(__file__).resolve().parents[1] / "data" / "golden"

import numpy as np  # noqa: E402

import config  # noqa: E402
import generate  # noqa: E402
import ops  # noqa: E402
import optim  # noqa: E402
import tokenizer  # noqa: E402
import transformer  # noqa: E402

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

X23 = arr([[1.0, 2.0], [-1.0, 0.5], [0.0, 3.0]])
W23 = arr([[0.5, -1.0, 2.0], [1.5, 0.25, -0.5]])
B3 = arr([0.1, -0.2, 0.3])
LOGITS = arr([[2.0, 1.0, 0.0], [0.5, 0.5, 3.0]])
LN_X = arr([[1.0, 2.0, 3.0, 6.0], [-2.0, 0.0, 0.5, 1.5], [4.0, 4.0, 4.0, 4.0]])
LN_G = arr([1.0, 0.5, 2.0, -1.0])
LN_B = arr([0.0, 0.25, -0.5, 1.0])
LN_DY = arr([[0.1, -0.2, 0.3, 0.4], [1.0, 0.5, -0.5, 0.0], [0.2, 0.2, -0.1, 0.3]])


def softmax_case(case_id, x):
    x = arr(x)
    return case(case_id, "ops.softmax_forward", {"x": x}, {"y": ops.softmax_forward(x)})


def cross_entropy_cases(case_id, logits, targets):
    targets = np.array(targets)
    loss, probs = ops.cross_entropy_forward(logits, targets)
    return [
        case(
            case_id,
            "ops.cross_entropy_forward",
            {"logits": logits, "targets": targets},
            {"loss": loss, "probs": probs},
        ),
        case(
            f"{case_id}.backward",
            "ops.cross_entropy_backward",
            {"probs": probs, "targets": targets},
            {"dZ": ops.cross_entropy_backward(probs, targets)},
        ),
    ]


def layer_norm_cases(case_id, epsilon):
    keyword = {} if epsilon is None else {"epsilon": epsilon}
    Y, (Xhat, s) = ops.layer_norm_forward(LN_X, LN_G, LN_B, **keyword)
    dX, dgamma, dbeta = ops.layer_norm_backward(LN_DY, (Xhat, s), LN_G)
    inputs = {"X": LN_X, "gamma": LN_G, "beta": LN_B}
    if epsilon is not None:
        inputs["epsilon"] = epsilon
    return [
        case(case_id, "ops.layer_norm_forward", inputs, {"Y": Y, "Xhat": Xhat, "s": s}),
        case(
            f"{case_id}.backward",
            "ops.layer_norm_backward",
            {"dY": LN_DY, "Xhat": Xhat, "s": s, "gamma": LN_G},
            {"dX": dX, "dgamma": dgamma, "dbeta": dbeta},
        ),
    ]


def ops_cases():
    dY = arr([[1.0, 0.0, -1.0], [0.5, 2.0, 0.25], [-0.5, 1.0, 1.5]])
    dX, dW, db = ops.linear_backward(X23, W23, dY)
    Z = arr([[-1.0, 0.0, 2.0], [3.0, -0.5, 0.0]])
    dA = arr([[0.5, 0.25, -1.0], [2.0, 4.0, 8.0]])
    y = ops.softmax_forward(arr([2.0, 1.0, 0.0]))
    y_rows = ops.softmax_forward(LOGITS)
    dY_vec = arr([0.5, -1.0, 2.0])
    dY_rows = arr([[0.5, -1.0, 2.0], [1.0, 1.0, -3.0]])
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


def bpe_cases():
    words = [["low", 5], ["lower", 2], ["newest", 6], ["widest", 3]]
    return [
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


# Writing and checking -----------------------------------------------------------------------------


def build():
    return {
        "ops": ops_cases(),
        "attention": attention_cases(),
        "optim": optim_cases(),
        "sampling": sampling_cases(),
        "bpe": bpe_cases(),
    }


def comparable(case_data):
    """NumPy's default sort is not stable, and its tie order may differ between machines, so `order`
    is not compared. Where the cutoff splits a tie, the kept tokens are compared as sorted values."""
    result = copy.deepcopy(case_data)
    outputs = result["outputs"]
    outputs.pop("order", None)
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
