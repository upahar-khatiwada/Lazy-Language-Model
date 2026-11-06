#!/usr/bin/env python3
"""Record the real shape of every tensor the repository's functions hold, for verify:content.

  bun run sync:shapes     writes data/golden/shapes.json
  bun run verify:shapes   recomputes and fails if the stored file differs

Two runs of the same real calls, on random inputs: one with a tiny config whose sizes are all
different, so a swapped dimension cannot go unnoticed, and one with the sizes in config.py. Only
shapes are stored, never values. V, N and W do not come from config.py, so they are small hand-picked
sizes in both runs. Repository modules are imported read-only, nothing is written outside
data/golden/, and training is not run: train.main is never called.
"""

import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = Path(__file__).resolve().parents[1] / "data" / "golden" / "shapes.json"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import config  # noqa: E402
import data  # noqa: E402
import generate  # noqa: E402
import gradcheck  # noqa: E402
import ops  # noqa: E402
import optim  # noqa: E402
import tokenizer  # noqa: E402
import train  # noqa: E402
import transformer  # noqa: E402

# Hand-picked. Enough text for N to exceed every T used below.
TEXT = "the cat sat on the mat and the dog sat on the log\n" * 4

TINY = {"V": 11, "Tmax": 7, "T": 5, "d": 6, "H": 2, "F": 13, "L": 2, "B": 4}
CONFIG = {
    "V": 17,
    "Tmax": config.context_len,
    "T": config.seq_len,
    "d": config.emb_dim,
    "H": config.num_heads,
    "F": config.ff_dim,
    "L": config.num_layers,
    "B": config.batch_size,
}

TRACED = [
    ops.xavier_uniform,
    ops.linear_forward,
    ops.linear_backward,
    ops.relu_forward,
    ops.relu_backward,
    ops.softmax_forward,
    ops.softmax_backward,
    ops.cross_entropy_forward,
    ops.cross_entropy_backward,
    ops.layer_norm_forward,
    ops.layer_norm_backward,
    transformer.init_params,
    transformer.embed_forward,
    transformer.embed_backward,
    transformer.split_heads,
    transformer.merge_heads,
    transformer.attention_forward,
    transformer.attention_backward,
    transformer.ffn_forward,
    transformer.ffn_backward,
    transformer.block_forward,
    transformer.block_backward,
    transformer.forward,
    transformer.loss_and_grads,
    train.batch_loss_and_grads,
    train.validation_loss,
    data.encode_corpus,
    data.split_ids,
    data.make_windows,
    data.iterate_batches,
    optim.Adam.step,
    optim.clip_grad_norm,
    generate.sample_next,
    generate.generate,
    gradcheck.check_params,
]
NAMES = {
    function.__code__: f"{function.__module__}.{function.__qualname__}" for function in TRACED
}


def size(value):
    """The shape of an array or a DataFrame, otherwise None."""
    if isinstance(value, (np.ndarray, pd.DataFrame)):
        return list(value.shape)
    return None


def record(store, name, value):
    found = size(value)
    if found is not None:
        shapes = store.setdefault(name, [])
        if found not in shapes:
            shapes.append(found)


def record_locals(store, frame):
    for name, value in frame.f_locals.items():
        record(store, name, value)
        if isinstance(value, dict):
            for key, item in value.items():
                # Every block has the same shapes, so the first one stands for all of them.
                if not str(key).startswith("b1."):
                    record(store, f'{name}["{key}"]', item)
        if name == "self":
            for attribute, item in vars(value).items():
                if isinstance(item, dict):
                    for key, inner in item.items():
                        if not str(key).startswith("b1."):
                            record(store, f'self.{attribute}["{key}"]', inner)


def run(dims):
    T, d, H, F, L, V, Tmax, B = (dims[name] for name in ("T", "d", "H", "F", "L", "V", "Tmax", "B"))
    rng = np.random.default_rng(0)
    functions = {}

    def tracer(frame, _event, _arg):
        name = NAMES.get(frame.f_code)
        if name is None:
            return None
        store = functions.setdefault(name, {})

        def on_event(inner, _inner_event, _inner_arg):
            record_locals(store, inner)
            return on_event

        return on_event

    cfg = transformer.ModelConfig(
        vocab_size=V, context_len=Tmax, emb_dim=d, num_heads=H, ff_dim=F, num_layers=L
    )
    tok = tokenizer.build_tokenizer(TEXT, 5)
    window = rng.integers(V, size=T)
    target = rng.integers(V, size=T)

    previous = sys.gettrace()
    sys.settrace(tracer)
    try:
        params = transformer.init_params(cfg, rng)
        _, grads = transformer.loss_and_grads(params, window, target, cfg)
        logits, _ = transformer.forward(params, window, cfg)

        ids = data.encode_corpus(TEXT, tok)
        data.split_ids(ids, config.val_fraction)
        tr_x, tr_y = data.make_windows(ids, T, config.stride)
        xb, yb = next(data.iterate_batches(tr_x, tr_y, B, rng))
        # Random ids in range for this model: the windows above only give the data shapes.
        batch_x = rng.integers(V, size=xb.shape)
        batch_y = rng.integers(V, size=yb.shape)
        train.batch_loss_and_grads(params, batch_x, batch_y, cfg)
        # All W windows, as train.main passes them. Ids in range for this model, as above.
        val_x = pd.DataFrame(rng.integers(V, size=tr_x.shape))
        val_y = pd.DataFrame(rng.integers(V, size=tr_y.shape))
        train.validation_loss(params, cfg, val_x, val_y)

        ops.linear_forward(rng.normal(size=(T, d)), rng.normal(size=(d, F)), rng.normal(size=F))
        adam = optim.Adam(params, weight_decay=config.weight_decay)
        adam.step(params, grads, config.lr_max)
        optim.clip_grad_norm(grads, config.grad_clip)
        generate.sample_next(logits[-1], config.temperature, 2, config.top_p, rng)
        # A seed of T ids, so forward sees exactly T: a shorter one would record other lengths too.
        seed_ids = [int(token) for token in window]
        generate.generate(params, cfg, seed_ids, 2, T, config.temperature, 0, config.top_p, rng)
        gradcheck.check_params(
            lambda current: transformer.loss_and_grads(current, window, target, cfg)[0],
            params,
            grads,
            rng,
            samples_per_param=1,
        )
    finally:
        sys.settrace(previous)

    return {
        "dims": {**dims, "dh": d // H, "N": len(ids), "W": len(tr_x)},
        "functions": {name: dict(sorted(functions[name].items())) for name in sorted(functions)},
    }


def build():
    return {
        "note": "Shapes from real calls on random inputs. No value is stored. Written by scripts/make_shapes.py.",
        "runs": {"tiny": run(TINY), "config": run(CONFIG)},
    }


def text(value):
    return json.dumps(value, indent=1, ensure_ascii=True) + "\n"


def main():
    fresh = text(build())
    if "--check" in sys.argv[1:]:
        if not OUT.exists() or OUT.read_text(encoding="ascii") != fresh:
            print("FAILED: data/golden/shapes.json differs from a fresh run. Run: bun run sync:shapes", file=sys.stderr)
            return 1
        print("OK: data/golden/shapes.json matches a fresh run.")
        return 0
    OUT.write_bytes(fresh.encode("ascii"))
    print("Wrote data/golden/shapes.json.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
