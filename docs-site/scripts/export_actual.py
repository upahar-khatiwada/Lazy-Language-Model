#!/usr/bin/env python3
"""Export real outputs of a trained checkpoint to data/generated/actual, for Actual Model Output panels.

The checkpoint and the corpus are only read. Without a checkpoint this writes actualAvailable: false,
says how to create one, and exits 0: it never trains unless --train is given. --train runs the
repository's own train.main with its prefix inside docs-site/.cache, never in checkpoints/.
"""

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

DOCS = Path(__file__).resolve().parents[1]
REPO = DOCS.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(DOCS / "scripts"))

import config  # noqa: E402
from checkpoint import load_checkpoint  # noqa: E402
from data import encode_corpus, load_corpus, make_windows, split_ids  # noqa: E402
from extract_repo import serialize  # noqa: E402
from generate import sample_next  # noqa: E402
from ops import cross_entropy_forward, perplexity, softmax_forward  # noqa: E402
from tokenizer import build_tokenizer  # noqa: E402
from transformer import count_params, embed_forward, forward  # noqa: E402

OUT = DOCS / "data" / "generated" / "actual"
OUTPUT_FILES = ("meta", "tokenizer", "embeddings", "attention", "probabilities", "losses")
SCHEMA = 1

DEFAULT_PREFIX = "checkpoints/model"
CACHE_DIR = "docs-site/.cache"
CACHE_PREFIX = f"{CACHE_DIR}/model"
SUFFIXES = (".npz", ".json", ".tok.json")

# Hand-picked. These choose what to look at. Every number below them comes from running the model.
PROMPT = "Once there was a little "
SENTENCE_COUNT = 3
SENTENCE_LENGTH = (30, 90)
LOSS_WINDOWS = 5
TOP = 10
MERGES_SHOWN = 10

# The rows sum to 1 up to round-off.
ROW_SUM_TOLERANCE = 1e-9

# Decision D1 in contract/DISCREPANCIES.md, for the repository's own checkpoint.
D1_DISCLOSURE = (
    "Read from {prefix}, an ignored local file, trained on {corpus} (git-ignored, and not at the "
    "pinned commit), with training settings that by inference differ from config.py."
)
OTHER_DISCLOSURE = (
    "Read from {prefix}, a local file that is not in the repository, using the corpus {corpus}. "
    "The checkpoint does not record its training settings."
)


class ExportError(Exception):
    pass


def how_to_create():
    return [
        "Train with the repository's own code, keeping the result out of checkpoints/:",
        f"  python docs-site/scripts/export_actual.py --train   (writes {CACHE_PREFIX}.*)",
        "or run training yourself, then export from that prefix:",
        f'  python -c "import train; train.main(\'{CACHE_PREFIX}\')"',
        f"  python docs-site/scripts/export_actual.py --checkpoint {CACHE_PREFIX}",
        "At the committed config.py this is a long run, so nothing starts it for you.",
    ]


def unavailable(reason):
    return {
        "meta": {
            "schema": SCHEMA,
            "actualAvailable": False,
            "reason": reason,
            "howToCreate": how_to_create(),
        }
    }


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args):
    result = subprocess.run(
        ["git", "--no-optional-locks", "-C", str(REPO), *args],
        capture_output=True,
        encoding="utf-8",
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def checkpoint_files(prefix):
    return [Path(str(REPO / prefix) + suffix) for suffix in SUFFIXES]


def config_values():
    return {
        name: value
        for name, value in vars(config).items()
        if not name.startswith("_") and isinstance(value, (int, float, str))
    }


# The newest commit outside docs-site, the same rule as data/generated/repo-meta.json.
def provenance(prefix, cfg, corpus_path, tokenizer_matches):
    sha = git("log", "-1", "--format=%H", "--", ".", ":(exclude)docs-site")
    status = git("status", "--porcelain=v1", "--", ".", ":(exclude)docs-site")
    template = D1_DISCLOSURE if prefix == DEFAULT_PREFIX else OTHER_DISCLOSURE
    return {
        "gitSha": sha or "not available",
        "gitShortSha": sha[:7] if sha else "not available",
        "gitDirty": bool(status),
        "checkpointPrefix": prefix,
        "checkpointFiles": [
            {"name": path.name, "sha256": sha256(path), "bytes": path.stat().st_size}
            for path in checkpoint_files(prefix)
        ],
        "corpus": {"path": config.corpus_path, "sha256": sha256(corpus_path)},
        "tokenizerMatchesCorpus": tokenizer_matches,
        # config.py's seed. A checkpoint does not record the seed it was trained with.
        "seed": config.seed,
        "config": config_values(),
        "modelConfig": asdict(cfg),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "disclosure": template.format(prefix=prefix, corpus=config.corpus_path),
    }


def pick_sentences(text):
    low, high = SENTENCE_LENGTH
    lines = [line for line in text.split("\n") if low <= len(line) <= high]
    return lines[:SENTENCE_COUNT]


def tokenizer_data(tok, sentences):
    def one(sentence):
        ids = tok.encode(sentence)
        unk = tok.stoi["<unk>"]
        return {
            "text": sentence,
            "ids": ids,
            "tokens": [tok.itos[i] for i in ids],
            "unknown": sum(1 for i in ids if i == unk),
            "roundTrip": tok.decode(ids) == sentence,
        }

    return {
        "vocabSize": tok.vocab_size,
        "mergeCount": len(tok.merges),
        "merges": [
            {
                "rank": rank,
                "pair": list(merge["pair"]),
                "merged": merge["merged"],
                "frequency": merge["frequency"],
            }
            for rank, merge in enumerate(tok.merges[:MERGES_SHOWN])
        ],
        "sentences": [one(sentence) for sentence in sentences],
    }


def check_attention(A, T, H):
    """Refuses to export a matrix that breaks what attention_forward promises."""
    if A.shape != (H, T, T):
        raise ExportError(f"attention weights have shape {A.shape}, expected {(H, T, T)}")
    largest_gap = float(np.abs(A.sum(axis=-1) - 1.0).max())
    if largest_gap > ROW_SUM_TOLERANCE:
        raise ExportError(f"a row of attention weights is off 1 by {largest_gap}")
    mask = np.triu(np.ones((T, T), dtype=bool), 1)
    largest_masked = float(np.abs(A[:, mask]).max()) if mask.any() else 0.0
    if largest_masked != 0.0:
        raise ExportError(f"a masked attention weight is {largest_masked}, not 0")
    return largest_gap


def attention_data(params, cfg, tok, x, y):
    _, cache = forward(params, x, cfg)
    T = len(x)
    layers = []
    largest_gap = 0.0
    for l, block in enumerate(cache["blocks"]):
        A = block["att"]["A"]
        largest_gap = max(largest_gap, check_attention(A, T, cfg.num_heads))
        layers.append(
            {
                "layer": l,
                "heads": [{"head": h, "A": A[h].tolist()} for h in range(cfg.num_heads)],
            }
        )
    if len(layers) != cfg.num_layers:
        raise ExportError(f"{len(layers)} layers of attention, expected {cfg.num_layers}")
    return {
        "window": {
            "split": "validation",
            "index": 0,
            "ids": [int(i) for i in x],
            "tokens": [tok.itos[i] for i in x],
            "targets": [int(i) for i in y],
            "text": tok.decode([int(i) for i in x]),
        },
        "T": T,
        "H": cfg.num_heads,
        "L": cfg.num_layers,
        "dh": cfg.emb_dim // cfg.num_heads,
        "largestRowSumGap": largest_gap,
        "layers": layers,
    }


class SpyRng:
    """Records the distribution sample_next draws from, without drawing."""

    def __init__(self):
        self.p = None

    def choice(self, n, p):
        self.p = np.array(p)
        return 0


def distribution(p, tok):
    order = np.argsort(-p, kind="stable")[:TOP]
    order = order[p[order] > 0]  # a token sampling removed is not one of its top tokens
    return {
        "probabilities": p.tolist(),
        "sum": float(p.sum()),
        "kept": int((p > 0).sum()),
        "top": [
            {"id": int(i), "token": tok.itos[int(i)], "probability": float(p[i])} for i in order
        ],
        "otherTokens": float(1.0 - p[order].sum()),
    }


def probabilities_data(params, cfg, tok):
    ids = tok.encode(PROMPT)
    window = ids[-config.seq_len :]
    logits, _ = forward(params, np.array(window), cfg)
    spy = SpyRng()
    sample_next(logits[-1], config.temperature, config.top_k, config.top_p, spy)
    return {
        "prompt": PROMPT,
        "ids": ids,
        "tokens": [tok.itos[i] for i in ids],
        "plain": distribution(softmax_forward(logits[-1]), tok),
        "sampling": {
            "temperature": config.temperature,
            "topK": config.top_k,
            "topP": config.top_p,
            **distribution(spy.p, tok),
        },
    }


def losses_data(params, cfg, tok, windows):
    rows = []
    for split, (inputs, targets) in windows.items():
        count = len(inputs)
        for index in np.linspace(0, count - 1, LOSS_WINDOWS, dtype=int):
            x = inputs.iloc[int(index)].to_numpy()
            y = targets.iloc[int(index)].to_numpy()
            logits, _ = forward(params, x, cfg)
            loss, _ = cross_entropy_forward(logits, y)
            rows.append(
                {
                    "split": split,
                    "index": int(index),
                    "windows": count,
                    "text": tok.decode([int(i) for i in x]),
                    "loss": float(loss),
                    "perplexity": float(perplexity(loss)),
                }
            )
    return {"uniformLoss": float(np.log(cfg.vocab_size)), "rows": rows}


def export(prefix):
    missing = [path.name for path in checkpoint_files(prefix) if not path.is_file()]
    if missing:
        return unavailable(f"No checkpoint at {prefix}. Missing: {', '.join(missing)}.")

    corpus_path = REPO / config.corpus_path
    if not corpus_path.is_file():
        return unavailable(f"The checkpoint exists, but the corpus {config.corpus_path} does not.")

    params, cfg, tok = load_checkpoint(str(REPO / prefix))
    text = load_corpus(str(corpus_path), config.corpus_column, config.max_rows)
    ids = encode_corpus(text, tok)
    train_ids, val_ids = split_ids(ids, config.val_fraction)
    train_x, train_y = make_windows(train_ids, config.seq_len, config.stride)
    val_x, val_y = make_windows(val_ids, config.seq_len, config.stride)
    x = val_x.iloc[0].to_numpy()
    y = val_y.iloc[0].to_numpy()

    matches = build_tokenizer(text, config.num_merges).itos == tok.itos

    meta = {
        "schema": SCHEMA,
        "actualAvailable": True,
        "vocabSize": tok.vocab_size,
        "paramCount": count_params(params),
        "tensorCount": len(params),
        "provenance": provenance(prefix, cfg, corpus_path, matches),
    }
    return {
        "meta": meta,
        "tokenizer": tokenizer_data(tok, pick_sentences(text)),
        "embeddings": {
            "V": cfg.vocab_size,
            "d": cfg.emb_dim,
            "Tmax": cfg.context_len,
            "T": len(x),
            "tokEmb": list(params["tok_emb"].shape),
            "posEmb": list(params["pos_emb"].shape),
            "X0": list(embed_forward(params, x).shape),
        },
        "attention": attention_data(params, cfg, tok, x, y),
        "probabilities": probabilities_data(params, cfg, tok),
        "losses": losses_data(params, cfg, tok, {"validation": (val_x, val_y), "training": (train_x, train_y)}),
    }


def write(outputs, folder):
    folder.mkdir(parents=True, exist_ok=True)
    for name in OUTPUT_FILES:
        path = folder / f"{name}.json"
        if name in outputs:
            path.write_bytes(serialize(outputs[name]))
        else:
            path.unlink(missing_ok=True)  # a stale file must not outlive its checkpoint


def relative_prefix(prefix):
    path = Path(prefix)
    full = path if path.is_absolute() else REPO / path
    try:
        return full.resolve().relative_to(REPO).as_posix()
    except ValueError:
        raise ExportError(f"--checkpoint must be inside the repository: {prefix}") from None


def train(prefix):
    if not (REPO / prefix).resolve().is_relative_to((REPO / CACHE_DIR).resolve()):
        raise ExportError(f"--train writes only under {CACHE_DIR}, not to {prefix}")
    if all(path.is_file() for path in checkpoint_files(prefix)):
        print(f"{prefix}.* already exists, so nothing was trained.")
        return
    import os

    import train as repository_train  # imported only here: it is the repository's own training

    os.chdir(REPO)  # train.main reads config.corpus_path relative to the repository
    print(f"Training with the repository's train.main, prefix {prefix}.")
    print(f"This is the full run at config.py: {config.epochs} epochs on every window.")
    repository_train.main(prefix)


def check(prefix):
    stored = OUT / "meta.json"
    if not stored.is_file():
        print("FAILED: data/generated/actual/meta.json does not exist. Run: bun run export:actual")
        return 1
    if json.loads(stored.read_text(encoding="utf-8"))["actualAvailable"] and not all(
        path.is_file() for path in checkpoint_files(prefix)
    ):
        print(f"note: no checkpoint at {prefix} on this machine, so stored data was not compared.")
        return 0
    with tempfile.TemporaryDirectory() as temp:
        fresh = Path(temp)
        write(export(prefix), fresh)
        differing = [
            f"{name}.json"
            for name in OUTPUT_FILES
            if (fresh / f"{name}.json").is_file() != (OUT / f"{name}.json").is_file()
            or (
                (fresh / f"{name}.json").is_file()
                and (fresh / f"{name}.json").read_bytes() != (OUT / f"{name}.json").read_bytes()
            )
        ]
    if differing:
        print(f"FAILED: stale or edited: {', '.join(differing)}. Run: bun run export:actual")
        return 1
    print(f"OK: data/generated/actual matches a fresh export from {prefix}.")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", help=f"checkpoint prefix (default: {DEFAULT_PREFIX})")
    parser.add_argument(
        "--train",
        action="store_true",
        help=f"run the repository's training first, with the prefix under {CACHE_DIR}",
    )
    parser.add_argument("--check", action="store_true", help="compare stored data, write nothing")
    args = parser.parse_args()

    try:
        prefix = relative_prefix(
            args.checkpoint or (CACHE_PREFIX if args.train else DEFAULT_PREFIX)
        )
        if args.check:
            return check(prefix)
        if args.train:
            train(prefix)
        outputs = export(prefix)
    except ExportError as error:
        print(f"FAILED: {error}")
        return 1

    write(outputs, OUT)
    meta = outputs["meta"]
    if not meta["actualAvailable"]:
        print(meta["reason"])
        print("Wrote actualAvailable: false. The site builds, and Actual panels stay disabled.")
        for line in meta["howToCreate"]:
            print(line)
        return 0

    source = meta["provenance"]
    print(f"Wrote data/generated/actual from {prefix}, at commit {source['gitShortSha']}.")
    print(f"V={meta['vocabSize']}, {meta['paramCount']} parameters.")
    attention = outputs["attention"]
    print(
        f"Attention: {attention['L']} layers, {attention['H']} heads, T={attention['T']}, "
        f"largest row-sum gap {attention['largestRowSumGap']:.1e}, masked cells exactly 0."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
