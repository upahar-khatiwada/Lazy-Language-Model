#!/usr/bin/env python3
"""Snapshots that pages need and that data/generated does not hold.

  bun run sync:derived     writes data/derived/{init-params,math-inventory,repo-inventory}.json
  bun run verify:derived   recomputes and fails if the stored files differ

init-params.json comes from calling transformer.init_params with prime-sized dimensions, so each
array size names exactly one symbol. Repository modules are imported read-only and training is not
run. math-inventory.json holds the rows of contract/MATH_INVENTORY.md, and repo-inventory.json the
file roles and symbol purposes of contract/REPO_INVENTORY.md: contract/ is git-ignored, so a fresh
clone can build without it. Without contract/, those files are left alone and not compared.
"""

import json
import re
import sys
from pathlib import Path

sys.dont_write_bytecode = True
SITE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SITE.parent))
OUT = SITE / "data" / "derived"
INVENTORY = SITE / "contract" / "MATH_INVENTORY.md"
REPO_INVENTORY = SITE / "contract" / "REPO_INVENTORY.md"

import numpy as np  # noqa: E402

import transformer  # noqa: E402

PROBE = {"V": 101, "Tmax": 103, "d": 107, "F": 109, "L": 2}


def init_kind(array):
    if np.all(array == 1):
        return "ones"
    if np.all(array == 0):
        return "zeros"
    return "xavier_uniform"


def init_params_snapshot():
    cfg = transformer.ModelConfig(
        vocab_size=PROBE["V"],
        context_len=PROBE["Tmax"],
        emb_dim=PROBE["d"],
        num_heads=1,
        ff_dim=PROBE["F"],
        num_layers=PROBE["L"],
    )
    params = transformer.init_params(cfg, np.random.default_rng(0))
    symbol = {PROBE[name]: name for name in ("V", "Tmax", "d", "F")}

    rows = {}
    for key, array in params.items():
        block = re.fullmatch(r"b(\d+)\.(.+)", key)
        name = f"b{{l}}.{block[2]}" if block else key
        row = {
            "name": name,
            "perBlock": block is not None,
            "dims": [symbol[size] for size in array.shape],
            "init": init_kind(array),
        }
        if rows.setdefault(name, row) != row:
            raise SystemExit(f"blocks disagree on {name}")

    return {
        "probe": PROBE,
        "countParams": transformer.count_params(params),
        "parameters": list(rows.values()),
    }


def inventory_snapshot():
    rows = []
    group = None
    for line in INVENTORY.read_text(encoding="utf-8").splitlines():
        heading = re.match(r"## (.+)", line)
        if heading:
            group = heading[1]
        row = re.match(r"\| (M\d+) \| (.+?) \|", line)
        if row:
            rows.append({"id": row[1], "name": row[2], "group": group})
    if not rows:
        raise SystemExit("contract/MATH_INVENTORY.md has no rows")
    return {"rows": rows}


def repo_inventory_snapshot():
    """Section 1 gives each file's role; the tables under each ### heading give symbol purposes."""
    files = {}
    symbols = {}
    section = None
    module = None
    for line in REPO_INVENTORY.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            section = line
            module = None
            continue
        heading = re.match(r"### (\S+\.py)$", line)
        if heading:
            module = heading[1].removesuffix(".py").replace("/", ".")
            continue
        file_row = re.match(r"\| `(\S+\.py)` \| yes \| \d+ \| (.+?) \|$", line)
        if file_row and section and section.startswith("## 1."):
            files[file_row[1]] = file_row[2]
            continue
        symbol_row = re.match(r"\| `(.+?)` \| [\d-]+ \| (.+?) \|", line)
        if symbol_row and module:
            name = re.split(r"[(\s]", symbol_row[1].removeprefix("class "))[0]
            symbols[f"{module}.{name}"] = symbol_row[2]
    if not files or not symbols:
        raise SystemExit("contract/REPO_INVENTORY.md has no file or symbol rows")
    return {"files": files, "symbols": symbols}


def text(value):
    return json.dumps(value, indent=2, ensure_ascii=False) + "\n"


def main():
    files = {"init-params.json": init_params_snapshot()}
    if INVENTORY.exists():
        files["math-inventory.json"] = inventory_snapshot()
    else:
        print("contract/MATH_INVENTORY.md is missing: math-inventory.json was not recomputed.")
    if REPO_INVENTORY.exists():
        files["repo-inventory.json"] = repo_inventory_snapshot()
    else:
        print("contract/REPO_INVENTORY.md is missing: repo-inventory.json was not recomputed.")

    if "--check" in sys.argv:
        failed = False
        for name, value in files.items():
            path = OUT / name
            if not path.exists() or path.read_text(encoding="utf-8") != text(value):
                print(f"differs: data/derived/{name}. Run: bun run sync:derived")
                failed = True
        if failed:
            return 1
        print(f"OK: {len(files)} derived file(s) match a fresh run.")
        return 0

    OUT.mkdir(exist_ok=True)
    for name, value in files.items():
        (OUT / name).write_text(text(value), encoding="utf-8", newline="\n")
    print(f"Wrote {len(files)} file(s) to data/derived/.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
