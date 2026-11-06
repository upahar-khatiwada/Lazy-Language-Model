#!/usr/bin/env python3
"""Snapshots that pages need and that data/generated does not hold.

  bun run sync:derived     writes data/derived/{init-params,math-inventory,repo-inventory,absent-techniques,discrepancies}.json
  bun run verify:derived   recomputes and fails if the stored files differ

init-params.json comes from calling transformer.init_params with prime-sized dimensions, so each
array size names exactly one symbol. Repository modules are imported read-only and training is not
run. math-inventory.json holds the rows of contract/MATH_INVENTORY.md, repo-inventory.json the file
roles and symbol purposes of contract/REPO_INVENTORY.md, absent-techniques.json the confirmed-absent
rows of contract/PDF_VS_CODE.md, and discrepancies.json the text of contract/DISCREPANCIES.md:
contract/ is git-ignored, so a fresh clone can build without it. Without contract/, those files are
left alone and not compared.
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
PDF_VS_CODE = SITE / "contract" / "PDF_VS_CODE.md"
DISCREPANCIES = SITE / "contract" / "DISCREPANCIES.md"

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


def table_cells(line):
    """The cells of one Markdown table row. A pipe escaped as \\| belongs to its cell."""
    cells = re.split(r"(?<!\\)\|", line.strip().strip("|"))
    return [cell.strip().replace("\\|", "|") for cell in cells]


def is_table_rule(line):
    return re.fullmatch(r"\|[\s:|-]+\|", line.strip()) is not None


def absent_techniques_snapshot():
    """Section 3 of contract/PDF_VS_CODE.md, keeping the rows confirmed absent: In code is `no`."""
    in_section = False
    rows = []
    for line in PDF_VS_CODE.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            in_section = line.startswith("## 3.")
            continue
        if not in_section or not line.startswith("|") or is_table_rule(line):
            continue
        cells = table_cells(line)
        if cells[0] == "Technique":
            if cells != ["Technique", "In PDF", "In code", "Used", "Document", "Note"]:
                raise SystemExit(f"contract/PDF_VS_CODE.md section 3 changed its columns: {cells}")
            continue
        if cells[2] == "no":
            rows.append({"technique": cells[0], "inPdf": cells[1], "note": cells[5]})
    if not rows:
        raise SystemExit("contract/PDF_VS_CODE.md section 3 has no confirmed-absent rows")
    return {"rows": rows}


def discrepancies_snapshot():
    """contract/DISCREPANCIES.md, section by section, as paragraphs, lists and tables."""
    sections = []
    blocks = None
    lines = DISCREPANCIES.read_text(encoding="utf-8").splitlines()
    preamble = []
    index = 0
    while index < len(lines):
        line = lines[index]
        heading = re.fullmatch(r"## (\d+)\. (.+)", line)
        if heading:
            blocks = []
            sections.append({"number": int(heading[1]), "title": heading[2], "blocks": blocks})
        elif line.startswith("# ") or not line.strip():
            pass
        elif line.startswith("|"):
            headers = table_cells(line)
            rows = []
            index += 2  # the header, then the rule under it
            while index < len(lines) and lines[index].startswith("|"):
                rows.append(table_cells(lines[index]))
                index += 1
            blocks.append({"type": "table", "headers": headers, "rows": rows})
            continue
        elif re.match(r"\d+\. ", line):
            items = []
            while index < len(lines) and re.match(r"\d+\. ", lines[index]):
                items.append(re.sub(r"^\d+\. ", "", lines[index]))
                index += 1
            blocks.append({"type": "list", "items": items})
            continue
        elif blocks is None:
            preamble.append(line)
        else:
            blocks.append({"type": "paragraph", "text": line})
        index += 1
    if not sections:
        raise SystemExit("contract/DISCREPANCIES.md has no numbered sections")
    return {"preamble": preamble, "sections": sections}


def text(value):
    return json.dumps(value, indent=2, ensure_ascii=False) + "\n"


def main():
    files = {"init-params.json": init_params_snapshot()}
    if PDF_VS_CODE.exists():
        files["absent-techniques.json"] = absent_techniques_snapshot()
    else:
        print("contract/PDF_VS_CODE.md is missing: absent-techniques.json was not recomputed.")
    if DISCREPANCIES.exists():
        files["discrepancies.json"] = discrepancies_snapshot()
    else:
        print("contract/DISCREPANCIES.md is missing: discrepancies.json was not recomputed.")
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
