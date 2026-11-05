#!/usr/bin/env python3
"""Fail if data/generated differs from a fresh extraction (stale or hand-edited data)."""

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True

from extract_repo import DEFAULT_OUT, DEFAULT_REPO, generate, serialize  # noqa: E402

# These describe the current checkout, not the data, so they are reported but not compared.
CHECKOUT_FIELDS = ("branch", "dirty", "dirtyFiles")


def compare_repo_meta(committed_bytes, fresh_bytes):
    """Returns (same, notes). Other fields and the formatting are still compared exactly."""
    try:
        committed = json.loads(committed_bytes)
        fresh = json.loads(fresh_bytes)
        notes = [
            f"note: repo-meta.json {name} is {committed[name]!r}, this checkout has {fresh[name]!r}"
            for name in CHECKOUT_FIELDS
            if committed[name] != fresh[name]
        ]
        for name in CHECKOUT_FIELDS:
            fresh[name] = committed[name]
    except (ValueError, KeyError, TypeError):
        return False, []
    return serialize(fresh) == committed_bytes, notes


def visible_files(folder):
    return {path.name for path in Path(folder).iterdir() if not path.name.startswith(".")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=DEFAULT_REPO, help="repository to read")
    parser.add_argument("--data", default=DEFAULT_OUT, help="committed generated data")
    args = parser.parse_args()

    data = Path(args.data)
    if not data.is_dir():
        print(f"FAILED: {data} does not exist. Run: bun run sync:repo")
        return 1

    problems = []
    with tempfile.TemporaryDirectory() as temp:
        generate(args.repo, temp)
        expected, actual = visible_files(temp), visible_files(data)
        problems += [f"missing: {name}" for name in sorted(expected - actual)]
        problems += [f"unexpected: {name}" for name in sorted(actual - expected)]
        for name in sorted(expected & actual):
            committed = (data / name).read_bytes()
            fresh = (Path(temp) / name).read_bytes()
            if name == "repo-meta.json":
                same, notes = compare_repo_meta(committed, fresh)
                for note in notes:
                    print(note)
            else:
                same = committed == fresh
            if not same:
                problems.append(f"differs: {name}")

    if problems:
        print("FAILED: generated data is stale or was edited by hand.")
        for problem in problems:
            print(f"  {problem}")
        print("Run: bun run sync:repo")
        return 1

    print(f"OK: {len(expected)} generated files match a fresh extraction.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
