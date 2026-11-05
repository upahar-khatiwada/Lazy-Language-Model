#!/usr/bin/env python3
"""Run repository functions on example inputs and report their real local values.

Reads a JSON list from stdin: [{"conceptId", "call", "input", "captures": [{"id", "capture"}]}].
Writes a JSON object to stdout: {conceptId: {"steps": {id: value}, "output": value}}.

Values come from the function's own frame, through sys.settrace, so no formula is repeated here.
Nothing is written to disk. sync_examples.ts starts this file; it is not used on its own.
"""

import importlib
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402


def to_json(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def from_json(value):
    return np.array(value, dtype=float) if isinstance(value, list) else value


def run_traced(function, kwargs):
    """Call function(**kwargs). Returns its return value and its locals at the moment it returned."""
    captured = {}

    def tracer(frame, event, _arg):
        if frame.f_code is not function.__code__:
            return None

        def on_line(inner, inner_event, _inner_arg):
            if inner_event == "return":
                captured.update(inner.f_locals)
            return on_line

        return on_line

    previous = sys.gettrace()
    sys.settrace(tracer)
    try:
        output = function(**kwargs)
    finally:
        sys.settrace(previous)
    return output, captured


def run_example(request):
    module_name, _, function_name = request["call"].rpartition(".")
    function = getattr(importlib.import_module(module_name), function_name)
    kwargs = {name: from_json(value) for name, value in request["input"].items()}
    output, local_values = run_traced(function, kwargs)

    steps = {}
    for step in request["captures"]:
        if step["capture"] not in local_values:
            raise KeyError(f'{request["call"]} has no local named {step["capture"]}')
        steps[step["id"]] = to_json(local_values[step["capture"]])
    return {"steps": steps, "output": to_json(output)}


def main():
    requests = json.load(sys.stdin)
    results = {request["conceptId"]: run_example(request) for request in requests}
    # allow_nan=False: a NaN or infinity is not valid JSON and must not be shown as a result.
    json.dump(results, sys.stdout, allow_nan=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
