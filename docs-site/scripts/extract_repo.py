#!/usr/bin/env python3
"""Generate machine-readable facts about the repository into data/generated.

Static analysis only: repository code is parsed, never imported or run.
Standard library only. Output is deterministic: sorted keys, stable order, no timestamps.
"""

import argparse
import ast
import builtins
import io
import json
import operator
import re
import subprocess
import sys
import tokenize
from dataclasses import dataclass, field
from pathlib import Path

DOCS_SITE = Path(__file__).resolve().parents[1]
DEFAULT_REPO = DOCS_SITE.parent
DEFAULT_OUT = DOCS_SITE / "data" / "generated"
EXCLUDED_DIRS = {"docs-site", "__pycache__", "checkpoints", "node_modules"}
SECTION_RE = re.compile(r"^-+\s*(.+?)\s*-+$")
ARITHMETIC = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
FUNCTION_NODES = (ast.FunctionDef, ast.AsyncFunctionDef)


def fail(message):
    raise SystemExit(f"extract_repo: {message}")


def serialize(data):
    text = json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
    return (text + "\n").encode("utf-8")


# ---------- git ----------


def git(repo, *args):
    # --no-optional-locks: reading status must not refresh .git/index.
    result = subprocess.run(
        ["git", "--no-optional-locks", "-C", str(repo), *args],
        capture_output=True,
        encoding="utf-8",
    )
    if result.returncode != 0:
        fail(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout


def normalize_remote(url):
    url = url.strip()
    match = re.match(r"^git@([^:]+):(.+)$", url)
    if match:
        url = f"https://{match.group(1)}/{match.group(2)}"
    url = re.sub(r"^https://[^/@]+@", "https://", url)
    return url.removesuffix("/").removesuffix(".git")


def dirty_files(repo):
    entries = git(repo, "status", "--porcelain=v1", "-z", "--", ".", ":(exclude)docs-site")
    entries = entries.split("\0")
    paths = []
    index = 0
    while index < len(entries):
        entry = entries[index]
        index += 1
        if not entry:
            continue
        paths.append(entry[3:])
        if entry[0] in "RC":  # a rename or copy is followed by its old path
            index += 1
    return sorted(paths)


def build_repo_meta(repo):
    # The newest commit outside docs-site, not literally HEAD: a commit cannot contain its own
    # SHA, so pinning to HEAD would make committed data stale the moment it is committed.
    pinned = git(repo, "log", "-1", "--format=%H%n%cI", "--", ".", ":(exclude)docs-site").split()
    if len(pinned) != 2:
        fail("no commit found outside docs-site")
    sha, commit_date = pinned
    changed = dirty_files(repo)
    return {
        "sha": sha,
        "shortSha": sha[:7],
        "commitDate": commit_date,
        "branch": git(repo, "rev-parse", "--abbrev-ref", "HEAD").strip(),
        "dirty": bool(changed),
        "dirtyFiles": changed,
        "githubBaseUrl": normalize_remote(git(repo, "remote", "get-url", "origin")),
    }


# ---------- parsing ----------


@dataclass
class Module:
    path: str
    id: str
    text: str
    lines: list
    tree: ast.Module
    tokens: list
    comments: dict  # line -> (text, standalone)
    is_test: bool
    names: dict = field(default_factory=dict)  # local name -> (kind, target)
    fixtures: dict = field(default_factory=dict)  # fixture name -> symbol id


@dataclass
class Symbol:
    id: str
    name: str
    kind: str  # function, method or class
    node: ast.AST
    module: Module
    owner: str | None = None  # class id, for methods


def is_test_path(path):
    parts = path.split("/")
    return parts[0] == "tests" or parts[-1].startswith("test_") or parts[-1].endswith("_test.py")


def find_python_files(repo):
    found = []
    for path in repo.rglob("*.py"):
        parts = path.relative_to(repo).parts[:-1]
        if not any(part in EXCLUDED_DIRS or part.startswith(".") for part in parts):
            found.append(path)
    return sorted(found, key=lambda p: p.relative_to(repo).as_posix())


def load_module(repo, path):
    relative = path.relative_to(repo).as_posix()
    # Working trees may be CRLF (core.autocrlf); git stores LF. Line numbers are unaffected.
    text = path.read_bytes().decode("utf-8").replace("\r\n", "\n")
    tokens = list(tokenize.generate_tokens(io.StringIO(text).readline))
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    comments = {}
    for token in tokens:
        if token.type == tokenize.COMMENT:
            row, col = token.start
            standalone = lines[row - 1][:col].strip() == ""
            comments[row] = (token.string[1:].strip(), standalone)
    return Module(
        path=relative,
        id=relative.removesuffix(".py").replace("/", "."),
        text=text,
        lines=lines,
        tree=ast.parse(text, filename=relative),
        tokens=tokens,
        comments=comments,
        is_test=is_test_path(relative),
    )


def collect_symbols(modules):
    symbols = {}
    for module in modules:
        for stmt in module.tree.body:
            if isinstance(stmt, FUNCTION_NODES):
                sid = f"{module.id}.{stmt.name}"
                symbols[sid] = Symbol(sid, stmt.name, "function", stmt, module)
            elif isinstance(stmt, ast.ClassDef):
                sid = f"{module.id}.{stmt.name}"
                symbols[sid] = Symbol(sid, stmt.name, "class", stmt, module)
                for item in stmt.body:
                    if isinstance(item, FUNCTION_NODES):
                        mid = f"{sid}.{item.name}"
                        symbols[mid] = Symbol(mid, item.name, "method", item, module, owner=sid)
        # Every function must be indexed; refuse to silently drop a nested definition.
        indexed = {id(s.node) for s in symbols.values() if s.module is module}
        for node in ast.walk(module.tree):
            if isinstance(node, (*FUNCTION_NODES, ast.ClassDef)) and id(node) not in indexed:
                fail(f"{module.path}:{node.lineno}: nested definition {node.name} is not supported")
    return symbols


def collect_names(modules, symbols):
    known = {module.id for module in modules}
    for module in modules:
        for node in ast.walk(module.tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    local = alias.asname or alias.name.split(".")[0]
                    internal = alias.name in known and (alias.asname or "." not in alias.name)
                    module.names[local] = ("module" if internal else "external", alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    fail(f"{module.path}:{node.lineno}: relative imports are not supported")
                for alias in node.names:
                    local = alias.asname or alias.name
                    target = f"{node.module}.{alias.name}"
                    if target in symbols:
                        module.names[local] = ("symbol", target)
                    elif node.module in known:
                        module.names[local] = ("value", target)
                    else:
                        module.names[local] = ("external", target)
        for stmt in module.tree.body:
            if isinstance(stmt, (*FUNCTION_NODES, ast.ClassDef)):
                sid = f"{module.id}.{stmt.name}"
                module.names[stmt.name] = ("symbol", sid)
                is_function = isinstance(stmt, FUNCTION_NODES)
                if module.is_test and is_function and has_decorator(stmt, "fixture"):
                    module.fixtures[stmt.name] = sid


def has_decorator(node, wanted):
    for decorator in node.decorator_list:
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        name = target.attr if isinstance(target, ast.Attribute) else getattr(target, "id", "")
        if name == wanted:
            return True
    return False


# ---------- static call resolution ----------


def argument_nodes(function):
    args = function.args
    named = [*args.posonlyargs, *args.args, *args.kwonlyargs]
    return named + [arg for arg in (args.vararg, args.kwarg) if arg]


def local_names(function):
    names = {arg.arg for arg in argument_nodes(function)}
    for stmt in function.body:
        for node in ast.walk(stmt):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                names.add(node.id)
            elif isinstance(node, ast.arg):
                names.add(node.arg)
            elif isinstance(node, ast.ExceptHandler) and node.name:
                names.add(node.name)
    return names


def flatten_text(module, node):
    source = ast.get_source_segment(module.text, node) or ""
    return " ".join(source.split())


class Resolver:
    """Resolves calls without guessing: a receiver of unknown type stays unresolved.

    Types are ("class", id) or ("tuple", (types...)). They come from constructors, annotations,
    self, function returns (solved to a fixed point) and pytest fixture parameters.
    """

    def __init__(self, symbols):
        self.symbols = symbols
        self.returns = {}
        self.solve_returns()

    def callable_symbols(self):
        return [s for s in self.symbols.values() if s.kind != "class"]

    def solve_returns(self):
        for _ in range(len(self.symbols) + 2):
            changed = False
            for symbol in self.callable_symbols():
                inferred = self.infer_return(symbol)
                if self.returns.get(symbol.id) != inferred:
                    self.returns[symbol.id] = inferred
                    changed = True
            if not changed:
                return
        fail("return type inference did not converge")

    def infer_return(self, symbol):
        env, names = self.environment(symbol)
        found = []
        for stmt in symbol.node.body:
            for node in ast.walk(stmt):
                if isinstance(node, ast.Return):
                    found.append(self.infer(node.value, env, names, symbol.module))
        if found and found[0] is not None and all(item == found[0] for item in found):
            return found[0]
        return None

    def environment(self, symbol):
        function, module = symbol.node, symbol.module
        names = local_names(function)
        env = {}
        arguments = argument_nodes(function)
        for arg in arguments:
            annotated = arg.annotation
            if isinstance(annotated, ast.Name):
                kind, target = module.names.get(annotated.id, (None, None))
                if kind == "symbol" and self.symbols[target].kind == "class":
                    env[arg.arg] = ("class", target)
            if arg.arg in module.fixtures:
                env[arg.arg] = self.returns.get(module.fixtures[arg.arg])
        if symbol.kind == "method" and arguments:
            env[arguments[0].arg] = ("class", symbol.owner)

        statements = [node for stmt in function.body for node in ast.walk(stmt)]
        statements = [node for node in statements if isinstance(node, (ast.Assign, ast.For))]
        for node in sorted(statements, key=lambda n: (n.lineno, n.col_offset)):
            if isinstance(node, ast.For):
                self.bind(node.target, None, env)
            else:
                value = self.infer(node.value, env, names, module)
                for target in node.targets:
                    self.bind(target, value, env)
        return env, names

    def bind(self, target, value, env):
        if isinstance(target, ast.Name):
            env[target.id] = value
        elif isinstance(target, ast.Tuple):
            elements = value[1] if value and value[0] == "tuple" else ()
            for index, element in enumerate(target.elts):
                known = elements[index] if len(elements) == len(target.elts) else None
                self.bind(element, known, env)

    def infer(self, node, env, names, module):
        if isinstance(node, ast.Name):
            return env.get(node.id)
        if isinstance(node, ast.Tuple):
            return ("tuple", tuple(self.infer(e, env, names, module) for e in node.elts))
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
            value = self.infer(node.value, env, names, module)
            index = node.slice.value
            if value and value[0] == "tuple" and isinstance(index, int):
                if -len(value[1]) <= index < len(value[1]):
                    return value[1][index]
            return None
        if isinstance(node, ast.Call):
            status, edges = self.resolve_call(node.func, env, names, module)
            if status == "resolved":
                target = edges[0][0]
                if self.symbols[target].kind == "class":
                    return ("class", target)
                return self.returns.get(target)
        return None

    def edges_for(self, target):
        if self.symbols[target].kind != "class":
            return [(target, "call")]
        edges = [(target, "instantiate")]
        if f"{target}.__init__" in self.symbols:
            edges.append((f"{target}.__init__", "instantiate"))
        return edges

    def resolve_call(self, func, env, names, module):
        """Returns (status, edges); status is resolved, external or unresolved."""
        if isinstance(func, ast.Name):
            if func.id in names:
                return "unresolved", []
            kind, target = module.names.get(func.id, (None, None))
            if kind == "symbol":
                return "resolved", self.edges_for(target)
            if kind == "external" or (kind is None and hasattr(builtins, func.id)):
                return "external", []
            return "unresolved", []
        if isinstance(func, ast.Attribute):
            if isinstance(func.value, ast.Constant):  # for example "".join
                return "external", []
            root = func.value
            while isinstance(root, ast.Attribute):
                root = root.value
            if isinstance(root, ast.Name) and root.id not in names:
                kind, target = module.names.get(root.id, (None, None))
                if kind == "external":
                    return "external", []
                if kind == "module" and root is func.value:
                    member = f"{target}.{func.attr}"
                    if member in self.symbols:
                        return "resolved", self.edges_for(member)
                    return "unresolved", []
            value = self.infer(func.value, env, names, module)
            if value and value[0] == "class":
                method = f"{value[1]}.{func.attr}"
                if method in self.symbols:
                    return "resolved", [(method, "call")]
        return "unresolved", []

    def reference_target(self, node, names, module):
        """A repository symbol used as a value, for example passed as a callback."""
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id not in names:
            kind, target = module.names.get(node.id, (None, None))
            return target if kind == "symbol" else None
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id not in names:
                kind, target = module.names.get(node.value.id, (None, None))
                member = f"{target}.{node.attr}"
                return member if kind == "module" and member in self.symbols else None
        return None

    def scan(self, nodes, env, names, module):
        nodes = list(nodes)
        called = {id(node.func) for node in nodes if isinstance(node, ast.Call)}
        edges, unresolved = {}, {}
        for node in nodes:
            if isinstance(node, ast.Call):
                status, found = self.resolve_call(node.func, env, names, module)
                if status == "unresolved":
                    text = flatten_text(module, node.func)
                    unresolved.setdefault(text, set()).add(node.lineno)
                for target, kind in found:
                    edges.setdefault((target, kind), set()).add(node.lineno)
            elif id(node) not in called and isinstance(node, (ast.Name, ast.Attribute)):
                target = self.reference_target(node, names, module)
                if target:
                    edges.setdefault((target, "reference"), set()).add(node.lineno)
        return edges, unresolved


def walk_statements(statements):
    for stmt in statements:
        yield from ast.walk(stmt)


def module_level_nodes(tree):
    """Everything evaluated on import: not function bodies, but decorators and defaults."""

    def visit(node):
        if isinstance(node, FUNCTION_NODES):
            defaults = [d for d in (*node.args.defaults, *node.args.kw_defaults) if d]
            for child in (*node.decorator_list, *defaults):
                yield from visit(child)
            return
        if not isinstance(node, ast.Module):
            yield node
        for child in ast.iter_child_nodes(node):
            yield from visit(child)

    yield from visit(tree)


def format_entry(edges, unresolved):
    return {
        "calls": [
            {"target": target, "kind": kind, "lines": sorted(lines)}
            for (target, kind), lines in sorted(edges.items())
        ],
        "unresolved": [
            {"text": text, "lines": sorted(lines)} for text, lines in sorted(unresolved.items())
        ],
    }


def build_call_graph(modules, symbols, resolver):
    raw = {}  # caller id -> (is_test, edges, unresolved)
    for symbol in resolver.callable_symbols():
        module = symbol.module
        env, names = resolver.environment(symbol)
        edges, unresolved = resolver.scan(walk_statements(symbol.node.body), env, names, module)
        for arg in argument_nodes(symbol.node):
            if arg.arg in module.fixtures:
                edges.setdefault((module.fixtures[arg.arg], "fixture"), set()).add(arg.lineno)
        raw[symbol.id] = (module.is_test, edges, unresolved)
    for module in modules:
        edges, unresolved = resolver.scan(module_level_nodes(module.tree), {}, set(), module)
        if edges or unresolved:
            raw[f"{module.id}.<module>"] = (module.is_test, edges, unresolved)

    graph = {"production": {}, "tests": {}}
    used_by = {sid: {"production": set(), "tests": set()} for sid in symbols}
    for caller, (is_test, edges, unresolved) in raw.items():
        group = "tests" if is_test else "production"
        graph[group][caller] = format_entry(edges, unresolved)
        for target, _ in edges:
            used_by[target][group].add(caller)
    graph["usedBy"] = {
        sid: {group: sorted(callers) for group, callers in entry.items()}
        for sid, entry in used_by.items()
    }
    return graph, {caller: edges for caller, (_, edges, _) in raw.items()}


# ---------- code-index.json ----------


def signature_text(module, node):
    """Verbatim header from `def` or `class` up to, not including, the closing colon."""
    start = (node.lineno, node.col_offset)
    depth = 0
    for token in module.tokens:
        if token.start < start or token.type != tokenize.OP:
            continue
        if token.string in "([{":
            depth += 1
        elif token.string in ")]}":
            depth -= 1
        elif token.string == ":" and depth == 0:
            end_row, end_col = token.start
            break
    else:
        fail(f"{module.path}:{node.lineno}: no signature end found")
    if end_row == node.lineno:
        return module.lines[end_row - 1][node.col_offset : end_col].rstrip()
    parts = [module.lines[node.lineno - 1][node.col_offset :]]
    parts += module.lines[node.lineno : end_row - 1]
    parts.append(module.lines[end_row - 1][:end_col])
    return " ".join(part.strip() for part in parts).strip()


def leading_comments(module, start_line):
    found = []
    line = start_line - 1
    while line in module.comments and module.comments[line][1]:
        found.append(module.comments[line][0])
        line -= 1
    return list(reversed(found))


def describe_symbol(symbol, meta):
    module, node = symbol.module, symbol.node
    start = min([node.lineno, *[d.lineno for d in node.decorator_list]])
    end = node.end_lineno
    return {
        "id": symbol.id,
        "name": symbol.name,
        "kind": symbol.kind,
        "signature": signature_text(module, node),
        "decorators": [f"@{flatten_text(module, d)}" for d in node.decorator_list],
        "startLine": start,
        "defLine": node.lineno,
        "endLine": end,
        "source": "\n".join(module.lines[start - 1 : end]),
        "leadingComments": leading_comments(module, start),
        "permalink": f"{meta['githubBaseUrl']}/blob/{meta['sha']}/{module.path}#L{start}-L{end}",
    }


def literal_or_none(node):
    """Returns (value, is_literal); a value JSON cannot hold counts as not literal."""
    try:
        value = ast.literal_eval(node)
        json.dumps(value, allow_nan=False, sort_keys=True)
        return value, True
    except (ValueError, TypeError, SyntaxError, RecursionError):
        return None, False


def module_assignments(module):
    assignments = []
    for stmt in module.tree.body:
        if isinstance(stmt, ast.Assign):
            targets, value = stmt.targets, stmt.value
        elif isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
            targets, value = [stmt.target], stmt.value
        else:
            continue
        for target in targets:
            if isinstance(target, ast.Name):
                assignments.append((target.id, stmt, value))
            elif isinstance(target, ast.Tuple):
                for element in target.elts:
                    if isinstance(element, ast.Name):
                        assignments.append((element.id, stmt, None))
    return assignments


def describe_import_entries(module, known):
    entries = []
    for node in ast.walk(module.tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                entries.append(
                    {
                        "line": node.lineno,
                        "module": alias.name,
                        "name": None,
                        "asName": alias.asname,
                        "internal": alias.name in known,
                    }
                )
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                entries.append(
                    {
                        "line": node.lineno,
                        "module": node.module,
                        "name": alias.name,
                        "asName": alias.asname,
                        "internal": node.module in known,
                    }
                )
    return sorted(entries, key=lambda e: (e["line"], e["module"], e["name"] or ""))


def dataclass_fields(symbol):
    fields = []
    for stmt in symbol.node.body:
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            fields.append(
                {
                    "name": stmt.target.id,
                    "annotation": flatten_text(symbol.module, stmt.annotation),
                    "default": flatten_text(symbol.module, stmt.value) if stmt.value else None,
                    "line": stmt.lineno,
                }
            )
    return fields


def build_code_index(modules, symbols, meta):
    known = {module.id for module in modules}
    files = []
    for module in modules:
        constants = []
        for name, stmt, value in module_assignments(module):
            literal, is_literal = literal_or_none(value) if value is not None else (None, False)
            constants.append(
                {
                    "name": name,
                    "startLine": stmt.lineno,
                    "endLine": stmt.end_lineno,
                    "source": "\n".join(module.lines[stmt.lineno - 1 : stmt.end_lineno]),
                    "value": literal,
                    "isLiteral": is_literal,
                }
            )
        functions, classes = [], []
        for stmt in module.tree.body:
            if isinstance(stmt, FUNCTION_NODES):
                functions.append(describe_symbol(symbols[f"{module.id}.{stmt.name}"], meta))
            elif isinstance(stmt, ast.ClassDef):
                symbol = symbols[f"{module.id}.{stmt.name}"]
                dataclass_flag = has_decorator(stmt, "dataclass")
                entry = describe_symbol(symbol, meta)
                entry["bases"] = [flatten_text(module, base) for base in stmt.bases]
                entry["isDataclass"] = dataclass_flag
                entry["fields"] = dataclass_fields(symbol) if dataclass_flag else []
                entry["methods"] = [
                    describe_symbol(symbols[f"{symbol.id}.{item.name}"], meta)
                    for item in stmt.body
                    if isinstance(item, FUNCTION_NODES)
                ]
                classes.append(entry)
        files.append(
            {
                "path": module.path,
                "module": module.id,
                "lineCount": len(module.lines),
                "docstring": ast.get_docstring(module.tree),
                "imports": describe_import_entries(module, known),
                "constants": constants,
                "functions": functions,
                "classes": classes,
            }
        )
    return {"sha": meta["sha"], "files": files}


# ---------- config.json ----------


def evaluate_derived(node, known):
    """Whitelisted arithmetic over literals and earlier constants. Never calls eval."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.Name) and node.id in known:
        return known[node.id]
    if isinstance(node, ast.BinOp) and type(node.op) in ARITHMETIC:
        left = evaluate_derived(node.left, known)
        right = evaluate_derived(node.right, known)
        return ARITHMETIC[type(node.op)](left, right)
    raise ValueError(ast.dump(node))


def config_reads(modules):
    reads = {}
    for module in modules:
        aliases = {n for n, (kind, t) in module.names.items() if kind == "module" and t == "config"}
        imported = {
            local: target.split(".", 1)[1]
            for local, (kind, target) in module.names.items()
            if kind == "value" and target.startswith("config.")
        }
        for node in ast.walk(module.tree):
            name = None
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                if node.value.id in aliases:
                    name = node.attr
            elif isinstance(node, ast.Name) and node.id in imported:
                name = imported[node.id]
            if name:
                reads.setdefault(name, set()).add((module.path, node.lineno))
    return reads


def build_config(modules):
    module = next((m for m in modules if m.path == "config.py"), None)
    if module is None:
        fail("config.py not found")
    reads = config_reads(modules)
    section_lines = sorted(
        (line, match.group(1))
        for line, (text, standalone) in module.comments.items()
        if standalone and (match := SECTION_RE.match(text))
    )
    known = {}
    constants = []
    for name, stmt, value in module_assignments(module):
        if value is None:
            fail(f"config.py:{stmt.lineno}: cannot read tuple assignment to {name}")
        try:
            resolved, derived = ast.literal_eval(value), False
        except ValueError:
            try:
                resolved, derived = evaluate_derived(value, known), True
            except (ValueError, ArithmeticError) as error:
                fail(f"config.py:{stmt.lineno}: cannot evaluate {name}: {error}")
        known[name] = resolved
        comment = module.comments.get(stmt.end_lineno)
        sections = [title for line, title in section_lines if line < stmt.lineno]
        constants.append(
            {
                "name": name,
                "value": resolved,
                "valueText": flatten_text(module, value),
                "line": stmt.lineno,
                "comment": comment[0] if comment and not comment[1] else None,
                "section": sections[-1] if sections else None,
                "derived": derived,
                "readBy": [
                    {"file": path, "line": line} for path, line in sorted(reads.get(name, ()))
                ],
            }
        )
    return {"path": module.path, "constants": constants}


# ---------- tests-index.json ----------


def parametrized_cases(node):
    """1 when not parametrized; null when the parameter values are not literal."""
    cases = 1
    for decorator in node.decorator_list:
        if not (isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Attribute)):
            continue
        if decorator.func.attr != "parametrize" or len(decorator.args) < 2:
            continue
        try:
            cases *= len(ast.literal_eval(decorator.args[1]))
        except (ValueError, TypeError):
            return None
    return cases


def test_functions(module, symbols):
    for stmt in module.tree.body:
        if isinstance(stmt, FUNCTION_NODES) and stmt.name.startswith("test_"):
            yield symbols[f"{module.id}.{stmt.name}"]
        elif isinstance(stmt, ast.ClassDef) and stmt.name.startswith("Test"):
            for item in stmt.body:
                if isinstance(item, FUNCTION_NODES) and item.name.startswith("test_"):
                    yield symbols[f"{module.id}.{stmt.name}.{item.name}"]


def build_tests_index(modules, symbols, edges_by_caller):
    known = {module.id for module in modules}
    files = []
    for module in modules:
        if not module.is_test:
            continue
        tests = list(test_functions(module, symbols))
        test_ids = {symbol.id for symbol in tests}
        entries = []
        for symbol in tests:
            edges = edges_by_caller[symbol.id]
            direct = {target for target, _ in edges}
            reached, seen, pending = set(), set(), list(direct)
            while pending:
                target = pending.pop()
                if target in seen:
                    continue
                seen.add(target)
                if symbols[target].module.is_test:
                    pending.extend(t for t, _ in edges_by_caller.get(target, {}))
                else:
                    reached.add(target)
            node = symbol.node
            in_tests = {t for t in direct if symbols[t].module.is_test}
            entries.append(
                {
                    "id": symbol.id,
                    "name": symbol.name,
                    "startLine": min([node.lineno, *[d.lineno for d in node.decorator_list]]),
                    "endLine": node.end_lineno,
                    "cases": parametrized_cases(node),
                    "fixtures": sorted(t for t, kind in edges if kind == "fixture"),
                    "helpers": sorted({t for t, kind in edges if kind != "fixture"} & in_tests),
                    "calls": sorted(direct - in_tests),
                    "reaches": sorted(reached),
                }
            )
        helpers = []
        for symbol in symbols.values():
            if symbol.module is module and symbol.id not in test_ids and symbol.kind != "method":
                is_fixture_symbol = symbol.id in module.fixtures.values()
                helpers.append(
                    {
                        "id": symbol.id,
                        "name": symbol.name,
                        "kind": "fixture" if is_fixture_symbol else symbol.kind,
                        "startLine": min(
                            [symbol.node.lineno, *[d.lineno for d in symbol.node.decorator_list]]
                        ),
                        "endLine": symbol.node.end_lineno,
                    }
                )
        imports = [
            {"module": e["module"], "name": e["name"], "line": e["line"]}
            for e in describe_import_entries(module, known)
            if e["internal"]
        ]
        files.append(
            {
                "path": module.path,
                "module": module.id,
                "repoImports": imports,
                "helpers": helpers,
                "tests": entries,
            }
        )
    return {"files": files}


# ---------- entry point ----------


def generate(repo, out):
    repo = Path(repo).resolve()
    out = Path(out).resolve()
    meta = build_repo_meta(repo)
    modules = [load_module(repo, path) for path in find_python_files(repo)]
    if not modules:
        fail(f"no Python files found in {repo}")
    symbols = collect_symbols(modules)
    collect_names(modules, symbols)
    resolver = Resolver(symbols)
    call_graph, edges_by_caller = build_call_graph(modules, symbols, resolver)

    stale = [path for path in meta["dirtyFiles"] if path.endswith(".py")]
    if stale:
        print(
            f"warning: uncommitted changes in {', '.join(stale)}: "
            "permalinks may not match these line numbers",
            file=sys.stderr,
        )

    outputs = {
        "repo-meta.json": meta,
        "code-index.json": build_code_index(modules, symbols, meta),
        "config.json": build_config(modules),
        "call-graph.json": call_graph,
        "tests-index.json": build_tests_index(modules, symbols, edges_by_caller),
    }
    out.mkdir(parents=True, exist_ok=True)
    for name, data in outputs.items():
        (out / name).write_bytes(serialize(data))
    return sorted(outputs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=DEFAULT_REPO, help="repository to read")
    parser.add_argument("--out", default=DEFAULT_OUT, help="folder to write generated JSON into")
    args = parser.parse_args()
    written = generate(args.repo, args.out)
    print(f"wrote {len(written)} files to {args.out}: {', '.join(written)}")


if __name__ == "__main__":
    main()
