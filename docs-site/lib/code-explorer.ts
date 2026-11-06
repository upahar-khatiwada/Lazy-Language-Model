import { readFileSync } from "node:fs";
import { join } from "node:path";
import { z } from "zod";
import { PIPELINE } from "@/data/pipeline";
import { loadSymbolIndex } from "./code-targets";
import { requireTarget } from "./code-targets";
import { loadConcepts } from "./concepts";
import { loadLinks, type Link } from "./links";
import { pageHref } from "./nav";
import { conceptLinks, readPageSources } from "./reference";
import { loadCallGraph, loadCodeIndex, loadConfig, loadTestsIndex, type CodeIndex } from "./repo";

// Everything the Code Explorer shows comes from data/generated, data/links and the contract snapshot
// in data/derived. Nothing here names a file or a function.

export type ExplorerFile = CodeIndex["files"][number];

// ---------- files ----------

// /code/ops for ops.py and /code/test_ops for tests/test_ops.py. Slugs must stay unique, and must not
// equal a written page under content/code/. tests/code-explorer.test.ts checks both.
export function fileSlug(path: string): string {
  return path.split("/").at(-1)!.replace(/\.py$/, "");
}

export function explorerHref(path: string, anchor?: string): string {
  const href = pageHref(`code/${fileSlug(path)}`);
  return anchor ? `${href}#${anchor}` : href;
}

export function explorerFiles(): ExplorerFile[] {
  return loadCodeIndex().files;
}

export function findExplorerFile(slug: string): ExplorerFile | undefined {
  return explorerFiles().find((file) => fileSlug(file.path) === slug);
}

export function isTestFile(path: string): boolean {
  return loadTestsIndex().files.some((file) => file.path === path);
}

export type TreeNode =
  | { type: "file"; name: string; path: string }
  | { type: "folder"; name: string; children: TreeNode[] };

// Files first, then folders, each in the order given. Folders come only from the paths.
export function buildFileTree(paths: string[]): TreeNode[] {
  const insert = (level: TreeNode[], parts: string[], path: string) => {
    const [head, ...rest] = parts as [string, ...string[]];
    if (rest.length === 0) {
      level.push({ type: "file", name: head, path });
      return;
    }
    let folder = level.find(
      (node): node is Extract<TreeNode, { type: "folder" }> =>
        node.type === "folder" && node.name === head,
    );
    if (!folder) {
      folder = { type: "folder", name: head, children: [] };
      level.push(folder);
    }
    insert(folder.children, rest, path);
  };
  const order = (nodes: TreeNode[]): TreeNode[] => [
    ...nodes.filter((node) => node.type === "file"),
    ...nodes
      .filter((node) => node.type === "folder")
      .map((node) => ({ ...node, children: order(node.children) })),
  ];

  const root: TreeNode[] = [];
  for (const path of paths) insert(root, path.split("/"), path);
  return order(root);
}

// ---------- contract text ----------

const RepoInventorySchema = z.strictObject({
  files: z.record(z.string(), z.string()),
  symbols: z.record(z.string(), z.string()),
});

let repoInventory: z.infer<typeof RepoInventorySchema> | undefined;

function loadRepoInventory() {
  repoInventory ??= RepoInventorySchema.parse(
    JSON.parse(
      readFileSync(join(process.cwd(), "data", "derived", "repo-inventory.json"), "utf-8"),
    ),
  );
  return repoInventory;
}

export const NOT_AVAILABLE = "Not available from the implementation.";

// The docstring first, then the contract text. Neither is written by hand here.
export function filePurpose(file: ExplorerFile): string {
  return file.docstring ?? loadRepoInventory().files[file.path] ?? NOT_AVAILABLE;
}

export function symbolPurpose(id: string): string | null {
  return loadRepoInventory().symbols[id] ?? null;
}

// ---------- symbols ----------

export type FileSymbol = {
  id: string;
  anchor: string;
  name: string;
  kind: "function" | "method" | "class" | "constant";
  signature: string | null;
  startLine: number;
  endLine: number;
  source: string;
  permalink: string;
};

// Every function, class, method and constant of a file, in line order.
export function fileSymbols(file: ExplorerFile): FileSymbol[] {
  const index = loadSymbolIndex();
  const entry = (
    id: string,
    kind: FileSymbol["kind"],
    signature: string | null,
    name: string,
  ): FileSymbol => {
    const symbol = index.get(id);
    if (!symbol) throw new Error(`Unknown code symbol: ${id}`);
    return {
      id,
      anchor: id.slice(file.module.length + 1),
      name,
      kind,
      signature,
      startLine: symbol.startLine,
      endLine: symbol.endLine,
      source: symbol.source,
      permalink: symbol.permalink,
    };
  };

  return [
    ...file.functions.map((fn) => entry(fn.id, "function", fn.signature, fn.name)),
    ...file.classes.flatMap((cls) => [
      entry(cls.id, "class", cls.signature, cls.name),
      ...cls.methods.map((method) => entry(method.id, "method", method.signature, method.name)),
    ]),
    ...file.constants.map((constant) =>
      entry(`${file.module}.${constant.name}`, "constant", null, constant.name),
    ),
  ].sort((a, b) => a.startLine - b.startLine || b.endLine - a.endLine);
}

export function callableSymbols(file: ExplorerFile): FileSymbol[] {
  return fileSymbols(file).filter((symbol) => symbol.kind !== "constant");
}

// Where a symbol from the call graph lives. "train.<module>" is the module-level code of a file.
export function locateSymbol(id: string): { path: string; anchor?: string } {
  const moduleLevel = id.match(/^(.+)\.<module>$/);
  if (moduleLevel) {
    const file = explorerFiles().find((candidate) => candidate.module === moduleLevel[1]);
    if (!file) throw new Error(`Unknown module in the call graph: ${id}`);
    return { path: file.path };
  }
  const symbol = loadSymbolIndex().get(id);
  if (!symbol) throw new Error(`Unknown code symbol: ${id}`);
  const file = explorerFiles().find((candidate) => candidate.path === symbol.file)!;
  return { path: file.path, anchor: id.slice(file.module.length + 1) };
}

// ---------- source lines ----------

export type SourceNode =
  | { kind: "line"; line: number }
  | {
      kind: "symbol";
      anchor: string;
      startLine: number;
      endLine: number;
      children: SourceNode[];
    };

// The file's lines, with each symbol's lines wrapped in one node, so an anchor covers exactly them.
// A method sits inside its class. Symbols that overlap without nesting cannot be wrapped, so they fail.
export function sourceTree(
  lineCount: number,
  symbols: Pick<FileSymbol, "anchor" | "startLine" | "endLine">[],
): SourceNode[] {
  const sorted = [...symbols].sort((a, b) => a.startLine - b.startLine || b.endLine - a.endLine);

  const build = (first: number, last: number, inside: typeof sorted): SourceNode[] => {
    const nodes: SourceNode[] = [];
    let line = first;
    let index = 0;
    while (line <= last) {
      const next = inside[index];
      if (!next || next.startLine > line) {
        nodes.push({ kind: "line", line });
        line += 1;
        continue;
      }
      if (next.endLine > last) {
        throw new Error(`${next.anchor} crosses the end of the range that contains it`);
      }
      let end = index + 1;
      while (end < inside.length && inside[end]!.startLine <= next.endLine) end += 1;
      nodes.push({
        kind: "symbol",
        anchor: next.anchor,
        startLine: next.startLine,
        endLine: next.endLine,
        children: build(next.startLine, next.endLine, inside.slice(index + 1, end)),
      });
      index = end;
      line = next.endLine + 1;
    }
    return nodes;
  };

  return build(1, lineCount, sorted);
}

// ---------- call graph ----------

export function usedBy(id: string): { production: string[]; tests: string[] } {
  return loadCallGraph().usedBy[id] ?? { production: [], tests: [] };
}

export function callsOf(id: string): string[] {
  const graph = loadCallGraph();
  const entry = graph.production[id] ?? graph.tests[id];
  return [...new Set((entry?.calls ?? []).map((call) => call.target))];
}

function closure(roots: string[]): Set<string> {
  const seen = new Set<string>();
  const pending = [...roots];
  while (pending.length > 0) {
    const id = pending.pop()!;
    if (seen.has(id)) continue;
    seen.add(id);
    pending.push(...callsOf(id));
  }
  return seen;
}

let reach: { train: Set<string>; talk: Set<string>; tests: Set<string> } | undefined;

// Static reachability through the call graph, which is the meaning of "reachable" in contract/.
export function reachSets() {
  reach ??= {
    train: closure(["train.main"]),
    talk: closure(["talk.main"]),
    tests: closure(Object.keys(loadCallGraph().tests)),
  };
  return reach;
}

export function reachedFrom(id: string): string[] {
  const { train, talk } = reachSets();
  return [...(train.has(id) ? ["train.main"] : []), ...(talk.has(id) ? ["talk.main"] : [])];
}

// Functions, classes and methods outside the tests that no test reaches, even indirectly.
export function untestedSymbols(): string[] {
  const { tests } = reachSets();
  return explorerFiles()
    .filter((file) => !isTestFile(file.path))
    .flatMap((file) => callableSymbols(file))
    .map((symbol) => symbol.id)
    .filter((id) => !tests.has(id));
}

// What a file's module-level code calls: the line that makes `python train.py` run main.
export function moduleEntry(file: ExplorerFile): { target: string; line: number } | null {
  const calls = loadCallGraph().production[`${file.module}.<module>`]?.calls ?? [];
  const call = calls[0];
  return call ? { target: call.target, line: call.lines[0]! } : null;
}

// Other files that use this one, from the call graph.
export function fileUsedBy(file: ExplorerFile): { production: string[]; tests: string[] } {
  const callers = (group: "production" | "tests") => {
    const paths = callableSymbols(file)
      .flatMap((symbol) => usedBy(symbol.id)[group])
      .map((id) => locateSymbol(id).path)
      .filter((path) => path !== file.path);
    return [...new Set(paths)].sort();
  };
  return { production: callers("production"), tests: callers("tests") };
}

export function importedBy(file: ExplorerFile): string[] {
  return explorerFiles()
    .filter((other) =>
      other.imports.some((entry) => entry.internal && entry.module === file.module),
    )
    .map((other) => other.path);
}

export function moduleFile(module: string): ExplorerFile | undefined {
  return explorerFiles().find((file) => file.module === module);
}

// ---------- config reads ----------

export function constantReads(file: ExplorerFile, name: string) {
  if (file.path !== loadConfig().path) return null;
  return loadConfig().constants.find((constant) => constant.name === name)?.readBy ?? [];
}

export function configReaders(file: ExplorerFile): { path: string; constants: string[] }[] {
  if (file.path !== loadConfig().path) return [];
  const byFile = new Map<string, Set<string>>();
  for (const constant of loadConfig().constants) {
    for (const read of constant.readBy) {
      if (!byFile.has(read.file)) byFile.set(read.file, new Set());
      byFile.get(read.file)!.add(constant.name);
    }
  }
  return [...byFile.entries()]
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([path, constants]) => ({ path, constants: [...constants] }));
}

// ---------- mathematics, from link records ----------

export type MathUse = {
  conceptId: string;
  title: string;
  status: "implemented" | "derived";
  href: string | undefined;
  roles: Link["role"][];
  lines: { start: number; end: number }[];
  symbols: string[];
};

function conceptOfEquation(): Map<string, ReturnType<typeof loadConcepts>[number]> {
  const map = new Map<string, ReturnType<typeof loadConcepts>[number]>();
  for (const concept of loadConcepts()) {
    for (const equation of concept.equations) map.set(equation.id, concept);
  }
  return map;
}

function mathUses(links: Link[]): MathUse[] {
  const concepts = conceptOfEquation();
  const pages = readPageSources();
  const uses = new Map<string, MathUse>();
  for (const link of links) {
    const concept = concepts.get(link.equationId);
    if (!concept) throw new Error(`Link names an unknown equation: ${link.equationId}`);
    const { startLine, endLine } = requireTarget(link);
    let use = uses.get(concept.id);
    if (!use) {
      use = {
        conceptId: concept.id,
        title: concept.title,
        status: concept.status,
        href: conceptLinks(concept.id, pages).explanation,
        roles: [],
        lines: [],
        symbols: [],
      };
      uses.set(concept.id, use);
    }
    if (!use.roles.includes(link.role)) use.roles.push(link.role);
    if (!use.symbols.includes(link.symbol)) use.symbols.push(link.symbol);
    if (!use.lines.some((line) => line.start === startLine && line.end === endLine)) {
      use.lines.push({ start: startLine, end: endLine });
    }
  }
  for (const use of uses.values()) use.lines.sort((a, b) => a.start - b.start);
  return [...uses.values()].sort((a, b) => a.lines[0]!.start - b.lines[0]!.start);
}

export function mathForFile(path: string): MathUse[] {
  return mathUses(loadLinks().filter((link) => link.file === path));
}

export function mathForSymbol(id: string): MathUse[] {
  return mathUses(loadLinks().filter((link) => link.symbol === id));
}

// ---------- pipeline ----------

export type PipelineStep = { phase: string; label: string; page: string };

// The pipeline steps whose code is in this file, or that call a function in it.
export function pipelineSteps(path: string): PipelineStep[] {
  const index = loadSymbolIndex();
  const steps: PipelineStep[] = [];
  for (const phase of PIPELINE) {
    for (const node of phase.nodes) {
      const called = node.call ? index.get(node.call)?.file : undefined;
      if (node.target.file === path || called === path) {
        steps.push({ phase: phase.title, label: node.label, page: node.page });
      }
    }
  }
  return steps;
}

export function reachCounts(file: ExplorerFile) {
  const symbols = callableSymbols(file);
  const { train, talk } = reachSets();
  return {
    total: symbols.length,
    train: symbols.filter((symbol) => train.has(symbol.id)).length,
    talk: symbols.filter((symbol) => talk.has(symbol.id)).length,
  };
}

// ---------- tests ----------

export type TestKind = "test" | "fixture" | "helper";

export function testKind(file: ExplorerFile, id: string): TestKind | null {
  const entry = loadTestsIndex().files.find((candidate) => candidate.path === file.path);
  if (!entry) return null;
  if (entry.tests.some((test) => test.id === id)) return "test";
  return entry.helpers.some((helper) => helper.id === id && helper.kind === "fixture")
    ? "fixture"
    : "helper";
}

export type TestCoverageRow = {
  path: string;
  testCount: number;
  caseCount: number;
  tests: { id: string; name: string; purpose: string | null; cases: number }[];
  exercised: string[];
};

// One row per test file. `exercised` holds what a test calls directly or through its fixtures and
// helpers, so it is the same as `reaches` in tests-index.json.
export function testCoverageRows(): TestCoverageRow[] {
  return loadTestsIndex().files.map((file) => {
    const tests = file.tests.map((test) => ({
      id: test.id,
      name: test.name,
      purpose: symbolPurpose(test.id),
      cases: test.cases ?? 1,
    }));
    return {
      path: file.path,
      testCount: tests.length,
      caseCount: tests.reduce((sum, test) => sum + test.cases, 0),
      tests,
      exercised: [...new Set(file.tests.flatMap((test) => test.reaches))].sort(),
    };
  });
}
