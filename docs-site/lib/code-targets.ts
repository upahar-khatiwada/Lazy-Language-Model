import { z } from "zod";
import { loadCodeIndex, loadRepoMeta, type CodeIndex, type RepoMeta } from "./repo";

// A code target names code by symbol and text. Line numbers are never stored: they are resolved
// from data/generated/code-index.json, so they always belong to the pinned commit.
export const codeTargetShape = {
  file: z.string().min(1),
  symbol: z.string().min(1),
  // A substring, or a regular expression when `regex` is true. It may span several lines.
  match: z.string().min(1),
  regex: z.boolean().optional(),
  // 0-based. Required when the match occurs more than once inside the symbol.
  occurrence: z.number().int().nonnegative().optional(),
};

export const CodeTargetSchema = z.strictObject(codeTargetShape);
export type CodeTarget = z.infer<typeof CodeTargetSchema>;

export type CodeSymbol = {
  id: string;
  name: string;
  file: string;
  startLine: number;
  endLine: number;
  source: string;
  permalink: string;
};

export type SymbolIndex = Map<string, CodeSymbol>;

export type ResolvedTarget = {
  symbol: CodeSymbol;
  startLine: number;
  endLine: number;
  permalink: string;
};

export type TargetResult = { ok: true; value: ResolvedTarget } | { ok: false; problem: string };

function linePermalink(repo: RepoMeta, file: string, startLine: number, endLine: number): string {
  const lines = startLine === endLine ? `L${startLine}` : `L${startLine}-L${endLine}`;
  return `${repo.githubBaseUrl}/blob/${repo.sha}/${file}#${lines}`;
}

export function buildSymbolIndex(codeIndex: CodeIndex, repo: RepoMeta): SymbolIndex {
  const index: SymbolIndex = new Map();
  const add = (file: string, symbol: Omit<CodeSymbol, "file" | "permalink">) => {
    index.set(symbol.id, {
      ...symbol,
      file,
      permalink: linePermalink(repo, file, symbol.startLine, symbol.endLine),
    });
  };

  for (const file of codeIndex.files) {
    for (const fn of file.functions) add(file.path, fn);
    for (const cls of file.classes) {
      add(file.path, cls);
      for (const method of cls.methods) add(file.path, method);
    }
    for (const constant of file.constants) {
      add(file.path, { ...constant, id: `${file.module}.${constant.name}` });
    }
  }
  return index;
}

let cachedIndex: SymbolIndex | undefined;

export function loadSymbolIndex(): SymbolIndex {
  cachedIndex ??= buildSymbolIndex(loadCodeIndex(), loadRepoMeta());
  return cachedIndex;
}

export function getSymbol(id: string, index: SymbolIndex = loadSymbolIndex()): CodeSymbol {
  const symbol = index.get(id);
  if (!symbol) throw new Error(`Unknown code symbol: ${id}`);
  return symbol;
}

// Offsets of every match, in order. Overlapping substrings count once, from the left.
function findMatches(
  source: string,
  target: CodeTarget,
): { start: number; end: number }[] | { problem: string } {
  if (target.regex) {
    let pattern: RegExp;
    try {
      pattern = new RegExp(target.match, "gm");
    } catch (error) {
      return { problem: `invalid regex "${target.match}": ${(error as Error).message}` };
    }
    const found = [...source.matchAll(pattern)].map((match) => ({
      start: match.index,
      end: match.index + match[0].length,
    }));
    if (found.some((match) => match.start === match.end)) {
      return { problem: `regex "${target.match}" can match empty text` };
    }
    return found;
  }

  const found: { start: number; end: number }[] = [];
  for (
    let start = source.indexOf(target.match);
    start !== -1;
    start = source.indexOf(target.match, start + target.match.length)
  ) {
    found.push({ start, end: start + target.match.length });
  }
  return found;
}

function countNewlines(text: string): number {
  return text.split("\n").length - 1;
}

export function resolveTarget(
  target: CodeTarget,
  index: SymbolIndex = loadSymbolIndex(),
): TargetResult {
  const label = `${target.symbol}: match "${target.match}"`;
  const symbol = index.get(target.symbol);
  if (!symbol) return { ok: false, problem: `symbol does not exist: ${target.symbol}` };
  if (symbol.file !== target.file) {
    return {
      ok: false,
      problem: `symbol ${target.symbol} is in ${symbol.file}, not ${target.file}`,
    };
  }

  const matches = findMatches(symbol.source, target);
  if ("problem" in matches) return { ok: false, problem: `${target.symbol}: ${matches.problem}` };
  if (matches.length === 0) return { ok: false, problem: `${label} not found` };
  if (target.occurrence === undefined && matches.length > 1) {
    return {
      ok: false,
      problem: `${label} is ambiguous: ${matches.length} matches and no occurrence index`,
    };
  }

  const occurrence = target.occurrence ?? 0;
  const match = matches[occurrence];
  if (!match) {
    return {
      ok: false,
      problem: `${label}: occurrence ${occurrence} is out of range, ${matches.length} found`,
    };
  }

  const startLine = symbol.startLine + countNewlines(symbol.source.slice(0, match.start));
  const endLine = symbol.startLine + countNewlines(symbol.source.slice(0, match.end - 1));
  const permalink = symbol.permalink.replace(
    /#.*$/,
    startLine === endLine ? `#L${startLine}` : `#L${startLine}-L${endLine}`,
  );
  return { ok: true, value: { symbol, startLine, endLine, permalink } };
}

// For rendering: verify_links has already reported bad targets by the time pages are built.
export function requireTarget(target: CodeTarget, index?: SymbolIndex): ResolvedTarget {
  const result = resolveTarget(target, index);
  if (!result.ok) throw new Error(result.problem);
  return result.value;
}

// The text of the resolved lines.
export function targetLines({ symbol, startLine, endLine }: ResolvedTarget): string {
  return symbol.source
    .split("\n")
    .slice(startLine - symbol.startLine, endLine - symbol.startLine + 1)
    .join("\n");
}

export function formatLineRange(startLine: number, endLine: number): string {
  return startLine === endLine ? `${startLine}` : `${startLine}-${endLine}`;
}
