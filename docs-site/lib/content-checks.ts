import { resolveTarget, type CodeTarget, type SymbolIndex } from "./code-targets";
import type { MathConcept } from "./concepts";
import { formatShape } from "./viz/shape";

// The seven checks behind `bun run verify:content`. Each is a pure function of its input, so a test
// can hand it a deliberately broken input. lib/content-input.ts loads the real one.

export type CheckResult = {
  id: number;
  title: string;
  problems: string[]; // any of these fails the command
  warnings: string[]; // shown, and does not fail the command
  report: string[]; // printed under the check whatever the outcome
};

function result(id: number, title: string, parts: Partial<CheckResult> = {}): CheckResult {
  return { id, title, problems: [], warnings: [], report: [], ...parts };
}

export type TextFile = { path: string; text: string };

function lineOf(text: string, offset: number): number {
  return text.slice(0, offset).split("\n").length;
}

// ---------- 1. Source symbols and link matches ----------

export type LinkFile = {
  path: string;
  links: (CodeTarget & { equationId: string })[];
};

export function checkSymbolsAndLinks(input: {
  concepts: MathConcept[];
  linkFiles: LinkFile[];
  symbols: SymbolIndex;
}): CheckResult {
  const problems: string[] = [];
  for (const concept of input.concepts) {
    for (const { file, symbol } of concept.source) {
      const found = input.symbols.get(symbol);
      if (!found) {
        problems.push(`data/concepts/${concept.id}.yaml: source symbol does not exist: ${symbol}`);
      } else if (found.file !== file) {
        problems.push(
          `data/concepts/${concept.id}.yaml: source symbol ${symbol} is in ${found.file}, not ${file}`,
        );
      }
    }
  }
  let links = 0;
  for (const { path, links: fileLinks } of input.linkFiles) {
    for (const link of fileLinks) {
      links++;
      const resolved = resolveTarget(link, input.symbols);
      if (!resolved.ok) problems.push(`${path}: link for ${link.equationId}: ${resolved.problem}`);
    }
  }
  return result(1, "Source symbols exist and link matches resolve", {
    problems,
    report: [`${input.concepts.length} concepts and ${links} links checked.`],
  });
}

// ---------- 2. Public function coverage ----------

export const COVERED_MODULES = [
  "tokenizer",
  "data",
  "ops",
  "gradcheck",
  "transformer",
  "optim",
  "generate",
  "checkpoint",
  "train",
  "talk",
];

export type PublicSymbol = { id: string; name: string; module: string };

export function isPublic(id: string): boolean {
  return id.split(".").every((part) => !part.startsWith("_"));
}

// References on a page: a symbol attribute, a quoted line, or the name in inline code. A bare name
// counts only when one module has it: `main` is in both train.py and talk.py.
function pageReferences(
  body: string,
  symbols: PublicSymbol[],
  codeRefs: Record<string, { symbol: string }>,
): Set<string> {
  const found = new Set<string>();
  for (const match of body.matchAll(/\bsymbol="([^"]+)"/g)) found.add(match[1]!);
  for (const match of body.matchAll(/<Code(?:Step|Line)\b[^>]*\bid="([^"]+)"/g)) {
    const ref = codeRefs[match[1]!];
    if (ref) found.add(ref.symbol);
  }
  const inline = new Set(
    [...body.matchAll(/`([^`\n]+)`/g)].map((match) => match[1]!.replace(/\(.*$/, "").trim()),
  );
  for (const symbol of symbols) {
    const shortId = symbol.id.slice(symbol.module.length + 1);
    if (inline.has(symbol.id) || (shortId.includes(".") && inline.has(shortId))) {
      found.add(symbol.id);
    } else if (
      inline.has(symbol.name) &&
      symbols.filter((other) => other.name === symbol.name).length === 1
    ) {
      found.add(symbol.id);
    }
  }
  return found;
}

export function checkCoverage(input: {
  symbols: PublicSymbol[];
  concepts: MathConcept[];
  links: { symbol: string }[];
  pages: { slug: string; body: string }[];
  codeRefs: Record<string, { symbol: string }>;
  allowlist: { symbol: string; reason: string }[];
}): CheckResult {
  const referenced = new Set<string>();
  for (const concept of input.concepts) {
    for (const { symbol } of concept.source) referenced.add(symbol);
    for (const { symbol } of concept.usedBy) referenced.add(symbol);
    for (const { target } of concept.shapes) referenced.add(target.symbol);
    if (concept.example) referenced.add(concept.example.call);
  }
  for (const { symbol } of input.links) referenced.add(symbol);
  for (const page of input.pages) {
    for (const id of pageReferences(page.body, input.symbols, input.codeRefs)) referenced.add(id);
  }

  const allowed = new Map(input.allowlist.map((entry) => [entry.symbol, entry.reason]));
  const problems: string[] = [];
  for (const { symbol, reason } of input.allowlist) {
    if (!input.symbols.some((candidate) => candidate.id === symbol)) {
      problems.push(`coverage allowlist names an unknown public function: ${symbol}`);
    }
    if (!reason.trim()) problems.push(`coverage allowlist entry ${symbol} has no written reason`);
  }

  const report: string[] = [];
  const allowedRows: string[] = [];
  const uncovered: string[] = [];
  for (const moduleName of COVERED_MODULES) {
    const inModule = input.symbols.filter((symbol) => symbol.module === moduleName);
    const covered = inModule.filter((symbol) => referenced.has(symbol.id));
    report.push(`${moduleName}: ${covered.length} of ${inModule.length} referenced`);
    for (const { id } of inModule) {
      if (referenced.has(id)) continue;
      const reason = allowed.get(id);
      if (reason !== undefined) allowedRows.push(`allowlisted ${id}: ${reason}`);
      else uncovered.push(id);
    }
  }
  report.push(...allowedRows);
  report.push(
    uncovered.length === 0
      ? "Uncovered functions: none."
      : `Uncovered functions (${uncovered.length}): ${uncovered.join(", ")}`,
  );
  for (const id of uncovered) {
    problems.push(`${id} is not referenced by any concept or page, and is not in the allowlist`);
  }
  return result(2, "Every public function is referenced or allowlisted", { problems, report });
}

// ---------- 3. Equations ----------

export function checkEquations(input: {
  concepts: MathConcept[];
  pages: { slug: string; body: string }[];
}): CheckResult {
  const problems: string[] = [];
  const known = new Set(
    input.concepts.flatMap((concept) => concept.equations.map((equation) => equation.id)),
  );
  let uses = 0;
  const pattern = /<MathBlock\b[^>]*?\bid="([^"]+)"|\b(?:equation|equationId)="([^"]+)"/g;
  for (const { slug, body } of input.pages) {
    for (const match of body.matchAll(pattern)) {
      const id = match[1] ?? match[2]!;
      uses++;
      if (!known.has(id)) {
        problems.push(
          `content/${slug}.mdx:${lineOf(body, match.index)}: equation id "${id}" does not exist`,
        );
      }
    }
  }

  const seen = new Map<string, string>();
  for (const concept of input.concepts) {
    for (const equation of concept.equations) {
      const key = equation.latex.replace(/\s+/g, "");
      const where = `${concept.id} (${equation.id})`;
      const other = seen.get(key);
      if (other) {
        problems.push(`duplicated equation LaTeX in ${other} and ${where}: ${equation.latex}`);
      } else seen.set(key, where);
    }
  }
  return result(3, "Equation ids in MDX exist and no LaTeX is duplicated", {
    problems,
    report: [`${uses} equation uses in MDX and ${seen.size} distinct equations checked.`],
  });
}

// ---------- 4. Labels on numeric examples ----------

export type ProvenanceFields = {
  gitSha: string;
  checkpointFiles: { name: string; sha256: string }[];
  seed: number;
};

// Text around a JSX attribute: from the "<Name" before it to the ">" that ends the tag, skipping any
// ">" inside {...}. Returns null when the attribute is not inside a tag.
export function jsxTagAround(text: string, offset: number): { name: string; tag: string } | null {
  const start = text.lastIndexOf("<", offset);
  if (start === -1) return null;
  const name = /^<([A-Z][A-Za-z0-9]*)/.exec(text.slice(start, start + 80))?.[1];
  if (!name) return null;
  let depth = 0;
  for (let index = start; index < text.length; index++) {
    const char = text[index];
    if (char === "{") depth++;
    else if (char === "}") depth--;
    else if (char === ">" && depth === 0) return { name, tag: text.slice(start, index + 1) };
  }
  return null;
}

export function checkLabels(input: {
  concepts: MathConcept[];
  sources: TextFile[]; // MDX and TSX
  // A bare badge states Actual Model Output with no chart beside it. Only these files may do that.
  actualBadgeFiles: { path: string; reason: string }[];
  provenance: ProvenanceFields | null;
}): CheckResult {
  const problems: string[] = [];
  let numericExamples = 0;
  let actualPanels = 0;

  for (const concept of input.concepts) {
    const example = concept.example;
    if (example?.kind === "actual" && !example.provenance?.trim()) {
      problems.push(
        `data/concepts/${concept.id}.yaml: Actual Model Output example has no provenance`,
      );
    }
  }

  const kinds = new Map(input.concepts.map((concept) => [concept.id, concept.example?.kind]));
  for (const { path, text } of input.sources) {
    for (const match of text.matchAll(/<NumericExample\b/g)) {
      const found = jsxTagAround(text, match.index + 1);
      if (!found) continue;
      numericExamples++;
      const where = `${path}:${lineOf(text, match.index)}`;
      const kind = /\bkind="([^"]*)"/.exec(found.tag)?.[1];
      // A kind passed as an expression comes from a component's own props, which MDX cannot do.
      if (/\bkind=\{/.test(found.tag)) continue;
      if (kind === undefined) {
        problems.push(`${where}: NumericExample does not declare a kind`);
        continue;
      }
      if (kind !== "illustrative" && kind !== "actual") {
        problems.push(`${where}: NumericExample has an unknown kind "${kind}"`);
        continue;
      }
      const concept = /\bconcept="([^"]+)"/.exec(found.tag)?.[1];
      if (concept && kinds.get(concept) !== kind) {
        problems.push(
          `${where}: NumericExample for ${concept} says ${kind}, the data says ${kinds.get(concept)}`,
        );
      }
    }

    for (const match of text.matchAll(/\bkind="actual"/g)) {
      const found = jsxTagAround(text, match.index);
      if (!found || found.name === "NumericExample") continue;
      const where = `${path}:${lineOf(text, match.index)}`;
      if (found.name === "Badge") {
        if (!input.actualBadgeFiles.some((allowed) => allowed.path === path)) {
          problems.push(`${where}: an Actual Model Output badge outside a panel with a provenance`);
        }
        continue;
      }
      actualPanels++;
      if (!/\bprovenance=/.test(found.tag)) {
        problems.push(`${where}: Actual Model Output panel <${found.name}> has no provenance`);
      }
    }
  }

  if (input.provenance) {
    const { gitSha, checkpointFiles, seed } = input.provenance;
    const where = "data/generated/actual/meta.json";
    if (!gitSha) problems.push(`${where}: provenance has no commit`);
    if (checkpointFiles.length === 0 || checkpointFiles.some((file) => !file.sha256)) {
      problems.push(`${where}: provenance has no checkpoint hash`);
    }
    if (typeof seed !== "number") problems.push(`${where}: provenance has no seed`);
  }

  return result(4, "NumericExample declares a kind and Actual panels have provenance", {
    problems,
    report: [
      `${numericExamples} NumericExample tags and ${actualPanels} Actual panels checked. ` +
        (input.provenance ? "Exported provenance is complete." : "Nothing has been exported."),
    ],
  });
}

// ---------- 5. Tensor shapes against the golden run ----------

export type ShapeRun = {
  dims: Record<string, number>;
  functions: Record<string, Record<string, number[][]>>;
};

export type ShapeRuns = { runs: Record<string, ShapeRun> };

export function checkShapes(input: {
  concepts: MathConcept[];
  runs: ShapeRuns["runs"];
  // Values the pages substitute today, from data/generated/config.json. V, N and W have none.
  configDims: Record<string, number>;
  aliases: Record<string, { fn: string; name: string }>;
  requiredFunctions: string[];
}): CheckResult {
  const problems: string[] = [];
  const unchecked: string[] = [];
  let stages = 0;
  let comparisons = 0;

  const configRun = input.runs.config;
  if (!input.runs.tiny) problems.push("data/golden/shapes.json has no tiny run");
  if (!configRun) problems.push("data/golden/shapes.json has no config run");
  for (const [name, value] of Object.entries(input.configDims)) {
    if (configRun && configRun.dims[name] !== undefined && configRun.dims[name] !== value) {
      problems.push(
        `data/golden/shapes.json is stale: config.py gives ${name} = ${value}, the run used ${configRun.dims[name]}. Run: bun run sync:shapes`,
      );
    }
  }
  for (const [runName, run] of Object.entries(input.runs)) {
    for (const fn of input.requiredFunctions) {
      if (!run.functions[fn]) problems.push(`data/golden/shapes.json: ${runName} run has no ${fn}`);
    }
  }

  for (const concept of input.concepts) {
    for (const shape of concept.shapes) {
      for (const stage of shape.stages) {
        stages++;
        const key = `${concept.id}/${shape.id}/${stage.name}`;
        const lookups = input.aliases[key]
          ? [input.aliases[key]]
          : [shape.target.symbol, ...concept.source.map((source) => source.symbol)].map((fn) => ({
              fn,
              name: stage.name,
            }));
        let checked = false;
        for (const [runName, run] of Object.entries(input.runs)) {
          const lookup = lookups.find(({ fn, name }) => run.functions[fn]?.[name]);
          if (!lookup) continue;
          checked = true;
          comparisons++;
          const recorded = run.functions[lookup.fn]![lookup.name]!;
          const expected = stage.dims.map((dim) => run.dims[dim]);
          const same = recorded.some(
            (found) =>
              found.length === expected.length && found.every((size, i) => size === expected[i]),
          );
          if (!same) {
            problems.push(
              `data/concepts/${concept.id}.yaml: shape ${shape.id}, stage "${stage.name}": ` +
                `${formatShape(stage.dims)} at the ${runName} config is ` +
                `${formatShape(expected as number[])}, but ${lookup.fn} recorded ${lookup.name} as ` +
                recorded.map((found) => formatShape(found)).join(" or "),
            );
          }
        }
        if (!checked) unchecked.push(key);
      }
    }
  }
  return result(5, "Tensor shapes match the golden run at the tiny and real configs", {
    problems,
    report: [
      `${stages - unchecked.length} of ${stages} stages checked (${comparisons} comparisons).`,
      unchecked.length === 0
        ? "Stages with no recorded tensor: none."
        : `Stages with no recorded tensor (${unchecked.length}): ${unchecked.join(", ")}`,
    ],
  });
}

// ---------- 6. Generated data against the repository ----------

export function checkFreshness(input: {
  repoMetaSha: string;
  codeIndexSha: string;
  actualSha: string | null;
  // The newest commit outside docs-site. Null when git could not be asked.
  headSha: string | null;
}): CheckResult {
  const warnings: string[] = [];
  const short = (sha: string) => sha.slice(0, 7);
  if (input.repoMetaSha !== input.codeIndexSha) {
    warnings.push(
      `data/generated disagrees with itself: repo-meta.json is at ${short(input.repoMetaSha)}, code-index.json at ${short(input.codeIndexSha)}. Run: bun run sync:repo`,
    );
  }
  if (input.headSha === null) {
    warnings.push("Could not read the repository HEAD, so staleness was not checked.");
  } else {
    if (input.repoMetaSha !== input.headSha) {
      warnings.push(
        `data/generated is stale: pinned to ${short(input.repoMetaSha)}, the repository is at ${short(input.headSha)}. Run: bun run sync:repo`,
      );
    }
    if (input.actualSha !== null && input.actualSha !== input.headSha) {
      warnings.push(
        `data/generated/actual is stale: exported at ${short(input.actualSha)}, the repository is at ${short(input.headSha)}. Run: bun run export:actual`,
      );
    }
  }
  return result(6, "Generated data matches the repository HEAD", {
    warnings,
    report: [
      `Pinned to ${short(input.repoMetaSha)}. Newest commit outside docs-site: ${input.headSha ? short(input.headSha) : "unknown"}.`,
    ],
  });
}

// ---------- 7. Forbidden topics ----------

export function termPattern(term: string): RegExp {
  const body = term.replace(/[.*+?^${}()|[\]\\]/g, "\\$&").replace(/[\s-]+/g, "[\\s-]+");
  const start = /^\w/.test(term) ? "\\b" : "";
  const end = /\w$/.test(term) ? "\\b" : "";
  return new RegExp(`${start}${body}${end}`, "gi");
}

export function checkForbiddenTopics(input: {
  absentRows: { technique: string }[];
  absentTerms: Record<string, string[]>;
  sources: TextFile[];
  repoFiles: TextFile[];
  exempt: { path: string; reason: string }[];
}): CheckResult {
  const problems: string[] = [];
  const techniques = input.absentRows.map((row) => row.technique);
  for (const technique of techniques) {
    if (!input.absentTerms[technique]?.length) {
      problems.push(
        `contract/PDF_VS_CODE.md lists "${technique}" as absent, but data/not-in-model.ts has no scan words for it`,
      );
    }
  }
  for (const technique of Object.keys(input.absentTerms)) {
    if (!techniques.includes(technique)) {
      problems.push(
        `data/not-in-model.ts has scan words for "${technique}", which is not a confirmed-absent row of contract/PDF_VS_CODE.md`,
      );
    }
  }

  // The contract's rows decide what is scanned. Words with no row are a problem above, not a scan.
  const terms = techniques.flatMap((technique) =>
    (input.absentTerms[technique] ?? []).map((term) => ({ technique, term })),
  );
  const isExempt = (path: string) =>
    input.exempt.some((entry) =>
      entry.path.endsWith("/") ? path.startsWith(entry.path) : path === entry.path,
    );

  let scanned = 0;
  for (const { path, text } of input.sources) {
    if (isExempt(path)) continue;
    scanned++;
    for (const { technique, term } of terms) {
      for (const match of text.matchAll(termPattern(term))) {
        problems.push(
          `${path}:${lineOf(text, match.index)}: "${match[0]}" names ${technique}, which this repository does not use. Move it to the Not in This Model page`,
        );
      }
    }
  }
  for (const { path, text } of input.repoFiles) {
    for (const { technique, term } of terms) {
      for (const match of text.matchAll(termPattern(term))) {
        problems.push(
          `${path}:${lineOf(text, match.index)}: "${match[0]}" is in the repository, so ${technique} may not be absent`,
        );
      }
    }
  }
  return result(7, "No forbidden topic outside Not in This Model", {
    problems,
    report: [
      `${terms.length} words for ${techniques.length} contract rows, over ${scanned} site files and ${input.repoFiles.length} repository files.`,
      ...input.exempt.map((entry) => `Not scanned: ${entry.path} (${entry.reason})`),
    ],
  });
}

// ---------- Running them ----------

// The command's exit code: warnings alone do not fail it.
export function exitCode(results: CheckResult[]): number {
  return results.some((check) => check.problems.length > 0) ? 1 : 0;
}

export function formatResults(results: CheckResult[]): string[] {
  const lines: string[] = [];
  for (const check of results) {
    const state = check.problems.length > 0 ? "FAIL" : check.warnings.length > 0 ? "WARN" : "PASS";
    lines.push(`[${state}] ${check.id}. ${check.title}`);
    for (const line of check.report) lines.push(`       ${line}`);
    for (const warning of check.warnings) lines.push(`       warning: ${warning}`);
    for (const problem of check.problems) lines.push(`       problem: ${problem}`);
  }
  const failed = results.filter((check) => check.problems.length > 0).length;
  const warned = results.filter((check) => check.warnings.length > 0).length;
  lines.push(
    failed > 0
      ? `FAILED: ${failed} of ${results.length} checks failed.`
      : `OK: ${results.length} checks passed, ${warned} with warnings.`,
  );
  return lines;
}
