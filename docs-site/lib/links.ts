import { z } from "zod";
import {
  codeTargetShape,
  requireTarget,
  resolveTarget,
  type CodeTarget,
  type ResolvedTarget,
  type SymbolIndex,
} from "./code-targets";
import { equationSlug, loadConcepts, readYamlFiles } from "./concepts";

export const LinkSchema = z.strictObject({
  equationId: z.string().min(1),
  ...codeTargetShape,
  // implements: the code computes the equation. inline: the same formula, written in place.
  // call: a call to the function that implements it. test: a test that checks it.
  role: z.enum(["implements", "inline", "call", "test"]),
});

export type Link = z.infer<typeof LinkSchema>;

const LinkFileSchema = z.strictObject({ links: z.array(LinkSchema) });

let links: Link[] | undefined;

export function loadLinks(): Link[] {
  links ??= readYamlFiles("links", LinkFileSchema).flatMap((file) => file.links);
  return links;
}

export type ResolvedLink = Link & { resolved: ResolvedTarget };

export function linksForEquation(equationId: string): ResolvedLink[] {
  return loadLinks()
    .filter((link) => link.equationId === equationId)
    .map((link) => ({ ...link, resolved: requireTarget(link) }));
}

// The code that computes an equation, leaving out call sites and tests.
export function implementationLinks(equationId: string): ResolvedLink[] {
  return linksForEquation(equationId).filter(
    (link) => link.role === "implements" || link.role === "inline",
  );
}

// Lines to highlight when showing a symbol. With an equation, only that equation's links.
// Without one, every link to the symbol and every declared inline use inside it.
export function highlightedLines(symbolId: string, equationId?: string): Set<number> {
  const targets: CodeTarget[] = loadLinks().filter(
    (link) => link.symbol === symbolId && (!equationId || link.equationId === equationId),
  );
  if (!equationId) {
    targets.push(
      ...loadConcepts()
        .flatMap((concept) => concept.usedBy)
        .filter((use) => use.symbol === symbolId),
    );
  }

  const lines = new Set<number>();
  for (const target of targets) {
    const { startLine, endLine } = requireTarget(target);
    for (let line = startLine; line <= endLine; line++) lines.add(line);
  }
  return lines;
}

// Every line a link covers. Empty when the link does not resolve; verify:links reports that.
export function linesCoveredByLink(link: Link, symbols?: SymbolIndex): number[] {
  const result = resolveTarget(link, symbols);
  if (!result.ok) return [];
  const { startLine, endLine } = result.value;
  return Array.from({ length: endLine - startLine + 1 }, (_, index) => startLine + index);
}

const ROLE_ORDER: Link["role"][] = ["implements", "inline", "call", "test"];

let byFile: Map<string, Map<number, string[]>> | undefined;

// The equations tied to each line of a file, as address slugs. Computed once.
export function equationsByLine(file: string): ReadonlyMap<number, string[]> {
  if (!byFile) {
    const built = new Map<string, Map<number, string[]>>();
    const ordered = [...loadLinks()].sort(
      (a, b) => ROLE_ORDER.indexOf(a.role) - ROLE_ORDER.indexOf(b.role),
    );
    for (const link of ordered) {
      const lines = built.get(link.file) ?? new Map<number, string[]>();
      built.set(link.file, lines);
      const slug = equationSlug(link.equationId);
      for (const line of linesCoveredByLink(link)) {
        const slugs = lines.get(line) ?? [];
        if (!slugs.includes(slug)) slugs.push(slug);
        lines.set(line, slugs);
      }
    }
    byFile = built;
  }
  return byFile.get(file) ?? new Map();
}

// The same, for a block of code. Plain data, so it can cross to a client component.
export function lineEquations(
  file: string,
  startLine: number,
  lineCount: number,
): Record<number, string[]> {
  const lines = equationsByLine(file);
  const found: Record<number, string[]> = {};
  for (let line = startLine; line < startLine + lineCount; line++) {
    const slugs = lines.get(line);
    if (slugs) found[line] = slugs;
  }
  return found;
}
