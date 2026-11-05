import { z } from "zod";
import {
  codeTargetShape,
  requireTarget,
  type CodeTarget,
  type ResolvedTarget,
} from "./code-targets";
import { loadConcepts, readYamlFiles } from "./concepts";

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
