import katex from "katex";
import { formatLineRange, loadSymbolIndex, type SymbolIndex } from "./code-targets";
import { explorerHref } from "./code-explorer";
import {
  equationAnchor,
  equationSlug,
  getEquation,
  loadConcepts,
  type MathConcept,
} from "./concepts";
import { linesCoveredByLink, linksForEquation, type Link } from "./links";
import { withEquation, type EquationDetail, type EquationLocation } from "./math-code";
import { pageHref } from "./nav";
import { conceptLinks, readPageSources, type PageSource } from "./reference";
import { loadCallGraph, type CallGraph } from "./repo";

// The server half of Math <-> Code linking. Line numbers are resolved from link records here and
// never stored. Nothing here names an equation or a file.

const ROLE_ORDER = ["implements", "inline", "call", "test"] as const;

export function equationLocations(equationId: string): EquationLocation[] {
  const slug = equationSlug(equationId);
  return linksForEquation(equationId)
    .sort((a, b) => ROLE_ORDER.indexOf(a.role) - ROLE_ORDER.indexOf(b.role))
    .map((link) => {
      const { symbol, startLine, endLine, permalink } = link.resolved;
      return {
        role: link.role,
        file: symbol.file,
        functionName: symbol.name,
        symbol: symbol.id,
        startLine,
        endLine,
        occurrence: link.occurrence ?? null,
        permalink,
        explorerHref: withEquation(explorerHref(symbol.file, `L${startLine}`), slug),
      };
    });
}

export function conceptLocations(concept: MathConcept) {
  return concept.equations.flatMap((equation) =>
    equationLocations(equation.id).map((location) => ({ equation, location })),
  );
}

let pagesCache: PageSource[] | undefined;
function cachedPages(): PageSource[] {
  pagesCache ??= readPageSources();
  return pagesCache;
}

// The page that renders the equation itself, with its anchor. Null when no page does.
function pageShowingEquation(equationId: string, pages: PageSource[]): string | null {
  const page = pages.find(
    ({ body }) =>
      body.includes(`<MathBlock id="${equationId}"`) ||
      body.includes(`<MathCodeLink equation="${equationId}"`),
  );
  return page ? `${pageHref(page.slug)}#${equationAnchor(equationId)}` : null;
}

// Where a code line sends the reader: the equation, or else the explanation of its concept.
export function equationHome(
  equationId: string,
  pages: PageSource[] = cachedPages(),
): string | null {
  const { concept } = getEquation(equationId);
  return (
    pageShowingEquation(equationId, pages) ?? conceptLinks(concept.id, pages).explanation ?? null
  );
}

// A note for an equation that has a helper function and is also written in place. It comes from
// the link records and the call graph, so it cannot drift from the code.
export function inlineEquivalentNote(
  equationId: string,
  callGraph: CallGraph = loadCallGraph(),
): string | null {
  const locations = equationLocations(equationId);
  const inline = locations.filter((location) => location.role === "inline");
  if (inline.length === 0) return null;
  const helpers = [
    ...new Set(
      locations.filter((location) => location.role === "implements").map(({ symbol }) => symbol),
    ),
  ].filter((symbol) => !inline.some((location) => location.symbol === symbol));
  if (helpers.length === 0) return null;

  const sites = [
    ...new Set(
      inline.map(
        ({ file, startLine, endLine }) => `${file}:${formatLineRange(startLine, endLine)}`,
      ),
    ),
  ].join(", ");
  const uncalled = helpers.every(
    (helper) => (callGraph.usedBy[helper]?.production ?? []).length === 0,
  );
  return uncalled
    ? `Inline equivalent: the pipeline does not call ${helpers.join(" or ")}. It computes the same formula in place at ${sites}.`
    : `Inline equivalent: the pipeline calls ${helpers.join(" and ")}, and also computes the same formula in place, with no call, at ${sites}.`;
}

export function equationDetail(slug: string): EquationDetail | null {
  for (const concept of loadConcepts()) {
    for (const equation of concept.equations) {
      if (equationSlug(equation.id) !== slug) continue;
      const home = equationHome(equation.id);
      return {
        slug,
        id: equation.id,
        label: equation.label,
        html: katex.renderToString(equation.latex, { displayMode: true, throwOnError: true }),
        conceptId: concept.id,
        conceptTitle: concept.title,
        homeHref: home ? withEquation(home, slug) : null,
        locations: equationLocations(equation.id),
        inlineNote: inlineEquivalentNote(equation.id),
      };
    }
  }
  return null;
}

export function allEquationSlugs(): string[] {
  return loadConcepts().flatMap((concept) =>
    concept.equations.map((equation) => equationSlug(equation.id)),
  );
}

export type ReachInput = {
  concepts: MathConcept[];
  links: Link[];
  pages: PageSource[];
  files: string[];
  symbols?: SymbolIndex;
};

// Every link record must be reachable from both directions. Runs in verify:links.
export function verifyEquationReach({
  concepts,
  links,
  pages,
  files,
  symbols = loadSymbolIndex(),
}: ReachInput): string[] {
  const problems: string[] = [];
  const known = new Set(files);

  const seen = new Map<string, string>();
  for (const concept of concepts) {
    for (const equation of concept.equations) {
      const slug = equationSlug(equation.id);
      const other = seen.get(slug);
      if (other && other !== equation.id) {
        problems.push(`equations ${other} and ${equation.id} share the deep link ?eq=${slug}`);
      }
      seen.set(slug, equation.id);
    }
    // From an equation to its code: the Trace block holds the Implemented in panel.
    if (!pages.some(({ body }) => body.includes(`<Trace concept="${concept.id}"`))) {
      problems.push(`concept ${concept.id}: no page shows its Implemented in panel`);
    }
  }

  for (const link of links) {
    const where = `link for ${link.equationId}`;
    if (!known.has(link.file)) problems.push(`${where}: ${link.file} is not in the Code Explorer`);
    if (linesCoveredByLink(link, symbols).length === 0) {
      problems.push(`${where}: resolves to no line`);
    }

    // From a code line back to its equation: the popover links to the page showing it.
    const concept = concepts.find((candidate) =>
      candidate.equations.some((equation) => equation.id === link.equationId),
    );
    if (
      !concept ||
      !(pageShowingEquation(link.equationId, pages) ?? conceptLinks(concept.id, pages).explanation)
    ) {
      problems.push(`${where}: no page to open from the code line`);
    }
  }
  return problems;
}
