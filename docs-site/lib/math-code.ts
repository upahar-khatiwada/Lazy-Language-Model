// The browser half of Math <-> Code linking: pure logic, no DOM and no file access, so tests can run
// it and a client component can import it. The server half is lib/equation-links.ts.

export const EQ_PARAM = "eq";

export type LinkRole = "implements" | "inline" | "call" | "test";

export const ROLE_LABELS: Record<LinkRole, string> = {
  implements: "Computes it",
  inline: "Written inline",
  call: "Calls it",
  test: "Tests it",
};

export type EquationLocation = {
  role: LinkRole;
  file: string;
  functionName: string;
  symbol: string;
  startLine: number;
  endLine: number;
  // The explicit index from the link record, when a match occurs more than once in the function.
  occurrence: number | null;
  permalink: string;
  explorerHref: string;
};

export type EquationDetail = {
  slug: string;
  id: string;
  label: string;
  html: string;
  conceptId: string;
  conceptTitle: string;
  homeHref: string | null;
  locations: EquationLocation[];
  inlineNote: string | null;
};

// Parameters go before the hash, whatever else the address already holds.
export function withEquation(href: string, slug: string | null): string {
  const hashAt = href.indexOf("#");
  const hash = hashAt === -1 ? "" : href.slice(hashAt);
  const rest = hashAt === -1 ? href : href.slice(0, hashAt);
  const queryAt = rest.indexOf("?");
  const path = queryAt === -1 ? rest : rest.slice(0, queryAt);
  const params = new URLSearchParams(queryAt === -1 ? "" : rest.slice(queryAt + 1));
  if (slug) params.set(EQ_PARAM, slug);
  else params.delete(EQ_PARAM);
  const query = params.toString();
  return `${path}${query ? `?${query}` : ""}${hash}`;
}

// A line lists its equations in a space-separated attribute.
export function splitEquations(attribute: string | undefined | null): string[] {
  return attribute ? attribute.split(" ").filter(Boolean) : [];
}

// Hover wins over keyboard focus, and either wins over the pinned equation while it lasts.
export function activeEquation(
  pinned: string | null,
  focused: string | null,
  hovered: string | null,
): string | null {
  return hovered ?? focused ?? pinned;
}

export function detailUrl(slug: string, basePath = ""): string {
  return `${basePath}/math-code/${slug}.json`;
}

export function lineCountText(count: number): string {
  if (count === 0) return "Its code is not on this page";
  return count === 1 ? "1 code line on this page" : `${count} code lines on this page`;
}

export function announcement(label: string, count: number): string {
  return `Equation highlighted: ${label}. ${lineCountText(count)}.`;
}

// The first line after the equation in page order, or the first on the page if none follows.
export function scrollTargetIndex(linesAfterEquation: boolean[]): number {
  if (linesAfterEquation.length === 0) return -1;
  const next = linesAfterEquation.indexOf(true);
  return next === -1 ? 0 : next;
}
