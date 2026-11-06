// What the search dialog needs before the index has loaded, and what the index builder shares with
// it. No file system access and no MiniSearch import here: lib/search-engine.ts is the part that is
// loaded on first open.

export const SEARCH_INDEX_FILE = "search-index.json";

// "absent" is last so that a technique this code does not use never comes before something it has.
export const RESULT_TYPES = ["concept", "equation", "page", "file", "function", "absent"] as const;
export type ResultType = (typeof RESULT_TYPES)[number];

export const TYPE_LABELS: Record<ResultType, string> = {
  concept: "Concept",
  equation: "Equation",
  page: "Page",
  file: "File",
  function: "Function",
  absent: "Not in This Model",
};

export type SearchDoc = {
  id: string;
  type: ResultType;
  title: string;
  names: string; // function names, ids and navigation labels
  body: string;
  latex: string; // indexed, not stored
  formula: string; // the LaTeX an Equation result shows
  detail: string; // shown under the title
  href: string;
};

export type SearchHit = Pick<SearchDoc, "type" | "title" | "detail" | "formula" | "href"> & {
  id: string;
  terms: string[]; // the indexed terms this result matched
};

export type SearchGroup = { type: ResultType; label: string; total: number; hits: SearchHit[] };
export type SearchResults = { total: number; groups: SearchGroup[]; flat: SearchHit[] };

export const NO_RESULTS: SearchResults = { total: 0, groups: [], flat: [] };

// Splits on everything but letters and digits, so softmax_forward gives softmax and forward, and
// \frac{e_i}{\sum_k e_k} gives frac, e, i, sum, k, e, k.
export function tokenize(text: string): string[] {
  return text.split(/[^\p{L}\p{N}]+/u).filter(Boolean);
}

// Case and punctuation do not count: "Softmax_Forward" and "softmax forward" are the same text.
function comparable(text: string): string {
  return tokenize(text.toLowerCase()).join(" ");
}

// Whether the query is this function's or file's name: a file by its path or base name, a function
// by its title or, for a method, the part after the class. Concepts, equations and pages have no
// such name.
export function isExactName(hit: Pick<SearchHit, "type" | "title">, query: string): boolean {
  const wanted = comparable(query);
  if (!wanted) return false;
  if (hit.type === "file") {
    return wanted === comparable(hit.title) || wanted === comparable(hit.title.split("/").at(-1)!);
  }
  if (hit.type === "function") {
    return wanted === comparable(hit.title) || wanted === comparable(hit.title.split(".").at(-1)!);
  }
  return false;
}

// How exactly a hit matches the whole query: a page's title, then a function or file name, then a
// concept's name. 0 for anything else. A heading is a "page" hit too, with # in its id, and does not
// count as a page title: most concepts have a heading of the same name.
export function exactRank(hit: Pick<SearchHit, "id" | "type" | "title">, query: string): number {
  if (isExactName(hit, query)) return 2;
  const wanted = comparable(query);
  if (!wanted || wanted !== comparable(hit.title)) return 0;
  if (hit.type === "page" && !hit.id.includes("#")) return 3;
  return hit.type === "concept" ? 1 : 0;
}

// `lead` is listed before the other groups, which keep their usual order. Enter opens the first row
// of the flat list, so the group holding that row has to come first on screen too.
export function groupHits(
  hits: SearchHit[],
  limitPerType: number,
  lead?: ResultType,
): SearchResults {
  const groups: SearchGroup[] = [];
  const order = lead ? [lead, ...RESULT_TYPES.filter((type) => type !== lead)] : RESULT_TYPES;
  for (const type of order) {
    const ofType = hits.filter((hit) => hit.type === type);
    if (ofType.length === 0) continue;
    groups.push({
      type,
      label: TYPE_LABELS[type],
      total: ofType.length,
      hits: ofType.slice(0, limitPerType),
    });
  }
  return { total: hits.length, groups, flat: groups.flatMap((group) => group.hits) };
}

export type TextPart = { text: string; match: boolean };

// Case-insensitive. The longest term wins where two overlap.
export function highlightParts(text: string, terms: string[]): TextPart[] {
  const wanted = [...new Set(terms.filter(Boolean))].sort((a, b) => b.length - a.length);
  if (wanted.length === 0) return text ? [{ text, match: false }] : [];
  const pattern = new RegExp(
    `(${wanted.map((term) => term.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|")})`,
    "giu",
  );
  // split with one capture group puts each match at an odd index.
  return text
    .split(pattern)
    .map((part, index) => ({ text: part, match: index % 2 === 1 }))
    .filter((part) => part.text);
}

// Moves the active option, wrapping at both ends. With nothing to select it stays at 0.
export function nextActive(current: number, delta: number, count: number): number {
  if (count === 0) return 0;
  return (current + delta + count) % count;
}

export type KeyInfo = {
  key: string;
  ctrlKey: boolean;
  metaKey: boolean;
  altKey: boolean;
  tagName?: string;
  isContentEditable?: boolean;
};

// Ctrl+K and Cmd+K work anywhere. "/" must still type a slash into a field.
export function opensSearch(event: KeyInfo): boolean {
  if ((event.ctrlKey || event.metaKey) && !event.altKey && event.key.toLowerCase() === "k") {
    return true;
  }
  if (event.key !== "/" || event.ctrlKey || event.metaKey || event.altKey) return false;
  const editable =
    event.isContentEditable === true ||
    ["INPUT", "TEXTAREA", "SELECT"].includes(event.tagName ?? "");
  return !editable;
}
