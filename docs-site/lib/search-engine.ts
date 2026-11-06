import MiniSearch, { type Options, type SearchOptions } from "minisearch";
import {
  exactRank,
  groupHits,
  tokenize,
  type SearchDoc,
  type SearchHit,
  type SearchResults,
} from "./search";

// The MiniSearch half of search. The dialog imports this module dynamically on first open, so
// MiniSearch stays out of the code every page loads.

const OPTIONS: Options<SearchDoc> = {
  fields: ["title", "names", "body", "latex"],
  storeFields: ["type", "title", "detail", "formula", "href"],
  tokenize,
  processTerm: (term) => term.toLowerCase(),
};

const SEARCH: SearchOptions = {
  // Titles and function names outrank the words around them.
  boost: { title: 6, names: 4 },
  combineWith: "AND",
  prefix: (term) => term.length >= 2,
  fuzzy: (term) => (term.length >= 5 ? 0.15 : false),
};

export const RESULTS_PER_TYPE = 12;

export function createIndex(documents: SearchDoc[]): MiniSearch<SearchDoc> {
  const index = new MiniSearch<SearchDoc>(OPTIONS);
  index.addAll(documents);
  return index;
}

export function loadIndex(json: string): MiniSearch<SearchDoc> {
  return MiniSearch.loadJSON<SearchDoc>(json, OPTIONS);
}

export function runSearch(
  index: MiniSearch<SearchDoc>,
  query: string,
  limitPerType = RESULTS_PER_TYPE,
): SearchResults {
  const wanted = tokenize(query.toLowerCase());
  if (wanted.length === 0) return groupHits([], limitPerType);

  const exactTitle = (title: string) =>
    tokenize(title.toLowerCase()).join(" ") === wanted.join(" ");
  const hits = index
    .search(query, SEARCH)
    .map((result) => ({
      hit: {
        id: String(result.id),
        type: result.type,
        title: result.title,
        detail: result.detail,
        formula: result.formula,
        href: result.href,
        terms: result.terms,
      } as SearchHit,
      score: result.score,
    }))
    // MiniSearch has no notion of "this is exactly what was typed", so that comes first, in the order
    // exactRank gives. Any other title that equals the query comes next.
    .sort(
      (a, b) =>
        exactRank(b.hit, query) - exactRank(a.hit, query) ||
        Number(exactTitle(b.hit.title)) - Number(exactTitle(a.hit.title)) ||
        b.score - a.score,
    )
    .map(({ hit }) => hit);
  const top = hits[0];
  return groupHits(hits, limitPerType, top && exactRank(top, query) > 0 ? top.type : undefined);
}
