import { compile } from "@mdx-js/mdx";
import remarkGfm from "remark-gfm";
import { NAV } from "@/data/nav";
import { ABSENT_TERMS } from "@/data/not-in-model";
import {
  callableSymbols,
  explorerFiles,
  explorerHref,
  filePurpose,
  symbolPurpose,
} from "./code-explorer";
import { equationSlug, loadConcepts } from "./concepts";
import { listContentSlugs, readPageBody, readPageMeta } from "./content";
import { loadAbsentTechniques, plainText, readerNote } from "./contract-pages";
import { equationHome } from "./equation-links";
import { withEquation } from "./math-code";
import { pageHref } from "./nav";
import { conceptLinks, readPageSources } from "./reference";
import { remarkHeadings, type PageHeading } from "./remark-headings";
import type { SearchDoc } from "./search";
import { createIndex } from "./search-engine";

// Builds the search documents from the same data the pages use. Nothing here names a concept.

// Over this, scripts/build_search.ts fails the build unless someone decides to raise it.
export const SEARCH_INDEX_LIMIT_BYTES = 500 * 1024;

const REFERENCE_HREF = pageHref("reference/mathematics");
const ABSENT_HREF = pageHref("reference/not-in-this-model");
const DETAIL_LENGTH = 150;

function shorten(text: string): string {
  const flat = text.replace(/\s+/g, " ").trim();
  if (flat.length <= DETAIL_LENGTH) return flat;
  return `${flat.slice(0, flat.lastIndexOf(" ", DETAIL_LENGTH))} ...`;
}

function join(...parts: (string | null | undefined)[]): string {
  return parts.filter(Boolean).join(" ");
}

export function conceptDocuments(): SearchDoc[] {
  const pages = readPageSources();
  return loadConcepts().map((concept) => ({
    id: `concept:${concept.id}`,
    type: "concept",
    title: concept.title,
    names: join(
      concept.id,
      ...concept.source.map((source) => source.symbol),
      ...concept.symbols.map((symbol) => symbol.codeName),
    ),
    body: join(
      concept.summary,
      ...concept.equations.map((equation) => equation.label),
      ...concept.symbols.map((symbol) => symbol.meaning),
    ),
    latex: join(
      ...concept.equations.map((equation) => equation.latex),
      ...concept.symbols.map((symbol) => symbol.symbol),
    ),
    formula: "",
    detail: shorten(concept.summary),
    href: conceptLinks(concept.id, pages).explanation ?? REFERENCE_HREF,
  }));
}

export function equationDocuments(): SearchDoc[] {
  const pages = readPageSources();
  return loadConcepts().flatMap((concept) =>
    concept.equations.map((equation): SearchDoc => {
      const home = equationHome(equation.id, pages);
      return {
        id: `equation:${equation.id}`,
        type: "equation",
        title: equation.label,
        names: equation.id,
        body: concept.title,
        latex: equation.latex,
        formula: equation.latex,
        detail: concept.title,
        href: home ? withEquation(home, equationSlug(equation.id)) : REFERENCE_HREF,
      };
    }),
  );
}

async function pageHeadings(slug: string): Promise<PageHeading[]> {
  const headings: PageHeading[] = [];
  await compile(readPageBody(slug), { remarkPlugins: [remarkGfm, remarkHeadings(headings)] });
  return headings;
}

// The titles of the concepts a page renders: the Attention page explains softmax, and its own
// description does not say so.
function conceptTitlesOnPage(body: string): string[] {
  const ids = new Set([...body.matchAll(/\bconcept="([^"]+)"/g)].map((match) => match[1]));
  return loadConcepts()
    .filter((concept) => ids.has(concept.id))
    .map((concept) => concept.title);
}

export async function pageDocuments(): Promise<SearchDoc[]> {
  const documents: SearchDoc[] = [];
  for (const slug of listContentSlugs()) {
    const meta = readPageMeta(slug);
    const group = NAV.find((candidate) => candidate.pages.some((page) => page.slug === slug));
    const navLabel = group?.pages.find((page) => page.slug === slug)?.title;
    documents.push({
      id: `page:${slug}`,
      type: "page",
      title: meta.title,
      names: join(navLabel, group?.title),
      body: join(meta.description, ...conceptTitlesOnPage(readPageBody(slug))),
      latex: "",
      formula: "",
      detail: shorten(meta.description),
      href: pageHref(slug),
    });
    for (const heading of await pageHeadings(slug)) {
      documents.push({
        id: `page:${slug}#${heading.id}`,
        type: "page",
        title: heading.text,
        names: "",
        body: meta.title,
        latex: "",
        formula: "",
        detail: meta.title,
        href: `${pageHref(slug)}#${heading.id}`,
      });
    }
  }
  return documents;
}

export function fileDocuments(): SearchDoc[] {
  return explorerFiles().map((file) => ({
    id: `file:${file.path}`,
    type: "file",
    title: file.path,
    // Function names here are why a search for a function also finds its file.
    names: join(
      file.module,
      ...callableSymbols(file).map((symbol) => symbol.name),
      ...file.constants.map((constant) => constant.name),
    ),
    body: filePurpose(file),
    latex: "",
    formula: "",
    detail: shorten(filePurpose(file)),
    href: explorerHref(file.path),
  }));
}

export function functionDocuments(): SearchDoc[] {
  return explorerFiles().flatMap((file) => {
    const comments = new Map(
      [...file.functions, ...file.classes.flatMap((cls) => [cls, ...cls.methods])].map((symbol) => [
        symbol.id,
        symbol.leadingComments.join(" "),
      ]),
    );
    return callableSymbols(file).map((symbol): SearchDoc => {
      const purpose = symbolPurpose(symbol.id);
      return {
        id: `function:${symbol.id}`,
        type: "function",
        title: symbol.anchor,
        names: symbol.id,
        // The repository's functions have no docstrings, so this is the comment above each one.
        body: join(purpose, comments.get(symbol.id)),
        latex: "",
        formula: "",
        detail: shorten(join(file.path, purpose ? `- ${purpose}` : null)),
        href: explorerHref(file.path, symbol.anchor),
      };
    });
  });
}

// The rows of Not in This Model. Each result says the code does not use it, whatever was searched.
export function absentDocuments(): SearchDoc[] {
  return loadAbsentTechniques().rows.map((row) => {
    const note = plainText(readerNote(row));
    return {
      id: `absent:${row.technique}`,
      type: "absent",
      title: plainText(row.technique),
      names: join(...(ABSENT_TERMS[row.technique] ?? [])),
      body: note,
      latex: "",
      formula: "",
      detail: shorten(join("Not used by this code.", note)),
      href: ABSENT_HREF,
    };
  });
}

export async function buildSearchDocuments(): Promise<SearchDoc[]> {
  return [
    ...conceptDocuments(),
    ...equationDocuments(),
    ...(await pageDocuments()),
    ...fileDocuments(),
    ...functionDocuments(),
    ...absentDocuments(),
  ];
}

export async function buildSearchIndexJson(): Promise<{ json: string; documents: number }> {
  const documents = await buildSearchDocuments();
  return { json: JSON.stringify(createIndex(documents)), documents: documents.length };
}
