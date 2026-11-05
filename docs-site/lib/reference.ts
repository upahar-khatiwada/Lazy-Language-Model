import { readFileSync } from "node:fs";
import { join } from "node:path";
import { z } from "zod";
import { formatLineRange, requireTarget, type SymbolIndex, loadSymbolIndex } from "./code-targets";
import { conceptAnchor, exampleAnchor, type ConceptCategory, type MathConcept } from "./concepts";
import { listContentSlugs, readPageBody } from "./content";
import { formatShape, resolveDimension, type DimensionName } from "./dimensions";
import { pageHref } from "./nav";

// The Mathematics Reference is generated: nothing here names a concept.

export const FILTER_LABELS = {
  tokenization: "Tokenization",
  data: "Data",
  operations: "Operations",
  embeddings: "Embeddings",
  attention: "Attention",
  transformer: "Transformer",
  loss: "Loss",
  optimization: "Optimization",
  generation: "Generation",
  // Decision D2: test-only mathematics is kept apart from the pipeline.
  testing: "Testing (test-only)",
} as const;

export type ReferenceFilter = keyof typeof FILTER_LABELS;

// A Record, so a new concept category does not compile until it has a filter.
const FILTER_OF: Record<ConceptCategory, ReferenceFilter> = {
  tokenizer: "tokenization",
  data: "data",
  initialization: "operations",
  "basic-operations": "operations",
  embeddings: "embeddings",
  attention: "attention",
  "feed-forward-and-block": "transformer",
  "whole-model": "transformer",
  loss: "loss",
  optimizer: "optimization",
  sampling: "generation",
  "gradient-checking": "testing",
};

export const DIRECTION_LABELS = {
  forward: "Forward",
  backward: "Backward",
  none: "Neither (not differentiated)",
} as const;

export type ReferenceDirection = keyof typeof DIRECTION_LABELS;

// The rows of contract/MATH_INVENTORY.md, from data/derived. contract/ itself is git-ignored.
const InventorySchema = z.strictObject({
  rows: z.array(z.strictObject({ id: z.string(), name: z.string(), group: z.string() })),
});

export type Inventory = z.infer<typeof InventorySchema>;

let inventory: Inventory | undefined;

export function loadInventory(): Inventory {
  inventory ??= InventorySchema.parse(
    JSON.parse(
      readFileSync(join(process.cwd(), "data", "derived", "math-inventory.json"), "utf-8"),
    ),
  );
  return inventory;
}

export function coverage(concepts: MathConcept[], rows: Inventory["rows"]) {
  const documented = new Set(concepts.flatMap((concept) => concept.inventory));
  return {
    documented: rows.filter((row) => documented.has(row.id)).length,
    total: rows.length,
  };
}

export type PageSource = { slug: string; body: string };

export function readPageSources(): PageSource[] {
  return listContentSlugs().map((slug) => ({ slug, body: readPageBody(slug) }));
}

// A page explains a concept if it renders one of the concept's components.
export function conceptLinks(conceptId: string, pages: PageSource[]) {
  const attribute = `concept="${conceptId}"`;
  const withExample = pages.find((page) => page.body.includes(`<NumericExample ${attribute}`));
  const explaining =
    pages.find((page) => page.body.includes(`<ConceptCard ${attribute}`)) ??
    pages.find((page) => page.body.includes(attribute));
  return {
    explanation:
      explaining &&
      `${pageHref(explaining.slug)}${
        explaining.body.includes(`<ConceptCard ${attribute}`) ? `#${conceptAnchor(conceptId)}` : ""
      }`,
    example: withExample && `${pageHref(withExample.slug)}#${exampleAnchor(conceptId)}`,
  };
}

function withValues(dims: DimensionName[]): string | null {
  const values = dims.map((name) => resolveDimension(name).value);
  if (values.every((value) => value === null)) return null;
  return formatShape(dims.map((name, index) => values[index] ?? name));
}

export type ReferenceRow = ReturnType<typeof buildReferenceRow>;

export function buildReferenceRow(
  concept: MathConcept,
  pages: PageSource[],
  symbols: SymbolIndex = loadSymbolIndex(),
) {
  const directions = [...new Set(concept.equations.map((equation) => equation.role))];
  const sources = concept.source.map(({ symbol }) => {
    const code = symbols.get(symbol);
    if (!code) throw new Error(`Concept ${concept.id}: unknown source symbol ${symbol}`);
    return {
      id: symbol,
      file: code.file,
      name: code.name,
      lines: formatLineRange(code.startLine, code.endLine),
      permalink: code.permalink,
    };
  });
  const shapes = concept.shapes.map((shape) => {
    const target = requireTarget(shape.target, symbols);
    return {
      id: shape.id,
      label: shape.label,
      permalink: target.permalink,
      stages: shape.stages.map((stage) => ({
        name: stage.name,
        symbolic: formatShape(stage.dims),
        values: withValues(stage.dims),
      })),
    };
  });
  const search = [
    concept.title,
    concept.summary,
    ...concept.equations.flatMap((equation) => [equation.label, equation.latex]),
    ...sources.flatMap((source) => [source.file, source.name]),
  ]
    .join(" ")
    .toLowerCase();

  return {
    id: concept.id,
    title: concept.title,
    status: concept.status,
    filter: FILTER_OF[concept.category],
    directions,
    equationIds: concept.equations.map((equation) => equation.id),
    hasExample: concept.example !== null,
    sources,
    shapes,
    search,
    ...conceptLinks(concept.id, pages),
  };
}
