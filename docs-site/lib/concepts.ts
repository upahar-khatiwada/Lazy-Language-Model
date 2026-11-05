import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { parse } from "yaml";
import { z } from "zod";
import { CodeTargetSchema, codeTargetShape } from "./code-targets";
import { DIMENSION_NAMES } from "./dimensions";

// Hand-written data. Every object is strict, so a stray `line` or `startLine` key is an error:
// line numbers are only ever resolved from data/generated/code-index.json.

const text = z.string().min(1);

// The groups of contract/MATH_INVENTORY.md.
export const CATEGORY_LABELS = {
  tokenizer: "Tokenizer",
  data: "Data",
  initialization: "Initialization",
  "basic-operations": "Basic operations",
  embeddings: "Embeddings",
  attention: "Attention",
  "feed-forward-and-block": "Feed-forward and block",
  "whole-model": "Whole model",
  optimizer: "Optimizer",
  sampling: "Sampling",
  "gradient-checking": "Gradient checking",
} as const;

const categories = Object.keys(CATEGORY_LABELS) as [
  keyof typeof CATEGORY_LABELS,
  ...(keyof typeof CATEGORY_LABELS)[],
];

const EquationSchema = z.strictObject({
  id: text,
  latex: text,
  // "none" is for operations outside differentiation: MATH_INVENTORY.md marks those "n/a".
  role: z.enum(["forward", "backward", "none"]),
  label: text,
});

const SymbolSchema = z.strictObject({
  symbol: text, // LaTeX
  meaning: text,
  codeName: text.nullable(), // null when the code does not name the value
  shape: text,
});

const SourceSchema = z.strictObject({ file: text, symbol: text });

const UsedBySchema = z.strictObject({ ...codeTargetShape, note: text });

const PdfRefSchema = z.strictObject({
  section: text,
  equations: z.array(text), // printed numbers; empty when the PDF only states the idea in prose
  page: z.number().int().positive(),
});

const ShapeSchema = z.strictObject({
  id: text,
  label: text,
  stages: z
    .array(z.strictObject({ name: text, dims: z.array(z.enum(DIMENSION_NAMES)).min(1) }))
    .min(2),
  target: CodeTargetSchema,
});

const NumericValueSchema = z.union([z.number(), z.array(z.number()), z.array(z.array(z.number()))]);

export type NumericValue = z.infer<typeof NumericValueSchema>;

const exampleShape = {
  call: text, // the repository function that computes the numbers
  input: z.record(z.string(), NumericValueSchema), // arguments by parameter name
  steps: z
    .array(
      z.strictObject({
        id: text,
        label: text,
        equationId: text,
        capture: text, // the name of a local variable in `call`
      }),
    )
    .min(1),
  output: z.strictObject({ label: text }),
  precision: z.number().int().min(0).max(8),
};

const ExampleSchema = z.discriminatedUnion("kind", [
  z.strictObject({ ...exampleShape, kind: z.literal("illustrative") }),
  // Decision D1: a value from a run must say where it came from.
  z.strictObject({ ...exampleShape, kind: z.literal("actual"), provenance: text }),
]);

export type Example = z.infer<typeof ExampleSchema>;
export type ExampleKind = Example["kind"];

export const MathConceptSchema = z
  .strictObject({
    id: text,
    title: text,
    category: z.enum(categories),
    status: z.enum(["implemented", "derived"]),
    summary: text,
    why: text,
    equations: z.array(EquationSchema).min(1),
    symbols: z.array(SymbolSchema),
    example: ExampleSchema.nullable(),
    shapes: z.array(ShapeSchema),
    source: z.array(SourceSchema).min(1),
    usedBy: z.array(UsedBySchema),
    pdfRef: PdfRefSchema.nullable(),
    notes: z.array(text),
  })
  .refine((concept) => concept.status === "derived" || concept.pdfRef !== null, {
    message: 'status "implemented" means the PDF states this mathematics, so pdfRef is required',
    path: ["pdfRef"],
  });

export type MathConcept = z.infer<typeof MathConceptSchema>;
export type Equation = MathConcept["equations"][number];

export const PipelineSchema = z.strictObject({
  id: text,
  title: text,
  steps: z
    .array(
      z.strictObject({
        label: text,
        dims: z.array(z.enum(DIMENSION_NAMES)).min(1).optional(),
        target: CodeTargetSchema,
      }),
    )
    .min(2),
});

export type Pipeline = z.infer<typeof PipelineSchema>;

export function readYamlFiles<T>(folder: string, schema: z.ZodType<T>): T[] {
  const dir = join(process.cwd(), "data", folder);
  return readdirSync(dir)
    .filter((name) => /\.ya?ml$/.test(name))
    .sort()
    .map((name) => {
      const result = schema.safeParse(parse(readFileSync(join(dir, name), "utf-8")));
      if (!result.success) {
        throw new Error(`Invalid data in data/${folder}/${name}: ${z.prettifyError(result.error)}`);
      }
      return result.data;
    });
}

let concepts: MathConcept[] | undefined;
let pipelines: Pipeline[] | undefined;

export function loadConcepts(): MathConcept[] {
  concepts ??= readYamlFiles("concepts", MathConceptSchema);
  return concepts;
}

export function loadPipelines(): Pipeline[] {
  pipelines ??= readYamlFiles("pipelines", PipelineSchema);
  return pipelines;
}

export function getConcept(id: string): MathConcept {
  const concept = loadConcepts().find((candidate) => candidate.id === id);
  if (!concept) throw new Error(`Unknown concept: ${id}`);
  return concept;
}

export function getEquation(id: string): { concept: MathConcept; equation: Equation } {
  for (const concept of loadConcepts()) {
    const equation = concept.equations.find((candidate) => candidate.id === id);
    if (equation) return { concept, equation };
  }
  throw new Error(`Unknown equation: ${id}`);
}

export function getPipeline(id: string): Pipeline {
  const pipeline = loadPipelines().find((candidate) => candidate.id === id);
  if (!pipeline) throw new Error(`Unknown pipeline: ${id}`);
  return pipeline;
}

// The anchor of an equation: eq-softmax-shift.
export function equationAnchor(id: string): string {
  return `eq-${id.replace(/[^a-zA-Z0-9]+/g, "-")}`;
}
