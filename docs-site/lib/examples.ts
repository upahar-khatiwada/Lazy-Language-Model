import { readFileSync } from "node:fs";
import { join } from "node:path";
import { z } from "zod";

// data/golden/examples.json holds numbers computed by running repository functions.
// `bun run sync:examples` writes it; never edit it by hand.

const value = z.union([
  z.number(),
  z.array(z.number()),
  z.array(z.array(z.number())),
  z.array(z.array(z.array(z.number()))),
]);

const StoredExampleSchema = z.strictObject({
  kind: z.enum(["illustrative", "actual"]),
  call: z.string(),
  input: z.record(z.string(), value),
  steps: z.record(z.string(), value),
  output: value,
});

export const StoredExamplesSchema = z.record(z.string(), StoredExampleSchema);

export type StoredExample = z.infer<typeof StoredExampleSchema>;
export type StoredExamples = z.infer<typeof StoredExamplesSchema>;

export const EXAMPLES_PATH = join(process.cwd(), "data", "golden", "examples.json");

let cached: StoredExamples | undefined;

export function loadStoredExamples(): StoredExamples {
  if (cached) return cached;
  const result = StoredExamplesSchema.safeParse(JSON.parse(readFileSync(EXAMPLES_PATH, "utf-8")));
  if (!result.success) {
    throw new Error(`Invalid data/golden/examples.json: ${z.prettifyError(result.error)}`);
  }
  cached = result.data;
  return cached;
}

export function getStoredExample(conceptId: string): StoredExample {
  const example = loadStoredExamples()[conceptId];
  if (!example) throw new Error(`No computed example for ${conceptId}. Run: bun run sync:examples`);
  return example;
}

type Nested = number | Nested[];

export function formatNumbers(input: Nested, precision: number): string {
  if (typeof input === "number") return input.toFixed(precision);
  return `[${input.map((item) => formatNumbers(item, precision)).join(", ")}]`;
}
