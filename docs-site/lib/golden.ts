import { readFileSync } from "node:fs";
import { join } from "node:path";
import { z } from "zod";

// data/golden/{ops,attention,optim,sampling,bpe}.json hold results of running the repository's real
// functions on hand-picked inputs. `bun run sync:golden` writes them; never edit them by hand.
// This module reads files, so only server components and tests may import it.

const CaseSchema = z.strictObject({
  id: z.string(),
  mirrors: z.string(),
  kind: z.literal("illustrative"),
  inputs: z.record(z.string(), z.unknown()),
  outputs: z.record(z.string(), z.unknown()),
  shapes: z.strictObject({
    inputs: z.record(z.string(), z.unknown()),
    outputs: z.record(z.string(), z.unknown()),
  }),
});

const FileSchema = z.strictObject({ cases: z.array(CaseSchema) });

export type GoldenCase = z.infer<typeof CaseSchema>;

export const GOLDEN_FILES = ["ops", "attention", "optim", "sampling", "bpe"] as const;
export type GoldenFile = (typeof GOLDEN_FILES)[number];

// JSON has no infinity, so make_golden.py stores -inf as this string.
function decode(value: unknown): unknown {
  if (value === "-Infinity") return -Infinity;
  if (Array.isArray(value)) return value.map(decode);
  if (typeof value === "object" && value !== null) {
    return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, decode(item)]));
  }
  return value;
}

const cached = new Map<GoldenFile, GoldenCase[]>();

export function loadGolden(file: GoldenFile): GoldenCase[] {
  const hit = cached.get(file);
  if (hit) return hit;
  const path = join(process.cwd(), "data", "golden", `${file}.json`);
  const parsed = FileSchema.safeParse(JSON.parse(readFileSync(path, "utf-8")));
  if (!parsed.success) {
    throw new Error(`Invalid data/golden/${file}.json: ${z.prettifyError(parsed.error)}`);
  }
  const cases = parsed.data.cases.map((item) => ({
    ...item,
    inputs: decode(item.inputs) as GoldenCase["inputs"],
    outputs: decode(item.outputs) as GoldenCase["outputs"],
  }));
  cached.set(file, cases);
  return cases;
}

export function getGoldenCase(file: GoldenFile, id: string): GoldenCase {
  const found = loadGolden(file).find((item) => item.id === id);
  if (!found) throw new Error(`No golden case ${id} in ${file}. Run: bun run sync:golden`);
  return found;
}
