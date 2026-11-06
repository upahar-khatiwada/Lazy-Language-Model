import { readFileSync } from "node:fs";
import { join } from "node:path";
import { z } from "zod";

// Snapshots of contract/PDF_VS_CODE.md and contract/DISCREPANCIES.md, from data/derived. contract/ is
// git-ignored, so scripts/sync_derived.py copies what the two reference pages need. Reads files, so
// only server components and tests may import it.

const AbsentSchema = z.strictObject({
  rows: z.array(z.strictObject({ technique: z.string(), inPdf: z.string(), note: z.string() })),
});

const BlockSchema = z.discriminatedUnion("type", [
  z.strictObject({ type: z.literal("paragraph"), text: z.string() }),
  z.strictObject({ type: z.literal("list"), items: z.array(z.string()) }),
  z.strictObject({
    type: z.literal("table"),
    headers: z.array(z.string()),
    rows: z.array(z.array(z.string())),
  }),
]);

const DiscrepanciesSchema = z.strictObject({
  preamble: z.array(z.string()),
  sections: z.array(
    z.strictObject({ number: z.number(), title: z.string(), blocks: z.array(BlockSchema) }),
  ),
});

export type AbsentTechniques = z.infer<typeof AbsentSchema>;
export type Discrepancies = z.infer<typeof DiscrepanciesSchema>;
export type DiscrepancyBlock = z.infer<typeof BlockSchema>;

function read<T>(name: string, schema: z.ZodType<T>): T {
  const path = join(process.cwd(), "data", "derived", name);
  return schema.parse(JSON.parse(readFileSync(path, "utf-8")));
}

let absent: AbsentTechniques | undefined;
let discrepancies: Discrepancies | undefined;

export function loadAbsentTechniques(): AbsentTechniques {
  absent ??= read("absent-techniques.json", AbsentSchema);
  return absent;
}

export function loadDiscrepancies(): Discrepancies {
  discrepancies ??= read("discrepancies.json", DiscrepanciesSchema);
  return discrepancies;
}

export type InlinePart = { text: string; style: "plain" | "code" | "bold" };

// The contract's inline Markdown: `code` and **bold**. A ** inside code stays code. Nothing else is
// interpreted, and no text is dropped.
export function inlineParts(text: string): InlinePart[] {
  const parts: InlinePart[] = [];
  text.split("`").forEach((piece, index) => {
    if (index % 2 === 1) {
      parts.push({ text: piece, style: "code" });
      return;
    }
    piece.split("**").forEach((inner, innerIndex) => {
      if (inner) parts.push({ text: inner, style: innerIndex % 2 === 1 ? "bold" : "plain" });
    });
  });
  return parts;
}
