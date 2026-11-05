import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, test } from "vitest";
import { loadCodeIndex } from "@/lib/repo";

const DIR = join(process.cwd(), "lib", "math");

const symbols = new Set<string>();
for (const file of loadCodeIndex().files) {
  for (const fn of file.functions) symbols.add(fn.id);
  for (const cls of file.classes) for (const method of cls.methods) symbols.add(method.id);
}

// Every exported function, with the comment block directly above it.
function exportedFunctions(source: string) {
  const lines = source.split(/\r?\n/);
  return lines.flatMap((line, index) => {
    const name = /^export function (\w+)/.exec(line)?.[1];
    if (!name) return [];
    const comment: string[] = [];
    for (let i = index - 1; i >= 0 && lines[i]!.startsWith("//"); i--) comment.unshift(lines[i]!);
    return [{ name, comment: comment.join(" ") }];
  });
}

const files = readdirSync(DIR).filter((name) => name.endsWith(".ts"));

test("the engine has the expected files", () => {
  expect(files.sort()).toEqual([
    "attention.ts",
    "bpe.ts",
    "data.ts",
    "embeddings.ts",
    "gradcheck.ts",
    "ops.ts",
    "optim.ts",
    "params.ts",
    "sampling.ts",
    "tensor.ts",
  ]);
});

describe.each(files)("%s", (file) => {
  const functions = exportedFunctions(readFileSync(join(DIR, file), "utf-8"));

  test("exports functions", () => {
    expect(functions.length).toBeGreaterThan(0);
  });

  test.each(functions)("$name names the Python function it mirrors", ({ comment }) => {
    expect(comment).toMatch(/^\/\/ Mirrors /);
    const named = comment
      .split(/\s+/)
      .map((word) => word.replace(/[.,:;]+$/, ""))
      .filter((word) => symbols.has(word));
    expect(named.length, comment).toBeGreaterThan(0);
  });

  test("uses no libraries, so it can run in the browser", () => {
    const imports = [...readFileSync(join(DIR, file), "utf-8").matchAll(/from "([^"]+)"/g)];
    for (const [, path] of imports) expect(path!.startsWith("./")).toBe(true);
  });
});
