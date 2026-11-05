// Computes every concept example by running the repository function.
//   bun run sync:examples    writes data/golden/examples.json
//   bun run verify:examples  recomputes and fails if the stored numbers differ
import { spawnSync } from "node:child_process";
import { readFileSync, writeFileSync } from "node:fs";
import { loadConcepts } from "@/lib/concepts";
import { EXAMPLES_PATH, StoredExamplesSchema, type StoredExamples } from "@/lib/examples";

const TOLERANCE = 1e-12;

function compute(): StoredExamples {
  const withExample = loadConcepts().flatMap((concept) =>
    concept.example ? [{ id: concept.id, example: concept.example }] : [],
  );

  const run = spawnSync("python", ["-B", "scripts/run_example.py"], {
    encoding: "utf-8",
    input: JSON.stringify(
      withExample.map(({ id, example }) => ({
        conceptId: id,
        call: example.call,
        input: example.input,
        captures: example.steps.map((step) => ({ id: step.id, capture: step.capture })),
      })),
    ),
  });
  if (run.status !== 0) throw new Error(`run_example.py failed:\n${run.stderr || run.error}`);

  const results = JSON.parse(run.stdout) as Record<
    string,
    { steps: StoredExamples[string]["steps"]; output: StoredExamples[string]["output"] }
  >;
  const stored: StoredExamples = {};
  for (const { id, example } of withExample) {
    const result = results[id];
    if (!result) throw new Error(`run_example.py returned nothing for ${id}`);
    stored[id] = {
      kind: example.kind,
      call: example.call,
      input: example.input,
      steps: result.steps,
      output: result.output,
    };
  }
  return StoredExamplesSchema.parse(stored);
}

function differences(path: string, a: unknown, b: unknown): string[] {
  if (typeof a === "number" && typeof b === "number") {
    return Math.abs(a - b) <= TOLERANCE ? [] : [`${path}: stored ${a}, computed ${b}`];
  }
  if (typeof a === "object" && a !== null && typeof b === "object" && b !== null) {
    const keys = new Set([...Object.keys(a), ...Object.keys(b)]);
    return [...keys].flatMap((key) =>
      differences(
        `${path}.${key}`,
        (a as Record<string, unknown>)[key],
        (b as Record<string, unknown>)[key],
      ),
    );
  }
  return a === b ? [] : [`${path}: stored ${String(a)}, computed ${String(b)}`];
}

function main(): number {
  const computed = compute();

  if (!process.argv.includes("--check")) {
    writeFileSync(EXAMPLES_PATH, `${JSON.stringify(computed, null, 2)}\n`);
    console.log(`Wrote ${Object.keys(computed).length} example(s) to data/golden/examples.json`);
    return 0;
  }

  const stored: unknown = JSON.parse(readFileSync(EXAMPLES_PATH, "utf-8"));
  const problems = differences("examples", stored, computed);
  if (problems.length > 0) {
    console.error("FAILED: stored examples differ from a fresh run. Run: bun run sync:examples");
    for (const problem of problems) console.error(`  ${problem}`);
    return 1;
  }
  console.log(`OK: ${Object.keys(computed).length} example(s) match a fresh run.`);
  return 0;
}

try {
  process.exitCode = main();
} catch (error) {
  console.error(`FAILED: ${(error as Error).message}`);
  process.exitCode = 1;
}
