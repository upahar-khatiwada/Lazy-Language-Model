// Fails the build when hand-written data no longer matches the code at the pinned commit.
import { loadSymbolIndex } from "@/lib/code-targets";
import { loadConcepts, loadPipelines } from "@/lib/concepts";
import { loadStoredExamples } from "@/lib/examples";
import { loadLinks } from "@/lib/links";
import { verifyData } from "@/lib/verify-links";

function main(): number {
  const input = {
    concepts: loadConcepts(),
    links: loadLinks(),
    pipelines: loadPipelines(),
    symbols: loadSymbolIndex(),
    examples: loadStoredExamples(),
  };
  const problems = verifyData(input);

  if (problems.length > 0) {
    console.error(`FAILED: ${problems.length} problem(s) in data/concepts, links or pipelines.`);
    for (const problem of problems) console.error(`  ${problem}`);
    return 1;
  }
  console.log(
    `OK: ${input.links.length} links, ${input.concepts.length} concepts and ` +
      `${input.pipelines.length} pipelines match the code.`,
  );
  return 0;
}

try {
  process.exitCode = main();
} catch (error) {
  console.error(`FAILED: ${(error as Error).message}`);
  process.exitCode = 1;
}
