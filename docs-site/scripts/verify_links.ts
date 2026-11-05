// Fails the build when hand-written data no longer matches the code at the pinned commit.
import { loadSymbolIndex } from "@/lib/code-targets";
import { loadConcepts, loadPipelines } from "@/lib/concepts";
import { loadStoredExamples } from "@/lib/examples";
import { loadLinks } from "@/lib/links";
import { navSlugs } from "@/lib/pages";
import { loadInitParams } from "@/lib/parameters";
import { loadInventory } from "@/lib/reference";
import { loadCallGraph, loadCodeIndex, loadConfig } from "@/lib/repo";
import { verifySiteData } from "@/lib/site-verify";
import { verifyData } from "@/lib/verify-links";

function main(): number {
  const input = {
    concepts: loadConcepts(),
    links: loadLinks(),
    pipelines: loadPipelines(),
    symbols: loadSymbolIndex(),
    examples: loadStoredExamples(),
  };
  const problems = [
    ...verifyData(input),
    ...verifySiteData({
      symbols: input.symbols,
      callGraph: loadCallGraph(),
      config: loadConfig(),
      navSlugs: navSlugs(),
      parameterNames: loadInitParams().parameters.map((parameter) => parameter.name),
      repoFiles: loadCodeIndex().files.map((file) => file.path),
      concepts: input.concepts,
      inventoryIds: loadInventory().rows.map((row) => row.id),
    }),
  ];

  if (problems.length > 0) {
    console.error(`FAILED: ${problems.length} problem(s) in hand-written data.`);
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
