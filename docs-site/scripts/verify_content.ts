// `bun run verify:content`: checks that the documentation matches the repository. The checks are in
// lib/content-checks.ts. Exits non-zero on any failure; a stale warning alone does not fail it.
import { exitCode, formatResults } from "@/lib/content-checks";
import { runContentChecks } from "@/lib/content-input";

try {
  const results = runContentChecks();
  for (const line of formatResults(results)) console.log(line);
  process.exitCode = exitCode(results);
} catch (error) {
  console.error(`FAILED: ${(error as Error).message}`);
  process.exitCode = 1;
}
