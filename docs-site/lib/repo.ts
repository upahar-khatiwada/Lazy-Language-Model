import { readFileSync } from "node:fs";
import { join } from "node:path";
import { z } from "zod";

// Schemas for data/generated, which scripts/extract_repo.py writes. Never edit that data by hand.

const RepoMetaSchema = z.strictObject({
  sha: z.string().regex(/^[0-9a-f]{40}$/),
  shortSha: z.string().regex(/^[0-9a-f]{7}$/),
  commitDate: z.string(),
  branch: z.string(),
  dirty: z.boolean(),
  dirtyFiles: z.array(z.string()),
  githubBaseUrl: z.url(),
});

const line = z.number().int().positive();

const symbolShape = {
  id: z.string(),
  name: z.string(),
  signature: z.string(),
  decorators: z.array(z.string()),
  startLine: line,
  defLine: line,
  endLine: line,
  source: z.string(),
  leadingComments: z.array(z.string()),
  permalink: z.url(),
};

const FunctionSymbolSchema = z.strictObject({
  ...symbolShape,
  kind: z.enum(["function", "method"]),
});

const ClassSymbolSchema = z.strictObject({
  ...symbolShape,
  kind: z.literal("class"),
  bases: z.array(z.string()),
  isDataclass: z.boolean(),
  fields: z.array(
    z.strictObject({
      name: z.string(),
      annotation: z.string(),
      default: z.string().nullable(),
      line,
    }),
  ),
  methods: z.array(FunctionSymbolSchema),
});

const CodeIndexSchema = z.strictObject({
  sha: z.string(),
  files: z.array(
    z.strictObject({
      path: z.string(),
      module: z.string(),
      lineCount: z.number().int().nonnegative(),
      docstring: z.string().nullable(),
      // The whole file, `lineCount` lines long. Its first line is line 1.
      source: z.string(),
      imports: z.array(
        z.strictObject({
          line,
          module: z.string(),
          name: z.string().nullable(),
          asName: z.string().nullable(),
          internal: z.boolean(),
        }),
      ),
      constants: z.array(
        z.strictObject({
          name: z.string(),
          startLine: line,
          endLine: line,
          source: z.string(),
          value: z.unknown(),
          isLiteral: z.boolean(),
        }),
      ),
      functions: z.array(FunctionSymbolSchema),
      classes: z.array(ClassSymbolSchema),
    }),
  ),
});

const ConfigSchema = z.strictObject({
  path: z.string(),
  constants: z.array(
    z.strictObject({
      name: z.string(),
      value: z.union([z.number(), z.string(), z.boolean(), z.null()]),
      valueText: z.string(),
      line,
      comment: z.string().nullable(),
      section: z.string().nullable(),
      derived: z.boolean(),
      readBy: z.array(z.strictObject({ file: z.string(), line })),
    }),
  ),
});

const CallerSchema = z.strictObject({
  calls: z.array(
    z.strictObject({
      target: z.string(),
      kind: z.enum(["call", "instantiate", "fixture", "reference"]),
      lines: z.array(line),
    }),
  ),
  unresolved: z.array(z.strictObject({ text: z.string(), lines: z.array(line) })),
});

const CallGraphSchema = z.strictObject({
  production: z.record(z.string(), CallerSchema),
  tests: z.record(z.string(), CallerSchema),
  usedBy: z.record(
    z.string(),
    z.strictObject({ production: z.array(z.string()), tests: z.array(z.string()) }),
  ),
});

const TestsIndexSchema = z.strictObject({
  files: z.array(
    z.strictObject({
      path: z.string(),
      module: z.string(),
      repoImports: z.array(
        z.strictObject({ module: z.string(), name: z.string().nullable(), line }),
      ),
      helpers: z.array(
        z.strictObject({
          id: z.string(),
          name: z.string(),
          kind: z.enum(["fixture", "function", "class"]),
          startLine: line,
          endLine: line,
        }),
      ),
      tests: z.array(
        z.strictObject({
          id: z.string(),
          name: z.string(),
          startLine: line,
          endLine: line,
          cases: z.number().int().nonnegative().nullable(),
          fixtures: z.array(z.string()),
          helpers: z.array(z.string()),
          calls: z.array(z.string()),
          reaches: z.array(z.string()),
        }),
      ),
    }),
  ),
});

export type RepoMeta = z.infer<typeof RepoMetaSchema>;
export type CodeIndex = z.infer<typeof CodeIndexSchema>;
export type FunctionSymbol = z.infer<typeof FunctionSymbolSchema>;
export type ClassSymbol = z.infer<typeof ClassSymbolSchema>;
export type RepoConfig = z.infer<typeof ConfigSchema>;
export type CallGraph = z.infer<typeof CallGraphSchema>;
export type TestsIndex = z.infer<typeof TestsIndexSchema>;

function loader<T>(fileName: string, schema: z.ZodType<T>): () => T {
  let cached: T | undefined;
  return () => {
    if (cached !== undefined) return cached;
    const path = join(process.cwd(), "data", "generated", fileName);
    const result = schema.safeParse(JSON.parse(readFileSync(path, "utf-8")));
    if (!result.success) {
      throw new Error(`Invalid data in ${fileName}: ${z.prettifyError(result.error)}`);
    }
    cached = result.data;
    return cached;
  };
}

export const loadRepoMeta = loader("repo-meta.json", RepoMetaSchema);
export const loadCodeIndex = loader("code-index.json", CodeIndexSchema);
export const loadConfig = loader("config.json", ConfigSchema);
export const loadCallGraph = loader("call-graph.json", CallGraphSchema);
export const loadTestsIndex = loader("tests-index.json", TestsIndexSchema);
