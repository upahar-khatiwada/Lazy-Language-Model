import { execFileSync } from "node:child_process";
import { readdirSync, readFileSync } from "node:fs";
import { join, relative, sep } from "node:path";
import { parse } from "yaml";
import { z } from "zod";
import { CODE_REFS } from "@/data/code-refs";
import { COVERAGE_ALLOWLIST } from "@/data/coverage-allowlist";
import { ABSENT_TERMS, SCAN_EXEMPT, TASK_NAMED_TERMS } from "@/data/not-in-model";
import { SHAPE_RUN_REQUIRED_FUNCTIONS, SHAPE_STAGE_ALIASES } from "@/data/shape-stages";
import { loadActual } from "./actual";
import { loadSymbolIndex } from "./code-targets";
import { loadConcepts } from "./concepts";
import {
  checkCoverage,
  checkEquations,
  checkForbiddenTopics,
  checkFreshness,
  checkLabels,
  checkShapes,
  checkSymbolsAndLinks,
  COVERED_MODULES,
  isPublic,
  type CheckResult,
  type LinkFile,
  type PublicSymbol,
  type ShapeRuns,
  type TextFile,
} from "./content-checks";
import { loadAbsentTechniques } from "./contract-pages";
import { resolveDimension, DIMENSION_NAMES } from "./dimensions";
import { LinkSchema, loadLinks } from "./links";
import { readPageSources } from "./reference";
import { loadCodeIndex, loadRepoMeta } from "./repo";

// Loads the real inputs of the checks in lib/content-checks.ts.

const SITE = process.cwd();

function walk(folder: string, extensions: string[]): TextFile[] {
  const root = join(SITE, folder);
  return readdirSync(root, { recursive: true, encoding: "utf-8" })
    .filter((path) => extensions.some((extension) => path.endsWith(extension)))
    .map((path) => {
      const absolute = join(root, path);
      return {
        path: relative(SITE, absolute).split(sep).join("/"),
        text: readFileSync(absolute, "utf-8"),
      };
    })
    .sort((a, b) => a.path.localeCompare(b.path));
}

function linkFiles(): LinkFile[] {
  const schema = z.strictObject({ links: z.array(LinkSchema) });
  return walk("data/links", [".yaml", ".yml"]).map(({ path, text }) => ({
    path,
    links: schema.parse(parse(text)).links,
  }));
}

export function publicSymbols(): PublicSymbol[] {
  return loadCodeIndex()
    .files.filter((file) => COVERED_MODULES.includes(file.module))
    .flatMap((file) =>
      [...file.functions, ...file.classes.flatMap((cls) => [cls, ...cls.methods])]
        .filter((symbol) => isPublic(symbol.id))
        .map((symbol) => ({ id: symbol.id, name: symbol.name, module: file.module })),
    );
}

function shapeRuns(): ShapeRuns["runs"] {
  const path = join(SITE, "data", "golden", "shapes.json");
  return (JSON.parse(readFileSync(path, "utf-8")) as ShapeRuns).runs;
}

function configDims(): Record<string, number> {
  const dims: Record<string, number> = {};
  for (const name of DIMENSION_NAMES) {
    const { value } = resolveDimension(name);
    if (value !== null) dims[name] = value;
  }
  return dims;
}

// A read-only question to git, using the same rule as data/generated/repo-meta.json: the newest
// commit outside docs-site, because a commit cannot contain its own hash.
function newestRepositoryCommit(): string | null {
  try {
    const output = execFileSync(
      "git",
      ["log", "-1", "--format=%H", "--", ".", ":(exclude)docs-site"],
      { cwd: join(SITE, ".."), encoding: "utf-8", stdio: ["ignore", "pipe", "ignore"] },
    );
    return output.trim() || null;
  } catch {
    return null;
  }
}

// Everything hand-written that reaches a reader. Generated data and contract snapshots are not here.
function scannedSources(): TextFile[] {
  return [
    ...walk("content", [".mdx"]),
    ...walk("data/concepts", [".yaml", ".yml"]),
    ...walk("data/links", [".yaml", ".yml"]),
    ...walk("data/pipelines", [".yaml", ".yml"]),
    ...readdirSync(join(SITE, "data"))
      .filter((name) => name.endsWith(".ts"))
      .map((name) => ({
        path: `data/${name}`,
        text: readFileSync(join(SITE, "data", name), "utf-8"),
      })),
    ...walk("components", [".tsx"]),
    ...walk("app", [".tsx"]),
    ...walk("lib", [".ts"]),
  ];
}

// A badge with no chart beside it. Each of these says what the panel holds.
const ACTUAL_BADGE_FILES = [
  {
    path: "components/ActualPanels.tsx",
    reason: "The panel shown when nothing has been exported.",
  },
  {
    path: "components/AttentionExamples.tsx",
    reason: "The slot's summary states the provenance, or that there is no data.",
  },
  { path: "components/Specimens.tsx", reason: "The design page shows every badge." },
  { path: "app/(docs)/page.tsx", reason: "The home page's legend says what each label means." },
];

export function runContentChecks(): CheckResult[] {
  const concepts = loadConcepts();
  const pages = readPageSources();
  const symbols = loadSymbolIndex();
  const actual = loadActual();
  const provenance = actual.available ? actual.meta.provenance : null;
  const repoMeta = loadRepoMeta();

  return [
    checkSymbolsAndLinks({ concepts, linkFiles: linkFiles(), symbols }),
    checkCoverage({
      symbols: publicSymbols(),
      concepts,
      links: loadLinks(),
      pages,
      codeRefs: CODE_REFS,
      allowlist: COVERAGE_ALLOWLIST,
    }),
    checkEquations({ concepts, pages }),
    checkLabels({
      concepts,
      sources: [
        ...walk("content", [".mdx"]),
        ...walk("components", [".tsx"]),
        ...walk("app", [".tsx"]),
      ],
      actualBadgeFiles: ACTUAL_BADGE_FILES,
      provenance,
    }),
    checkShapes({
      concepts,
      runs: shapeRuns(),
      configDims: configDims(),
      aliases: SHAPE_STAGE_ALIASES,
      requiredFunctions: SHAPE_RUN_REQUIRED_FUNCTIONS,
    }),
    checkFreshness({
      repoMetaSha: repoMeta.sha,
      codeIndexSha: loadCodeIndex().sha,
      actualSha: provenance?.gitSha ?? null,
      headSha: newestRepositoryCommit(),
    }),
    checkForbiddenTopics({
      absentRows: loadAbsentTechniques().rows,
      absentTerms: ABSENT_TERMS,
      taskNamedTerms: TASK_NAMED_TERMS,
      sources: scannedSources(),
      repoFiles: loadCodeIndex().files.map((file) => ({ path: file.path, text: file.source })),
      exempt: SCAN_EXEMPT,
    }),
  ];
}
