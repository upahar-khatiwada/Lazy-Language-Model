import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { z } from "zod";
import { attentionProblems, shortHash, type ActualAttentionData } from "./actual-view";

// data/generated/actual, which scripts/export_actual.py writes. Never edit it by hand. Pages read it
// at build time, and a site with no export builds with every Actual panel off.

const CheckpointFileSchema = z.strictObject({
  name: z.string(),
  sha256: z.string().regex(/^[0-9a-f]{64}$/),
  bytes: z.number().int().positive(),
});

const ProvenanceSchema = z.strictObject({
  gitSha: z.string(),
  gitShortSha: z.string(),
  gitDirty: z.boolean(),
  checkpointPrefix: z.string(),
  checkpointFiles: z.array(CheckpointFileSchema).length(3),
  corpus: z.strictObject({ path: z.string(), sha256: z.string() }),
  tokenizerMatchesCorpus: z.boolean(),
  seed: z.number().int(),
  config: z.record(z.string(), z.union([z.number(), z.string()])),
  modelConfig: z.record(z.string(), z.number()),
  environment: z.strictObject({ python: z.string(), numpy: z.string(), pandas: z.string() }),
  disclosure: z.string().min(1),
});

const MetaSchema = z.discriminatedUnion("actualAvailable", [
  z.strictObject({
    schema: z.literal(1),
    actualAvailable: z.literal(false),
    reason: z.string(),
    howToCreate: z.array(z.string()),
  }),
  z.strictObject({
    schema: z.literal(1),
    actualAvailable: z.literal(true),
    vocabSize: z.number().int().positive(),
    paramCount: z.number().int().positive(),
    tensorCount: z.number().int().positive(),
    provenance: ProvenanceSchema,
  }),
]);

const TokenizerSchema = z.strictObject({
  vocabSize: z.number().int().positive(),
  mergeCount: z.number().int().nonnegative(),
  merges: z.array(
    z.strictObject({
      rank: z.number().int().nonnegative(),
      pair: z.array(z.string()).length(2),
      merged: z.string(),
      frequency: z.number().int().positive(),
    }),
  ),
  sentences: z.array(
    z.strictObject({
      text: z.string(),
      ids: z.array(z.number().int()),
      tokens: z.array(z.string()),
      unknown: z.number().int().nonnegative(),
      roundTrip: z.boolean(),
    }),
  ),
});

const EmbeddingsSchema = z.strictObject({
  V: z.number().int().positive(),
  d: z.number().int().positive(),
  Tmax: z.number().int().positive(),
  T: z.number().int().positive(),
  tokEmb: z.array(z.number().int()).length(2),
  posEmb: z.array(z.number().int()).length(2),
  X0: z.array(z.number().int()).length(2),
});

const AttentionSchema: z.ZodType<ActualAttentionData> = z.strictObject({
  window: z.strictObject({
    split: z.string(),
    index: z.number().int(),
    ids: z.array(z.number().int()),
    tokens: z.array(z.string()),
    targets: z.array(z.number().int()),
    text: z.string(),
  }),
  T: z.number().int().positive(),
  H: z.number().int().positive(),
  L: z.number().int().positive(),
  dh: z.number().int().positive(),
  largestRowSumGap: z.number(),
  layers: z.array(
    z.strictObject({
      layer: z.number().int(),
      heads: z.array(z.strictObject({ head: z.number().int(), A: z.array(z.array(z.number())) })),
    }),
  ),
});

const distributionShape = {
  probabilities: z.array(z.number()),
  sum: z.number(),
  kept: z.number().int(),
  top: z.array(
    z.strictObject({ id: z.number().int(), token: z.string(), probability: z.number() }),
  ),
  otherTokens: z.number(),
};

const ProbabilitiesSchema = z.strictObject({
  prompt: z.string(),
  ids: z.array(z.number().int()),
  tokens: z.array(z.string()),
  plain: z.strictObject(distributionShape),
  sampling: z.strictObject({
    temperature: z.number(),
    topK: z.number().int(),
    topP: z.number(),
    ...distributionShape,
  }),
});

const LossesSchema = z.strictObject({
  uniformLoss: z.number(),
  rows: z.array(
    z.strictObject({
      split: z.enum(["validation", "training"]),
      index: z.number().int(),
      windows: z.number().int(),
      text: z.string(),
      loss: z.number(),
      perplexity: z.number(),
    }),
  ),
});

export type ActualMeta = z.infer<typeof MetaSchema>;
export type AvailableMeta = Extract<ActualMeta, { actualAvailable: true }>;
export type Provenance = AvailableMeta["provenance"];

export type ActualData = {
  available: true;
  meta: AvailableMeta;
  tokenizer: z.infer<typeof TokenizerSchema>;
  embeddings: z.infer<typeof EmbeddingsSchema>;
  attention: ActualAttentionData;
  probabilities: z.infer<typeof ProbabilitiesSchema>;
  losses: z.infer<typeof LossesSchema>;
};

export type NoActualData = { available: false; reason: string; howToCreate: string[] };

export type Actual = ActualData | NoActualData;

const NOT_EXPORTED: NoActualData = {
  available: false,
  reason: "Nothing has been exported.",
  howToCreate: ["python docs-site/scripts/export_actual.py"],
};

function readJson<T>(folder: string, name: string, schema: z.ZodType<T>): T {
  const path = join(folder, `${name}.json`);
  const result = schema.safeParse(JSON.parse(readFileSync(path, "utf-8")));
  if (!result.success) {
    throw new Error(`Invalid data in ${path}: ${z.prettifyError(result.error)}`);
  }
  return result.data;
}

export function readActual(folder: string): Actual {
  if (!existsSync(join(folder, "meta.json"))) return NOT_EXPORTED;
  const meta = readJson(folder, "meta", MetaSchema);
  if (!meta.actualAvailable) {
    return { available: false, reason: meta.reason, howToCreate: meta.howToCreate };
  }
  const data: ActualData = {
    available: true,
    meta,
    tokenizer: readJson(folder, "tokenizer", TokenizerSchema),
    embeddings: readJson(folder, "embeddings", EmbeddingsSchema),
    attention: readJson(folder, "attention", AttentionSchema),
    probabilities: readJson(folder, "probabilities", ProbabilitiesSchema),
    losses: readJson(folder, "losses", LossesSchema),
  };
  const problems = attentionProblems(data.attention);
  if (problems.length > 0) {
    throw new Error(`Exported attention weights are wrong: ${problems.slice(0, 5).join("; ")}`);
  }
  return data;
}

let cached: Actual | undefined;

export function loadActual(): Actual {
  cached ??= readActual(join(process.cwd(), "data", "generated", "actual"));
  return cached;
}

// The line every Actual panel shows: commit, checkpoint hash and seed, then decision D1's disclosure.
export function provenanceLine(source: Provenance): string {
  const weights = source.checkpointFiles.find(({ name }) => name.endsWith(".npz"))!;
  const commit = `Commit ${source.gitShortSha}${source.gitDirty ? " (with local changes)" : ""}`;
  const hash = `Checkpoint ${weights.name} sha256 ${shortHash(weights.sha256)}`;
  const seed = `Seed ${source.seed}, from config.py, since a checkpoint does not record its own`;
  return `${commit}. ${hash}. ${seed}. ${source.disclosure}`;
}
