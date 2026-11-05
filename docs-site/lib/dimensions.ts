import { loadConfig, type RepoConfig } from "./repo";

// Names used for tensor dimensions, and the config.py constant behind each one.
// V, N and W have no constant: they depend on the corpus, so no number is shown for them.
const DIMENSIONS = {
  d: { config: "emb_dim", note: null },
  H: { config: "num_heads", note: null },
  dh: { config: "head_dim", note: null },
  F: { config: "ff_dim", note: null },
  L: { config: "num_layers", note: null },
  T: {
    config: "seq_len",
    note: "T is at most seq_len: generation uses the last seq_len ids, or fewer for a short prompt.",
  },
  Tmax: { config: "context_len", note: null },
  V: { config: null, note: "V depends on the corpus. It is not a value in config.py." },
  N: { config: null, note: "N is the number of ids. It depends on the corpus." },
  W: { config: null, note: "W is the number of windows. It depends on the corpus." },
  B: { config: "batch_size", note: "A final short batch has fewer rows." },
} as const;

export type DimensionName = keyof typeof DIMENSIONS;

export const DIMENSION_NAMES = Object.keys(DIMENSIONS) as [DimensionName, ...DimensionName[]];

export type ResolvedDimension = {
  name: DimensionName;
  configName: string | null;
  value: number | null;
  note: string | null;
};

export function resolveDimension(
  name: DimensionName,
  config: RepoConfig = loadConfig(),
): ResolvedDimension {
  const { config: configName, note } = DIMENSIONS[name];
  if (configName === null) return { name, configName: null, value: null, note };

  const value = config.constants.find((constant) => constant.name === configName)?.value;
  if (typeof value !== "number") {
    throw new Error(`config.json has no numeric ${configName} for dimension ${name}`);
  }
  return { name, configName, value, note };
}

export { formatShape } from "./viz/shape";
