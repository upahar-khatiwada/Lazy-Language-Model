import { readFileSync } from "node:fs";
import { join } from "node:path";
import { z } from "zod";
import { formatShape } from "./viz/shape";

// Built from data/derived/init-params.json, which comes from calling transformer.init_params.
// Counts are worked out here from the shapes, so no formula or total is typed by hand.

const DIM = z.enum(["V", "Tmax", "d", "F"]);
export type ParamDim = z.infer<typeof DIM>;

const InitParamsSchema = z.strictObject({
  probe: z.strictObject({
    V: z.number(),
    Tmax: z.number(),
    d: z.number(),
    F: z.number(),
    L: z.number(),
  }),
  countParams: z.number(),
  parameters: z.array(
    z.strictObject({
      name: z.string(),
      perBlock: z.boolean(),
      dims: z.array(DIM).min(1),
      init: z.enum(["ones", "zeros", "xavier_uniform"]),
    }),
  ),
});

export type InitParams = z.infer<typeof InitParamsSchema>;
export type InitParameter = InitParams["parameters"][number];

let cached: InitParams | undefined;

export function loadInitParams(): InitParams {
  cached ??= InitParamsSchema.parse(
    JSON.parse(readFileSync(join(process.cwd(), "data", "derived", "init-params.json"), "utf-8")),
  );
  return cached;
}

// Every key of the dictionary, in the order init_params inserts them, for this many blocks. The block
// part of the list repeats once per block, as the loop in init_params does.
export function expandBlockName(
  parameters: InitParameter[],
  L: number,
): { key: string; dims: ParamDim[] }[] {
  const start = parameters.findIndex((parameter) => parameter.perBlock);
  const end = parameters.findLastIndex((parameter) => parameter.perBlock);
  const once = ({ name, dims }: InitParameter) => ({ key: name, dims });
  return [
    ...parameters.slice(0, start).map(once),
    ...Array.from({ length: L }, (_, l) =>
      parameters.slice(start, end + 1).map(({ name, dims }) => ({
        key: name.replace("{l}", String(l)),
        dims,
      })),
    ).flat(),
    ...parameters.slice(end + 1).map(once),
  ];
}

export type DimValues = Record<ParamDim | "L", number>;

// The display order of a product, and the spelling the task asks for.
const ORDER: ParamDim[] = ["V", "Tmax", "d", "F"];
const TEXT: Record<ParamDim, string> = { V: "V", Tmax: "T_max", d: "d", F: "F" };
const TEX: Record<ParamDim, string> = { V: "V", Tmax: "T_{\\max}", d: "d", F: "F" };

const sorted = (dims: ParamDim[]) => [...dims].sort((a, b) => ORDER.indexOf(a) - ORDER.indexOf(b));

export function symbolicShape(dims: ParamDim[]): string {
  return formatShape(dims.map((dim) => TEXT[dim]));
}

// V stays a symbol: it depends on the corpus and config.py has no value for it.
export function realShape(dims: ParamDim[], values: Omit<DimValues, "V" | "L">): string {
  return formatShape(dims.map((dim) => (dim === "V" ? "V" : values[dim])));
}

export function symbolicCount(dims: ParamDim[]): string {
  return sorted(dims)
    .map((dim) => TEXT[dim])
    .join(" × ");
}

const number = (value: number) => value.toLocaleString("en-US");

// Scalars in one tensor, with V left as a symbol when it appears.
export function realCount(dims: ParamDim[], values: Omit<DimValues, "V" | "L">): string {
  const known = dims.filter((dim) => dim !== "V") as Exclude<ParamDim, "V">[];
  const product = known.reduce((total, dim) => total * values[dim], 1);
  if (!dims.includes("V")) return number(product);
  return product === 1 ? "V" : `${number(product)}V`;
}

type Term = { coefficient: number; dims: ParamDim[] };

function collect(parameters: InitParameter[]): Term[] {
  const terms = new Map<string, Term>();
  for (const { dims } of parameters) {
    const key = sorted(dims).join("*");
    const term = terms.get(key) ?? { coefficient: 0, dims: sorted(dims) };
    term.coefficient += 1;
    terms.set(key, term);
  }
  return [...terms.values()].sort((a, b) => b.dims.length - a.dims.length);
}

function termTex({ coefficient, dims }: Term): string {
  const powers = ORDER.flatMap((dim) => {
    const power = dims.filter((candidate) => candidate === dim).length;
    return power === 0 ? [] : [power === 1 ? TEX[dim] : `${TEX[dim]}^{${power}}`];
  });
  return `${coefficient === 1 ? "" : coefficient}${powers.join(" ")}`;
}

// The total as a polynomial, collected from the shapes.
export function countFormulaTex(parameters: InitParameter[]): string {
  const once = collect(parameters.filter((p) => !p.perBlock)).map(termTex);
  const block = collect(parameters.filter((p) => p.perBlock)).map(termTex);
  return `${once.join(" + ")} + L\\left(${block.join(" + ")}\\right)`;
}

export function evaluateCount(parameters: InitParameter[], values: DimValues): number {
  return parameters.reduce(
    (total, { dims, perBlock }) =>
      total + dims.reduce((product, dim) => product * values[dim], 1) * (perBlock ? values.L : 1),
    0,
  );
}

// The total as `a V + b`, for the real configuration.
export function countInV(
  parameters: InitParameter[],
  values: Omit<DimValues, "V">,
): { perVocabulary: number; constant: number } {
  const constant = evaluateCount(parameters, { ...values, V: 0 });
  return { perVocabulary: evaluateCount(parameters, { ...values, V: 1 }) - constant, constant };
}
