// The parameter count as a formula in the shapes init_params creates. tests/math.golden.test.ts
// compares it with count_params on small configurations.

export type ParameterDims = { V: number; Tmax: number; d: number; F: number; L: number };

// Mirrors transformer.init_params, the scalars one pass of its layer loop creates:
// ln1 2d, four (d, d) matrices with four d biases, ln2 2d, ff1 dF + F, ff2 Fd + d.
export function blockParameterCount({ d, F }: Pick<ParameterDims, "d" | "F">): number {
  return 4 * d * d + 2 * d * F + F + 9 * d;
}

// Mirrors transformer.count_params, without building any array.
export function parameterCount({ V, Tmax, d, F, L }: ParameterDims): number {
  return (2 * d + 1) * V + (Tmax + 2) * d + L * blockParameterCount({ d, F });
}
