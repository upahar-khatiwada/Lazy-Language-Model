// No file access here, unlike lib/dimensions.ts, so client components can import this.

// The shape of nested arrays, like ndarray.shape.
export function shapeOf(value: unknown): number[] {
  const shape: number[] = [];
  let current = value;
  while (Array.isArray(current)) {
    shape.push(current.length);
    current = current[0];
  }
  return shape;
}

// (H, T, T), and (V,) for one dimension, as Python prints a tuple.
export function formatShape(parts: (string | number)[]): string {
  return parts.length === 1 ? `(${parts[0]},)` : `(${parts.join(", ")})`;
}
