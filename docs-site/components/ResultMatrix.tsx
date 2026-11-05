import type { Matrix } from "@/lib/math/tensor";
import { formatShape, shapeOf } from "@/lib/viz/shape";
import styles from "./Examples.module.css";

export const showNumber = (value: number) => String(Number(value.toFixed(4)));

// A computed matrix. The screen shows four decimals; data-exact keeps the full value, so tests can
// compare what is on the page with the Python result at full precision.
export function ResultMatrix({
  label,
  testId,
  values,
}: {
  label: string;
  testId: string;
  values: Matrix;
}) {
  return (
    <table className={styles.compact} data-testid={testId} data-exact={JSON.stringify(values)}>
      <caption>
        {label}, shape {formatShape(shapeOf(values))}
      </caption>
      <tbody>
        {values.map((row, t) => (
          <tr key={t}>
            {row.map((value, c) => (
              <td key={c}>{showNumber(value)}</td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
}
