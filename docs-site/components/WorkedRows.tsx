import { DataTable } from "./DataTable";

// More decimals than other pages, because these values are small. No server-only imports, so client
// components can use it. data-exact holds the full value for tests.
export const fine = (value: number) => String(Number(value.toFixed(8)));

export function WorkedRows({
  label,
  rows,
}: {
  label: string;
  rows: { code: string; value: number; testId: string }[];
}) {
  return (
    <DataTable label={label} data-testid="calc-rows">
      <thead>
        <tr>
          <th scope="col">Code, in order</th>
          <th scope="col">Result</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((row) => (
          <tr key={row.testId}>
            <td>
              <code>{row.code}</code>
            </td>
            <td>
              <code data-testid={row.testId} data-exact={JSON.stringify(row.value)}>
                {fine(row.value)}
              </code>
            </td>
          </tr>
        ))}
      </tbody>
    </DataTable>
  );
}
