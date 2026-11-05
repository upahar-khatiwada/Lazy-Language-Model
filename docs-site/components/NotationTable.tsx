import { NOTATION } from "@/data/notation";
import { formatLineRange, requireTarget } from "@/lib/code-targets";
import { resolveDimension } from "@/lib/dimensions";
import { DataTable } from "./DataTable";
import { TexInline } from "./TexDisplay";

export function NotationTable() {
  return (
    <DataTable label="Math symbols and their names in the code" data-testid="notation-table">
      <thead>
        <tr>
          <th scope="col">Symbol</th>
          <th scope="col">Meaning</th>
          <th scope="col">config.py</th>
          <th scope="col">Value</th>
          <th scope="col">In the model code</th>
        </tr>
      </thead>
      <tbody>
        {NOTATION.map((row) => {
          const target = requireTarget(row.target);
          const { value } = resolveDimension(row.dimension);
          return (
            <tr key={row.dimension} data-dimension={row.dimension}>
              <td>
                <TexInline tex={row.tex} />
              </td>
              <td>
                {row.meaning}
                {row.note ? <p>{row.note}</p> : null}
              </td>
              <td>{row.config ? <code>{row.config}</code> : "not in config.py"}</td>
              <td data-testid="notation-value">{value ?? "depends on the corpus"}</td>
              <td>
                <code>{row.code}</code>{" "}
                <a className="code-link" href={target.permalink}>
                  {target.symbol.file}:{formatLineRange(target.startLine, target.endLine)}
                </a>
              </td>
            </tr>
          );
        })}
      </tbody>
    </DataTable>
  );
}
