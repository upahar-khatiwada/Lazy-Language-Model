import { getConcept } from "@/lib/concepts";
import { DataTable } from "./DataTable";
import { TexInline } from "./TexDisplay";

export function Symbols({ concept }: { concept: string }) {
  const { title, symbols } = getConcept(concept);
  return (
    <DataTable label={`Symbols used in ${title}`}>
      <thead>
        <tr>
          <th scope="col">Symbol</th>
          <th scope="col">Meaning</th>
          <th scope="col">In code</th>
          <th scope="col">Shape</th>
        </tr>
      </thead>
      <tbody>
        {symbols.map((row) => (
          <tr key={row.symbol}>
            <td>
              <TexInline tex={row.symbol} />
            </td>
            <td>{row.meaning}</td>
            <td>{row.codeName ? <code>{row.codeName}</code> : "not named in the code"}</td>
            <td>{row.shape}</td>
          </tr>
        ))}
      </tbody>
    </DataTable>
  );
}
