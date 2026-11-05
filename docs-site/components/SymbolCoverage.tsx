import GithubSlugger from "github-slugger";
import { SYMBOL_COVERAGE } from "@/data/coverage";
import { formatLineRange, getSymbol } from "@/lib/code-targets";
import { DataTable } from "./DataTable";

// Every symbol of a file: which section of this page shows it, or why it is not shown.
export function SymbolCoverage({ file }: { file: string }) {
  const rows = SYMBOL_COVERAGE[file];
  if (!rows) throw new Error(`data/coverage.ts has no entry for ${file}`);
  return (
    <DataTable label={`Every symbol in ${file}`} data-testid="symbol-coverage">
      <thead>
        <tr>
          <th scope="col">Symbol</th>
          <th scope="col">Lines</th>
          <th scope="col">On this page</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((row) => {
          const code = getSymbol(row.symbol);
          return (
            <tr key={row.symbol} data-symbol={row.symbol}>
              <td>
                <code>{row.symbol}</code>
              </td>
              <td>
                <a className="code-link" href={code.permalink}>
                  {code.file}:{formatLineRange(code.startLine, code.endLine)}
                </a>
              </td>
              <td>
                {"section" in row ? (
                  <a href={`#${new GithubSlugger().slug(row.section)}`}>{row.section}</a>
                ) : (
                  <>Skipped. {row.skip}</>
                )}
              </td>
            </tr>
          );
        })}
      </tbody>
    </DataTable>
  );
}
