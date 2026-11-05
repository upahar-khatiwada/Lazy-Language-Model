import { formatLineRange, targetLines } from "@/lib/code-targets";
import { equationAnchor, getConcept } from "@/lib/concepts";
import { linksForEquation } from "@/lib/links";
import { DataTable } from "./DataTable";

const ROLE = {
  implements: "Computes it",
  inline: "Written inline",
  call: "Calls it",
  test: "Tests it",
} as const;

// The entries of data/links for one concept: each equation beside the exact lines tied to it.
export function EquationCode({ concept }: { concept: string }) {
  const { title, equations } = getConcept(concept);
  return (
    <DataTable label={`Code for each equation in ${title}`} data-testid="equation-code">
      <thead>
        <tr>
          <th scope="col">Equation</th>
          <th scope="col">Role</th>
          <th scope="col">Lines</th>
          <th scope="col">Code</th>
        </tr>
      </thead>
      <tbody>
        {equations.flatMap((equation) =>
          linksForEquation(equation.id).map((link) => {
            const { resolved } = link;
            return (
              <tr key={`${equation.id}:${link.symbol}:${link.match}:${link.occurrence ?? 0}`}>
                <td>
                  <a href={`#${equationAnchor(equation.id)}`}>{equation.label}</a>
                </td>
                <td>{ROLE[link.role]}</td>
                <td>
                  <a className="code-link" href={resolved.permalink}>
                    {resolved.symbol.file}:{formatLineRange(resolved.startLine, resolved.endLine)}
                  </a>
                </td>
                <td>
                  <code>{targetLines(resolved).trim()}</code>
                </td>
              </tr>
            );
          }),
        )}
      </tbody>
    </DataTable>
  );
}
