import Link from "next/link";
import { formatLineRange } from "@/lib/code-targets";
import { equationSlug, getConcept } from "@/lib/concepts";
import { conceptLocations, inlineEquivalentNote } from "@/lib/equation-links";
import { ROLE_LABELS } from "@/lib/math-code";
import { Callout } from "./Callout";
import { DataTable } from "./DataTable";
import styles from "./Trace.module.css";

const lineText = (startLine: number, endLine: number) =>
  `${startLine === endLine ? "line" : "lines"} ${formatLineRange(startLine, endLine)}`;

// From a concept to the lines that compute each of its equations.
export function ImplementedIn({ concept }: { concept: string }) {
  const found = getConcept(concept);
  const rows = conceptLocations(found).filter(
    ({ location }) => location.role === "implements" || location.role === "inline",
  );
  const notes = [
    ...new Set(
      found.equations
        .map((equation) => inlineEquivalentNote(equation.id))
        .filter((note): note is string => note !== null),
    ),
  ];
  return (
    <div className={styles.panel} data-testid="implemented-in">
      <p className={styles.panelTitle}>Implemented in</p>
      {notes.map((note) => (
        <Callout key={note} kind="note">
          {note}
        </Callout>
      ))}
      <DataTable label={`Where ${found.title} is implemented`}>
        <thead>
          <tr>
            <th scope="col">Equation</th>
            <th scope="col">Role</th>
            <th scope="col">File</th>
            <th scope="col">Function</th>
            <th scope="col">Lines</th>
            <th scope="col">GitHub</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(({ equation, location }) => (
            <tr
              key={`${equation.id}:${location.symbol}:${location.startLine}:${location.occurrence}`}
              data-eq-hover={equationSlug(equation.id)}
              data-eq-row={equation.id}
            >
              <td>{equation.label}</td>
              <td>{ROLE_LABELS[location.role]}</td>
              <td>
                <code>{location.file}</code>
              </td>
              <td>
                <code>{location.functionName}</code>
              </td>
              <td>
                <Link href={location.explorerHref} prefetch={false}>
                  {formatLineRange(location.startLine, location.endLine)}
                </Link>
                {location.occurrence !== null ? (
                  <span data-occurrence={location.occurrence}>
                    {" "}
                    (occurrence {location.occurrence})
                  </span>
                ) : null}
              </td>
              <td>
                <a href={location.permalink}>
                  GitHub
                  <span className="visually-hidden">
                    : {location.file}, {lineText(location.startLine, location.endLine)}
                  </span>
                </a>
              </td>
            </tr>
          ))}
        </tbody>
      </DataTable>
    </div>
  );
}

// The call sites and tests tied to this concept's equations by link records.
export function LinkedUses({ concept }: { concept: string }) {
  const rows = conceptLocations(getConcept(concept)).filter(
    ({ location }) => location.role === "call" || location.role === "test",
  );
  if (rows.length === 0) return null;
  return (
    <ul className={styles.uses} data-testid="linked-uses">
      {rows.map(({ equation, location }) => (
        <li
          key={`${equation.id}:${location.symbol}:${location.startLine}:${location.occurrence}`}
          data-eq-hover={equationSlug(equation.id)}
        >
          {ROLE_LABELS[location.role]}:{" "}
          <Link href={location.explorerHref} prefetch={false}>
            <code>{location.file}</code>, {lineText(location.startLine, location.endLine)}
          </Link>{" "}
          (<code>{location.functionName}</code>), for {equation.label}
          {location.occurrence !== null ? `, occurrence ${location.occurrence}` : ""}
        </li>
      ))}
    </ul>
  );
}
