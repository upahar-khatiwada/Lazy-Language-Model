import { formatLineRange, getSymbol, loadSymbolIndex, requireTarget } from "@/lib/code-targets";
import { getConcept } from "@/lib/concepts";
import { loadCallGraph } from "@/lib/repo";
import { Badge } from "./Badge";
import styles from "./Trace.module.css";

function SymbolLink({ id }: { id: string }) {
  const symbol = loadSymbolIndex().get(id);
  if (!symbol) return <code>{id}</code>;
  return (
    <a href={symbol.permalink}>
      <code>{id}</code>
    </a>
  );
}

function SymbolList({ title, ids }: { title: string; ids: string[] }) {
  if (ids.length === 0) return null;
  return (
    <p>
      {title}:{" "}
      {ids.map((id, index) => (
        <span key={id}>
          {index > 0 ? ", " : null}
          <SymbolLink id={id} />
        </span>
      ))}
    </p>
  );
}

// From a concept to its evidence. Callers come from call-graph.json; inline uses are declared
// in the concept, because a formula written in place leaves no call behind.
export function Trace({ concept }: { concept: string }) {
  const found = getConcept(concept);
  const usedBy = loadCallGraph().usedBy;
  const callers = found.source.flatMap(({ symbol }) => usedBy[symbol]?.production ?? []);
  const tests = found.source.flatMap(({ symbol }) => usedBy[symbol]?.tests ?? []);
  const pdf = found.pdfRef;

  return (
    <dl className={styles.trace} data-testid="trace">
      <div className={styles.row}>
        <dt>Source</dt>
        <dd>
          {found.source.map(({ symbol }) => {
            const code = getSymbol(symbol);
            return (
              <p key={symbol}>
                <a href={code.permalink}>
                  <code>{code.file}</code>, lines {formatLineRange(code.startLine, code.endLine)}
                </a>
                , <code>{code.name}</code>
              </p>
            );
          })}
        </dd>
      </div>
      <div className={styles.row}>
        <dt>PDF reference</dt>
        <dd>
          {pdf ? (
            <p>
              Section {pdf.section}
              {pdf.equations.length > 0 ? `, Eq. ${pdf.equations.join(", ")}` : ""}, p. {pdf.page}
            </p>
          ) : (
            <p>Not stated in the PDF</p>
          )}
        </dd>
      </div>
      <div className={styles.row}>
        <dt>Implementation</dt>
        <dd>
          {callers.length === 0 ? <p>Not called by the pipeline.</p> : null}
          <SymbolList title="Called by" ids={callers} />
          <SymbolList title="Tested by" ids={tests} />
          {found.usedBy.map((use) => {
            const target = requireTarget(use);
            return (
              <p key={`${use.symbol}:${use.match}`}>
                Written inline in{" "}
                <a href={target.permalink}>
                  <code>{target.symbol.file}</code>, line{" "}
                  {formatLineRange(target.startLine, target.endLine)}
                </a>
                . {use.note}
              </p>
            );
          })}
        </dd>
      </div>
      <div className={styles.row}>
        <dt>Status</dt>
        <dd>
          <Badge kind={found.status} />
        </dd>
      </div>
    </dl>
  );
}
