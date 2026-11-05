import { formatLineRange, getSymbol } from "@/lib/code-targets";
import { getConcept } from "@/lib/concepts";
import { Badge } from "./Badge";
import styles from "./SourceBadge.module.css";

// Compact: where a concept is implemented and whether the PDF states it.
export function SourceBadge({ concept }: { concept: string }) {
  const found = getConcept(concept);
  return (
    <span className={styles.badge} data-testid="source-badge">
      {found.source.map(({ symbol }) => {
        const code = getSymbol(symbol);
        return (
          <a className="code-link" href={code.permalink} key={symbol}>
            {code.file}:{formatLineRange(code.startLine, code.endLine)} {code.name}
          </a>
        );
      })}
      <Badge kind={found.status} />
    </span>
  );
}
