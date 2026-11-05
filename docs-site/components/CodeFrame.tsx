import type { CodeToken } from "@/lib/highlight";
import styles from "./CodeBlock.module.css";
import { CopyButton } from "./CopyButton";

export type CodeFrameProps = {
  lines: CodeToken[][];
  source: string;
  filePath: string;
  functionName?: string;
  // Both come from data/generated. Never pass hand-written values.
  startLine: number;
  permalink: string;
  // Real line numbers in the file, from resolved links.
  highlightLines?: ReadonlySet<number>;
};

// Code that is already highlighted. No Shiki here, so a client component can change the highlighted
// lines without bringing the highlighter into the browser.
export function CodeFrame({
  lines,
  source,
  filePath,
  functionName,
  startLine,
  permalink,
  highlightLines,
}: CodeFrameProps) {
  return (
    <figure className={styles.block}>
      <figcaption className={styles.chrome}>
        <span className={styles.file}>{filePath}</span>
        {functionName ? <span className={styles.function}>{functionName}</span> : null}
        <span className={styles.actions}>
          <a className={styles.githubLink} href={permalink}>
            GitHub<span className="visually-hidden">: {filePath}</span>
          </a>
          <CopyButton text={source} />
        </span>
      </figcaption>
      <pre className={styles.pre} tabIndex={0} aria-label={`Source of ${filePath}`}>
        <code className={styles.code}>
          {lines.map((tokens, index) => {
            const lineNumber = startLine + index;
            const highlighted = highlightLines?.has(lineNumber) ?? false;
            return (
              <span
                className={highlighted ? `${styles.line} ${styles.highlighted}` : styles.line}
                data-line={lineNumber}
                data-highlighted={highlighted ? "true" : undefined}
                key={index}
              >
                <span className={styles.number} aria-hidden="true">
                  {lineNumber}
                </span>
                {tokens.map((token, tokenIndex) => (
                  <span key={tokenIndex} style={{ color: token.color }}>
                    {token.content}
                  </span>
                ))}
              </span>
            );
          })}
        </code>
      </pre>
    </figure>
  );
}
