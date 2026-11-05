import { highlightPython } from "@/lib/highlight";
import styles from "./CodeBlock.module.css";
import { CopyButton } from "./CopyButton";

type Props = {
  source: string;
  filePath: string;
  functionName?: string;
  // Both come from data/generated. Never pass hand-written values.
  startLine: number;
  permalink: string;
};

export async function CodeBlock({ source, filePath, functionName, startLine, permalink }: Props) {
  const lines = await highlightPython(source);
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
        <code>
          {lines.map((tokens, index) => (
            <span className={styles.line} key={index}>
              <span className={styles.number} aria-hidden="true">
                {startLine + index}
              </span>
              {tokens.map((token, tokenIndex) => (
                <span key={tokenIndex} style={{ color: token.color }}>
                  {token.content}
                </span>
              ))}
            </span>
          ))}
        </code>
      </pre>
    </figure>
  );
}
