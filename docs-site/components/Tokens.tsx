import styles from "./Examples.module.css";

// Whitespace would be invisible in a token box.
export function showToken(token: string): string {
  return token.replaceAll(" ", "␠").replaceAll("\n", "\\n").replaceAll("\t", "\\t");
}

export function Tokens({ tokens, testId }: { tokens: string[]; testId?: string }) {
  return (
    <ul className={styles.tokens} data-testid={testId}>
      {tokens.map((token, i) => (
        <li key={i}>
          <code>{showToken(token)}</code>
        </li>
      ))}
    </ul>
  );
}
