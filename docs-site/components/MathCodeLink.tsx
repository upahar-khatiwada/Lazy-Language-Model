import { implementationLinks } from "@/lib/links";
import { CodeReference } from "./CodeReference";
import { MathBlock } from "./MathBlock";
import styles from "./MathCodeLink.module.css";

// An equation beside the code that computes it. Stacks on mobile.
export function MathCodeLink({ equation }: { equation: string }) {
  const symbols = [...new Set(implementationLinks(equation).map((link) => link.symbol))];
  if (symbols.length === 0) throw new Error(`Equation ${equation} has no link to code`);
  return (
    <div className={styles.link} data-testid="math-code-link">
      <MathBlock id={equation} />
      <div className={styles.code}>
        {symbols.map((symbol) => (
          <CodeReference key={symbol} symbol={symbol} equation={equation} />
        ))}
      </div>
    </div>
  );
}
