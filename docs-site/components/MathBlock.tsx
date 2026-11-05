import { equationAnchor, getEquation } from "@/lib/concepts";
import { Badge } from "./Badge";
import { CopyButton } from "./CopyButton";
import styles from "./MathBlock.module.css";
import { TexDisplay } from "./TexDisplay";

// Takes an equation id. The equation itself lives in data/concepts.
export function MathBlock({ id }: { id: string }) {
  const { equation } = getEquation(id);
  const anchor = equationAnchor(id);
  return (
    <figure className={styles.block} id={anchor} data-equation={id}>
      <figcaption className={styles.chrome}>
        <span>
          {equation.label} {equation.derived ? <Badge kind="derived" /> : null}
        </span>
        <span className={styles.actions}>
          <a className={styles.anchor} href={`#${anchor}`}>
            #<span className="visually-hidden">Link to equation: {equation.label}</span>
          </a>
          <CopyButton text={equation.latex} label="Copy LaTeX" />
        </span>
      </figcaption>
      <TexDisplay tex={equation.latex} label={equation.label} />
    </figure>
  );
}
