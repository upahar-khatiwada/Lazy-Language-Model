import katex from "katex";
import styles from "./MathBlock.module.css";

// Rendered at build time, so no KaTeX JavaScript reaches the browser. `tabIndex` lets keyboard
// users scroll a wide equation.
export function MathBlock({ tex, label }: { tex: string; label: string }) {
  const html = katex.renderToString(tex, { displayMode: true, throwOnError: true });
  return (
    <div
      className={styles.math}
      role="group"
      aria-label={label}
      tabIndex={0}
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
}
