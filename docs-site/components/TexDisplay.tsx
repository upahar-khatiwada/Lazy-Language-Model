import katex from "katex";
import styles from "./TexDisplay.module.css";

// The low-level KaTeX renderers. Pages and concepts use MathBlock, which takes an equation id.
// This is not exposed to MDX. Rendered at build time, so no KaTeX JavaScript reaches the browser.
// `tabIndex` lets keyboard users scroll a wide equation.
export function TexDisplay({ tex, label }: { tex: string; label: string }) {
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

export function TexInline({ tex }: { tex: string }) {
  const html = katex.renderToString(tex, { throwOnError: true });
  return <span dangerouslySetInnerHTML={{ __html: html }} />;
}
