import { CATEGORY_LABELS, conceptAnchor, getConcept } from "@/lib/concepts";
import { Badge } from "./Badge";
import styles from "./ConceptCard.module.css";

export function ConceptCard({ concept }: { concept: string }) {
  const found = getConcept(concept);
  return (
    <section
      className={styles.card}
      id={conceptAnchor(found.id)}
      aria-label={`Concept: ${found.title}`}
    >
      <div className={styles.top}>
        <span className={styles.title}>{found.title}</span>
        <Badge kind={found.status} />
      </div>
      <p className={styles.category}>{CATEGORY_LABELS[found.category]}</p>
      <p>{found.summary}</p>
      <p className={styles.why}>{found.why}</p>
    </section>
  );
}
