import { CATEGORY_LABELS, conceptAnchor, getConcept } from "@/lib/concepts";
import { Badge } from "./Badge";
import styles from "./ConceptCard.module.css";

// `split` leaves out the "why" text, for pages that show it under its own heading with ConceptWhy.
export function ConceptCard({ concept, split = false }: { concept: string; split?: boolean }) {
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
      {split ? null : <p className={styles.why}>{found.why}</p>}
    </section>
  );
}

export function ConceptWhy({ concept }: { concept: string }) {
  return <p>{getConcept(concept).why}</p>;
}
