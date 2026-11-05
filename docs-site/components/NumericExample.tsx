import { exampleAnchor, getConcept, type ExampleKind } from "@/lib/concepts";
import { formatNumbers, getStoredExample } from "@/lib/examples";
import { Badge } from "./Badge";
import styles from "./NumericExample.module.css";

// `kind` is required so that a page author states the label, and it must agree with the data.
// MDX is not type-checked, so a missing kind also fails at build time, not only in TypeScript.
export function NumericExample({ concept, kind }: { concept: string; kind: ExampleKind }) {
  if (kind !== "illustrative" && kind !== "actual") {
    throw new Error(`NumericExample for ${concept} needs kind="illustrative" or kind="actual"`);
  }
  const example = getConcept(concept).example;
  if (!example) throw new Error(`Concept ${concept} has no example`);
  if (example.kind !== kind) {
    throw new Error(
      `NumericExample for ${concept}: kind is ${kind}, the data says ${example.kind}`,
    );
  }
  const stored = getStoredExample(concept);
  const format = (value: Parameters<typeof formatNumbers>[0]) =>
    formatNumbers(value, example.precision);

  return (
    <section
      className={styles.example}
      id={exampleAnchor(concept)}
      aria-label={`Numeric example: ${concept}`}
    >
      <div className={styles.header}>
        <Badge kind={kind} />
        <span className={styles.note}>
          {example.kind === "actual" ? (
            example.provenance
          ) : (
            <>
              Hand-picked input. Values computed by running <code>{example.call}</code>; they are
              not learned values.
            </>
          )}
        </span>
      </div>
      <dl className={styles.rows}>
        {Object.entries(example.input).map(([name, value]) => (
          <div className={styles.row} key={name}>
            <dt>
              Input <code>{name}</code>
            </dt>
            <dd>
              <code>{format(value)}</code>
            </dd>
          </div>
        ))}
        {example.steps.map((step) => {
          const value = stored.steps[step.id];
          if (value === undefined) throw new Error(`${concept}: no value for step ${step.id}`);
          return (
            <div className={styles.row} key={step.id}>
              <dt>
                {step.label} <code>{step.capture}</code>
              </dt>
              <dd>
                <code>{format(value)}</code>
              </dd>
            </div>
          );
        })}
        <div className={styles.row}>
          <dt>{example.output.label}</dt>
          <dd>
            <code data-testid="example-output">{format(stored.output)}</code>
          </dd>
        </div>
      </dl>
    </section>
  );
}
