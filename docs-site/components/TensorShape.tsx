import { formatLineRange, requireTarget } from "@/lib/code-targets";
import { getConcept } from "@/lib/concepts";
import { formatShape, resolveDimension, type ResolvedDimension } from "@/lib/dimensions";
import styles from "./TensorShape.module.css";

function legend(dimension: ResolvedDimension): string {
  const base = dimension.configName
    ? `${dimension.name} = ${dimension.configName} = ${dimension.value}.`
    : "";
  return [base, dimension.note].filter(Boolean).join(" ");
}

// Boxes and arrows. Sizes come from data/generated/config.json; dimensions without a config value
// stay symbolic.
export function TensorShape({ concept, shape }: { concept: string; shape: string }) {
  const found = getConcept(concept).shapes.find((candidate) => candidate.id === shape);
  if (!found) throw new Error(`Concept ${concept} has no shape ${shape}`);

  const target = requireTarget(found.target);
  const dimensions = new Map(
    found.stages.flatMap((stage) =>
      stage.dims.map((name) => [name, resolveDimension(name)] as const),
    ),
  );

  return (
    <figure data-testid={`tensor-shape-${shape}`}>
      <figcaption className={styles.caption}>
        <span>{found.label}</span>
        <a className="code-link" href={target.permalink}>
          {target.symbol.file}:{formatLineRange(target.startLine, target.endLine)}
        </a>
      </figcaption>
      <ol className={styles.stages}>
        {found.stages.map((stage) => (
          <li className={styles.stage} key={stage.name}>
            <div className={styles.box}>
              <code className={styles.name}>{stage.name}</code>
              <code>{formatShape(stage.dims)}</code>
              {stage.dims.some((name) => dimensions.get(name)?.value !== null) ? (
                <code className={styles.values}>
                  {formatShape(stage.dims.map((name) => dimensions.get(name)?.value ?? name))}
                </code>
              ) : null}
            </div>
          </li>
        ))}
      </ol>
      <p className={styles.legend}>
        {[...dimensions.values()].map(legend).filter(Boolean).join(" ")}
      </p>
    </figure>
  );
}
