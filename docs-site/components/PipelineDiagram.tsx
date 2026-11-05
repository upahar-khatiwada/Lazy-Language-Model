import { formatLineRange, requireTarget } from "@/lib/code-targets";
import { getPipeline } from "@/lib/concepts";
import { formatShape } from "@/lib/dimensions";
import styles from "./PipelineDiagram.module.css";

// Drawn entirely from data/pipelines. Each step links to its resolved lines.
export function PipelineDiagram({ id }: { id: string }) {
  const pipeline = getPipeline(id);
  return (
    <figure data-testid={`pipeline-${id}`}>
      <figcaption className={styles.title}>{pipeline.title}</figcaption>
      <ol className={styles.steps}>
        {pipeline.steps.map((step) => {
          const target = requireTarget(step.target);
          return (
            <li className={styles.step} key={step.label}>
              <div className={styles.box}>
                <span className={styles.label}>{step.label}</span>
                {step.dims ? <code>{formatShape(step.dims)}</code> : null}
                <a className="code-link" href={target.permalink}>
                  {target.symbol.file}:{formatLineRange(target.startLine, target.endLine)}
                </a>
              </div>
            </li>
          );
        })}
      </ol>
    </figure>
  );
}
