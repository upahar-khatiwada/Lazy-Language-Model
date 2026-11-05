import { formatLineRange, requireTarget } from "@/lib/code-targets";
import { getPipeline } from "@/lib/concepts";
import { formatShape, resolveDimension, type DimensionName } from "@/lib/dimensions";
import { PIPELINE, type PipelineNode, type PipelinePhase } from "@/data/pipeline";
import { PageLink } from "./PageLink";
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

function ShapeValues({ dims }: { dims: DimensionName[] }) {
  const values = dims.map((name) => resolveDimension(name).value ?? name);
  return (
    <span className={styles.shape}>
      <code>{formatShape(dims)}</code>
      {dims.some((name) => resolveDimension(name).value !== null) ? (
        <code className={styles.values}>{formatShape(values)}</code>
      ) : null}
    </span>
  );
}

function Node({
  node,
  number,
  showCode,
}: {
  node: PipelineNode;
  number: number;
  showCode: boolean;
}) {
  const target = requireTarget(node.target);
  const code = node.inline ?? node.call?.split(".").at(-1);
  return (
    <li className={styles.node} data-testid={`node-${node.id}`}>
      <div className={styles.nodeBox}>
        <span className={styles.nodeLabel}>
          <span className={styles.number}>{number}.</span>{" "}
          <PageLink page={node.page}>{node.label}</PageLink>
        </span>
        {showCode && code ? <code className={styles.call}>{code}</code> : null}
        {node.dims ? <ShapeValues dims={node.dims} /> : null}
        {node.note ? <span className={styles.note}>{node.note}</span> : null}
        <a className="code-link" href={target.permalink}>
          {target.symbol.file}:{formatLineRange(target.startLine, target.endLine)}
        </a>
      </div>
    </li>
  );
}

function RepeatTitle({ phase }: { phase: PipelinePhase }) {
  const { repeat } = phase;
  if (!repeat) return <div className={styles.phaseTitle}>{phase.title}</div>;
  const { configName, value } = resolveDimension(repeat.dimension);
  const target = requireTarget(repeat.target);
  return (
    <div className={styles.phaseTitle}>
      {phase.title}, repeated <code>{configName}</code> = {value} times{" "}
      <a className="code-link" href={target.permalink}>
        {target.symbol.file}:{formatLineRange(target.startLine, target.endLine)}
      </a>
    </div>
  );
}

// The pipeline from data/pipeline.ts. Each node links to its page and to its resolved lines.
export function FullPipelineDiagram({ only }: { only?: "forward" }) {
  const phases = only === "forward" ? PIPELINE.filter((phase) => phase.forward) : PIPELINE;
  let number = 0;
  return (
    <figure className={styles.flow} data-testid={only ? "pipeline-forward" : "pipeline-full"}>
      {phases.map((phase) => (
        <section
          className={phase.repeat ? `${styles.phase} ${styles.repeat}` : styles.phase}
          key={phase.id}
          aria-label={phase.title}
          data-phase={phase.id}
        >
          <RepeatTitle phase={phase} />
          <ol
            className={phase.repeat ? `${styles.nodes} ${styles.threes}` : styles.nodes}
            start={number + 1}
          >
            {phase.nodes.map((node) => (
              <Node node={node} number={++number} showCode={only === "forward"} key={node.id} />
            ))}
          </ol>
        </section>
      ))}
      <figcaption className={styles.key}>
        Plain names have no page yet. The link under each name opens the real lines at the pinned
        commit.
      </figcaption>
    </figure>
  );
}
