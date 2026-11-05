import type { Metadata } from "next";
import Link from "next/link";
import { Badge } from "@/components/Badge";
import { Button, buttonClass } from "@/components/Button";
import { DataTable } from "@/components/DataTable";
import { DocMain } from "@/components/DocMain";
import styles from "@/components/Home.module.css";
import { OnThisPage } from "@/components/OnThisPage";
import { PageLink } from "@/components/PageLink";
import { FullPipelineDiagram } from "@/components/PipelineDiagram";
import { GLANCE, QUESTIONS } from "@/data/home";
import { buildGlance } from "@/lib/model-glance";
import { pageHref } from "@/lib/nav";
import { firstWrittenInGroup, writtenHref } from "@/lib/pages";
import { loadConfig, loadRepoMeta } from "@/lib/repo";

export const metadata: Metadata = {
  title: "LLM Developer Docs",
  description:
    "The mathematics behind a small language model built from scratch, and the code that implements it.",
};

const HEADINGS = [
  { id: "the-model-at-a-glance", text: "The model at a glance", depth: 2 },
  { id: "the-pipeline", text: "The pipeline", depth: 2 },
  { id: "questions-this-site-answers", text: "Questions this site answers", depth: 2 },
  { id: "labels-used-on-this-site", text: "Labels used on this site", depth: 2 },
] as const;

const [GLANCE_HEADING, PIPELINE_HEADING, QUESTIONS_HEADING, LABELS_HEADING] = HEADINGS;

function HeroButton({
  href,
  label,
  primary,
}: {
  href: string | null;
  label: string;
  primary?: boolean;
}) {
  const variant = primary ? "primary" : "secondary";
  if (href) {
    return (
      <Link className={buttonClass(variant)} href={href}>
        {label}
      </Link>
    );
  }
  return (
    <Button variant={variant} disabled title="This page is not written yet">
      {label}
    </Button>
  );
}

export default function HomePage() {
  const firstMathematics = firstWrittenInGroup("Mathematics");
  const glance = buildGlance(GLANCE, loadConfig(), loadRepoMeta());

  return (
    <>
      <DocMain>
        <header className={styles.hero}>
          <h1>LLM Developer Docs</h1>
          <p className={styles.heroText}>
            This is a small language model built from scratch. This documentation shows the
            mathematics behind it, the code that implements the mathematics, and how the pieces work
            together.
          </p>
          <div className={styles.actions}>
            <HeroButton
              primary
              label="Start learning"
              href={firstMathematics && pageHref(firstMathematics.slug)}
            />
            <HeroButton label="Explore the code" href={writtenHref("code/code-explorer")} />
          </div>
        </header>

        <section className={styles.section}>
          <h2 id={GLANCE_HEADING.id}>{GLANCE_HEADING.text}</h2>
          <p className={styles.lead}>
            Read from <code>config.py</code> at the pinned commit.
          </p>
          <DataTable label="The model at a glance" data-testid="glance">
            <thead>
              <tr>
                <th scope="col">Setting</th>
                <th scope="col">
                  Name in <code>config.py</code>
                </th>
                <th scope="col">Value</th>
              </tr>
            </thead>
            {glance.map((group) => (
              <tbody key={group.title}>
                <tr className={styles.groupRow}>
                  <th scope="rowgroup" colSpan={3}>
                    {group.title}
                  </th>
                </tr>
                {group.rows.map((row) => (
                  <tr key={row.name} data-config={row.name}>
                    <td>
                      {row.label}
                      {row.note ? <span className={styles.rowNote}> {row.note}</span> : null}
                      {row.unread ? (
                        <span className={styles.rowNote}> No module reads this constant.</span>
                      ) : null}
                    </td>
                    <td>
                      <a className="code-link" href={row.permalink}>
                        {row.name}
                      </a>
                    </td>
                    <td>
                      <code>{row.value}</code>
                    </td>
                  </tr>
                ))}
              </tbody>
            ))}
          </DataTable>
        </section>

        <section className={styles.section}>
          <h2 id={PIPELINE_HEADING.id}>{PIPELINE_HEADING.text}</h2>
          <p className={styles.lead}>
            Every step the code runs, from text to generated tokens. Shapes are for one sequence:
            the model has no batch dimension.
          </p>
          <FullPipelineDiagram />
        </section>

        <section className={styles.section}>
          <h2 id={QUESTIONS_HEADING.id}>{QUESTIONS_HEADING.text}</h2>
          <ul className={styles.questions}>
            {QUESTIONS.map(({ question, page }) => (
              <li key={question}>
                <PageLink page={page}>{question}</PageLink>
              </li>
            ))}
          </ul>
        </section>

        <section className={styles.section}>
          <h2 id={LABELS_HEADING.id}>{LABELS_HEADING.text}</h2>
          <dl className={styles.legend}>
            <div>
              <dt>
                <Badge kind="illustrative" />
              </dt>
              <dd>Hand-picked numbers. These are not learned values.</dd>
            </div>
            <div>
              <dt>
                <Badge kind="actual" />
              </dt>
              <dd>Numbers produced by running the repository, with their source stated.</dd>
            </div>
            <div>
              <dt>
                <Badge kind="derived" />
              </dt>
              <dd>Mathematics the code implements but the reference PDF does not state.</dd>
            </div>
          </dl>
        </section>
      </DocMain>
      <OnThisPage headings={[...HEADINGS]} />
    </>
  );
}
