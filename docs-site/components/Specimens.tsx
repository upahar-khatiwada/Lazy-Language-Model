import { useId, type ReactNode } from "react";
import { loadCodeIndex } from "@/lib/repo";
import { Badge } from "./Badge";
import { Button, buttonClass } from "./Button";
import { Callout } from "./Callout";
import { CodeBlock } from "./CodeBlock";
import { DataTable } from "./DataTable";
import { MathBlock } from "./MathBlock";
import { SearchField } from "./SearchField";
import { SidebarItem } from "./SidebarItem";
import styles from "./Specimens.module.css";

const COLOUR_TOKENS = [
  "--bg",
  "--surface",
  "--text",
  "--muted",
  "--line",
  "--accent",
  "--highlight",
  "--highlight-line",
  "--code-bg",
  "--hover",
  "--active-bg",
  "--syntax-keyword",
  "--syntax-string",
  "--syntax-number",
  "--syntax-comment",
  "--syntax-function",
];

const SPACE_TOKENS = [1, 2, 3, 4, 5, 6, 7, 8].map((step) => `--space-${step}`);

const SAMPLE_PATH = "tests/test_gradcheck.py";
const SAMPLE_FUNCTION = "test_key_bias_gradient_is_zero";

// Real source with a keyword, string, number, comment and function, so every syntax token shows.
function sampleSymbol() {
  const file = loadCodeIndex().files.find((candidate) => candidate.path === SAMPLE_PATH);
  const symbol = file?.functions.find((candidate) => candidate.name === SAMPLE_FUNCTION);
  if (!symbol) throw new Error(`${SAMPLE_PATH} has no function ${SAMPLE_FUNCTION}`);
  return symbol;
}

function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className={styles.section} data-specimen={title}>
      <h3>{title}</h3>
      {children}
    </section>
  );
}

export function Specimens({ theme }: { theme: "light" | "dark" }) {
  const headingId = useId();
  const symbol = sampleSymbol();

  return (
    <section
      className={styles.panel}
      data-theme={theme}
      data-testid={`panel-${theme}`}
      aria-labelledby={headingId}
    >
      <h2 id={headingId}>{theme === "light" ? "Light theme" : "Dark theme"}</h2>

      <Section title="Typography">
        <p className={styles.sampleH1}>Heading level 1</p>
        <p className={styles.sampleH2}>Heading level 2</p>
        <p className={styles.sampleH3}>Heading level 3</p>
        <div className="prose" data-testid="body-sample">
          <p>
            Body text is IBM Plex Sans at 15px with a line height of 1.65. Prose keeps a measure
            near sixty-five characters so that long lines stay easy to follow from one edge to the
            other. Short paragraphs and long ones alike wrap at the same comfortable width, whether
            the column around them is narrow or wide, and tables and code blocks may use more room.
          </p>
        </div>
        <p className={styles.muted}>Muted text is used for secondary information.</p>
        <p>
          Inline code such as <code>tokenizer.py</code> uses IBM Plex Mono.
        </p>
        <p>
          A <a href="#design-top">standard link</a> uses the accent colour. A{" "}
          <a className="code-link" href="#design-top">
            code or math link
          </a>{" "}
          uses the amber highlight.
        </p>
      </Section>

      <Section title="Colour tokens">
        <ul className={styles.swatches}>
          {COLOUR_TOKENS.map((token) => (
            <li className={styles.swatch} key={token}>
              <span className={styles.chip} style={{ background: `var(${token})` }} />
              {token}
            </li>
          ))}
        </ul>
      </Section>

      <Section title="Spacing, radius and shadow">
        <ul className={styles.spacing}>
          {SPACE_TOKENS.map((token) => (
            <li className={styles.spacingRow} key={token}>
              <span className={styles.bar} style={{ width: `var(${token})` }} />
              {token}
            </li>
          ))}
        </ul>
        <p className={styles.shadowSample}>6px radius, 1px border, the single subtle shadow</p>
      </Section>

      <Section title="Badges">
        <div className={styles.row}>
          <Badge kind="illustrative" />
          <Badge kind="actual" />
          <Badge kind="derived" />
        </div>
      </Section>

      <Section title="Callouts">
        <Callout kind="note">
          <p>A note adds information that helps but is not essential.</p>
        </Callout>
        <Callout kind="warning">
          <p>A warning points out something that is easy to get wrong.</p>
        </Callout>
        <Callout kind="source">
          <p>A source callout names where a statement comes from.</p>
        </Callout>
      </Section>

      <Section title="Tables">
        <DataTable label="Example labels">
          <thead>
            <tr>
              <th scope="col">Label</th>
              <th scope="col">Meaning</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>Illustrative Example</td>
              <td>Hand-picked numbers</td>
            </tr>
            <tr>
              <td>Actual Model Output</td>
              <td>Produced by running the repository</td>
            </tr>
          </tbody>
        </DataTable>
      </Section>

      <Section title="Code block">
        <CodeBlock
          source={symbol.source}
          filePath={SAMPLE_PATH}
          functionName={symbol.name}
          startLine={symbol.startLine}
          permalink={symbol.permalink}
        />
      </Section>

      <Section title="Math block">
        <p className={styles.muted}>Typography specimen only. This is not model mathematics.</p>
        <MathBlock
          label="Typography specimen"
          tex={String.raw`\sum_{i=1}^{n} \frac{a_i^2}{\sqrt{b_i}} = \begin{bmatrix} x & y \\ z & w \end{bmatrix}`}
        />
      </Section>

      <Section title="Sidebar item states">
        <div className={styles.sidebarSample}>
          <SidebarItem
            title="Default"
            sourceFile="ops.py"
            href="#design-top"
            state="default"
            preview={{}}
          />
          <SidebarItem
            title="Hover"
            sourceFile="ops.py"
            href="#design-top"
            state="default"
            preview={{ hover: true }}
          />
          <SidebarItem
            title="Active"
            sourceFile="ops.py"
            href="#design-top"
            state="active"
            preview={{}}
          />
          <SidebarItem title="Disabled" sourceFile="ops.py" href="#design-top" state="disabled" />
        </div>
      </Section>

      <Section title="Focus ring">
        <p className={styles.focusSample}>Keyboard focus looks like this</p>
      </Section>

      <Section title="Buttons">
        <div className={styles.row}>
          <Button variant="primary">Primary</Button>
          <Button>Secondary</Button>
          <Button disabled>Disabled</Button>
          <a className={buttonClass()} href="#design-top">
            Link as button
          </a>
        </div>
      </Section>

      <Section title="Search field">
        <SearchField />
      </Section>
    </section>
  );
}
