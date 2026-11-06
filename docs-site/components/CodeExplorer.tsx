import Link from "next/link";
import type { ReactNode } from "react";
import { CODE_REFS } from "@/data/code-refs";
import { formatLineRange, requireTarget } from "@/lib/code-targets";
import {
  buildFileTree,
  callableSymbols,
  callsOf,
  configReaders,
  constantReads,
  explorerFiles,
  explorerHref,
  fileSlug,
  filePurpose,
  fileSymbols,
  fileUsedBy,
  importedBy,
  isTestFile,
  locateSymbol,
  mathForFile,
  mathForSymbol,
  moduleEntry,
  moduleFile,
  pipelineSteps,
  reachCounts,
  reachedFrom,
  sourceTree,
  symbolPurpose,
  testKind,
  usedBy,
  type ExplorerFile,
  type FileSymbol,
  type MathUse,
  type SourceNode,
  type TreeNode,
} from "@/lib/code-explorer";
import { highlightPython, type CodeToken } from "@/lib/highlight";
import { equationsByLine } from "@/lib/links";
import { Badge } from "./Badge";
import codeStyles from "./CodeBlock.module.css";
import styles from "./CodeExplorer.module.css";
import { EquationMark } from "./CodeFrame";
import { CopyButton } from "./CopyButton";
import { DataTable } from "./DataTable";
import { InlineCode } from "./InlineCode";
import { PageLink } from "./PageLink";

// ---------- links ----------

// A same-page link stays a plain anchor, so the browser sets :target and fires hashchange.
function ExplorerLink({
  path,
  anchor,
  from,
  children,
}: {
  path: string;
  anchor?: string;
  from?: string;
  children: ReactNode;
}) {
  if (path === from && anchor) return <a href={`#${anchor}`}>{children}</a>;
  return (
    <Link href={explorerHref(path, anchor)} prefetch={false}>
      {children}
    </Link>
  );
}

// For MDX: a repository file or symbol, opened in the Code Explorer.
export function CodeFileLink({
  file,
  symbol,
  codeRef,
  children,
}: {
  file?: string;
  symbol?: string;
  // An id from data/code-refs.ts, which links to the first of its lines.
  codeRef?: string;
  children?: ReactNode;
}) {
  if (codeRef) {
    const target = CODE_REFS[codeRef];
    if (!target) throw new Error(`Unknown code reference: ${codeRef}`);
    const { symbol: found, startLine } = requireTarget(target);
    return (
      <ExplorerLink path={found.file} anchor={`L${startLine}`}>
        {children}
      </ExplorerLink>
    );
  }
  const place: { path: string; anchor?: string } = symbol ? locateSymbol(symbol) : { path: file! };
  return (
    <ExplorerLink path={place.path} anchor={place.anchor}>
      {children ?? <code>{symbol ?? file}</code>}
    </ExplorerLink>
  );
}

function SymbolRef({ id, from }: { id: string; from?: string }) {
  const { path, anchor } = locateSymbol(id);
  return (
    <ExplorerLink path={path} anchor={anchor} from={from}>
      <code className={styles.name} data-symbol-ref={id}>
        {id}
      </code>
    </ExplorerLink>
  );
}

function SymbolRefs({ ids, from }: { ids: string[]; from?: string }) {
  return (
    <>
      {ids.map((id, index) => (
        <span key={id}>
          {index > 0 ? ", " : null}
          <SymbolRef id={id} from={from} />
        </span>
      ))}
    </>
  );
}

function FileRefs({ paths }: { paths: string[] }) {
  return (
    <>
      {paths.map((path, index) => (
        <span key={path}>
          {index > 0 ? ", " : null}
          <ExplorerLink path={path}>
            <code>{path}</code>
          </ExplorerLink>
        </span>
      ))}
    </>
  );
}

function TestLabel() {
  return <span className={styles.tag}>Test</span>;
}

// ---------- /code ----------

function TreeRows({ nodes, depth }: { nodes: TreeNode[]; depth: number }) {
  const files = new Map(explorerFiles().map((file) => [file.path, file]));
  return nodes.map((node) =>
    node.type === "folder" ? (
      <TreeFolder key={node.name} name={node.name} nodes={node.children} depth={depth} />
    ) : (
      <tr key={node.path} data-file={node.path}>
        <td style={{ paddingLeft: `calc(var(--space-3) + ${depth * 2}ch)` }}>
          <Link className="code-link" href={explorerHref(node.path)} prefetch={false}>
            {node.name}
          </Link>
        </td>
        <td>{isTestFile(node.path) ? <TestLabel /> : "Source"}</td>
        <td>{files.get(node.path)!.lineCount}</td>
        <td>{callableSymbols(files.get(node.path)!).length}</td>
        <td>
          <InlineCode text={filePurpose(files.get(node.path)!)} />
        </td>
      </tr>
    ),
  );
}

function TreeFolder({ name, nodes, depth }: { name: string; nodes: TreeNode[]; depth: number }) {
  return (
    <>
      <tr data-folder={name}>
        <td colSpan={5} style={{ paddingLeft: `calc(var(--space-3) + ${depth * 2}ch)` }}>
          <strong>
            <code>{name}/</code>
          </strong>
        </td>
      </tr>
      <TreeRows nodes={nodes} depth={depth + 1} />
    </>
  );
}

export function CodeTree() {
  const tree = buildFileTree(explorerFiles().map((file) => file.path));
  return (
    <DataTable label="Repository files" data-testid="code-tree">
      <thead>
        <tr>
          <th scope="col">File</th>
          <th scope="col">Kind</th>
          <th scope="col">Lines</th>
          <th scope="col">Functions</th>
          <th scope="col">Purpose</th>
        </tr>
      </thead>
      <tbody>
        <TreeRows nodes={tree} depth={0} />
      </tbody>
    </DataTable>
  );
}

// ---------- the file list on the left ----------

function FileListNodes({ nodes, current }: { nodes: TreeNode[]; current: string }) {
  return (
    <ul className={styles.fileListItems}>
      {nodes.map((node) =>
        node.type === "folder" ? (
          <li key={node.name}>
            <span className={styles.folder}>
              {node.name}/{node.name === "tests" ? <TestLabel /> : null}
            </span>
            <FileListNodes nodes={node.children} current={current} />
          </li>
        ) : (
          <li key={node.path}>
            <Link
              className={node.path === current ? styles.currentFile : styles.fileLink}
              href={explorerHref(node.path)}
              aria-current={node.path === current ? "page" : undefined}
              prefetch={false}
            >
              {node.name}
            </Link>
          </li>
        ),
      )}
    </ul>
  );
}

export const FILE_SECTIONS = [
  { id: "file-pipeline", title: "Where it fits" },
  { id: "file-imports", title: "Imports and used by" },
  { id: "file-math", title: "Mathematics used" },
  { id: "file-symbols", title: "Functions and constants" },
  { id: "file-source", title: "Source" },
] as const;

function FileList({ file }: { file: ExplorerFile }) {
  const tree = buildFileTree(explorerFiles().map(({ path }) => path));
  const symbols = fileSymbols(file);
  return (
    <div className={styles.fileList}>
      <nav aria-label="Repository files">
        <div className={styles.listTitle}>Files</div>
        <FileListNodes nodes={tree} current={file.path} />
      </nav>
      <nav aria-label="In this file">
        <div className={styles.listTitle}>In this file</div>
        <ul className={styles.fileListItems}>
          {FILE_SECTIONS.map(({ id, title }) => (
            <li key={id}>
              <a className={styles.fileLink} href={`#${id}`}>
                {title}
              </a>
            </li>
          ))}
        </ul>
        {symbols.length > 0 ? (
          <ul className={`${styles.fileListItems} ${styles.symbolList}`}>
            {symbols.map((symbol) => (
              <li key={symbol.anchor}>
                <a className={styles.symbolLink} href={`#${symbol.anchor}`}>
                  {symbol.anchor}
                </a>
              </li>
            ))}
          </ul>
        ) : null}
      </nav>
    </div>
  );
}

// ---------- where it fits ----------

function WhereItFits({ file }: { file: ExplorerFile }) {
  const steps = pipelineSteps(file.path);
  const counts = reachCounts(file);
  const entry = moduleEntry(file);
  const readers = configReaders(file);
  const used = fileUsedBy(file);

  if (isTestFile(file.path)) {
    return (
      <p data-testid="fit-text">
        A test file. It is not part of the pipeline: neither <code>train.main</code> nor{" "}
        <code>talk.main</code> reaches it. It is listed with what it verifies on{" "}
        <PageLink page="code/checkpoints-tests" hash="test-suite">
          Checkpoints & Tests
        </PageLink>
        .
      </p>
    );
  }

  return (
    <>
      {entry ? (
        <p data-testid="fit-text">
          An entry point. Running <code>python {file.path}</code> calls{" "}
          <SymbolRef id={entry.target} from={file.path} /> at{" "}
          <a href={`#L${entry.line}`}>line {entry.line}</a>.
        </p>
      ) : null}
      {counts.total > 0 ? (
        <p data-testid="fit-counts">
          {counts.train} of {counts.total} functions, classes and methods here are reachable from{" "}
          <code>train.main</code>, and {counts.talk} from <code>talk.main</code>.
          {counts.train === 0 && counts.talk === 0 && used.tests.length > 0
            ? " Only tests use this file."
            : null}
        </p>
      ) : null}
      {readers.length > 0 ? (
        <ul data-testid="fit-readers">
          {readers.map(({ path, constants }) => (
            <li key={path}>
              <ExplorerLink path={path}>
                <code>{path}</code>
              </ExplorerLink>{" "}
              reads{" "}
              {constants.map((name, index) => (
                <span key={name}>
                  {index > 0 ? ", " : null}
                  <a href={`#${name}`}>
                    <code>{name}</code>
                  </a>
                </span>
              ))}
              .
            </li>
          ))}
        </ul>
      ) : null}
      {steps.length > 0 ? (
        <DataTable label="Pipeline steps in this file" data-testid="fit-steps">
          <thead>
            <tr>
              <th scope="col">Pipeline stage</th>
              <th scope="col">Step</th>
              <th scope="col">Explained on</th>
            </tr>
          </thead>
          <tbody>
            {steps.map(({ phase, label, page }, index) => (
              <tr key={index}>
                <td>{phase}</td>
                <td>{label}</td>
                <td>
                  <PageLink page={page}>{page.split("/").at(-1)!.replaceAll("-", " ")}</PageLink>
                </td>
              </tr>
            ))}
          </tbody>
        </DataTable>
      ) : null}
    </>
  );
}

// ---------- imports ----------

function ImportsAndUsedBy({ file }: { file: ExplorerFile }) {
  const internal = [
    ...new Set(file.imports.filter((entry) => entry.internal).map((entry) => entry.module)),
  ];
  const external = [
    ...new Set(file.imports.filter((entry) => !entry.internal).map((entry) => entry.module)),
  ];
  const used = fileUsedBy(file);
  const importers = importedBy(file);
  // Constants have no calls, so the call graph has no record of who uses them.
  const readers = configReaders(file).map(({ path }) => path);
  return (
    <dl className={styles.facts} data-testid="file-relations">
      <dt>Imports from this repository</dt>
      <dd>
        {internal.length > 0 ? (
          <FileRefs paths={internal.map((module) => moduleFile(module)!.path)} />
        ) : (
          "None"
        )}
      </dd>
      <dt>Other imports</dt>
      <dd>
        {external.length > 0
          ? external.map((module, index) => (
              <span key={module}>
                {index > 0 ? ", " : null}
                <code>{module}</code>
              </span>
            ))
          : "None"}
      </dd>
      <dt>Imported by</dt>
      <dd>{importers.length > 0 ? <FileRefs paths={importers} /> : "No file"}</dd>
      <dt>Used by, in the pipeline</dt>
      <dd>
        {used.production.length > 0 ? (
          <FileRefs paths={used.production} />
        ) : readers.length > 0 ? (
          <>
            No calls, because this file defines only constants. <FileRefs paths={readers} /> read
            them.
          </>
        ) : (
          "No file"
        )}
      </dd>
      <dt>Used by, in tests</dt>
      <dd>{used.tests.length > 0 ? <FileRefs paths={used.tests} /> : "No file"}</dd>
    </dl>
  );
}

// ---------- mathematics ----------

const ROLE_LABEL = {
  implements: "Computes it",
  inline: "Writes it in place",
  call: "Calls it",
  test: "Tests it",
} as const;

function ConceptName({ use }: { use: MathUse }) {
  return use.href ? (
    <Link href={use.href} prefetch={false}>
      {use.title}
    </Link>
  ) : (
    <span>{use.title}</span>
  );
}

function MathTable({ file }: { file: ExplorerFile }) {
  const uses = mathForFile(file.path);
  if (uses.length === 0) {
    return <p data-testid="math-none">No link record points to this file.</p>;
  }
  return (
    <DataTable label="Mathematics in this file" data-testid="math-table">
      <thead>
        <tr>
          <th scope="col">Concept</th>
          <th scope="col">What this file does</th>
          <th scope="col">Lines</th>
          <th scope="col">Status</th>
        </tr>
      </thead>
      <tbody>
        {uses.map((use) => (
          <tr key={use.conceptId} data-concept={use.conceptId}>
            <td>
              <ConceptName use={use} />
            </td>
            <td>{use.roles.map((role) => ROLE_LABEL[role]).join(", ")}</td>
            <td>
              {use.lines.map(({ start, end }, index) => (
                <span key={start}>
                  {index > 0 ? ", " : null}
                  <a href={`#L${start}`}>{formatLineRange(start, end)}</a>
                </span>
              ))}
            </td>
            <td>
              <Badge kind={use.status} />
            </td>
          </tr>
        ))}
      </tbody>
    </DataTable>
  );
}

// ---------- functions ----------

function SymbolCard({ symbol, file }: { symbol: FileSymbol; file: ExplorerFile }) {
  const used = usedBy(symbol.id);
  const calls = callsOf(symbol.id);
  const math = mathForSymbol(symbol.id);
  const purpose = symbolPurpose(symbol.id);
  const reached = reachedFrom(symbol.id);
  const kind = testKind(file, symbol.id);
  const test = isTestFile(file.path);
  return (
    <article
      className={symbol.kind === "method" ? `${styles.card} ${styles.method}` : styles.card}
      data-symbol-card={symbol.anchor}
    >
      <div className={styles.cardHead}>
        <a className={styles.signature} href={`#${symbol.anchor}`}>
          <code>{symbol.signature}</code>
        </a>
        <span className={styles.cardMeta}>
          {kind ? <span className={styles.tag}>{kind}</span> : null}
          <span data-lines>Lines {formatLineRange(symbol.startLine, symbol.endLine)}</span>
          <a href={symbol.permalink}>
            GitHub<span className="visually-hidden">: {symbol.id}</span>
          </a>
        </span>
      </div>
      {purpose ? (
        <p>
          <InlineCode text={purpose} />
        </p>
      ) : null}
      <dl className={styles.facts}>
        {test ? null : (
          <>
            <dt>Reached from</dt>
            <dd>
              {reached.length > 0
                ? reached.map((id, index) => (
                    <span key={id}>
                      {index > 0 ? ", " : null}
                      <code>{id}</code>
                    </span>
                  ))
                : "Neither train.main nor talk.main"}
            </dd>
          </>
        )}
        <dt>Used by, in the pipeline</dt>
        <dd data-used-by="production">
          {used.production.length > 0 ? (
            <SymbolRefs ids={used.production} from={file.path} />
          ) : (
            "Nothing"
          )}
        </dd>
        <dt>Used by, in tests</dt>
        <dd data-used-by="tests">
          {used.tests.length > 0 ? <SymbolRefs ids={used.tests} from={file.path} /> : "Nothing"}
        </dd>
        {calls.length > 0 ? (
          <>
            <dt>Calls</dt>
            <dd data-calls>
              <SymbolRefs ids={calls} from={file.path} />
            </dd>
          </>
        ) : null}
        {math.length > 0 ? (
          <>
            <dt>Mathematics</dt>
            <dd>
              {math.map((use, index) => (
                <span key={use.conceptId}>
                  {index > 0 ? ", " : null}
                  <ConceptName use={use} />
                </span>
              ))}
            </dd>
          </>
        ) : null}
      </dl>
    </article>
  );
}

function ConstantsTable({ file, symbols }: { file: ExplorerFile; symbols: FileSymbol[] }) {
  const isConfig = constantReads(file, symbols[0]!.name) !== null;
  return (
    <DataTable label="Constants" data-testid="constants">
      <thead>
        <tr>
          <th scope="col">Constant</th>
          <th scope="col">Lines</th>
          {isConfig ? <th scope="col">Read by</th> : null}
        </tr>
      </thead>
      <tbody>
        {symbols.map((symbol) => {
          const reads = constantReads(file, symbol.name);
          return (
            <tr key={symbol.anchor} data-constant={symbol.anchor}>
              <td>
                <a className={styles.name} href={`#${symbol.anchor}`}>
                  <code>{symbol.source}</code>
                </a>
              </td>
              <td>
                <a href={symbol.permalink}>{formatLineRange(symbol.startLine, symbol.endLine)}</a>
              </td>
              {isConfig ? (
                <td data-read-by>
                  {reads && reads.length > 0
                    ? reads.map((read, index) => (
                        <span key={`${read.file}:${read.line}`}>
                          {index > 0 ? ", " : null}
                          <ExplorerLink path={read.file} anchor={`L${read.line}`}>
                            <code>
                              {read.file}:{read.line}
                            </code>
                          </ExplorerLink>
                        </span>
                      ))
                    : "Nothing"}
                </td>
              ) : null}
            </tr>
          );
        })}
      </tbody>
    </DataTable>
  );
}

// ---------- source ----------

function renderSource(
  nodes: SourceNode[],
  lines: CodeToken[][],
  equations: ReadonlyMap<number, string[]>,
): ReactNode {
  return nodes.map((node) => {
    if (node.kind === "symbol") {
      return (
        <span
          className={styles.symbolRange}
          id={node.anchor}
          data-symbol={node.anchor}
          data-start={node.startLine}
          data-end={node.endLine}
          key={`${node.anchor}`}
        >
          {renderSource(node.children, lines, equations)}
        </span>
      );
    }
    const slugs = equations.get(node.line);
    return (
      <span
        className={`${codeStyles.line} ${styles.sourceLine}`}
        id={`L${node.line}`}
        data-line={node.line}
        data-eqs={slugs?.join(" ")}
        key={node.line}
      >
        {slugs ? <EquationMark line={node.line} slugs={slugs} /> : null}
        <span className={codeStyles.number} aria-hidden="true">
          {node.line}
        </span>
        {(lines[node.line - 1] ?? []).map((token, index) => (
          <span key={index} style={{ color: token.color }}>
            {token.content}
          </span>
        ))}
      </span>
    );
  });
}

export async function SourceListing({
  file,
  permalink,
}: {
  file: ExplorerFile;
  permalink: string;
}) {
  const lines = await highlightPython(file.source);
  const symbols = fileSymbols(file);
  return (
    <figure className={`${codeStyles.block} ${styles.listing}`} data-source-listing>
      <figcaption className={codeStyles.chrome}>
        <span className={codeStyles.file}>{file.path}</span>
        <span className={codeStyles.function}>{file.lineCount} lines</span>
        <span className={codeStyles.actions}>
          <a className={codeStyles.githubLink} href={permalink}>
            GitHub<span className="visually-hidden">: {file.path}</span>
          </a>
          <CopyButton text={file.source} />
        </span>
      </figcaption>
      <pre
        className={`${codeStyles.pre} ${styles.sourcePre}`}
        tabIndex={0}
        aria-label={`Source of ${file.path}`}
      >
        <code className={codeStyles.code}>
          {renderSource(sourceTree(file.lineCount, symbols), lines, equationsByLine(file.path))}
        </code>
      </pre>
    </figure>
  );
}

// ---------- the page ----------

export function explorerNeighbours(file: ExplorerFile) {
  const files = explorerFiles();
  const index = files.findIndex((candidate) => candidate.path === file.path);
  const page = (other?: ExplorerFile) =>
    other
      ? {
          title: other.path,
          slug: `code/${fileSlug(other.path)}`,
          href: explorerHref(other.path),
          enabled: true,
        }
      : undefined;
  return { prev: page(files[index - 1]), next: page(files[index + 1]) };
}

export function CodeFileBody({
  file,
  children,
}: {
  file: ExplorerFile;
  // The source listing, which is asynchronous and rendered by the page.
  children: ReactNode;
}) {
  const symbols = fileSymbols(file);
  const callable = symbols.filter((symbol) => symbol.kind !== "constant");
  const constants = symbols.filter((symbol) => symbol.kind === "constant");
  return (
    <div className={styles.layout}>
      <FileList file={file} />
      <div className={`prose ${styles.body}`}>
        <h2 id="file-pipeline">Where it fits</h2>
        <WhereItFits file={file} />
        <h2 id="file-imports">Imports and used by</h2>
        <ImportsAndUsedBy file={file} />
        <h2 id="file-math">Mathematics used</h2>
        <MathTable file={file} />
        <h2 id="file-symbols">Functions and constants</h2>
        {callable.length > 0 ? (
          <div className={styles.cards}>
            {callable.map((symbol) => (
              <SymbolCard key={symbol.anchor} symbol={symbol} file={file} />
            ))}
          </div>
        ) : (
          <p>This file defines no functions.</p>
        )}
        {constants.length > 0 ? <ConstantsTable file={file} symbols={constants} /> : null}
        <h2 id="file-source">Source</h2>
        {children}
      </div>
    </div>
  );
}
