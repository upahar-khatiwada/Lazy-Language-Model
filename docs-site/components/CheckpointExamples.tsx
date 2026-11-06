import Link from "next/link";
import { CHECKPOINT_FILES } from "@/data/checkpoint";
import {
  explorerHref,
  locateSymbol,
  moduleEntry,
  moduleFile,
  testCoverageRows,
  untestedSymbols,
  usedBy,
} from "@/lib/code-explorer";
import { loadCodeIndex } from "@/lib/repo";
import { getGoldenCase } from "@/lib/golden";
import styles from "./CheckpointExamples.module.css";
import { CodeLine } from "./CodeStep";
import { DataTable } from "./DataTable";
import { InlineCode } from "./InlineCode";
import { VizFigure } from "./VizFigure";

function SymbolLink({ id }: { id: string }) {
  const { path, anchor } = locateSymbol(id);
  return (
    <Link href={explorerHref(path, anchor)} prefetch={false}>
      <code className={styles.name}>{id}</code>
    </Link>
  );
}

function SymbolLinks({ ids }: { ids: string[] }) {
  return (
    <>
      {ids.map((id, index) => (
        <span key={id}>
          {index > 0 ? ", " : null}
          <SymbolLink id={id} />
        </span>
      ))}
    </>
  );
}

export function CheckpointFiles() {
  return (
    <DataTable label="Files in one checkpoint" data-testid="checkpoint-files">
      <thead>
        <tr>
          <th scope="col">File</th>
          <th scope="col">Format</th>
          <th scope="col">What it holds</th>
          <th scope="col">Written at</th>
          <th scope="col">Read at</th>
        </tr>
      </thead>
      <tbody>
        {CHECKPOINT_FILES.map(({ extension, format, holds, write, read }) => (
          <tr key={extension} data-extension={extension}>
            <td>
              <code>prefix{extension}</code>
            </td>
            <td>{format}</td>
            <td>
              <InlineCode text={holds} />
            </td>
            <td>
              <CodeLine id={write} />
            </td>
            <td>
              <CodeLine id={read} />
            </td>
          </tr>
        ))}
      </tbody>
    </DataTable>
  );
}

export function ModelConfigFields() {
  const cls = loadCodeIndex()
    .files.flatMap((file) => file.classes)
    .find((candidate) => candidate.id === "transformer.ModelConfig")!;
  return (
    <DataTable label="Fields of ModelConfig" data-testid="model-config-fields">
      <thead>
        <tr>
          <th scope="col">Key in the file</th>
          <th scope="col">Type</th>
          <th scope="col">Default</th>
        </tr>
      </thead>
      <tbody>
        {cls.fields.map((field) => (
          <tr key={field.name}>
            <td>
              <code>{field.name}</code>
            </td>
            <td>
              <code>{field.annotation}</code>
            </td>
            <td>{field.default ? <code>{field.default}</code> : "None, so it must be passed"}</td>
          </tr>
        ))}
      </tbody>
    </DataTable>
  );
}

type CheckpointOutputs = {
  folders: string[];
  writes: string[];
  arrays: { key: string; shape: number[] }[];
  config_json: string;
  tokenizer_json: string;
};

export function checkpointCase() {
  const { inputs, outputs } = getGoldenCase("checkpoint", "checkpoint.tiny_model");
  return {
    inputs: inputs as { text: string; max_merges: number; prefix: string },
    outputs: outputs as CheckpointOutputs,
  };
}

export function CheckpointExample() {
  const { inputs, outputs } = checkpointCase();
  return (
    <VizFigure
      title={`What save_checkpoint writes for a tokenizer trained on "${inputs.text}"`}
      kind="illustrative"
      caption={`The text and a model with 1 block and emb_dim 4 are hand-picked. The real save_checkpoint ran with its file calls captured, so no file was written. These are not the settings in config.py.`}
    >
      <div className={styles.example} data-testid="checkpoint-example">
        <p>
          Folder it would create: <code>{outputs.folders.join(", ")}</code>. Files, in the order
          they are written: {outputs.writes.map((path) => path).join(", ")}.
        </p>
        <p>
          <code>{outputs.writes[1]}</code>
        </p>
        <pre className={styles.json} tabIndex={0} data-testid="checkpoint-config-json">
          {outputs.config_json}
        </pre>
        <p>
          <code>{outputs.writes[2]}</code>
        </p>
        <pre className={styles.json} tabIndex={0} data-testid="checkpoint-tokenizer-json">
          {outputs.tokenizer_json}
        </pre>
        <p>
          <code>{outputs.writes[0]}</code>: {outputs.arrays.length} arrays
        </p>
        <ul className={styles.arrays} data-testid="checkpoint-arrays">
          {outputs.arrays.map(({ key, shape }) => (
            <li key={key}>
              <code>{key}</code>{" "}
              <span>
                ({shape.join(", ")}
                {shape.length === 1 ? "," : ""})
              </span>
            </li>
          ))}
        </ul>
      </div>
    </VizFigure>
  );
}

// Where a symbol is used, from the call graph.
export function SymbolUsage({ symbol }: { symbol: string }) {
  const used = usedBy(symbol);
  return (
    <dl className={styles.usage} data-testid={`usage-${symbol}`}>
      <dt>Called by</dt>
      <dd>
        {used.production.length > 0 ? (
          <SymbolLinks ids={used.production} />
        ) : (
          "Nothing in the pipeline"
        )}
      </dd>
      <dt>Tested by</dt>
      <dd>
        {used.tests.length > 0 ? <SymbolLinks ids={used.tests} /> : "No test calls it directly"}
      </dd>
    </dl>
  );
}

// The line of a script that starts main, from the call graph.
export function EntryLine({ module }: { module: string }) {
  const file = moduleFile(module);
  const entry = file && moduleEntry(file);
  if (!file || !entry) throw new Error(`No module-level call in ${module}`);
  return (
    <Link className="code-link" href={explorerHref(file.path, `L${entry.line}`)} prefetch={false}>
      {file.path}:{entry.line}
    </Link>
  );
}

export function TestTotals() {
  const rows = testCoverageRows();
  const tests = rows.reduce((sum, row) => sum + row.testCount, 0);
  const cases = rows.reduce((sum, row) => sum + row.caseCount, 0);
  return (
    <span data-testid="test-totals">
      {rows.length} test files hold {tests} test functions, which make {cases} cases because
      parametrized tests run once per value
    </span>
  );
}

export function TestCoverage() {
  return (
    <DataTable label="Test files" data-testid="test-coverage" className={styles.coverage}>
      <thead>
        <tr>
          <th scope="col">Test file</th>
          <th scope="col">Test functions</th>
          <th scope="col">What it verifies</th>
          <th scope="col">Repository functions it exercises</th>
        </tr>
      </thead>
      <tbody>
        {testCoverageRows().map((row) => (
          <tr key={row.path} data-test-file={row.path}>
            <td>
              <Link href={explorerHref(row.path)} prefetch={false}>
                <code className={styles.fileName}>
                  {row.path.split("/").map((part, index, parts) => (
                    <span key={part}>
                      {part}
                      {index < parts.length - 1 ? (
                        <>
                          /<wbr />
                        </>
                      ) : null}
                    </span>
                  ))}
                </code>
              </Link>
            </td>
            <td data-test-count>
              {row.testCount}
              {row.caseCount !== row.testCount ? ` (${row.caseCount} cases)` : null}
            </td>
            <td>
              <ul className={styles.verifies}>
                {row.tests.map((test) => (
                  <li key={test.id} data-test-id={test.id}>
                    <Link
                      href={explorerHref(row.path, test.name)}
                      prefetch={false}
                      title={test.name}
                    >
                      <InlineCode text={test.purpose ?? test.name} />
                    </Link>
                  </li>
                ))}
              </ul>
            </td>
            <td data-exercised>
              <SymbolLinks ids={row.exercised} />
            </td>
          </tr>
        ))}
      </tbody>
    </DataTable>
  );
}

export function UntestedSymbols() {
  return (
    <ul data-testid="untested-symbols">
      {untestedSymbols().map((id) => (
        <li key={id}>
          <SymbolLink id={id} />
        </li>
      ))}
    </ul>
  );
}
