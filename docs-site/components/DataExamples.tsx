import { exampleAnchor } from "@/lib/concepts";
import { getGoldenCase, loadGolden } from "@/lib/golden";
import { batchSizes, joinRows, makeWindows, range, splitIds, windowCount } from "@/lib/math/data";
import { DataTable } from "./DataTable";
import { VizFigure } from "./VizFigure";
import { WindowExplorer } from "./WindowExplorer";

// Inputs are the inputs of golden cases. Results come from lib/math/data.ts, which
// tests/math.golden.test.ts compares with the real functions, except for the shuffled order.

export function CorpusExample() {
  const { rows, max_rows } = getGoldenCase("data", "load_corpus.skips_missing").inputs as {
    rows: (string | null)[];
    max_rows: number;
  };
  return (
    <VizFigure
      title="From rows to one text"
      kind="illustrative"
      caption="Hand-picked rows, not from the corpus."
    >
      <DataTable label="load_corpus input and output" data-testid="corpus-example">
        <thead>
          <tr>
            <th scope="col">Rows</th>
            <th scope="col">Text returned</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>
              <code>{JSON.stringify(rows)}</code>
            </td>
            <td>
              <code data-testid="corpus-text">{JSON.stringify(joinRows(rows, max_rows))}</code>
            </td>
          </tr>
        </tbody>
      </DataTable>
    </VizFigure>
  );
}

export function SplitExample({ concept }: { concept: string }) {
  const cases = loadGolden("data").filter(({ mirrors }) => mirrors === "data.split_ids");
  return (
    <div id={exampleAnchor(concept)}>
      <VizFigure
        title="Splitting before windowing"
        kind="illustrative"
        caption="Hand-picked lengths, with stride 1. These are not counts from the corpus. The last row uses the N of tests/test_data.py with the settings in config.py."
      >
        <DataTable label="The split on small streams" data-testid="split-example">
          <thead>
            <tr>
              <th scope="col">N</th>
              <th scope="col">val_fraction</th>
              <th scope="col">T</th>
              <th scope="col">cut</th>
              <th scope="col">Train windows</th>
              <th scope="col">Validation windows</th>
              <th scope="col">Windows crossing the cut, if windowing came first</th>
            </tr>
          </thead>
          <tbody>
            {cases.map(({ id, inputs }) => {
              const { n, val_fraction, seq_len } = inputs as {
                n: number;
                val_fraction: number;
                seq_len: number;
              };
              const { cut, train, val } = splitIds(range(0, n), val_fraction);
              const whole = makeWindows(range(0, n), seq_len);
              const crossing = whole.starts.filter(
                (_, w) => whole.inputs[w]![0]! < cut && whole.targets[w]!.at(-1)! >= cut,
              ).length;
              return (
                <tr key={id} data-case={id}>
                  <td>{n.toLocaleString("en-US")}</td>
                  <td>{val_fraction}</td>
                  <td>{seq_len}</td>
                  <td data-testid="split-cut">{cut.toLocaleString("en-US")}</td>
                  <td data-testid="split-train">
                    {windowCount(train.length, seq_len).toLocaleString("en-US")}
                  </td>
                  <td data-testid="split-val">{windowCount(val.length, seq_len)}</td>
                  <td data-testid="split-crossing">{crossing}</td>
                </tr>
              );
            })}
          </tbody>
        </DataTable>
      </VizFigure>
    </div>
  );
}

export function WindowExample({ concept }: { concept: string }) {
  const { ids, seq_len, stride } = getGoldenCase("data", "make_windows.test_file").inputs as {
    ids: number[];
    seq_len: number;
    stride: number;
  };
  return (
    <div id={exampleAnchor(concept)}>
      <WindowExplorer ids={ids} seqLen={seq_len} stride={stride} />
    </div>
  );
}

export function WindowCountExample({ concept }: { concept: string }) {
  const cases = loadGolden("data").filter(
    ({ mirrors, outputs }) => mirrors === "data.make_windows" && outputs.count !== undefined,
  );
  return (
    <div id={exampleAnchor(concept)}>
      <VizFigure
        title="The formula and make_windows"
        kind="illustrative"
        caption="Hand-picked lengths. The last column is the number of rows the real make_windows returned."
      >
        <DataTable label="Window count against make_windows" data-testid="count-example">
          <thead>
            <tr>
              <th scope="col">N</th>
              <th scope="col">T</th>
              <th scope="col">k</th>
              <th scope="col">Formula</th>
              <th scope="col">make_windows</th>
            </tr>
          </thead>
          <tbody>
            {cases.map(({ id, inputs, outputs }) => {
              const { ids, seq_len, stride } = inputs as {
                ids: number[];
                seq_len: number;
                stride: number;
              };
              return (
                <tr key={id} data-case={id}>
                  <td>{ids.length}</td>
                  <td>{seq_len}</td>
                  <td>{stride}</td>
                  <td data-testid="count-formula">{windowCount(ids.length, seq_len, stride)}</td>
                  <td data-testid="count-real">{outputs.count as number}</td>
                </tr>
              );
            })}
          </tbody>
        </DataTable>
      </VizFigure>
    </div>
  );
}

export function BatchExample({ concept }: { concept: string }) {
  const { inputs, outputs } = getGoldenCase("data", "iterate_batches.shuffled");
  const { n, seq_len, batch_size, seed } = inputs as {
    n: number;
    seq_len: number;
    batch_size: number;
    seed: number;
  };
  const epochs = outputs.epoch_starts as number[][][];
  const windows = windowCount(n, seq_len);
  return (
    <div id={exampleAnchor(concept)}>
      <VizFigure
        title="Two epochs of iterate_batches"
        kind="illustrative"
        caption={`Hand-picked sizes: ${windows} windows, batch_size ${batch_size}, generator seed ${seed}. The order is the stored result of running the real function, because it comes from pandas and NumPy. Each number is the start of one window. This is not the order of a real training run, which also depends on every earlier use of the generator.`}
      >
        <p data-testid="batch-sizes">
          Batch sizes: <code>{batchSizes(windows, batch_size).join(", ")}</code>
        </p>
        <DataTable label="Windows in each batch" data-testid="batch-example">
          <thead>
            <tr>
              <th scope="col">Epoch</th>
              {epochs[0]!.map((_, b) => (
                <th scope="col" key={b}>
                  Batch {b + 1}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {epochs.map((batches, e) => (
              <tr key={e}>
                <th scope="row">{e + 1}</th>
                {batches.map((batch, b) => (
                  <td key={b}>
                    <code>{batch.join(", ")}</code>
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </DataTable>
      </VizFigure>
    </div>
  );
}
