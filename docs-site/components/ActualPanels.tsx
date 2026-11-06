import { loadActual, provenanceLine, type NoActualData, type Provenance } from "@/lib/actual";
import { attentionProblems, shortHash, visibleToken } from "@/lib/actual-view";
import styles from "./AttentionLab.module.css";
import { Badge } from "./Badge";
import { DataTable } from "./DataTable";
import { ProbabilityBars } from "./ProbabilityBars";
import { Tokens } from "./Tokens";
import { VizFigure } from "./VizFigure";

const shape = (dims: number[]) => `(${dims.join(", ")})`;

// Shown in place of a panel when nothing has been exported.
export function ActualUnavailable({ none }: { none: NoActualData }) {
  return (
    <section
      className={styles.slot}
      aria-label="Actual Model Output slot"
      data-testid="actual-unavailable"
    >
      <div className={styles.slotTop}>
        <Badge kind="actual" />
        <span>No data</span>
      </div>
      <p>{none.reason} Nothing in this panel comes from a run of the model.</p>
      <p>
        To fill it, run <code>python docs-site/scripts/export_actual.py</code> with a checkpoint,
        then build the site again. The script prints how to create a checkpoint.
      </p>
    </section>
  );
}

// Every value in a file behind a panel, for a reader who wants to reproduce it.
export function ProvenanceDetails({ source }: { source: Provenance }) {
  return (
    <details data-testid="actual-provenance-details">
      <summary>Where these numbers came from</summary>
      <DataTable label="Provenance of Actual Model Output">
        <tbody>
          <tr>
            <th scope="row">Repository commit</th>
            <td>
              <code>{source.gitSha}</code>
            </td>
          </tr>
          {source.checkpointFiles.map(({ name, sha256 }) => (
            <tr key={name}>
              <th scope="row">
                <code>{name}</code> sha256
              </th>
              <td>
                <code>{sha256}</code>
              </td>
            </tr>
          ))}
          <tr>
            <th scope="row">
              Corpus <code>{source.corpus.path}</code> sha256
            </th>
            <td>
              <code>{source.corpus.sha256}</code>
            </td>
          </tr>
          <tr>
            <th scope="row">Tokenizer rebuilt from this corpus equals the checkpoint&apos;s</th>
            <td>{source.tokenizerMatchesCorpus ? "Yes" : "No"}</td>
          </tr>
          <tr>
            <th scope="row">Model shape, from the checkpoint</th>
            <td>
              <code>{JSON.stringify(source.modelConfig)}</code>
            </td>
          </tr>
          <tr>
            <th scope="row">
              <code>config.py</code>, when this was exported
            </th>
            <td>
              <code>{JSON.stringify(source.config)}</code>
            </td>
          </tr>
          <tr>
            <th scope="row">Seed</th>
            <td>
              <code>{source.seed}</code>
            </td>
          </tr>
          <tr>
            <th scope="row">Environment</th>
            <td>
              <code>
                Python {source.environment.python}, NumPy {source.environment.numpy}, pandas{" "}
                {source.environment.pandas}
              </code>
            </td>
          </tr>
        </tbody>
      </DataTable>
    </details>
  );
}

export function ActualTokenizerPanel() {
  const actual = loadActual();
  if (!actual.available) return <ActualUnavailable none={actual} />;
  const { tokenizer, meta } = actual;
  const source = meta.provenance;
  return (
    <div data-testid="actual-tokenizer">
      <VizFigure
        title="The first merges in the checkpoint"
        kind="actual"
        provenance={provenanceLine(source)}
        caption={`${tokenizer.mergeCount} merges and a vocabulary of ${tokenizer.vocabSize} tokens, read from the checkpoint's tokenizer.`}
      >
        <DataTable label="Real merges, in the order they were learned">
          <thead>
            <tr>
              <th scope="col">Merge</th>
              <th scope="col">Pair</th>
              <th scope="col">New token</th>
              <th scope="col">Frequency</th>
            </tr>
          </thead>
          <tbody>
            {tokenizer.merges.map(({ rank, pair, merged, frequency }) => (
              <tr key={rank} data-merge={rank}>
                <td>{rank}</td>
                <td>
                  <code>
                    {visibleToken(pair[0]!)} + {visibleToken(pair[1]!)}
                  </code>
                </td>
                <td>
                  <code>{visibleToken(merged)}</code>
                </td>
                <td>{frequency.toLocaleString("en-US")}</td>
              </tr>
            ))}
          </tbody>
        </DataTable>
      </VizFigure>
      <VizFigure
        title="Sentences from the corpus, encoded by that tokenizer"
        kind="actual"
        provenance={provenanceLine(source)}
      >
        <DataTable label="Real tokens and ids" data-testid="actual-sentences">
          <thead>
            <tr>
              <th scope="col">Sentence</th>
              <th scope="col">Tokens</th>
              <th scope="col">Ids</th>
            </tr>
          </thead>
          <tbody>
            {tokenizer.sentences.map(({ text, tokens, ids }) => (
              <tr key={text}>
                <td>{text}</td>
                <td>
                  <Tokens tokens={tokens} />
                </td>
                <td>
                  <code>{ids.join(" ")}</code>
                </td>
              </tr>
            ))}
          </tbody>
        </DataTable>
      </VizFigure>
      <ProvenanceDetails source={source} />
    </div>
  );
}

export function ActualEmbeddingsPanel() {
  const actual = loadActual();
  if (!actual.available) return <ActualUnavailable none={actual} />;
  const { embeddings, meta, attention } = actual;
  return (
    <div data-testid="actual-embeddings">
      <VizFigure
        title="Embedding shapes in the checkpoint"
        kind="actual"
        provenance={provenanceLine(meta.provenance)}
        caption={`The checkpoint holds ${meta.tensorCount} tensors and ${meta.paramCount.toLocaleString("en-US")} parameters. X0 is the real result of embed_forward on one window of ${attention.T} tokens.`}
      >
        <DataTable label="Real embedding shapes">
          <thead>
            <tr>
              <th scope="col">Value</th>
              <th scope="col">Shape</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>
                <code>tok_emb</code>
              </td>
              <td data-shape="tok_emb">
                <code>{shape(embeddings.tokEmb)}</code>
              </td>
            </tr>
            <tr>
              <td>
                <code>pos_emb</code>
              </td>
              <td data-shape="pos_emb">
                <code>{shape(embeddings.posEmb)}</code>
              </td>
            </tr>
            <tr>
              <td>
                <code>X0 = tok_emb[ids] + pos_emb[:T]</code>
              </td>
              <td data-shape="X0">
                <code>{shape(embeddings.X0)}</code>
              </td>
            </tr>
          </tbody>
        </DataTable>
      </VizFigure>
      <ProvenanceDetails source={meta.provenance} />
    </div>
  );
}

export function ActualGenerationPanel() {
  const actual = loadActual();
  if (!actual.available) return <ActualUnavailable none={actual} />;
  const { probabilities, meta } = actual;
  const { plain, sampling, prompt, tokens } = probabilities;
  const source = meta.provenance;
  const line = provenanceLine(source);
  const bars = (distribution: typeof plain) => ({
    labels: [
      ...distribution.top.map(({ token }) => visibleToken(token)),
      ...(distribution.otherTokens > 0 ? ["all other tokens"] : []),
    ],
    probabilities: [
      ...distribution.top.map(({ probability }) => probability),
      ...(distribution.otherTokens > 0 ? [distribution.otherTokens] : []),
    ],
  });
  const promptNote = `The prompt is hand-picked: "${prompt}", which encodes to ${tokens.length} tokens. The probabilities are what the model returned for this one prompt.`;
  return (
    <div data-testid="actual-generation">
      <ProbabilityBars
        title="Next-token probabilities, softmax of the last logits"
        kind="actual"
        provenance={line}
        caption={`${promptNote} The ${plain.top.length} most likely tokens are listed, and the other ${plain.kept - plain.top.length} tokens share one bar.`}
        {...bars(plain)}
      />
      <ProbabilityBars
        title="What sample_next draws from, at the config.py settings"
        kind="actual"
        provenance={line}
        caption={`The same prompt, with temperature ${sampling.temperature}, top_k ${sampling.topK} and top_p ${sampling.topP}, read from the real sample_next before it draws. ${sampling.kept} tokens keep a non-zero probability.`}
        {...bars(sampling)}
      />
      <ProvenanceDetails source={source} />
    </div>
  );
}

// The summary beside the lab's switch. The matrices themselves are in the lab.
export function ActualAttentionSummary() {
  const actual = loadActual();
  if (!actual.available) return null;
  const { attention, meta } = actual;
  const source = meta.provenance;
  const weights = source.checkpointFiles.find(({ name }) => name.endsWith(".npz"))!;
  return (
    <>
      <p data-testid="actual-provenance">{provenanceLine(source)}</p>
      <p>
        In the lab above, choose Actual Model Output to see the weights <code>A</code> for every
        layer and head. They come from running <code>forward</code> on one window of {attention.T}{" "}
        real tokens: <code>{visibleToken(attention.window.text)}</code>
      </p>
      <p data-testid="actual-checks">
        Checks on all {attention.L * attention.H} matrices, each {attention.T} by {attention.T}:{" "}
        {attentionProblems(attention).length === 0
          ? `the shapes are right, every row sums to 1 (largest gap ${attention.largestRowSumGap.toExponential(1)}), and every masked cell is exactly 0.`
          : "they failed."}{" "}
        Checkpoint weights file: <code>{shortHash(weights.sha256)}</code>.
      </p>
      <ProvenanceDetails source={source} />
    </>
  );
}
