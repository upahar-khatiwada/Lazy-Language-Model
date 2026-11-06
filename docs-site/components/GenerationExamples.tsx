import { exampleAnchor } from "@/lib/concepts";
import { getGoldenCase } from "@/lib/golden";
import {
  applyTemperature,
  applyTopK,
  applyTopP,
  greedyIndex,
  sampleNextDistribution,
  topKCutoff,
  type SampleSettings,
} from "@/lib/math/sampling";
import type { Matrix, Vector } from "@/lib/math/tensor";
import { configValueText } from "@/lib/model-glance";
import { loadConfig, loadRepoMeta } from "@/lib/repo";
import { DataTable } from "./DataTable";
import { GenerationStepper, type StepperInputs, type StepperMode } from "./GenerationStepper";
import examples from "./Examples.module.css";
import { ProbabilityLab, type LabPreset } from "./ProbabilityLab";
import { showNumber } from "./ResultMatrix";
import { VizFigure } from "./VizFigure";

// Inputs are the inputs of golden cases. Results come from lib/math/sampling.ts, which
// tests/math.golden.test.ts compares with the real generate.sample_next at 1e-9.

type SampleInputs = { logits: Vector; temperature: number; top_k: number; top_p: number };

const sampleInputs = (id: string) => getGoldenCase("sampling", id).inputs as SampleInputs;

const settingsOf = ({ temperature, top_k, top_p }: SampleInputs): SampleSettings => ({
  temperature,
  topK: top_k,
  topP: top_p,
});

function Numbers({ values, testId }: { values: Vector; testId?: string }) {
  return (
    <code data-testid={testId} data-exact={JSON.stringify(values)}>
      [{values.map(showNumber).join(", ")}]
    </code>
  );
}

// Greedy -------------------------------------------------------------------------------------------

export function greedyRows() {
  return ["greedy.zero", "greedy.below_guard", "greedy.tie_goes_to_first"].map((id) => {
    const inputs = sampleInputs(id);
    return { id, ...inputs, result: greedyIndex(inputs.logits) };
  });
}

export function GreedyExample() {
  const atGuard = sampleInputs("greedy.at_guard_is_not_greedy");
  const result = sampleNextDistribution(atGuard.logits, settingsOf(atGuard));
  if (result.kind !== "distribution") throw new Error("temperature 1e-6 must not be greedy");
  return (
    <VizFigure
      title="The guard, on hand-picked logits"
      kind="illustrative"
      caption="Hand-picked logits, not model output."
    >
      <DataTable label="Greedy cases" data-testid="greedy-table">
        <thead>
          <tr>
            <th scope="col">logits</th>
            <th scope="col">temperature</th>
            <th scope="col">Returned token</th>
          </tr>
        </thead>
        <tbody>
          {greedyRows().map((row) => (
            <tr key={row.id}>
              <td>
                <Numbers values={row.logits} />
              </td>
              <td>
                <code>{row.temperature}</code>
              </td>
              <td>
                <code data-testid={`greedy-${row.id}`}>{row.result}</code>
              </td>
            </tr>
          ))}
        </tbody>
      </DataTable>
      <p>
        At exactly <code>{atGuard.temperature}</code> the test <code>temperature {"<"} 1e-6</code>{" "}
        is false, so the code computes a distribution:{" "}
        <Numbers values={result.p} testId="greedy-at" />. It is close to greedy, because dividing by
        so small a number leaves one exponential far above the rest, but it still draws from the
        generator.
      </p>
    </VizFigure>
  );
}

// Temperature --------------------------------------------------------------------------------------

export function temperatureRows() {
  return ["temperature.half", "temperature.one", "temperature.two"].map((id) => {
    const { logits, temperature } = sampleInputs(id);
    return { id, logits, temperature, p: applyTemperature(logits, temperature) };
  });
}

export function TemperatureExample() {
  const rows = temperatureRows();
  return (
    <VizFigure
      title="One set of logits at three temperatures"
      kind="illustrative"
      caption={
        <>
          Hand-picked logits <Numbers values={rows[0]!.logits} />, not model output.
        </>
      }
    >
      <DataTable label="Probabilities by temperature" data-testid="temperature-table">
        <thead>
          <tr>
            <th scope="col">temperature</th>
            <th scope="col">logits / temperature</th>
            <th scope="col">p</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.id}>
              <td>
                <code>{row.temperature}</code>
              </td>
              <td>
                <Numbers values={row.logits.map((value) => value / row.temperature)} />
              </td>
              <td>
                <Numbers values={row.p} testId={`temperature-${row.id}`} />
              </td>
            </tr>
          ))}
        </tbody>
      </DataTable>
    </VizFigure>
  );
}

// Top-k --------------------------------------------------------------------------------------------

export function topKRows(id: string) {
  const inputs = sampleInputs(id);
  const p = applyTemperature(inputs.logits, inputs.temperature);
  const cutoff = topKCutoff(p, inputs.top_k);
  return { k: inputs.top_k, p, cutoff, after: applyTopK(p, inputs.top_k) };
}

function TopKCase({ id, title }: { id: string; title: string }) {
  const { k, p, cutoff, after } = topKRows(id);
  const sorted = [...p].sort((a, b) => b - a);
  return (
    <DataTable label={title} data-testid={`top-k-${id}`}>
      <thead>
        <tr>
          <th scope="col">Quantity</th>
          <th scope="col">Value</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>
            <code>p</code>
          </td>
          <td>
            <Numbers values={p} />
          </td>
        </tr>
        <tr>
          <td>
            <code>p</code>, sorted largest first
          </td>
          <td>
            <Numbers values={sorted} />
          </td>
        </tr>
        <tr>
          <td>
            <code>cutoff</code>, for <code>top_k = {k}</code>
          </td>
          <td>
            <code>{showNumber(cutoff)}</code>
          </td>
        </tr>
        <tr>
          <td>Tokens kept</td>
          <td>
            <code>{after.filter((value) => value > 0).length}</code>
          </td>
        </tr>
        <tr>
          <td>
            <code>p</code> after renormalising
          </td>
          <td>
            <Numbers values={after} testId={`top-k-after-${id}`} />
          </td>
        </tr>
      </tbody>
    </DataTable>
  );
}

export function TopKExample() {
  return (
    <VizFigure
      title="Top-k without and with a tie"
      kind="illustrative"
      caption="Hand-picked logits at temperature 1, not model output."
    >
      <div className={examples.stack}>
        <TopKCase id="top_k.two" title="No tie at the cut-off" />
        <TopKCase id="top_k.ties_are_all_kept" title="Three tokens tied at the cut-off" />
      </div>
    </VizFigure>
  );
}

// Top-p --------------------------------------------------------------------------------------------

export function topPRows(id: string) {
  const inputs = sampleInputs(id);
  const p = applyTemperature(inputs.logits, inputs.temperature);
  const result = applyTopP(p, inputs.top_p);
  return {
    topP: inputs.top_p,
    keepCount: result.keepCount,
    rows: result.order.map((token, position) => ({
      token,
      position,
      probability: p[token]!,
      cum: result.cum[position]!,
      kept: position < result.keepCount,
      final: result.p[token]!,
    })),
  };
}

function TopPCase({ id, title }: { id: string; title: string }) {
  const { rows, topP } = topPRows(id);
  return (
    <DataTable label={`${title}, top_p = ${topP}`} data-testid={`top-p-${id}`}>
      <thead>
        <tr>
          <th scope="col">Sorted position</th>
          <th scope="col">Token</th>
          <th scope="col">p</th>
          <th scope="col">cum</th>
          <th scope="col">Kept</th>
          <th scope="col">After renormalising</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((row) => (
          <tr key={row.token} className={row.kept ? examples.chosen : undefined}>
            <td>
              <code>{row.position}</code>
            </td>
            <td>
              <code>{row.token}</code>
            </td>
            <td>
              <code>{showNumber(row.probability)}</code>
            </td>
            <td>
              <code>{showNumber(row.cum)}</code>
            </td>
            <td>{row.kept ? "Yes" : "No"}</td>
            <td>
              <code>{showNumber(row.final)}</code>
            </td>
          </tr>
        ))}
      </tbody>
    </DataTable>
  );
}

export function TopPExample() {
  return (
    <VizFigure
      title="Top-p: the token that crosses is kept"
      kind="illustrative"
      caption="Hand-picked logits at temperature 1, not model output."
    >
      <div className={examples.stack}>
        <TopPCase id="top_p.keeps_crossing_token" title="The sum passes top_p" />
        <TopPCase id="top_p.exact_boundary" title="The sum lands exactly on top_p" />
      </div>
    </VizFigure>
  );
}

// Random choice ------------------------------------------------------------------------------------

export function ChoiceExample() {
  const { inputs, outputs } = getGoldenCase("sampling", "choice.repeated_draws") as unknown as {
    inputs: SampleInputs & { seed: number };
    outputs: { first_draws: number[]; counts: number[]; total: number };
  };
  const result = sampleNextDistribution(inputs.logits, settingsOf(inputs));
  if (result.kind !== "distribution") throw new Error("expected a distribution");
  return (
    <VizFigure
      title={`${outputs.total} draws from one distribution`}
      kind="illustrative"
      caption={
        <>
          Hand-picked logits, not model output. The settings are those in <code>config.py</code>.
          The draws were recorded by calling the real <code>sample_next</code> with{" "}
          <code>np.random.default_rng({inputs.seed})</code>, because that generator is not
          reproduced in this site&apos;s engine. <code>p</code> comes from the engine.
        </>
      }
    >
      <p>
        First draws: <code data-testid="choice-first">[{outputs.first_draws.join(", ")}]</code>
      </p>
      <DataTable label="Draws by token" data-testid="choice-table">
        <thead>
          <tr>
            <th scope="col">Token</th>
            <th scope="col">p</th>
            <th scope="col">Times drawn</th>
          </tr>
        </thead>
        <tbody>
          {result.p.map((value, token) => (
            <tr key={token}>
              <td>
                <code>{token}</code>
              </td>
              <td>
                <code>{showNumber(value)}</code>
              </td>
              <td>
                <code data-testid={`choice-count-${token}`}>{outputs.counts[token]}</code>
              </td>
            </tr>
          ))}
        </tbody>
      </DataTable>
    </VizFigure>
  );
}

// Probability lab ----------------------------------------------------------------------------------

export const LAB_PRESETS = [
  { id: "lab.config_defaults", label: "config.py settings" },
  { id: "lab.greedy", label: "Greedy" },
  { id: "lab.no_truncation", label: "No truncation" },
  { id: "lab.top_k", label: "Top-k" },
  { id: "lab.top_k_tie", label: "Top-k with a tie" },
  { id: "lab.top_p_tie", label: "Top-p splits a tie" },
  { id: "lab.top_k_then_top_p", label: "Top-k then top-p" },
] as const;

export function labPresets(): LabPreset[] {
  return LAB_PRESETS.map(({ id, label }) => {
    const inputs = sampleInputs(id);
    return { id, label, logits: inputs.logits, settings: settingsOf(inputs) };
  });
}

export function ProbabilityLabExample() {
  return (
    <section id={exampleAnchor("sampling.lab")} aria-label="Probability lab">
      <ProbabilityLab presets={labPresets()} />
    </section>
  );
}

// Autoregressive stepper ---------------------------------------------------------------------------

type StepperCase = {
  inputs: {
    tokens: string[];
    table: Matrix;
    seed_ids: number[];
    max_new_tokens: number;
    window: number;
    temperature: number;
    top_k: number;
    top_p: number;
    seed: number;
  };
  outputs: { ids: number[]; windows: number[][]; distributions: Matrix; draws: number[] };
};

export function stepperCase(id: string) {
  return getGoldenCase("sampling", id) as unknown as StepperCase;
}

export function stepperInputs(id: string): StepperInputs {
  const { inputs, outputs } = stepperCase(id);
  return {
    tokens: inputs.tokens,
    table: inputs.table,
    seedIds: inputs.seed_ids,
    maxNewTokens: inputs.max_new_tokens,
    window: inputs.window,
    settings: settingsOf({
      logits: [],
      temperature: inputs.temperature,
      top_k: inputs.top_k,
      top_p: inputs.top_p,
    }),
    draws: outputs.draws,
  };
}

export function GenerationStepperExample() {
  const config = stepperCase("stepper.config_defaults").inputs;
  const greedy = stepperCase("stepper.greedy").inputs;
  const shared = `The logits come from the hand-picked table below, not from the model. Each row depends only on the last token, while a real model's logits depend on the whole window. The window is ${config.window} here so that it fills quickly. The code passes a larger one.`;
  const modes: StepperMode[] = [
    {
      id: "config",
      label: `temperature ${config.temperature}, top_k ${config.top_k}, top_p ${config.top_p}, seed ${config.seed}`,
      note: `${shared} The settings and seed are those in config.py. The draws were recorded by running the real generate.generate with forward replaced by the table.`,
      inputs: stepperInputs("stepper.config_defaults"),
    },
    {
      id: "greedy",
      label: `temperature ${greedy.temperature}, greedy`,
      note: `${shared} At this temperature every pick is the largest logit, so no draw is made.`,
      inputs: stepperInputs("stepper.greedy"),
    },
  ];
  return <GenerationStepper modes={modes} />;
}

// Defaults -----------------------------------------------------------------------------------------

function constantOf(name: string) {
  const constant = loadConfig().constants.find((candidate) => candidate.name === name);
  if (!constant) throw new Error(`config.json has no constant named ${name}`);
  return constant;
}

// Whether each branch runs follows from the value and from the conditions in sample_next.
export function samplingDefaults() {
  const value = (name: string) => constantOf(name).value as number;
  const greedy = value("temperature") < 1e-6;
  // len(p) is V, which depends on the corpus. 0 < top_k fails first at any V when top_k is 0.
  const topK = value("top_k") > 0;
  const topP = value("top_p") < 1;
  return [
    {
      name: "temperature",
      state: greedy ? "Greedy" : "On",
      effect: greedy
        ? "The guard returns the arg-max token."
        : "The guard does not fire, and the logits are divided by this value.",
    },
    {
      name: "top_k",
      state: topK ? "On" : "Off",
      effect: topK
        ? "Top-k runs if this is below V."
        : "0 < top_k is false, so the top-k branch is skipped.",
    },
    {
      name: "top_p",
      state: topP ? "On" : "Off",
      effect: topP
        ? "top_p < 1.0 is true, so the top-p branch runs."
        : "The top-p branch is skipped.",
    },
    {
      name: "max_new_tokens",
      state: "talk.py only",
      effect: "train.py passes its own literal instead.",
    },
    {
      name: "seq_len",
      state: "On",
      effect: "Both callers pass it as window. It is also the training window length.",
    },
    {
      name: "seed",
      state: "On",
      effect: "Both callers create their generator from it.",
    },
  ].map((row) => {
    const constant = constantOf(row.name);
    return { ...row, text: configValueText(constant), line: constant.line };
  });
}

export function SamplingDefaults() {
  const repo = loadRepoMeta();
  return (
    <DataTable label="Generation settings in config.py" data-testid="sampling-defaults">
      <thead>
        <tr>
          <th scope="col">Name</th>
          <th scope="col">Value</th>
          <th scope="col">State</th>
          <th scope="col">Effect</th>
          <th scope="col">Line</th>
        </tr>
      </thead>
      <tbody>
        {samplingDefaults().map((row) => (
          <tr key={row.name}>
            <td>
              <code>{row.name}</code>
            </td>
            <td>
              <code data-config-value={row.name}>{row.text}</code>
            </td>
            <td data-testid={`default-state-${row.name}`}>{row.state}</td>
            <td>{row.effect}</td>
            <td>
              <a
                className="code-link"
                href={`${repo.githubBaseUrl}/blob/${repo.sha}/config.py#L${row.line}`}
              >
                config.py:{row.line}
              </a>
            </td>
          </tr>
        ))}
      </tbody>
    </DataTable>
  );
}
