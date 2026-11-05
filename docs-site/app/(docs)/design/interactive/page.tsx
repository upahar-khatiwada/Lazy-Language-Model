import type { Metadata } from "next";
import { DocHeader, DocMain } from "@/components/DocMain";
import { OnThisPage } from "@/components/OnThisPage";
import { VizSpecimens, type SpecimenInputs } from "@/components/VizSpecimens";
import { getGoldenCase } from "@/lib/golden";
import type { AttentionParams } from "@/lib/math/attention";
import type { Matrix, Vector } from "@/lib/math/tensor";
import type { PageHeading } from "@/lib/remark-headings";

// Hidden, like /design/components: not in data/nav.ts and marked noindex.
export const metadata: Metadata = {
  title: "Interactive kit",
  robots: { index: false, follow: false },
};

const HEADINGS: PageHeading[] = [
  { id: "matrix-input", text: "MatrixInput and ShapeFlow", depth: 2 },
  { id: "step-player", text: "StepPlayer and Heatmap", depth: 2 },
  { id: "selected-row", text: "Selected row and BarChart", depth: 2 },
  { id: "probability-bars", text: "ProbabilityBars", depth: 2 },
  { id: "line-chart", text: "LineChart", depth: 2 },
];

// Inputs come from the golden files, values from lib/math. No number is typed here.
function specimenInputs(): SpecimenInputs {
  const { Xn, H, ...params } = getGoldenCase("attention", "attention_forward.t3_d4_h2").inputs as {
    Xn: Matrix;
    H: number;
  } & AttentionParams;
  const sampling = getGoldenCase("sampling", "sample_next.config_defaults").inputs as {
    logits: Vector;
    temperature: number;
    top_k: number;
    top_p: number;
  };
  const schedule = getGoldenCase("optim", "lr_schedule.curve").inputs as {
    steps: number[];
    warmup: number;
    total: number;
    lr_max: number;
    lr_min: number;
  };
  return {
    attention: { Xn, H, params },
    sampling: {
      logits: sampling.logits,
      temperature: sampling.temperature,
      topK: sampling.top_k,
      topP: sampling.top_p,
    },
    schedule: {
      steps: schedule.steps,
      warmup: schedule.warmup,
      total: schedule.total,
      lrMax: schedule.lr_max,
      lrMin: schedule.lr_min,
    },
  };
}

export default function InteractiveKitPage() {
  return (
    <>
      <DocMain>
        <DocHeader
          title="Interactive kit"
          description="Visual primitives driven by lib/math. Hand-picked inputs, so every chart is an Illustrative Example; none of these are learned values."
        />
        <VizSpecimens inputs={specimenInputs()} />
      </DocMain>
      <OnThisPage headings={HEADINGS} />
    </>
  );
}
