import { exampleAnchor } from "@/lib/concepts";
import { getGoldenCase } from "@/lib/golden";
import { loadConfig } from "@/lib/repo";
import { BpeStepper } from "./BpeStepper";

// The stepper's texts are the inputs of golden cases, so its merges have Python results to compare.
export function BpeExample({ concept }: { concept: string }) {
  const texts = ["train_bpe_on_text.stepper", "train_bpe_on_text.test_file"].map(
    (id) => getGoldenCase("bpe", id).inputs.text as string,
  );
  const numMerges = loadConfig().constants.find(({ name }) => name === "num_merges")?.value;
  if (typeof numMerges !== "number") throw new Error("config.json has no numeric num_merges");
  return (
    <div id={exampleAnchor(concept)}>
      <BpeStepper texts={texts} numMerges={numMerges} />
    </div>
  );
}
