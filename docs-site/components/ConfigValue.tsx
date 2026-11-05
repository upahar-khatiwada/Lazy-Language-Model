import { configValueText } from "@/lib/model-glance";
import { loadConfig } from "@/lib/repo";

// A value from config.py, for use in page text. Pages never type these numbers.
export function ConfigValue({ name }: { name: string }) {
  const constant = loadConfig().constants.find((candidate) => candidate.name === name);
  if (!constant) throw new Error(`config.json has no constant named ${name}`);
  return <code data-config-value={name}>{configValueText(constant)}</code>;
}
