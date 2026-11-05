import { CODE_REFS } from "@/data/code-refs";
import { formatLineRange, requireTarget, targetLines } from "@/lib/code-targets";
import { CodeBlock } from "./CodeBlock";

function resolveRef(id: string) {
  const target = CODE_REFS[id];
  if (!target) throw new Error(`Unknown code reference: ${id}`);
  return requireTarget(target);
}

// The real lines of one step. Text and line numbers come from data/generated/code-index.json.
export function CodeStep({ id }: { id: string }) {
  const resolved = resolveRef(id);
  return (
    <CodeBlock
      source={targetLines(resolved)}
      filePath={resolved.symbol.file}
      functionName={resolved.symbol.name}
      startLine={resolved.startLine}
      permalink={resolved.permalink}
    />
  );
}

// An inline link to the real lines.
export function CodeLine({ id }: { id: string }) {
  const { symbol, startLine, endLine, permalink } = resolveRef(id);
  return (
    <a className="code-link" href={permalink}>
      {symbol.file}:{formatLineRange(startLine, endLine)}
    </a>
  );
}
