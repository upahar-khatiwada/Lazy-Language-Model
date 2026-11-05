import { getSymbol } from "@/lib/code-targets";
import { highlightedLines } from "@/lib/links";
import { CodeBlock } from "./CodeBlock";

// Shows a repository symbol. Source, line numbers and permalink come from code-index.json, and
// highlighted lines from links. With `equation`, only that equation's lines are highlighted.
export function CodeReference({ symbol, equation }: { symbol: string; equation?: string }) {
  const found = getSymbol(symbol);
  return (
    <CodeBlock
      source={found.source}
      filePath={found.file}
      functionName={found.name}
      startLine={found.startLine}
      permalink={found.permalink}
      highlightLines={highlightedLines(symbol, equation)}
    />
  );
}
