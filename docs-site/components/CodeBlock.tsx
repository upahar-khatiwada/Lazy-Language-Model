import { highlightPython } from "@/lib/highlight";
import { lineEquations } from "@/lib/links";
import { CodeFrame, type CodeFrameProps } from "./CodeFrame";

export async function CodeBlock({ source, ...props }: Omit<CodeFrameProps, "lines">) {
  const lines = await highlightPython(source);
  return (
    <CodeFrame
      lines={lines}
      source={source}
      lineEquations={lineEquations(props.filePath, props.startLine, lines.length)}
      {...props}
    />
  );
}
