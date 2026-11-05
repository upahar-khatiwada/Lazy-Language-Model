import { highlightPython } from "@/lib/highlight";
import { CodeFrame, type CodeFrameProps } from "./CodeFrame";

export async function CodeBlock({ source, ...props }: Omit<CodeFrameProps, "lines">) {
  const lines = await highlightPython(source);
  return <CodeFrame lines={lines} source={source} {...props} />;
}
