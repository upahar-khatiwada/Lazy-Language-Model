import { createCssVariablesTheme, createHighlighter, type Highlighter } from "shiki";

// Token colours stay CSS variables, so one highlight pass works in both themes.
// components/CodeBlock.module.css maps the --shiki-* variables to the --syntax-* design tokens.
const theme = createCssVariablesTheme({ name: "docs-tokens", variablePrefix: "--shiki-" });

let highlighter: Promise<Highlighter> | undefined;

export type CodeToken = { content: string; color?: string };

export async function highlightPython(source: string): Promise<CodeToken[][]> {
  highlighter ??= createHighlighter({ themes: [theme], langs: ["python"] });
  const { tokens } = (await highlighter).codeToTokens(source.replace(/\n+$/, ""), {
    lang: "python",
    theme: "docs-tokens",
  });
  return tokens.map((line) =>
    line.map((token) => ({ content: token.content, color: token.color })),
  );
}
