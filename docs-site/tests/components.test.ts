import { createElement, type ReactElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { afterEach, describe, expect, test, vi } from "vitest";
import { CodeBlock } from "@/components/CodeBlock";
import { CodeReference } from "@/components/CodeReference";
import { MathBlock } from "@/components/MathBlock";
import { NumericExample } from "@/components/NumericExample";
import { TensorShape } from "@/components/TensorShape";
import { mdxComponents } from "@/components/mdx";
import { getSymbol } from "@/lib/code-targets";
import * as concepts from "@/lib/concepts";
import { highlightedLines } from "@/lib/links";
import { loadRepoMeta } from "@/lib/repo";

const render = (element: ReactElement) => renderToStaticMarkup(element);

afterEach(() => {
  vi.restoreAllMocks();
});

describe("NumericExample", () => {
  const CONCEPT = "ops.softmax_forward";

  test("is a type error without a kind", () => {
    // @ts-expect-error `kind` is required. `bun run typecheck` fails if this ever compiles.
    const element = createElement(NumericExample, { concept: CONCEPT });
    expect(() => render(element)).toThrow('needs kind="illustrative" or kind="actual"');
  });

  test("rejects a kind that is not one of the two labels", () => {
    // @ts-expect-error Only the two labels are allowed.
    const element = createElement(NumericExample, { concept: CONCEPT, kind: "estimated" });
    expect(() => render(element)).toThrow('needs kind="illustrative" or kind="actual"');
  });

  test("cannot claim Actual Model Output for illustrative data", () => {
    const element = createElement(NumericExample, { concept: CONCEPT, kind: "actual" });
    expect(() => render(element)).toThrow("the data says illustrative");
  });

  test("renders the Illustrative Example badge and computed values", () => {
    const html = render(createElement(NumericExample, { concept: CONCEPT, kind: "illustrative" }));
    expect(html).toContain("Illustrative Example");
    expect(html).not.toContain("Actual Model Output");
    expect(html).toContain("[0.6652, 0.2447, 0.0900]");
    expect(html).toContain("[1.0000, 0.3679, 0.1353]");
  });

  test("renders the Actual Model Output badge and its provenance for actual data", () => {
    const real = concepts.getConcept(CONCEPT);
    const provenance = "Test fixture only, not a real run.";
    vi.spyOn(concepts, "getConcept").mockReturnValue({
      ...real,
      example: { ...real.example!, kind: "actual", provenance },
    });
    const html = render(createElement(NumericExample, { concept: CONCEPT, kind: "actual" }));
    expect(html).toContain("Actual Model Output");
    expect(html).not.toContain("Illustrative Example");
    expect(html).toContain(provenance);
  });
});

describe("MathBlock", () => {
  test("renders by id, with an anchor and no JavaScript KaTeX", () => {
    const html = render(createElement(MathBlock, { id: "softmax.shift" }));
    expect(html).toContain('id="eq-softmax-shift"');
    expect(html).toContain('href="#eq-softmax-shift"');
    expect(html).toContain("katex-display");
    expect(html).toContain("Copy LaTeX");
  });

  test("fails on an unknown id", () => {
    expect(() => render(createElement(MathBlock, { id: "nope" }))).toThrow("Unknown equation");
  });

  test("every equation on the site has MathML, a label and no KaTeX error node", () => {
    const equations = concepts.loadConcepts().flatMap((concept) => concept.equations);
    expect(equations.length).toBeGreaterThan(100);
    for (const equation of equations) {
      const html = render(createElement(MathBlock, { id: equation.id }));
      expect(html, equation.id).toContain("<math");
      expect(html, equation.id).toMatch(/role="group" aria-label="[^"]+"/);
      expect(html, equation.id).not.toContain("katex-error");
    }
  });
});

describe("CodeReference", () => {
  const SYMBOL = "ops.softmax_forward";

  // CodeBlock is async, which renderToStaticMarkup cannot wait for, so it is called here directly.
  const renderReference = async (equation?: string) => {
    const { props } = CodeReference({ symbol: SYMBOL, equation }) as ReactElement<
      Parameters<typeof CodeBlock>[0]
    >;
    return render(await CodeBlock(props));
  };
  const numbers = (html: string) =>
    [...html.matchAll(/data-line="(\d+)"/g)].map((match) => Number(match[1]));

  test("shows the real source with its real line numbers and a pinned permalink", async () => {
    const found = getSymbol(SYMBOL);
    const html = await renderReference();
    const count = found.source.split("\n").length;
    expect(numbers(html)).toEqual(Array.from({ length: count }, (_, i) => found.startLine + i));
    expect(html).toContain(`href="${found.permalink}"`);
    expect(found.permalink).toContain(`/blob/${loadRepoMeta().sha}/${found.file}#L`);
    expect(html).toContain(`data-code-file="${found.file}"`);
    expect(html).toContain("softmax_forward");
    // The copy button copies the source, and the scrolling block can take keyboard focus.
    expect(html).toContain('tabindex="0"');
    expect(html).toContain(`aria-label="Source of ${found.file}"`);
  });

  test("highlights exactly the lines of the link records", async () => {
    const html = await renderReference();
    const lines = [...html.matchAll(/data-line="(\d+)" data-highlighted="true"/g)].map((m) =>
      Number(m[1]),
    );
    expect(lines).toEqual([...highlightedLines(SYMBOL)].sort((a, b) => a - b));
    expect(lines.length).toBeGreaterThan(0);
  });

  test("with an equation, only that equation's lines", async () => {
    const html = await renderReference("softmax.normalize");
    const lines = [...html.matchAll(/data-line="(\d+)" data-highlighted="true"/g)].map((m) =>
      Number(m[1]),
    );
    expect(lines).toEqual([...highlightedLines(SYMBOL, "softmax.normalize")]);
    expect(lines.length).toBe(1);
  });

  test("fails on a symbol that does not exist", () => {
    expect(() => CodeReference({ symbol: "ops.nope" })).toThrow();
  });
});

test("TensorShape shows config values and keeps V symbolic", () => {
  const attention = render(
    createElement(TensorShape, { concept: "ops.softmax_forward", shape: "attention" }),
  );
  expect(attention).toContain("(H, T, T)");
  expect(attention).toContain("(2, 16, 16)");
  expect(attention).toContain("H = num_heads = 2.");

  const sampling = render(
    createElement(TensorShape, { concept: "ops.softmax_forward", shape: "sampling" }),
  );
  expect(sampling).toContain("(V,)");
  expect(sampling).toContain("depends on the corpus");
});

test("MDX gets no way to pass raw LaTeX or source text", () => {
  expect(Object.keys(mdxComponents)).not.toContain("TexDisplay");
  expect(Object.keys(mdxComponents)).not.toContain("CodeBlock");
});
