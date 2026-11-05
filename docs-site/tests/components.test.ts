import { createElement, type ReactElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { afterEach, describe, expect, test, vi } from "vitest";
import { MathBlock } from "@/components/MathBlock";
import { NumericExample } from "@/components/NumericExample";
import { TensorShape } from "@/components/TensorShape";
import { mdxComponents } from "@/components/mdx";
import * as concepts from "@/lib/concepts";

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
