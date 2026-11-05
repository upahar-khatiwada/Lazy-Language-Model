import { createElement, type ReactElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, test } from "vitest";
import { BarChart } from "@/components/BarChart";
import { Heatmap } from "@/components/Heatmap";
import { LineChart } from "@/components/LineChart";
import { ProbabilityBars } from "@/components/ProbabilityBars";
import { ShapeFlow } from "@/components/ShapeFlow";
import { formatShape as dimensionsFormatShape } from "@/lib/dimensions";
import { linearPercent, sequentialStep, shortNumber } from "@/lib/viz/scale";
import { formatShape, shapeOf } from "@/lib/viz/shape";

const render = (element: ReactElement) => renderToStaticMarkup(element);

const KIND = "illustrative" as const;

const heatmap = {
  title: "H",
  kind: KIND,
  rowLabels: ["a", "b"],
  colLabels: ["x", "y"],
  xLabel: "col",
  yLabel: "row",
  values: [
    [1, -Infinity],
    [0, 2],
  ],
};

describe("scales", () => {
  test("linearPercent is linear, and a flat domain does not give NaN", () => {
    expect(linearPercent([0, 10], 2.5)).toBe(25);
    expect(linearPercent([-1, 1], 0)).toBe(50);
    expect(linearPercent([3, 3], 3)).toBe(0);
  });

  test("sequentialStep covers all steps and stays in range", () => {
    expect(sequentialStep(0, 1, 0)).toBe(0);
    expect(sequentialStep(0, 1, 1)).toBe(5);
    expect(sequentialStep(0, 1, 0.5)).toBe(3);
    expect(sequentialStep(2, 2, 2)).toBe(0);
  });

  test("shortNumber", () => {
    expect(shortNumber(0.001)).toBe("0.001");
    expect(shortNumber(12.34567)).toBe("12.35");
  });

  test("shapes print as Python tuples", () => {
    expect(shapeOf([[[1], [2]]])).toEqual([1, 2, 1]);
    expect(formatShape([3])).toBe("(3,)");
    expect(formatShape([2, 3])).toBe("(2, 3)");
    expect(dimensionsFormatShape).toBe(formatShape);
  });
});

describe("a chart cannot go without its label", () => {
  test("a missing kind is a type error and a render error", () => {
    // @ts-expect-error `kind` is required. `bun run typecheck` fails if this ever compiles.
    const element = createElement(BarChart, { title: "B", labels: ["a"], values: [1] });
    expect(() => render(element)).toThrow('needs kind="illustrative" or kind="actual"');
  });

  test("Actual Model Output needs a provenance", () => {
    const element = createElement(Heatmap, { ...heatmap, kind: "actual" });
    expect(() => render(element)).toThrow("needs a provenance");
  });

  test.each([
    ["Heatmap", createElement(Heatmap, heatmap)],
    ["BarChart", createElement(BarChart, { title: "B", kind: KIND, labels: ["a"], values: [1] })],
    [
      "ProbabilityBars",
      createElement(ProbabilityBars, {
        title: "P",
        kind: KIND,
        labels: ["a"],
        probabilities: [1],
      }),
    ],
    [
      "LineChart",
      createElement(LineChart, {
        title: "L",
        kind: KIND,
        points: [{ x: 1, y: 1 }],
        xLabel: "x",
        yLabel: "y",
      }),
    ],
    [
      "ShapeFlow",
      createElement(ShapeFlow, {
        title: "S",
        kind: KIND,
        stages: [{ name: "X", shape: [1, 2] }],
      }),
    ],
  ])("%s shows the badge", (_name, element) => {
    expect(render(element)).toContain("Illustrative Example");
  });
});

describe("Heatmap", () => {
  test("masked cells are marked in words, not colour alone", () => {
    const html = render(createElement(Heatmap, heatmap));
    expect(html).toContain(">-inf<");
    expect(html).toContain("-inf (masked)");
    expect(html).toContain('data-masked="true"');
  });

  test("the scale ignores masked cells, and the legend shows real values", () => {
    const html = render(createElement(Heatmap, heatmap));
    expect(html).toContain("Low 0.00");
    expect(html).toContain("High 2.00");
    expect(html).toContain("fill:var(--seq-5)");
  });

  test("the selected row is drawn only when asked for", () => {
    expect(render(createElement(Heatmap, heatmap))).not.toContain("heatmap-selected-row");
    expect(render(createElement(Heatmap, { ...heatmap, selectedRow: 1 }))).toContain(
      "heatmap-selected-row",
    );
  });
});

describe("BarChart", () => {
  test("every bar starts from the zero line, also for negatives", () => {
    const html = render(
      createElement(BarChart, { title: "B", kind: KIND, labels: ["a", "b"], values: [-1, 3] }),
    );
    expect(html).toContain('x1="25%"');
    expect(html).toContain('x="0%"');
    expect(html).toContain('x="25%"');
    expect(html).toContain('width="75%"');
  });

  test("ProbabilityBars keeps the 0 to 1 axis and says removed in words", () => {
    const html = render(
      createElement(ProbabilityBars, {
        title: "P",
        kind: KIND,
        labels: ["a", "b"],
        probabilities: [0.5, 0],
        showRemoved: true,
      }),
    );
    expect(html).toContain('width="50%"');
    expect(html).toContain("0.0000 removed");
  });
});

describe("LineChart", () => {
  const points = [
    { x: 0, y: 2 },
    { x: 5, y: 7 },
    { x: 10, y: 4 },
  ];
  const chart = (labelled: number[]) =>
    render(
      createElement(LineChart, {
        title: "L",
        kind: KIND,
        points,
        xLabel: "x",
        yLabel: "y",
        labelled,
        yDomain: [0, 8],
      }),
    );

  test("the line and its markers use the same scale", () => {
    const html = chart([1]);
    expect(html).toContain("0,75 50,12.5 100,50");
    expect(html).toContain('cx="50%"');
    expect(html).toContain('cy="12.5%"');
  });

  test("only the axis ends and the requested points are printed on the plot", () => {
    const plot = chart([1]).split("<details")[0]!;
    const printed = [...plot.matchAll(/<(?:text|span)[^>]*>([^<]*\d[^<]*)<\/(?:text|span)>/g)].map(
      (match) => match[1],
    );
    expect(printed.sort()).toEqual(["0", "0", "10", "5: 7", "8"].sort());
  });

  test("an unknown labelled point is an error, not a silent gap", () => {
    expect(() => chart([9])).toThrow("has no point 9");
  });
});
