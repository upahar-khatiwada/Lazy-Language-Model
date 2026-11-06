import { createElement, type ReactElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, test } from "vitest";
import { CodeFrame } from "@/components/CodeFrame";
import { ImplementedIn } from "@/components/ImplementedIn";
import { MathBlock } from "@/components/MathBlock";
import { Trace } from "@/components/Trace";
import { requireTarget } from "@/lib/code-targets";
import { equationSlug, getConcept, loadConcepts } from "@/lib/concepts";
import {
  allEquationSlugs,
  equationDetail,
  equationHome,
  equationLocations,
  inlineEquivalentNote,
  verifyEquationReach,
} from "@/lib/equation-links";
import { equationsByLine, lineEquations, linesCoveredByLink, loadLinks } from "@/lib/links";
import {
  activeEquation,
  announcement,
  detailUrl,
  lineCountText,
  scrollTargetIndex,
  splitEquations,
  withEquation,
} from "@/lib/math-code";
import { readPageSources } from "@/lib/reference";
import { loadCodeIndex } from "@/lib/repo";

const render = (element: ReactElement) => renderToStaticMarkup(element);

describe("every link record, in both directions", () => {
  const links = loadLinks();

  test.each(links.map((link, index) => [index, link.equationId, link] as const))(
    "#%i %s",
    (_index, equationId, link) => {
      const { startLine, endLine, symbol } = requireTarget(link);
      const slug = equationSlug(equationId);

      // From the equation to the code: the record is listed, opening the exact line with the
      // highlight on, and its concept shows it.
      const location = equationLocations(equationId).find(
        (candidate) =>
          candidate.symbol === link.symbol &&
          candidate.startLine === startLine &&
          candidate.role === link.role,
      );
      expect(location).toBeDefined();
      expect(location!.explorerHref).toContain(`?eq=${slug}#L${startLine}`);

      // From the code to the equation: each covered line names it, and it leads somewhere.
      for (let line = startLine; line <= endLine; line++) {
        expect(equationsByLine(symbol.file).get(line)).toContain(slug);
      }
      expect(equationHome(equationId)).not.toBeNull();
    },
  );

  test("the real data passes the check that verify:links runs", () => {
    expect(
      verifyEquationReach({
        concepts: loadConcepts(),
        links,
        pages: readPageSources(),
        files: loadCodeIndex().files.map((file) => file.path),
      }),
    ).toEqual([]);
  });

  const reachInput = () => ({
    concepts: loadConcepts(),
    links,
    pages: readPageSources(),
    files: loadCodeIndex().files.map((file) => file.path),
  });

  test("a concept no page shows is reported", () => {
    const input = reachInput();
    const pages = input.pages.map((page) => ({
      ...page,
      body: page.body.replaceAll('<Trace concept="attention.scale"', "<Other"),
    }));
    expect(verifyEquationReach({ ...input, pages }).join("\n")).toContain(
      "concept attention.scale: no page shows its Implemented in panel",
    );
  });

  test("a file outside the Code Explorer is reported", () => {
    const input = reachInput();
    const files = input.files.filter((file) => file !== "ops.py");
    expect(verifyEquationReach({ ...input, files }).join("\n")).toContain(
      "ops.py is not in the Code Explorer",
    );
  });

  test("an equation with no page is reported", () => {
    const input = reachInput();
    const concepts = input.concepts.filter((concept) => concept.id !== "attention.scale");
    expect(verifyEquationReach({ ...input, concepts }).join("\n")).toContain(
      "no page to open from the code line",
    );
  });

  test("two equations with the same deep link are reported", () => {
    const input = reachInput();
    const copy = structuredClone(input.concepts);
    copy[0]!.equations[0]!.id = "x.y";
    copy[1]!.equations[0]!.id = "x-y";
    expect(verifyEquationReach({ ...input, concepts: copy }).join("\n")).toContain(
      "share the deep link ?eq=x-y",
    );
  });

  test("a link that does not resolve is reported as covering no line", () => {
    expect(linesCoveredByLink({ ...links[0]!, match: "not in the code at all" })).toEqual([]);
  });
});

describe("deep links", () => {
  test("the slug is the address form of the id, and every equation has one file", () => {
    expect(equationSlug("attention.scale")).toBe("attention-scale");
    const slugs = allEquationSlugs();
    expect(new Set(slugs).size).toBe(slugs.length);
    expect(equationDetail("attention-scale")!.id).toBe("attention.scale");
    expect(equationDetail("nope")).toBeNull();
  });

  test("the parameter goes before the hash and keeps other parameters", () => {
    expect(withEquation("/a/#L3", "x")).toBe("/a/?eq=x#L3");
    expect(withEquation("/a/?b=1#h", "x")).toBe("/a/?b=1&eq=x#h");
    expect(withEquation("/a/?eq=y", "x")).toBe("/a/?eq=x");
    expect(withEquation("/a/?eq=y#h", null)).toBe("/a/#h");
    expect(withEquation("/a/", null)).toBe("/a/");
  });

  test("the file address respects a base path", () => {
    expect(detailUrl("a-b")).toBe("/math-code/a-b.json");
    expect(detailUrl("a-b", "/base")).toBe("/base/math-code/a-b.json");
  });
});

describe("what is highlighted and said", () => {
  test("hover beats focus, and focus beats the pinned equation", () => {
    expect(activeEquation("p", null, null)).toBe("p");
    expect(activeEquation("p", "f", null)).toBe("f");
    expect(activeEquation("p", "f", "h")).toBe("h");
    expect(activeEquation(null, null, null)).toBeNull();
  });

  test("the announcement states the equation and the lines", () => {
    expect(lineCountText(0)).toContain("not on this page");
    expect(lineCountText(1)).toBe("1 code line on this page");
    expect(announcement("Scale", 2)).toBe(
      "Equation highlighted: Scale. 2 code lines on this page.",
    );
  });

  test("scrolling picks the first line after the equation, then the first on the page", () => {
    expect(scrollTargetIndex([])).toBe(-1);
    expect(scrollTargetIndex([false, true, true])).toBe(1);
    expect(scrollTargetIndex([false, false])).toBe(0);
  });

  test("a line attribute splits into equations", () => {
    expect(splitEquations("a b")).toEqual(["a", "b"]);
    expect(splitEquations(undefined)).toEqual([]);
  });
});

describe("what the pages carry", () => {
  test("an equation names itself, and offers a real button", () => {
    const html = render(createElement(MathBlock, { id: "attention.scale" }));
    expect(html).toContain('data-eq="attention-scale"');
    expect(html).toMatch(/<button[^>]*data-eq-show/);
  });

  test("a linked code line names its equations and has a keyboard control", () => {
    const html = render(
      createElement(CodeFrame, {
        lines: [[{ content: "one", color: "red" }], [{ content: "two", color: "red" }]],
        source: "one\ntwo",
        filePath: "m.py",
        startLine: 10,
        permalink: "https://example.test",
        lineEquations: { 11: ["a-b", "c"] },
      }),
    );
    expect(html).toContain('data-line="11" data-eqs="a-b c"');
    expect(html).toContain('aria-label="Line 11: show its 2 equations"');
    expect(html).not.toMatch(/data-line="10"[^>]*data-eqs/);
    expect(html.match(/data-eq-mark/g)).toHaveLength(1);
  });

  test("the mark adds no text, so a line stays exactly the source", () => {
    const html = render(
      createElement(CodeFrame, {
        lines: [[{ content: "x = 1", color: "red" }]],
        source: "x = 1",
        filePath: "m.py",
        startLine: 1,
        permalink: "https://example.test",
        lineEquations: { 1: ["a"] },
      }),
    );
    expect(html).toMatch(/<button[^>]*data-eq-mark[^>]*><\/button>/);
  });

  test("a block of code gets only the lines of its file and range", () => {
    const [link] = loadLinks().filter((candidate) => candidate.role === "implements");
    const { startLine, symbol } = requireTarget(link!);
    const slug = equationSlug(link!.equationId);
    expect(lineEquations(symbol.file, startLine, 1)[startLine]).toContain(slug);
    expect(lineEquations(symbol.file, startLine + 100000, 5)).toEqual({});
    expect(lineEquations("no/such/file.py", 1, 5)).toEqual({});
  });
});

describe("Implemented in and Used by", () => {
  test("every concept shows file, function, lines and a GitHub link for its equations", () => {
    for (const concept of loadConcepts()) {
      const html = render(createElement(Trace, { concept: concept.id }));
      expect(html, concept.id).toContain('data-testid="implemented-in"');
      expect(html, concept.id).toContain('data-testid="used-by"');
      for (const equation of concept.equations) {
        const computed = equationLocations(equation.id).filter(
          ({ role }) => role === "implements" || role === "inline",
        );
        expect(computed.length, equation.id).toBeGreaterThan(0);
        for (const location of computed) {
          expect(html, equation.id).toContain(location.permalink);
          expect(html, equation.id).toContain(location.functionName);
          // Outside the real router next/link leaves out the trailing slash.
          expect(html, equation.id).toContain(location.explorerHref.replace("/?", "?"));
        }
      }
    }
  });

  test("call sites and tests from link records are under Used by", () => {
    const concept = loadConcepts().find((candidate) =>
      candidate.equations.some((equation) =>
        equationLocations(equation.id).some(({ role }) => role === "test"),
      ),
    )!;
    const html = render(createElement(Trace, { concept: concept.id }));
    expect(html).toContain('data-testid="linked-uses"');
    expect(html).toContain("Tests it");
  });

  test("formulas written inside another function appear through declared links", () => {
    const html = render(createElement(ImplementedIn, { concept: "attention.projections" }));
    const q = equationLocations("attention.q").find(({ role }) => role === "implements")!;
    expect(q.symbol).toBe("transformer.attention_forward");
    expect(html).toContain(q.permalink);
  });

  test("an explicit occurrence index is shown", () => {
    const withIndex = loadLinks().find(
      (link) =>
        link.occurrence !== undefined && (link.role === "implements" || link.role === "inline"),
    );
    if (withIndex) {
      const concept = loadConcepts().find((candidate) =>
        candidate.equations.some((equation) => equation.id === withIndex.equationId),
      )!;
      const html = render(createElement(ImplementedIn, { concept: concept.id }));
      expect(html).toContain(`data-occurrence="${withIndex.occurrence}"`);
    }
    const testLink = loadLinks().find(
      (link) => link.occurrence !== undefined && link.role === "test",
    )!;
    const location = equationLocations(testLink.equationId).find(
      ({ symbol }) => symbol === testLink.symbol,
    )!;
    expect(location.occurrence).toBe(testLink.occurrence);
  });
});

describe("inline equivalent", () => {
  test("a helper the pipeline never calls is named, with the real places", () => {
    const note = inlineEquivalentNote("linear.forward")!;
    expect(note).toContain("Inline equivalent: the pipeline does not call ops.linear_forward");
    for (const { file, startLine, role } of equationLocations("linear.forward")) {
      if (role === "inline") expect(note).toContain(`${file}:${startLine}`);
    }
  });

  test("the note follows the call graph, so a helper that gets a caller changes it", () => {
    const usedBy = { "ops.linear_forward": { production: ["x.y"], tests: [] } };
    const note = inlineEquivalentNote("linear.forward", { usedBy } as never)!;
    expect(note).toContain("the pipeline calls ops.linear_forward");
    expect(note).not.toContain("does not call");
  });

  test("an equation with no inline site has no note", () => {
    expect(inlineEquivalentNote("attention.q")).toBeNull();
  });

  test("the panel and the popover data both carry the note", () => {
    const html = render(createElement(ImplementedIn, { concept: "ops.linear_forward" }));
    expect(html).toContain("the pipeline does not call ops.linear_forward");
    expect(equationDetail("linear-forward")!.inlineNote).toContain("ops.linear_forward");
  });

  test("every note names real inline sites of its equation", () => {
    for (const concept of loadConcepts()) {
      for (const equation of getConcept(concept.id).equations) {
        const note = inlineEquivalentNote(equation.id);
        if (!note) continue;
        expect(note, equation.id).toMatch(/^Inline equivalent: the pipeline /);
        expect(equationLocations(equation.id).some(({ role }) => role === "inline")).toBe(true);
      }
    }
  });
});
