import { isDeepStrictEqual } from "node:util";
import katex from "katex";
import { resolveTarget, type CodeTarget, type SymbolIndex } from "./code-targets";
import type { MathConcept, Pipeline } from "./concepts";
import type { StoredExamples } from "./examples";
import type { Link } from "./links";

export type VerifyInput = {
  concepts: MathConcept[];
  links: Link[];
  pipelines: Pipeline[];
  symbols: SymbolIndex;
  examples: StoredExamples;
};

function duplicates(values: string[]): string[] {
  return [...new Set(values.filter((value, index) => values.indexOf(value) !== index))];
}

function badLatex(latex: string): string | null {
  try {
    katex.renderToString(latex, { throwOnError: true });
    return null;
  } catch (error) {
    return (error as Error).message;
  }
}

// Returns every problem found. An empty list means the data is consistent with the code.
export function verifyData({ concepts, links, pipelines, symbols, examples }: VerifyInput) {
  const problems: string[] = [];

  const checkTarget = (where: string, target: CodeTarget) => {
    const result = resolveTarget(target, symbols);
    if (!result.ok) problems.push(`${where}: ${result.problem}`);
  };

  const equationIds = concepts.flatMap((concept) =>
    concept.equations.map((equation) => equation.id),
  );

  for (const id of duplicates(concepts.map((concept) => concept.id))) {
    problems.push(`duplicate concept id: ${id}`);
  }
  for (const id of duplicates(equationIds)) problems.push(`duplicate equation id: ${id}`);
  for (const id of duplicates(pipelines.map((pipeline) => pipeline.id))) {
    problems.push(`duplicate pipeline id: ${id}`);
  }

  for (const link of links) {
    const where = `link for ${link.equationId}`;
    if (!equationIds.includes(link.equationId)) {
      problems.push(`${where}: equation does not exist`);
    }
    checkTarget(where, link);
  }

  for (const concept of concepts) {
    const where = `concept ${concept.id}`;

    for (const equation of concept.equations) {
      const hasCode = links.some(
        (link) =>
          link.equationId === equation.id && (link.role === "implements" || link.role === "inline"),
      );
      if (!hasCode) problems.push(`${where}: equation ${equation.id} has no link to code`);

      const latexError = badLatex(equation.latex);
      if (latexError) problems.push(`${where}: equation ${equation.id}: ${latexError}`);
    }

    for (const symbol of concept.symbols) {
      const latexError = badLatex(symbol.symbol);
      if (latexError) problems.push(`${where}: symbol ${symbol.symbol}: ${latexError}`);
    }

    for (const { file, symbol } of concept.source) {
      const found = symbols.get(symbol);
      if (!found) problems.push(`${where}: source symbol does not exist: ${symbol}`);
      else if (found.file !== file) {
        problems.push(`${where}: source symbol ${symbol} is in ${found.file}, not ${file}`);
      }
    }

    for (const use of concept.usedBy) checkTarget(`${where}: usedBy`, use);
    for (const shape of concept.shapes) checkTarget(`${where}: shape ${shape.id}`, shape.target);

    const example = concept.example;
    const stored = examples[concept.id];
    if (!example) {
      if (stored) problems.push(`${where}: a computed example is stored but none is declared`);
      continue;
    }
    if (!symbols.has(example.call)) {
      problems.push(`${where}: example calls an unknown symbol: ${example.call}`);
    }
    for (const step of example.steps) {
      if (!concept.equations.some((equation) => equation.id === step.equationId)) {
        problems.push(`${where}: example step ${step.id} names unknown ${step.equationId}`);
      }
    }
    if (!stored) {
      problems.push(`${where}: no computed example. Run: bun run sync:examples`);
    } else if (
      stored.kind !== example.kind ||
      stored.call !== example.call ||
      !isDeepStrictEqual(stored.input, example.input) ||
      !isDeepStrictEqual(
        Object.keys(stored.steps).sort(),
        example.steps.map((step) => step.id).sort(),
      )
    ) {
      problems.push(`${where}: the computed example is stale. Run: bun run sync:examples`);
    }
  }

  for (const id of Object.keys(examples)) {
    if (!concepts.some((concept) => concept.id === id)) {
      problems.push(`computed example has no concept: ${id}`);
    }
  }

  for (const pipeline of pipelines) {
    pipeline.steps.forEach((step, index) => {
      checkTarget(`pipeline ${pipeline.id} step ${index + 1}`, step.target);
    });
  }

  return problems;
}
