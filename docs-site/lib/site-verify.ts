import { CODE_REFS } from "@/data/code-refs";
import { GLANCE, QUESTIONS } from "@/data/home";
import { PARAMETER_ROLES, REPO_MAP } from "@/data/parameters";
import { PIPELINE, type PipelinePhase } from "@/data/pipeline";
import {
  resolveTarget,
  targetLines,
  type CodeTarget,
  type ResolvedTarget,
  type SymbolIndex,
} from "./code-targets";
import type { MathConcept } from "./concepts";
import type { CallGraph, RepoConfig } from "./repo";

export type SiteDataInput = {
  symbols: SymbolIndex;
  callGraph: CallGraph;
  config: RepoConfig;
  navSlugs: string[];
  pipeline?: PipelinePhase[];
  codeRefs?: Record<string, CodeTarget>;
  parameterNames: string[];
  parameterRoles?: Record<string, { target: CodeTarget }>;
  repoFiles: string[];
  repoMap?: Record<string, string>;
  concepts?: MathConcept[];
  inventoryIds?: string[];
};

// Returns every problem found. Runs in verify:links, so a problem stops the build.
export function verifySiteData(input: SiteDataInput): string[] {
  const {
    symbols,
    callGraph,
    config,
    navSlugs,
    pipeline = PIPELINE,
    codeRefs = CODE_REFS,
    parameterNames,
    parameterRoles = PARAMETER_ROLES,
    repoFiles,
    repoMap = REPO_MAP,
    concepts = [],
    inventoryIds = [],
  } = input;
  const problems: string[] = [];

  const resolve = (where: string, target: CodeTarget): ResolvedTarget | null => {
    const result = resolveTarget(target, symbols);
    if (result.ok) return result.value;
    problems.push(`${where}: ${result.problem}`);
    return null;
  };

  const checkPage = (where: string, page: string) => {
    if (!navSlugs.includes(page)) problems.push(`${where}: page is not in the navigation: ${page}`);
  };

  const checkCall = (where: string, target: ResolvedTarget, call: string) => {
    const edge = callGraph.production[target.symbol.id]?.calls.find((c) => c.target === call);
    const inRange = edge?.lines.some((line) => line >= target.startLine && line <= target.endLine);
    if (!inRange) {
      problems.push(
        `${where}: ${target.symbol.id} does not call ${call} at lines ` +
          `${target.startLine}-${target.endLine}`,
      );
    }
  };

  // Pipeline: consecutive nodes in one function must follow its line order.
  let previous: ResolvedTarget | null = null;
  for (const phase of pipeline) {
    if (phase.repeat) {
      const where = `pipeline phase ${phase.id}`;
      const target = resolve(where, phase.repeat.target);
      if (target) checkCall(where, target, phase.repeat.call);
    }
    for (const node of phase.nodes) {
      const where = `pipeline node ${node.id}`;
      checkPage(where, node.page);
      if (phase.forward && Boolean(node.call) === Boolean(node.inline)) {
        problems.push(`${where}: a forward box needs exactly one of call and inline`);
      }
      const target = resolve(where, node.target);
      if (!target) continue;
      if (node.call) checkCall(where, target, node.call);
      if (node.inline && !targetLines(target).includes(node.inline)) {
        problems.push(`${where}: "${node.inline}" is not in the resolved lines`);
      }
      if (previous?.symbol.id === target.symbol.id && previous.startLine >= target.startLine) {
        problems.push(`${where}: comes before the previous node in ${target.symbol.id}`);
      }
      previous = target;
    }
  }

  for (const group of GLANCE) {
    for (const row of group.rows) {
      if (!config.constants.some((constant) => constant.name === row.config)) {
        problems.push(`homepage row ${row.label}: config.json has no ${row.config}`);
      }
    }
  }
  for (const { question, page } of QUESTIONS) checkPage(`question "${question}"`, page);

  // Quoted lines. A walkthrough (train., talk.) must follow the line order of its function.
  const lastLine = new Map<string, number>();
  for (const [id, target] of Object.entries(codeRefs)) {
    const resolved = resolve(`code reference ${id}`, target);
    if (!resolved) continue;
    const prefix = id.split(".")[0]!;
    if (prefix === "train" || prefix === "talk") {
      if ((lastLine.get(prefix) ?? 0) >= resolved.startLine) {
        problems.push(`code reference ${id}: comes before the previous ${prefix} step`);
      }
      lastLine.set(prefix, resolved.endLine);
    }
  }

  const sameSet = (where: string, actual: string[], declared: string[]) => {
    for (const name of actual.filter((item) => !declared.includes(item))) {
      problems.push(`${where}: nothing declared for ${name}`);
    }
    for (const name of declared.filter((item) => !actual.includes(item))) {
      problems.push(`${where}: ${name} does not exist in the repository`);
    }
  };

  sameSet("parameter roles", parameterNames, Object.keys(parameterRoles));
  for (const [name, { target }] of Object.entries(parameterRoles)) {
    resolve(`parameter ${name}`, target);
  }
  sameSet("repository map", repoFiles, Object.keys(repoMap));

  for (const concept of concepts) {
    for (const row of concept.inventory) {
      if (!inventoryIds.includes(row)) {
        problems.push(`concept ${concept.id}: ${row} is not a row of the mathematics inventory`);
      }
    }
  }

  return problems;
}
