import { PARAMETER_ROLES, REPO_MAP } from "@/data/parameters";
import { formatLineRange, requireTarget, type CodeTarget } from "@/lib/code-targets";
import { resolveDimension } from "@/lib/dimensions";
import { pageHref } from "@/lib/nav";
import { writtenHref } from "@/lib/pages";
import {
  countFormulaTex,
  countInV,
  loadInitParams,
  realCount,
  realShape,
  symbolicCount,
  symbolicShape,
} from "@/lib/parameters";
import { loadCodeIndex, loadRepoMeta } from "@/lib/repo";
import { Badge } from "./Badge";
import styles from "./ArchitectureTables.module.css";
import { DataTable } from "./DataTable";
import { TexDisplay } from "./TexDisplay";

const INIT_LABEL = { ones: "ones", zeros: "zeros", xavier_uniform: "Xavier uniform" } as const;

function value(name: "d" | "F" | "Tmax" | "L"): number {
  const { value: found } = resolveDimension(name);
  if (found === null) throw new Error(`Dimension ${name} has no value`);
  return found;
}

function realValues() {
  return { d: value("d"), F: value("F"), Tmax: value("Tmax") };
}

// `names` limits the table to those parameters, for a page about one operation.
export function ParameterTable({ names }: { names?: string[] } = {}) {
  const values = realValues();
  const roles: Record<string, { role: string; target: CodeTarget }> = PARAMETER_ROLES;
  const { parameters } = loadInitParams();
  const shown = names ? parameters.filter(({ name }) => names.includes(name)) : parameters;
  if (names && shown.length !== names.length) {
    throw new Error(`ParameterTable: a name is not a parameter: ${names.join(", ")}`);
  }
  return (
    <DataTable
      label="Parameters from init_params"
      data-testid="parameter-table"
      className={styles.parameters}
    >
      <thead>
        <tr>
          <th scope="col">Name</th>
          <th scope="col">Shape</th>
          <th scope="col">With config</th>
          <th scope="col">Scalars</th>
          <th scope="col">Role</th>
        </tr>
      </thead>
      <tbody>
        {shown.map(({ name, dims, init, perBlock }) => {
          const entry = roles[name];
          if (!entry) throw new Error(`data/parameters.ts has no role for ${name}`);
          const target = requireTarget(entry.target);
          return (
            <tr key={name} data-parameter={name}>
              <td>
                <code>{name}</code>
                <a className={`code-link ${styles.line}`} href={target.permalink}>
                  {target.symbol.file}:{formatLineRange(target.startLine, target.endLine)}
                </a>
              </td>
              <td>
                <code>{symbolicShape(dims)}</code>
              </td>
              <td>
                <code>{realShape(dims, values)}</code>
              </td>
              <td>
                <code>{symbolicCount(dims)}</code>
                <span className={styles.muted}>
                  {realCount(dims, values)}
                  {perBlock ? " per block" : ""}
                </span>
              </td>
              <td>
                {entry.role}
                <span className={styles.muted}>
                  Starts as {INIT_LABEL[init]}. {dims.length >= 2 ? "Weight decay." : "No decay."}
                </span>
              </td>
            </tr>
          );
        })}
      </tbody>
    </DataTable>
  );
}

// Every parameter, grouped by how init_params sets it. The grouping comes from running init_params.
export function InitialValues() {
  const { parameters } = loadInitParams();
  const roles: Record<string, { target: CodeTarget }> = PARAMETER_ROLES;
  return (
    <DataTable label="Starting value of every parameter" data-testid="initial-values">
      <thead>
        <tr>
          <th scope="col">Starts as</th>
          <th scope="col">Parameters</th>
        </tr>
      </thead>
      <tbody>
        {(Object.keys(INIT_LABEL) as (keyof typeof INIT_LABEL)[]).map((init) => (
          <tr key={init} data-init={init}>
            <td>{INIT_LABEL[init]}</td>
            <td>
              {parameters
                .filter((parameter) => parameter.init === init)
                .map(({ name }, index) => {
                  const entry = roles[name];
                  if (!entry) throw new Error(`data/parameters.ts has no role for ${name}`);
                  return (
                    <span key={name}>
                      {index > 0 ? ", " : null}
                      <a href={requireTarget(entry.target).permalink}>
                        <code>{name}</code>
                      </a>
                    </span>
                  );
                })}
            </td>
          </tr>
        ))}
      </tbody>
    </DataTable>
  );
}

export function ParameterCount({ configOnly = false }: { configOnly?: boolean }) {
  const { parameters } = loadInitParams();
  const { perVocabulary, constant } = countInV(parameters, { ...realValues(), L: value("L") });
  return (
    <div className={styles.count} data-testid="parameter-count">
      <Badge kind="derived" />
      {configOnly ? null : (
        <TexDisplay
          tex={`N = ${countFormulaTex(parameters)}`}
          label="Parameter count from shapes"
        />
      )}
      <TexDisplay
        tex={`N = ${perVocabulary.toLocaleString("en-US")}\\,V + ${constant.toLocaleString("en-US").replace(",", "{,}")}`}
        label="Parameter count with config values"
      />
    </div>
  );
}

export function RepoMap() {
  const repo = loadRepoMeta();
  const explorer = writtenHref("code/code-explorer") ? pageHref("code/code-explorer") : null;
  return (
    <DataTable label="Repository map" data-testid="repo-map" className={styles.repoMap}>
      <thead>
        <tr>
          <th scope="col">File</th>
          <th scope="col">Lines</th>
          <th scope="col">What it holds</th>
        </tr>
      </thead>
      <tbody>
        {loadCodeIndex().files.map(({ path, lineCount }) => {
          const role = REPO_MAP[path];
          if (!role) throw new Error(`data/parameters.ts REPO_MAP has no entry for ${path}`);
          // The Code Explorer takes over these links as soon as its page exists.
          const href = explorer
            ? `${explorer}#${path}`
            : `${repo.githubBaseUrl}/blob/${repo.sha}/${path}`;
          return (
            <tr key={path} data-file={path}>
              <td>
                <a className="code-link" href={href}>
                  {path}
                </a>
              </td>
              <td>{lineCount}</td>
              <td>{role}</td>
            </tr>
          );
        })}
      </tbody>
    </DataTable>
  );
}
