import { PARAMETER_GROUPS } from "@/data/parameters";
import { resolveDimension } from "@/lib/dimensions";
import { exampleAnchor } from "@/lib/concepts";
import { loadGolden } from "@/lib/golden";
import { parameterCount, type ParameterDims } from "@/lib/math/params";
import { evaluateCount, loadInitParams, realCount, symbolicCount } from "@/lib/parameters";
import { DataTable } from "./DataTable";
import { VizFigure } from "./VizFigure";

function value(name: "d" | "F" | "Tmax"): number {
  const { value: found } = resolveDimension(name);
  if (found === null) throw new Error(`Dimension ${name} has no value`);
  return found;
}

const number = (count: number) => count.toLocaleString("en-US");

// Sums init_params group by group. Names and shapes come from data/derived/init-params.json, sizes
// from config.json. Only the grouping is hand-written, and site-verify checks that it is complete.
export function ParameterDerivation() {
  const { parameters } = loadInitParams();
  const values = { d: value("d"), F: value("F"), Tmax: value("Tmax") };
  const block = parameters.filter((parameter) => parameter.perBlock);
  const rows = PARAMETER_GROUPS.map(({ label, names }) => ({
    label,
    names,
    members: names.map((name) => {
      const found = parameters.find((parameter) => parameter.name === name);
      if (!found) throw new Error(`init_params creates no parameter named ${name}`);
      return found;
    }),
  }));

  return (
    <DataTable label="Scalars in each group of parameters" data-testid="parameter-derivation">
      <thead>
        <tr>
          <th scope="col">Group</th>
          <th scope="col">Names</th>
          <th scope="col">Scalars</th>
          <th scope="col">With config</th>
          <th scope="col">Repeats</th>
        </tr>
      </thead>
      <tbody>
        {rows.map(({ label, names, members }) => (
          <tr key={label} data-group={label}>
            <td>{label}</td>
            <td>
              <code>{names.join(", ")}</code>
            </td>
            <td>
              <code>{members.map(({ dims }) => symbolicCount(dims)).join(" + ")}</code>
            </td>
            <td>
              <code>{members.map(({ dims }) => realCount(dims, values)).join(" + ")}</code>
            </td>
            <td>{members[0]!.perBlock ? "Once per block" : "Once"}</td>
          </tr>
        ))}
        <tr data-group="one-block">
          <th scope="row" colSpan={3}>
            One block, the rows marked once per block
          </th>
          <td colSpan={2}>
            <code data-testid="block-total">
              {number(evaluateCount(block, { ...values, V: 0, L: 1 }))}
            </code>
          </td>
        </tr>
      </tbody>
    </DataTable>
  );
}

// The formula beside the real count_params, on small configurations.
export function ParameterExample({ concept }: { concept: string }) {
  const cases = loadGolden("params");
  return (
    <div id={exampleAnchor(concept)}>
      <VizFigure
        title="The formula and count_params on small configurations"
        kind="illustrative"
        caption="Hand-picked configurations. The count_params column comes from running init_params and count_params in Python. These are not this model's parameters, except that the last row uses the shape values in config.py, with the V that tests/test_transformer.py uses."
      >
        <DataTable label="Formula against count_params" data-testid="parameter-example">
          <thead>
            <tr>
              <th scope="col">V</th>
              <th scope="col">T_max</th>
              <th scope="col">d</th>
              <th scope="col">F</th>
              <th scope="col">L</th>
              <th scope="col">Formula</th>
              <th scope="col">count_params</th>
            </tr>
          </thead>
          <tbody>
            {cases.map((item) => {
              const dims = item.inputs as ParameterDims;
              return (
                <tr key={item.id} data-case={item.id}>
                  <td>{dims.V}</td>
                  <td>{dims.Tmax}</td>
                  <td>{dims.d}</td>
                  <td>{dims.F}</td>
                  <td>{dims.L}</td>
                  <td data-testid="formula-count">{number(parameterCount(dims))}</td>
                  <td data-testid="real-count">{number(item.outputs.count as number)}</td>
                </tr>
              );
            })}
          </tbody>
        </DataTable>
      </VizFigure>
    </div>
  );
}
