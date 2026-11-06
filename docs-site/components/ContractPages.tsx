import { READER_NOTES } from "@/data/not-in-model";
import {
  inlineParts,
  loadAbsentTechniques,
  loadDiscrepancies,
  type DiscrepancyBlock,
} from "@/lib/contract-pages";
import { DataTable } from "./DataTable";

function ContractText({ text }: { text: string }) {
  return (
    <>
      {inlineParts(text).map((part, index) => {
        if (part.style === "code") return <code key={index}>{part.text}</code>;
        if (part.style === "bold") return <strong key={index}>{part.text}</strong>;
        return part.text;
      })}
    </>
  );
}

export function AbsentTechniques() {
  const { rows } = loadAbsentTechniques();
  return (
    <DataTable label="Techniques not in this model" data-testid="absent-techniques">
      <thead>
        <tr>
          <th scope="col">Technique</th>
          <th scope="col">In the reference PDF</th>
          <th scope="col">Note</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((row) => (
          <tr key={row.technique}>
            <td>
              <ContractText text={row.technique} />
            </td>
            <td>
              <ContractText text={row.inPdf} />
            </td>
            <td>
              <ContractText text={READER_NOTES[row.technique] ?? row.note} />
            </td>
          </tr>
        ))}
      </tbody>
    </DataTable>
  );
}

function Block({ block, label }: { block: DiscrepancyBlock; label: string }) {
  if (block.type === "paragraph") {
    return (
      <p>
        <ContractText text={block.text} />
      </p>
    );
  }
  if (block.type === "list") {
    return (
      <ol>
        {block.items.map((item) => (
          <li key={item}>
            <ContractText text={item} />
          </li>
        ))}
      </ol>
    );
  }
  return (
    <DataTable label={label}>
      <thead>
        <tr>
          {block.headers.map((header) => (
            <th scope="col" key={header}>
              <ContractText text={header} />
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {block.rows.map((row, rowIndex) => (
          <tr key={rowIndex}>
            {row.map((cell, cellIndex) => (
              <td key={cellIndex}>
                <ContractText text={cell} />
              </td>
            ))}
          </tr>
        ))}
      </tbody>
    </DataTable>
  );
}

// The text before the first numbered section of the contract file.
export function DiscrepancyPreamble() {
  return (
    <>
      {loadDiscrepancies().preamble.map((text) => (
        <p key={text}>
          <ContractText text={text} />
        </p>
      ))}
    </>
  );
}

// One numbered section. The page writes the heading, so that the on-this-page rail lists it, and
// tests/content-checks.test.ts compares that heading with the contract.
export function DiscrepancySection({ number }: { number: number }) {
  const section = loadDiscrepancies().sections.find((candidate) => candidate.number === number);
  if (!section) throw new Error(`contract/DISCREPANCIES.md has no section ${number}`);
  return (
    <>
      {section.blocks.map((block, index) => (
        <Block key={index} block={block} label={section.title} />
      ))}
    </>
  );
}
