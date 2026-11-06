// Writes public/search-index.json and reports its size. `bun run build` and `bun run dev` run this.
import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { SEARCH_INDEX_FILE } from "@/lib/search";
import { buildSearchIndexJson, SEARCH_INDEX_LIMIT_BYTES } from "@/lib/search-index";

async function main(): Promise<number> {
  const { json, documents } = await buildSearchIndexJson();
  const bytes = Buffer.byteLength(json, "utf-8");
  const kilobytes = (size: number) => `${(size / 1024).toFixed(1)} KB`;
  const report =
    `Search index: ${kilobytes(bytes)}, ${documents} documents ` +
    `(limit ${kilobytes(SEARCH_INDEX_LIMIT_BYTES)}).`;
  if (bytes > SEARCH_INDEX_LIMIT_BYTES) {
    console.error(`FAILED: ${report} Reduce what is indexed, or justify and raise the limit.`);
    return 1;
  }
  const folder = join(process.cwd(), "public");
  mkdirSync(folder, { recursive: true });
  writeFileSync(join(folder, SEARCH_INDEX_FILE), json);
  console.log(report);
  return 0;
}

main().then(
  (code) => {
    process.exitCode = code;
  },
  (error: Error) => {
    console.error(`FAILED: ${error.message}`);
    process.exitCode = 1;
  },
);
