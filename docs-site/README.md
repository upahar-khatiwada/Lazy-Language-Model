# LLM Developer Docs

Static documentation site for the from-scratch NumPy language model in this repository. Run every command from `docs-site/`.

## Commands

Requires Bun, Node.js and Python 3 on `PATH`. Next.js does not build on Bun's `node` stand-in, so Node.js must be installed.

| Command                    | What it does                                                                        |
| -------------------------- | ----------------------------------------------------------------------------------- |
| `bun install`              | Install the pinned dependencies                                                     |
| `bun run dev`              | Start the development server                                                        |
| `bun run build`            | Build the static export into `out/`                                                 |
| `bun run start`            | Serve `out/` at `http://127.0.0.1:3000` (without a base path)                       |
| `bun run lint`             | Run ESLint and Prettier checks                                                      |
| `bun run typecheck`        | Type-check with TypeScript                                                          |
| `bun run test`             | Run the Vitest tests. Do not use a bare `bun test`: that ignores Vitest             |
| `bun run test:e2e`         | Run Playwright against `out/`. Needs a build and `bunx playwright install chromium` |
| `bun run sync:repo`        | Regenerate `data/generated/` from the repository                                    |
| `bun run verify:generated` | Fail if `data/generated/` is stale or was edited by hand                            |
| `bun run verify:content`   | Stub: not implemented yet                                                           |

To build for GitHub Pages, set the base path first:

```sh
NEXT_PUBLIC_BASE_PATH=/Lazy-Language-Model-LLM- bun run build
```

Git Bash on Windows rewrites a value starting with `/` into a Windows path, which fails the build. There, add `MSYS2_ENV_CONV_EXCL=NEXT_PUBLIC_BASE_PATH` before the command, or use PowerShell.

## Folders

| Path              | Contents                                                      |
| ----------------- | ------------------------------------------------------------- |
| `app/`            | Next.js routes                                                |
| `components/`     | React components                                              |
| `styles/`         | `tokens.css` holds every colour literal; `base.css` is global |
| `content/`        | MDX pages at `content/<slug>.mdx`, with frontmatter           |
| `data/nav.ts`     | Navigation tree. A page is disabled until its MDX file exists |
| `data/concepts/`  | Hand-written structured concept data                          |
| `data/links/`     | Hand-written structured link data                             |
| `data/generated/` | Machine-generated repository facts. Never edit by hand        |
| `data/golden/`    | Golden reference data                                         |
| `lib/`            | Shared code. `lib/repo.ts` loads and validates generated data |
| `scripts/`        | Extraction and verification scripts                           |
| `tests/`          | Vitest tests; Playwright tests are in `tests/e2e/`            |
| `contract/`       | Repository inventories, architecture and discrepancies        |

## Design system

`/design` is a hidden style guide that renders every element in both themes. It is not in the navigation and is marked `noindex`. Components take colours from the tokens in `styles/tokens.css` only; `tests/tokens.test.ts` enforces that and checks WCAG AA contrast.

## Generated data

`scripts/extract_repo.py` reads the repository statically, using only the Python standard library, and never runs its code. It writes `repo-meta.json`, `code-index.json`, `config.json`, `call-graph.json` and `tests-index.json`.

`sha` in `repo-meta.json` is the newest commit that touches anything outside `docs-site/`. Commits that only change `docs-site/` do not move it, so committing the generated data does not make it stale.

`verify:generated` compares every file byte for byte, except for `branch`, `dirty` and `dirtyFiles` in `repo-meta.json`. Those describe the current checkout and are reported as notes.
