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
| `bun run verify:links`     | Fail if hand-written data no longer matches the code. `build` runs this             |
| `bun run sync:examples`    | Run repository functions and write `data/golden/examples.json`. Needs NumPy         |
| `bun run verify:examples`  | Fail if the stored example numbers differ from a fresh run. Needs NumPy             |
| `bun run sync:golden`      | Run repository functions and write the golden files for `lib/math`. Needs NumPy     |
| `bun run verify:golden`    | Fail if the stored golden numbers differ from a fresh run. Needs NumPy              |
| `bun run sync:derived`     | Call `init_params`, snapshot the math inventory into `data/derived/`. Needs NumPy   |
| `bun run verify:derived`   | Fail if `data/derived/` differs from a fresh run. Needs NumPy                       |
| `bun run verify:content`   | Stub: not implemented yet                                                           |

To build for GitHub Pages, set the base path first:

```sh
NEXT_PUBLIC_BASE_PATH=/Lazy-Language-Model-LLM- bun run build
```

Git Bash on Windows rewrites a value starting with `/` into a Windows path, which fails the build. There, add `MSYS2_ENV_CONV_EXCL=NEXT_PUBLIC_BASE_PATH` before the command, or use PowerShell.

## Folders

| Path              | Contents                                                       |
| ----------------- | -------------------------------------------------------------- |
| `app/`            | Next.js routes                                                 |
| `components/`     | React components                                               |
| `styles/`         | `tokens.css` holds every colour literal; `base.css` is global  |
| `content/`        | MDX pages at `content/<slug>.mdx`, with frontmatter            |
| `data/nav.ts`     | Navigation tree. A page is disabled until its MDX file exists  |
| `data/concepts/`  | Hand-written structured concept data                           |
| `data/links/`     | Hand-written structured link data                              |
| `data/generated/` | Machine-generated repository facts. Never edit by hand         |
| `data/pipelines/` | Hand-written data for pipeline diagrams                        |
| `data/golden/`    | Numbers computed by running the repository. Never edit by hand |
| `lib/`            | Shared code. `lib/repo.ts` loads and validates generated data  |
| `scripts/`        | Extraction and verification scripts                            |
| `tests/`          | Vitest tests; Playwright tests are in `tests/e2e/`             |
| `contract/`       | Repository inventories, architecture and discrepancies         |

## Design system

`/design` is a hidden style guide that renders every element in both themes. It is not in the navigation and is marked `noindex`. Components take colours from the tokens in `styles/tokens.css` only; `tests/tokens.test.ts` enforces that and checks WCAG AA contrast.

The hidden page `/design/components/` renders every documentation component from the sample concept `ops.softmax_forward`. Its source is `content/_samples/softmax-forward.mdx`. Folders in `content/` starting with `_` get no route of their own.

## Concepts and links

Pages reference data by id, for example `<MathBlock id="softmax.shift" />`. Equations, symbols, shapes and examples live in `data/concepts/*.yaml`. `data/links/*.yaml` ties each equation to code by `file`, `symbol` and `match`, with `regex` and `occurrence` when needed.

Never store a line number. Lines are resolved at build time from `code-index.json`, and the schemas reject line-number keys. `verify:links` fails if a symbol does not exist, a match is not found, a match is ambiguous without an `occurrence`, or an equation has no link to code.

An example declares its input and which local variables to capture. `sync:examples` runs the real function and stores the values. Every `<NumericExample>` needs a `kind` that agrees with the data.

## Math and code navigation

`MathBlock` puts `data-eq` on an equation. `CodeFrame` and the Code Explorer put `data-eqs` on each linked line, computed on the server from `data/links` by `lib/links.ts`. One client component, `MathCodeNavigator`, is mounted in the docs layout and reads those attributes, so no page or component wires anything up.

- A click, or Show code, pins the equation in the address as `?eq=attention-scale`. Opening that address restores it. Hover and keyboard focus show the same highlight without moving the page. Escape clears it.
- A click on a linked line, or Enter on its gutter mark, opens a popover. `/math-code/<slug>.json`, one small static file per equation, holds the rendered equation, so no KaTeX JavaScript reaches the browser.
- To link a formula written inside another function, or one of several matches in a function, add a record in `data/links` with `role: inline` and `occurrence` if needed. `Implemented in`, `Used by`, the inline equivalent note and both directions of navigation follow from that record.

`verify:links` fails if a link is not reachable both ways: its concept must have a `<Trace>` on a page, its file must be in the Code Explorer, and its equation must have a page to open.

## Actual Model Output

`bun run export:actual` reads a checkpoint, read-only, and writes `data/generated/actual/*.json`: sizes, real tokens and merges, the attention weights of one real window for every layer and head, next-token probabilities for one prompt, and losses on a few windows. `--checkpoint PREFIX` chooses another checkpoint. It takes about a minute.

With no checkpoint it writes `actualAvailable: false`, prints how to create one and exits 0. The site then builds with every Actual panel off. It never trains unless given `--train`, which runs the repository's `train.main` with its prefix in `docs-site/.cache/` and refuses anywhere else. At the committed `config.py` that is a long run.

`bun run verify:actual` compares the stored files with a fresh export. Run the export again after a change to the checkpoint, the corpus or the code, because `build` only reads the files and checks the attention weights. `lib/actual.ts` loads and validates them. Illustrative and Actual values live in separate components, and nothing outside `data/generated/actual` feeds an Actual panel.

## Math engine and interactive kit

`lib/math/` is a pure TypeScript copy of the repository's NumPy operations, for interactive pages. Each function names the Python function it mirrors, and `tests/math.golden.test.ts` compares it with the real Python to within 1e-9, using `data/golden/*.json`. No number in an example may be typed into JSX: take inputs from data and values from `lib/math`.

`np.argsort` in `generate.sample_next` is not stable. When the top-p cutoff falls inside a group of equal probabilities, Python may keep different token ids from the engine. The kept count and the probabilities still match, and `applyTopP` sets `tieDependent` so a page can say so.

`rng.choice`, `attention_backward`, `apply_merges` and `count_words` are not mirrored.

`Heatmap`, `BarChart`, `ProbabilityBars`, `LineChart` and `ShapeFlow` require a `kind`, and `actual` also requires a `provenance`. `MatrixInput` and `StepPlayer` are controls. The hidden page `/design/interactive/` shows them all.

## Generated data

`scripts/extract_repo.py` reads the repository statically, using only the Python standard library, and never runs its code. It writes `repo-meta.json`, `code-index.json`, `config.json`, `call-graph.json` and `tests-index.json`.

`sha` in `repo-meta.json` is the newest commit that touches anything outside `docs-site/`. Commits that only change `docs-site/` do not move it, so committing the generated data does not make it stale.

`verify:generated` compares every file byte for byte, except for `branch`, `dirty` and `dirtyFiles` in `repo-meta.json`. Those describe the current checkout and are reported as notes.
