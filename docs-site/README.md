# LLM Developer Docs

Static documentation site for the from-scratch NumPy language model in this repository. Every equation on the site is tied to the lines of Python that compute it.

Run every command from `docs-site/`.

## From clone to running site

You need [Bun](https://bun.sh), [Node.js](https://nodejs.org) and Python 3 on `PATH`. Next.js does not build on Bun's `node` stand-in, so Node.js must be installed. The build downloads the IBM Plex fonts once, so it needs network access.

```sh
git clone https://github.com/upahar-khatiwada/Lazy-Language-Model-LLM-.git
cd Lazy-Language-Model-LLM-/docs-site
bun install
bun run dev
```

Open the address `bun run dev` prints, usually `http://localhost:3000`. To see the static site as it will be published:

```sh
bun run build
bun run start
```

That is enough to read and edit the site, because all generated data is committed. To regenerate that data from the Python code, you also need NumPy and pandas. The repository has no requirements file:

```sh
pip install numpy pandas
```

Then check your setup:

```sh
bun run lint
bun run typecheck
bun run test
```

Two things are git-ignored at the repository root and will not be in a fresh clone:

- `docs-site/contract/` holds the inventories behind the site. Without it, `verify:derived` checks less, and it says so. `data/derived/` holds the snapshots the build reads.
- `data/corpus.csv` and `checkpoints/` are the corpus and the trained model. `verify:actual` needs them, and it notes that it did not compare when there is no checkpoint. Reading, building and testing the site do not.

## Data flow

The site never takes a number, line or shape from a person's memory. Each kind of fact has one source:

```text
 repository .py files
      |
      |  sync:repo      read statically, never run        data/generated/*.json
      |  sync:golden    real functions, hand-picked input  data/golden/*.json
      |  sync:examples  real functions, hand-picked input  data/golden/examples.json
      |  sync:shapes    real functions, tiny and real size data/golden/shapes.json
      |  sync:derived   init_params, contract/ snapshots   data/derived/*.json
      |  export:actual  a trained checkpoint               data/generated/actual/*.json
      v
 hand-written data: data/concepts/*.yaml (equations, symbols, shapes, examples)
                    data/links/*.yaml    (each equation -> file, symbol, matched code)
                    data/nav.ts          (the sidebar)
                    content/<slug>.mdx   (the pages, which refer to all of the above by id)
      |
      |  build          verify:links, sync:search, next build
      v
 out/               static HTML, plus public/search-index.json copied in
```

`lib/math/` is a TypeScript copy of the NumPy operations, so the labs can recompute in the browser. `tests/math.golden.test.ts` compares it with the real Python to within 1e-9, and `tests/interactive.test.ts` compares what each lab shows with the same values.

## Commands

| Command                    | What it does                                                                                                              |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| `bun install`              | Install the pinned dependencies                                                                                           |
| `bun run dev`              | Start the development server                                                                                              |
| `bun run build`            | Build the static export into `out/`                                                                                       |
| `bun run start`            | Serve `out/` at `http://127.0.0.1:3000` (without a base path)                                                             |
| `bun run lint`             | Run ESLint and Prettier checks                                                                                            |
| `bun run typecheck`        | Type-check with TypeScript                                                                                                |
| `bun run test`             | Run the Vitest tests. Do not use a bare `bun test`: that ignores Vitest                                                   |
| `bun run test:e2e`         | Run Playwright against `out/`. Needs a build and `bunx playwright install chromium`. See [Tests](#tests)                  |
| `bun run sync:repo`        | Regenerate `data/generated/` from the repository                                                                          |
| `bun run verify:generated` | Fail if `data/generated/` is stale or was edited by hand                                                                  |
| `bun run verify:links`     | Fail if hand-written data no longer matches the code. `build` runs this                                                   |
| `bun run sync:examples`    | Run repository functions and write `data/golden/examples.json`. Needs NumPy                                               |
| `bun run verify:examples`  | Fail if the stored example numbers differ from a fresh run. Needs NumPy                                                   |
| `bun run sync:golden`      | Run repository functions and write the golden files for `lib/math`. Needs NumPy and pandas                                |
| `bun run verify:golden`    | Fail if the stored golden numbers differ from a fresh run. Needs NumPy and pandas                                         |
| `bun run sync:derived`     | Call `init_params`, snapshot the contract into `data/derived/`. Needs NumPy                                               |
| `bun run verify:derived`   | Fail if `data/derived/` differs from a fresh run. Needs NumPy                                                             |
| `bun run sync:shapes`      | Run repository functions at a tiny config and at `config.py`, and write `data/golden/shapes.json`. Needs NumPy and pandas |
| `bun run verify:shapes`    | Fail if `data/golden/shapes.json` differs from a fresh run                                                                |
| `bun run export:actual`    | Read a checkpoint and write `data/generated/actual/`. Needs NumPy, pandas and the corpus                                  |
| `bun run verify:actual`    | Compare `data/generated/actual/` with a fresh export. Without a checkpoint it notes that and exits 0                      |
| `bun run sync:search`      | Build `public/search-index.json` and report its size. `build` and `dev` run this                                          |
| `bun run verify:content`   | Seven checks that the docs match the repository, with a coverage report. Needs git for check 6                            |
| `bun run verify:site`      | Check the built `out/`: links, anchors, permalinks, metadata and static accessibility rules. Run after `build`            |

## Folders

| Path              | Contents                                                            |
| ----------------- | ------------------------------------------------------------------- |
| `app/`            | Next.js routes                                                      |
| `components/`     | React components                                                    |
| `styles/`         | `tokens.css` holds every colour literal; `base.css` is global       |
| `content/`        | MDX pages at `content/<slug>.mdx`, with frontmatter                 |
| `data/nav.ts`     | Navigation tree. A page is disabled until its MDX file exists       |
| `data/concepts/`  | Hand-written structured concept data                                |
| `data/links/`     | Hand-written structured link data                                   |
| `data/pipelines/` | Hand-written data for pipeline diagrams                             |
| `data/generated/` | Machine-generated repository facts. Never edit by hand              |
| `data/golden/`    | Numbers computed by running the repository. Never edit by hand      |
| `data/derived/`   | Results of `init_params`, and snapshots of `contract/`. Never edit  |
| `lib/`            | Shared code. `lib/repo.ts` loads and validates generated data       |
| `scripts/`        | Extraction and verification scripts                                 |
| `tests/`          | Vitest tests; Playwright tests are in `tests/e2e/`                  |
| `contract/`       | Repository inventories, architecture and discrepancies. Git-ignored |

## How to add a concept

A concept is one operation the code performs. Say the code gains `ops.gelu_forward`.

1. **Add `data/concepts/ops.gelu_forward.yaml`.** Copy `data/concepts/ops.softmax_forward.yaml`. It holds the title, category, the `inventory` row, the summary, why the model needs it, each equation's LaTeX, each symbol with its `codeName`, the tensor shapes and the example input. Write each equation only here, once.
2. **Add `data/links/ops.gelu_forward.yaml`.** One record per equation, with `file`, `symbol` and `match`, the text of the code line. Add `occurrence` if the match appears more than once, and `regex` if it needs one. Never write a line number: lines are resolved at build time from `code-index.json`, and the schemas reject line-number keys.
3. **Compute the example.** Run `bun run sync:examples`. It runs the real function on your input and stores the values. Label the example `illustrative` if you picked the input, and `actual` only if it comes from a real run. Never mix the two.
4. **Mirror it if a lab needs it.** Add the function to `lib/math/`, naming the Python function it mirrors, and add a case to `scripts/make_golden.py`. Then run `bun run sync:golden`. `tests/math.golden.test.ts` fails until the function has a case.
5. **Check.** `bun run verify:links` fails if a symbol does not exist, a match is not found or is ambiguous, or an equation has no link. `bun run verify:content` fails if a public function has no concept.

If the PDF does not state an equation, mark it `derived`. If the code does not provide something, write "not available from the implementation". Never document mathematics that only the PDF has, or that is merely common in transformers.

## How to add a page

1. **Add `content/<slug>.mdx`,** with this frontmatter. `status` is `published`, `draft` or `placeholder`.

   ```yaml
   ---
   title: GELU
   description: One sentence for search results and link previews.
   sourceFile: ops.py
   status: published
   ---
   ```

2. **Add it to `data/nav.ts`.** A page stays disabled in the sidebar until its file exists. The title goes in the sidebar, and `sourceFile` is checked against the code index.
3. **Write the concept sections in the nine-step order** of the page protocol, each as a `<ConceptStep>`. `content/mathematics/softmax-cross-entropy.mdx` is a good model:

   ```mdx
   ## GELU

   <ConceptStep>What is it?</ConceptStep>

   <ConceptCard concept="ops.gelu_forward" split />

   <ConceptStep>Why does the model need it?</ConceptStep>

   <ConceptWhy concept="ops.gelu_forward" />

   <ConceptStep>Mathematics</ConceptStep>

   <MathBlock id="gelu.forward" />

   <ConceptStep>What does each symbol mean?</ConceptStep>

   <Symbols concept="ops.gelu_forward" />

   <ConceptStep>Tiny numerical example</ConceptStep>

   <NumericExample concept="ops.gelu_forward" kind="illustrative" />

   <ConceptStep>Tensor shapes</ConceptStep>

   <TensorShape concept="ops.gelu_forward" shape="gelu" />

   <ConceptStep>Actual implementation</ConceptStep>

   <CodeReference symbol="ops.gelu_forward" />

   <ConceptStep>Code and math connection</ConceptStep>

   <EquationCode concept="ops.gelu_forward" />

   <ConceptStep>Where is it used?</ConceptStep>

   <Trace concept="ops.gelu_forward" />
   ```

4. **Do not write folds by hand.** `lib/remark-collapse.ts` folds the long steps on any page that uses `<ConceptStep>`, and leaves the source alone.
5. **Use only what `components/mdx.tsx` offers.** It has no way to pass raw LaTeX or source text, so a page cannot copy an equation or a code listing.
6. **Check.** Run `bun run build`, then `bun run verify:site` and `bun run verify:content`. A new component that holds numbers needs a label: `Heatmap`, `BarChart`, `ProbabilityBars`, `LineChart` and `ShapeFlow` require `kind`, and `actual` also requires `provenance`.

## Tests

`bun run test` runs Vitest. It holds the golden comparisons, and `tests/interactive.test.ts`, which mounts every interactive component in a simulated DOM, drives it with real events and compares what it shows with the real Python results. Each lab there has a test that fails if its formula is wrong.

`tests/e2e/` holds Playwright specs. By the rule in `CLAUDE.md` the site is desktop-only and these specs are not being run or updated, so they do not describe the site as it is today. `verify:site` covers links, metadata and static accessibility rules without a browser. It cannot see colour, layout or anything rendered only by JavaScript.

## Design system

`/design` is a hidden style guide that renders every element in both themes. It is not in the navigation and is marked `noindex`. Components take colours from the tokens in `styles/tokens.css` only; `tests/tokens.test.ts` enforces that and checks WCAG AA contrast.

The hidden page `/design/components/` renders every documentation component from the sample concept `ops.softmax_forward`. Its source is `content/_samples/softmax-forward.mdx`. Folders in `content/` starting with `_` get no route of their own.

## Concepts and links

Pages reference data by id, for example `<MathBlock id="softmax.shift" />`. Equations, symbols, shapes and examples live in `data/concepts/*.yaml`. `data/links/*.yaml` ties each equation to code by `file`, `symbol` and `match`, with `regex` and `occurrence` when needed.

An example declares its input and which local variables to capture. `sync:examples` runs the real function and stores the values. Every `<NumericExample>` needs a `kind` that agrees with the data.

## Math and code navigation

`MathBlock` puts `data-eq` on an equation. `CodeFrame` and the Code Explorer put `data-eqs` on each linked line, computed on the server from `data/links` by `lib/links.ts`. One client component, `MathCodeNavigator`, is mounted in the docs layout and reads those attributes, so no page or component wires anything up.

- A click, or Show code, pins the equation in the address as `?eq=attention-scale`. Opening that address restores it. Hover and keyboard focus show the same highlight without moving the page. Escape clears it.
- A click on a linked line, or Enter on its gutter mark, opens a popover. `/math-code/<slug>.json`, one small static file per equation, holds the rendered equation, so no KaTeX JavaScript reaches the browser.
- To link a formula written inside another function, or one of several matches in a function, add a record in `data/links` with `role: inline` and `occurrence` if needed. `Implemented in`, `Used by`, the inline equivalent note and both directions of navigation follow from that record.

`verify:links` fails if a link is not reachable both ways: its concept must have a `<Trace>` on a page, its file must be in the Code Explorer, and its equation must have a page to open.

## Long pages

`lib/remark-collapse.ts` folds these parts of a page that uses `<ConceptStep>`, with a plain `<details>`: Tiny numerical example, Actual implementation, Code and math connection, every step of a backward section except its first two, and a full code listing outside the steps. The first section with something to fold starts open.

Folded content stays in the page, so find-in-page and printing see it. `lib/reveal.ts` opens a fold when an anchor, a search result or a `?eq=` highlight points inside it, and all of them while printing. `DataTable` scrolls a wide or tall table inside its own container.

## Actual Model Output

`bun run export:actual` reads a checkpoint, read-only, and writes `data/generated/actual/*.json`: sizes, real tokens and merges, the attention weights of one real window for every layer and head, next-token probabilities for one prompt, and losses on a few windows. `--checkpoint PREFIX` chooses another checkpoint. It takes about a minute.

With no checkpoint it writes `actualAvailable: false`, prints how to create one and exits 0. The site then builds with every Actual panel off. It never trains unless given `--train`, which runs the repository's `train.main` with its prefix in `docs-site/.cache/` and refuses anywhere else. At the committed `config.py` that is a long run.

`bun run verify:actual` compares the stored files with a fresh export. Run the export again after a change to the checkpoint, the corpus or the code, because `build` only reads the files and checks the attention weights. `lib/actual.ts` loads and validates them. Illustrative and Actual values live in separate components, and nothing outside `data/generated/actual` feeds an Actual panel.

## Math engine and interactive kit

`lib/math/` is a pure TypeScript copy of the repository's NumPy operations, for interactive pages. Each function names the Python function it mirrors, and `tests/math.golden.test.ts` compares it with the real Python to within 1e-9, using `data/golden/*.json`. No number in an example may be typed into JSX: take inputs from data and values from `lib/math`.

`np.argsort` in `generate.sample_next` is not stable. When the top-p cutoff falls inside a group of equal probabilities, Python may keep different token ids from the engine. The kept count and the probabilities still match, and `applyTopP` sets `tieDependent` so a page can say so.

`rng.choice`, `attention_backward`, `apply_merges` and `count_words` are not mirrored.

`MatrixInput` and `StepPlayer` are controls. The hidden page `/design/interactive/` shows them and every chart.

## Generated data

`scripts/extract_repo.py` reads the repository statically, using only the Python standard library, and never runs its code. It writes `repo-meta.json`, `code-index.json`, `config.json`, `call-graph.json` and `tests-index.json`.

`sha` in `repo-meta.json` is the newest commit that touches anything outside `docs-site/`. Commits that only change `docs-site/` do not move it, so committing the generated data does not make it stale.

`verify:generated` compares every file byte for byte, except for `branch`, `dirty` and `dirtyFiles` in `repo-meta.json`. Those describe the current checkout and are reported as notes.

## When the Python code changes, and deploying

See [MAINTENANCE.md](MAINTENANCE.md).
