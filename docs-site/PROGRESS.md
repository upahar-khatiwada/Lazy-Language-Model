# PROGRESS

Append one entry per finished task. Newest at the bottom.

## Template
### Task NN: title (YYYY-MM-DD)
- Files created or changed:
- Commands run and results:
- Assumptions:
- Open issues:
- Decisions needed from the human:

### Task 01: Project Discovery & Documentation Contract (2026-11-05)
- Files created or changed:
  - Created `docs-site/contract/`: `REPO_INVENTORY.md`, `ARCHITECTURE.md`, `HYPERPARAMETERS.md`, `MATH_INVENTORY.md`, `PDF_VS_CODE.md`, `DISCREPANCIES.md`.
  - Changed `docs-site/PROGRESS.md` (this entry).
  - Nothing outside `docs-site/` was changed. No application code was written.
- Commands run and results:
  - `git rev-parse HEAD`: `c17aa75afd287dc623009168f50c4544fc1d4a32`, committed 2026-11-03T16:04:10+05:45.
  - `git status --short` at start: `M data/corpus.csv`, `?? docs-site/`. The corpus change was already there before this task.
  - `.venv/Scripts/python.exe -B -m pytest tests -q -p no:cacheprovider --basetemp=<scratchpad>`: `67 passed in 4.11s`; rerun after writing: `67 passed in 3.21s`. File snapshot before and after the first run: no differences.
  - Scratchpad validators, not kept in the repository: inventory against `ast`: `inventory symbols checked 130 bad 0`; math rows: `math rows 61 bad 0`; PDF: `equations 89 unique 89 complete True`, `technique rows 70`; config: `hyper rows 29 bad 0`.
  - Static call graph from `train.main` and `talk.main`: 53 of 58 symbols reachable; unreached: `ops.linear_forward`, `transformer.count_params`, `gradcheck.numerical_grad`, `gradcheck.check_params`, `gradcheck.relative_error`.
  - Read-only checkpoint and corpus analysis: 38 tensors, 63,871 parameters, V=575. Working-tree corpus reproduces the checkpoint's `itos`; committed corpus gives V=564. `pos_emb` rows 16-31 ratio 0.9078093812857 solves to 17,590 total steps.
  - Training was not run.
- Assumptions:
  - The PDF was read at `C:\Users\upahar\Downloads\LLM_study_guide_v2_streamlined.pdf`, as `pdftotext` text rather than rendered pages (the Read tool failed: `pdftoppm` not installed). Superscripts were reconstructed from context.
  - "Used" comes from a static, name-based call graph. It was not traced at run time, since that would need training.
  - The defaults in `contract/DISCREPANCIES.md` section 1 apply until the human decides otherwise.
- Open issues:
  - The existing checkpoint does not match the pinned commit: different corpus and, by inference, different training settings. It is also git-ignored. See `contract/DISCREPANCIES.md` section 7.
  - Possible LLM issues are listed, not fixed, in `contract/DISCREPANCIES.md` section 8. The main ones: `config.ln_eps` and `config.head_dim` are never read.
  - Plain `pytest` was not tried; only `python -m pytest` was run.
- Decisions needed from the human: none. D1 to D5 from `contract/DISCREPANCIES.md` section 1 were decided by the human on 2026-11-05, accepting all defaults. These decisions replace the "NEEDS HUMAN DECISION" status in that file:
  - D1: No training-history output on the site. Any value derived from the checkpoint must disclose its source. The corpus will not be committed and the model will not be retrained for now.
  - D2: Yes. Test-only mathematics (`gradcheck.py`, `linear_forward`, `count_params`) may be documented, only in a labelled testing section.
  - D3: Yes. PDF worked examples may be reused, labelled Illustrative Example, only where the code agrees.
  - D4: No. Later tasks do not reopen the PDF; they rely on `contract/PDF_VS_CODE.md`.
  - D5: Links use the real remote name, `Lazy-Language-Model-LLM-`.
- Status: Task 01 finished.

### Task 02: App Scaffold & Repository Extraction (2026-11-05)
- Files created or changed (all under `docs-site/`):
  - App: `package.json`, `bun.lock`, `next.config.ts`, `tsconfig.json`, `eslint.config.mjs`, `.prettierrc`, `.prettierignore`, `vitest.config.mts`, `playwright.config.ts`, `.gitignore`, `.gitattributes`, `README.md`, `app/layout.tsx`, `app/page.tsx`, `.gitkeep` in `components/`, `content/`, `data/concepts/`, `data/links/`, `data/golden/`.
  - Extraction: `scripts/extract_repo.py`, `scripts/verify_generated.py`, `scripts/verify_content.ts` (stub), `lib/repo.ts`, `data/generated/{repo-meta,code-index,config,call-graph,tests-index}.json`.
  - Tests: `tests/smoke.test.ts`, `tests/repo.test.ts`, `tests/e2e/home.spec.ts`.
  - This entry.
- Changed outside the repository, with the human's approval: Node.js v26.13.0 installed with `winget install OpenJS.NodeJS.LTS`; Playwright Chromium downloaded to the user profile by `bunx playwright install chromium`.
- Commands run and results (Node v26.13.0, Bun 1.3.13, after deleting `node_modules`, `.next` and `out`):
  - `bun install --frozen-lockfile`: `358 packages installed`, exit 0.
  - `bun run lint`: `All matched files use Prettier code style!`, exit 0.
  - `bun run typecheck`: exit 0.
  - `bun run test`: `Test Files 2 passed (2)`, `Tests 16 passed (16)`, exit 0.
  - `bun run build`: `(Static) prerendered as static content`, exit 0; produced `out/index.html`.
  - `bun run test:e2e`: `1 passed (3.8s)`, exit 0.
  - `python -B -m http.server 3111 --bind 127.0.0.1 --directory out` and `curl`: `HTTP 200`; body contains `<h1>LLM Developer Docs</h1>`.
  - `$env:NEXT_PUBLIC_BASE_PATH = "/x"; bun run build` in PowerShell: exit 0, 9 asset URLs carry `/x/_next/`. Rebuild without it: exit 0, 0 carry it.
  - `bun run sync:repo` twice, then `sha256sum` and `cmp`: `IDENTICAL (5 files)`.
  - `bun run verify:generated`: `OK: 5 generated files match a fresh extraction.`, exit 0. After editing each of the 5 generated files in turn: `FAILED: ...`, `differs: <file>`, exit 1 each time. With an extra file: `unexpected: extra.json`, exit 1. Restored: OK, and hashes equal the pre-edit hashes.
  - `sed -n 'START,ENDp' <file> | tr -d '\r'` against stored source, all `MATCH`: `ops.linear_forward` (7-8), `ops.layer_norm_backward` (49-57), `transformer.ModelConfig` (7-15), `optim.Adam.step` (13-30), `tests.test_tokenizer.test_round_trip` (48-50), `transformer.forward` (155-168).
  - Generated facts checked by `tests/repo.test.ts`: `usedBy` of `ops.linear_forward` is `production: []`, `tests: [tests.test_ops.test_linear_worked_example]`. 58 root symbols, 53 reachable, and the same 5 unreached as `contract/REPO_INVENTORY.md` section 5. 64 test functions, 67 cases. `ln_eps` and `head_dim` have empty `readBy`.
  - `git status --short`: `M data/corpus.csv` (already there), `?? docs-site/`. `git diff --stat -- '*.py'`: empty. No `package-lock.json`, `yarn.lock` or `pnpm-lock.yaml`.
  - Failures seen on the way:
    - Without Node, on Bun's `node` stand-in, `bun run build` failed with `TypeError: Expected CommonJS module to have a function wrapper ... this is a bug in Bun`. A one-off `next build --webpack` failed the same way. Installing Node fixed it.
    - TypeScript 7.0.2 failed lint: `typescript-eslint does not support TS 7.0`.
    - `NEXT_PUBLIC_BASE_PATH=/x bun run build` in Git Bash failed: `basePath has to start with a /, found "X:/"`. Git Bash converts the value; PowerShell and `MSYS2_ENV_CONV_EXCL` both work.
- Assumptions and approved deviations:
  - Next runs on Node, as the task says. The human first chose Bun only, then approved installing Node after the build failed.
  - TypeScript is pinned to 6.0.3, not the latest 7.0.2, because `typescript-eslint` (used by `eslint-config-next`) does not support 7.0 yet.
  - `sha` in `repo-meta.json` is the newest commit outside `docs-site/`, not literally HEAD, because a commit cannot contain its own SHA. Today both are `c17aa75`.
  - `verify:generated` reports but does not compare `branch`, `dirty` and `dirtyFiles`. Editing only those three fields is not detected.
  - `dirty` ignores `docs-site/`. It is `true` because of `data/corpus.csv`.
  - A leading comment is an unbroken comment block directly above a symbol.
  - Additions beyond the task's field list: `readBy` and `section` in `config.json`, `reference` edges and `unresolved` lists in `call-graph.json`, `dirtyFiles` in `repo-meta.json`.
  - TaskCreate and a todo tool were not available in this session; work was tracked against the approved plan.
- Open issues:
  - The call graph is static. Calls on receivers of unknown type stay in `unresolved` (for example `rng.choice`, or a callback parameter such as `loss_fn`) and create no `usedBy` entry. No name-based guessing is done.
  - `sha` staying put after a docs-only commit was not tested end to end, because this task may not commit. Check `bun run verify:generated` after the first commit.
  - `extract_repo.py` stops with an error on nested definitions and relative imports, rather than dropping them. The repository has neither today.
  - `bun run start` serves `out/` without a base path, so a base-path build cannot be browsed correctly with it.
  - A terminal or editor opened before the Node install needs a restart to find `node`.
  - `verify:content` is a stub. The repository does not record the number of the prompt that implements it.
  - No new LLM issues found. `contract/DISCREPANCIES.md` section 8 still stands.
- Decisions needed from the human: none.
- Status: Task 02 finished. Nothing was committed.
