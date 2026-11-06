# Maintaining the site

Run every command from `docs-site/`. Setup is in [README.md](README.md).

## When the LLM code changes

Any commit that touches a file outside `docs-site/` moves the commit the site is pinned to. Line numbers, permalinks and generated numbers then need refreshing. `bun run verify:content` prints a warning when the pin is behind:

```text
warning: data/generated is stale: pinned to <old>, the repository is at <new>.
```

Work through these in order. The commands that run Python need NumPy and pandas.

### 1. Regenerate

```sh
bun run sync:repo      # data/generated/: symbols, source, line numbers, config, call graph, tests
bun run sync:golden    # data/golden/: results of the real functions, used to test lib/math
bun run sync:examples  # data/golden/examples.json: the worked examples
bun run sync:shapes    # data/golden/shapes.json: tensor shapes, at a tiny config and at config.py
bun run sync:derived   # data/derived/: init_params results, and contract snapshots
```

If the checkpoint, the corpus or the model code changed, and you have the corpus and a checkpoint:

```sh
bun run export:actual  # data/generated/actual/: the only source of Actual Model Output
```

Without a checkpoint it writes `actualAvailable: false` and the site turns its Actual panels off. It never trains unless you pass `--train`.

### 2. Verify, and fix what fails

```sh
bun run verify:generated
bun run verify:golden
bun run verify:examples
bun run verify:shapes
bun run verify:derived
bun run verify:actual
bun run verify:links
bun run verify:content
```

| What fails                                        | What it means                                     | What to do                                                                              |
| ------------------------------------------------- | ------------------------------------------------- | --------------------------------------------------------------------------------------- |
| `verify:links`: `match "..." not found`           | The code line a link points at changed            | Update `match` in `data/links/`. Then read the page: the equation may need to change    |
| `verify:links`: `is ambiguous: N matches`         | The match now appears more than once              | Add `occurrence`, or a longer `match`                                                   |
| `verify:links`: `symbol does not exist`           | A function was renamed or removed                 | Update the concept and link files, and any `symbol=` or `<CodeStep>` on a page          |
| `verify:content`: `not referenced by any concept` | A new public function has no documentation        | Add a concept, as in the README                                                         |
| `verify:content`: a documented shape disagrees    | A tensor shape changed                            | Update the shape in `data/concepts/`                                                    |
| `verify:content`: a forbidden word on a page      | A page mentions a technique the code does not use | Remove it. If the code now uses it, update `contract/PDF_VS_CODE.md` and `sync:derived` |
| `bun run test`: a golden comparison               | The Python changed, and `lib/math` did not follow | Change the mirror in `lib/math/` to do what the Python now does. Do not edit the JSON   |

Never edit `data/generated/`, `data/golden/` or `data/derived/` by hand, and never weaken a test to make it pass. When the code and a page disagree, the code is right: change the page. When the code and the PDF disagree, follow the code and record the difference in `contract/DISCREPANCIES.md`, then run `sync:derived`.

If a change adds or alters mathematics, also update its concept, including the labelled tiny example, and look at the page. The checks prove that numbers and lines match the code. They do not prove the sentences around them are still true.

### 3. Rebuild and check everything

```sh
bun run lint
bun run typecheck
bun run test
bun run build
bun run verify:site
```

`bun run build` runs `verify:links` and `sync:search` itself. `verify:site` reads the built `out/`, so it must run after the build.

The `sync` and `verify` commands that run Python, and `verify:content`, are not part of `build`. CI should run them, because a change to a repository function's numbers will otherwise build without complaint.

## Deploying to GitHub Pages

The site is a static export, so it needs only a host for the `out/` folder. On GitHub Pages it is served from a subfolder named after the repository, so it must be built with a base path:

```sh
NEXT_PUBLIC_BASE_PATH=/Lazy-Language-Model-LLM- bun run build
NEXT_PUBLIC_BASE_PATH=/Lazy-Language-Model-LLM- bun run verify:site
```

In PowerShell, set the variable first: `$env:NEXT_PUBLIC_BASE_PATH = "/Lazy-Language-Model-LLM-"`. Git Bash on Windows rewrites a value starting with `/` into a Windows path, which fails the build. There, add `MSYS2_ENV_CONV_EXCL=NEXT_PUBLIC_BASE_PATH` before the command.

`bun run start` serves `out/` with no base path, so it cannot show a base-path build correctly.

To publish on each push, save this workflow as `.github/workflows/docs.yml` at the repository root. It is here as text because it lives outside `docs-site/`. In the repository's settings, under Pages, set the source to GitHub Actions.

```yaml
name: Deploy docs

on:
  push:
    branches: [main]
    paths: ["docs-site/**", "*.py"]
  workflow_dispatch:

permissions:
  contents: read
  pages: write
  id-token: write

concurrency:
  group: pages
  cancel-in-progress: true

jobs:
  build:
    runs-on: ubuntu-latest
    defaults:
      run:
        working-directory: docs-site
    env:
      NEXT_PUBLIC_BASE_PATH: /${{ github.event.repository.name }}
    steps:
      - uses: actions/checkout@v4
        with:
          # verify:content and the generated data compare against git history.
          fetch-depth: 0
      - uses: oven-sh/setup-bun@v2
      - uses: actions/setup-node@v4
        with:
          node-version: lts/*
      - uses: actions/setup-python@v5
        with:
          python-version: "3.x"
      - run: pip install numpy pandas
      - run: bun install --frozen-lockfile
      - run: bun run verify:generated
      - run: bun run verify:golden
      - run: bun run verify:content
      - run: bun run test
      - run: bun run build
      - run: bun run verify:site
      - uses: actions/upload-pages-artifact@v3
        with:
          path: docs-site/out

  deploy:
    needs: build
    runs-on: ubuntu-latest
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    steps:
      - id: deployment
        uses: actions/deploy-pages@v4
```

Notes on this workflow:

- It has not been run. It is a starting point, and the action versions should be checked when you add it.
- `verify:actual` is left out on purpose, because CI has no corpus or checkpoint. The stored `data/generated/actual/` files are what get published.
- Publishing with this workflow does not run Jekyll, so the `_next` folder is served as it is. If you publish from a branch instead, add an empty `.nojekyll` file to `out/`, or GitHub Pages will drop every folder starting with an underscore.
- `out/404.html` is what GitHub Pages serves for an unknown address.
- GitHub Pages compresses text with gzip, which matters here: the Attention page's HTML is about 1.3 MB as a file and about 136 KB when compressed. Measure performance on a server that compresses, not on `bun run start`.
- The site sets no absolute URL, so link previews carry a title and a description but no canonical address and no image. There is no card image, because none is available from the repository.

## Before committing

- `data/generated/actual/` holds a few sentences and short windows of `data/corpus.csv`, which is git-ignored at the repository root. Decide whether those excerpts should be public.
- Check that nothing outside `docs-site/` changed by accident: `git status --short -- ':(exclude)docs-site'`.
