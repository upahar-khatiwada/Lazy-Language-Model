# LLM Developer Docs: project instructions for Claude Code

This file lives at docs-site/CLAUDE.md. It applies to every task in this series.

Project
"LLM Developer Docs": a static documentation website for an existing from-scratch language model written in Python and NumPy (no PyTorch, no autograd; every forward and backward pass is hand-written). Repository: upahar-khatiwada/Lazy-Language-Model-LLM. The website lives in the docs-site/ folder of that repository unless a task says otherwise. Reference document: LLM_study_guide_v2_streamlined.pdf. The repository was last inspected at commit c17aa75; always read the real HEAD and pin to it.

What the documentation is for
A developer or student should be able to move from "What does this LLM do?" to "How does attention work?", "What is the equation?", "Can I calculate it with a tiny example?", "Where is it implemented?", "Show me the actual Python code", "How does the gradient flow through it?" and "How does this fit into the whole model?".

Source of truth, in strict order
1. The actual GitHub code.
2. The mathematics that code actually implements and uses.
3. The reference document.
4. General ML knowledge.
If 3 disagrees with 1, show the discrepancy and follow 1. Never document mathematics that appears only in the PDF, or only because it is commonly taught for Transformers. General knowledge may explain an implemented operation; label it ("In general, ...") and never word it as something this project does.

Hard rules
- Do NOT modify the LLM. Do not edit, reformat, refactor, "fix" or "improve" any file in the repository outside docs-site/ (this includes every .py file, tests/, data/, config.py, .gitignore). If you think you found a bug in the LLM, report it in your summary and leave it alone.
- Every mathematical explanation must trace to code: file, function, and real line numbers from the pinned commit. Never hand-write line numbers; resolve them from generated data.
- Label every numeric example either "Illustrative Example" (hand-picked numbers) or "Actual Model Output" (produced by running the repository). Never mix them. Never present illustrative numbers as learned values. Never say what the model "focuses on" unless an Actual Model Output panel shows it.
- Do not invent equations, files, functions, tensor shapes, hyperparameters, components, optimizers or outputs. If the repository does not provide something, write "not available from the implementation".
- Write each equation once, in structured data, and reuse it through components. Do not copy equations across pages.
- Mark mathematics that the code implements but the PDF does not state as "Derived from implementation".

Design direction
Developer documentation. Three columns (sidebar, content, on-this-page). IBM Plex Sans and IBM Plex Mono, KaTeX for math, light and dark themes, restrained colour, no decorative effects. Readable first.

Architecture in place
Next.js (static export) + TypeScript + MDX + KaTeX + Shiki (build time) + MiniSearch. Hand-written structured data in docs-site/data/concepts and docs-site/data/links. Machine-generated data in docs-site/data/generated (never edit by hand). Page content in docs-site/content. Start every task by reading docs-site/contract/ (inventories, architecture, discrepancies) and the files named in the task.

CLAUDE CODE WORKING RULES
- Run from the repository root. All new files go under docs-site/. Never edit any file outside docs-site/ (all root .py files, tests/, data/, config.py, .gitignore, anything else). If a task seems to need that, stop and ask.
- Read-only access to the LLM code is expected: use Read, Grep and Glob. Python scripts you write under docs-site/scripts/ may import repository modules but must not write outside docs-site/.
- Do NOT run git commit, git push, git add, git stash, git checkout, git reset or any command that changes git state. Never commit unless the user explicitly asks. Reading (git status, git diff, git log, git rev-parse) is fine.
- Do not run training (python train.py) unless the task says so. Running the existing tests with pytest is allowed if it does not write into the repository.
- Use Bash only for bun / bunx, python scripts under docs-site/scripts/, pytest, and read-only inspection. No curl or wget to other sites.
- Track the task with TaskCreate/TaskUpdate (or the todo list): one item per requirement plus a final "verify" item. Mark items done only after the check ran.
- Fresh session each task: the only memory between tasks is the repository, docs-site/CLAUDE.md and docs-site/PROGRESS.md. At the very start of a task, read docs-site/CLAUDE.md and docs-site/PROGRESS.md (and docs-site/contract/ once it exists). At the very end, append an entry to docs-site/PROGRESS.md: task number, date, files created or changed, commands run with results, assumptions, open issues.
- Large files (the PDF, big generated JSON): read in ranges or use a subagent, and never paste them whole into the conversation.
- Never say a check passed unless you ran it in this session and saw it pass. Quote the command and the result line.