---
name: commit
description: Stage, lint, test, and commit changes following robot_simulator conventions. Use when ready to commit working code.
---

## Context

- Status: !`git status --short`
- Diff stat: !`git diff --stat`
- Current branch: !`git branch --show-current`
- Last commit: !`git log -1 --oneline`

## Task

$ARGUMENTS

## Steps

Execute in order. **Do NOT skip a HARD GATE.**

### Step 1: Stage files

Stage specific files only — never `git add .` or `git add -A`.

```bash
git add <file1> <file2> ...
```

### Step 2: HARD GATE — Lint & format (pre-commit)

```bash
pre-commit run --files <staged files>
```

If pre-commit auto-fixes files: re-stage them, then re-run this gate.
If exit code != 0 after fixes: **STOP**. Fix reported issues before continuing.

### Step 3: HARD GATE — Tests

```bash
python -m pytest tests/ -x -q
```

If any test fails: **STOP**. Fix before continuing.
If `tests/` is empty (no test files yet): skip with a note in the commit message.

### Step 4: Commit

Commit message format: `type: short description` (conventional commits, lowercase).

Valid types: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`, `perf`

```bash
git commit -m "type: short description"
```

Examples:
- `feat: add warp kernel for ABA forward pass`
- `fix: correct gravity sign in ABA root initialization`
- `test: add free-fall accuracy unit test`
- `docs: update PROGRESS.md for Phase 2 completion`

### Step 5: Report

Output:
- `BRANCH: <branch-name>`
- `COMMIT: <full commit hash + message>`
- `FILES: <comma-separated list of committed files>`
