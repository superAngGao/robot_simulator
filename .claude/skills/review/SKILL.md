---
name: review
description: Review staged or recent changes for test coverage and code quality. Use before committing new modules or significant changes.
---

## Context

- Staged diff: !`git diff --staged --stat`
- Recent diff: !`git diff HEAD --stat`
- Current branch: !`git branch --show-current`
- Test files: !`find tests/ -name "*.py" ! -name "__init__.py" | sort`

## Task

$ARGUMENTS

## Steps

### Step 1: Identify changed modules

List all `.py` files added or modified in the staged diff (or HEAD diff if nothing staged).

### Step 2: HARD CHECK — Test coverage

For each changed/new module in `physics/`, `rendering/`, `rl_env/`, or `domain_rand/`:

1. Check if a corresponding test file exists in `tests/` (e.g., `physics/contact.py` → `tests/test_contact.py`).
2. If **no test file exists**: flag as `⚠ MISSING TESTS`.
3. If a test file exists: scan it briefly — does it cover the new/changed functionality?

### Step 3: Code quality scan

Read the changed files and check:

- [ ] Physics functions cite a reference (equation, paper, section) in their docstring.
- [ ] No magic numbers without a comment explaining the value.
- [ ] Public API functions/classes have at least a one-line docstring.
- [ ] No dead code (commented-out blocks, unused imports).

Run linter:

```bash
ruff check $(git diff --staged --name-only | grep '\.py$' | tr '\n' ' ')
```

### Step 3.5: MANIFEST.md staleness check

Compare the current changes against `MANIFEST.md`. Flag as `⚠ MANIFEST STALE` if any of:
- A new phase was completed or started
- Architecture changed (new pipeline, new layer, new subsystem)
- Solver matrix changed (new solver added or retired)
- Test count changed significantly (new module with dedicated test file)

If stale, update `MANIFEST.md` before proceeding.

### Step 4: Report

Output a structured report:

```
## Review Report

### Test Coverage
- physics/foo.py  →  tests/test_foo.py  ✅ exists
- physics/bar.py  →  tests/test_bar.py  ⚠ MISSING — create before merging

### Code Quality
- [ issue 1 ]
- [ issue 2 ]

### Verdict
PASS / NEEDS WORK
Recommended action: [commit as-is / add tests first / fix issues first]
```

If verdict is NEEDS WORK, **do not proceed to commit**. Resolve issues first.
