---
name: run-tests
description: Run the full test suite and report results. Use after any physics or integrator change.
---

## Context

- Test files: !`find tests/ -name "test_*.py" | sort`
- Recent changes: !`git diff HEAD --stat`

## Task

$ARGUMENTS

## Steps

### Step 1: Run full suite

```bash
python -m pytest tests/ -v
```

### Step 2: If tests/ is empty

If no test files exist yet, report:

```
⚠ No unit tests found in tests/.
Suggested tests to add for Phase 2:
- tests/test_free_fall.py     — analytic free-fall vs ABA (already validated manually)
- tests/test_pendulum.py      — single pendulum energy conservation
- tests/test_contact.py       — contact force direction and magnitude
- tests/test_joint_limits.py  — penalty torque at/beyond limits
```

### Step 3: Report

```
## Test Results

Passed: N / Total
Failed: [list]
Warnings: [list]

Verdict: GREEN / RED
```
