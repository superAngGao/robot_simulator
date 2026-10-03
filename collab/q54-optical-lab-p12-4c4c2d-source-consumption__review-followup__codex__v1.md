# C2d0 exact bootstrap and core identity repair

Initiative: q54-optical-lab-p12-4c4c2d-source-consumption
Stage: review-followup
Author: codex
Version: v1
Date: 2026-10-03
Status: in_review
Related Files: source-consumption design-request v1; code-review v1;
optical_worker/native_bootstrap.py; tests/unit/optics/test_native_bootstrap.py
Owner Summary: Address independent F1/F2 without changing trust or activation.
Fresh independent re-review is required; no implementation approval is claimed.

Original base: `29b2055061d38211546774f510e70545701bc0e9`.
Rejected first candidate: `0fae0a4c6df4f5527196009ce994aee3f0b7385d`.
The complete fresh-session report is archived verbatim as code-review v1,
SHA-256 `9486ae73be138940fd90bc393879b6ea332a49f500494caa1933843a26486dbe`.
Both findings are P2; passing regressions did not prove the missing identities.

## Repair

- F1: each source session now strongly retains the trusted initial bootstrap
  module and its loader/spec/package identities. Every live check requires the
  exact `sys.modules['__main__']` owner plus canonical name/file and frozen
  metadata. This does not claim that the interpreter's initial script execution
  was performed or retroactively authorized by the captured-source loader.
- F2: actual core audits now compare spec name/parent and `__package__` against
  the literal package/module layout, in addition to module, loader, origin and
  package-path checks. Failure poisons the session; repeat verification rejects.
- Tests add independent initializer/leaf spec-name and package drift; bootstrap
  deletion, path-only impersonation, full copied-dictionary impersonation and
  name/file/package/loader/spec replacement. They inspect failed state and repeat
  rejection, not only an exception. Existing reload/re-exec/cache tests remain.
- Repin the complete bootstrap source and derived native policy/catalog digests.
  Loader/finder class pins, C0 policy/protocol/files and every activation gate
  remain unchanged. No new product callback or fault selector is introduced.

Bootstrap source SHA-256 after repair:
`18e7bf22e88fa71dc150ddfe85151cc05498514784d85ac59571aab0fee0b10a`.

## Author validation before revised freeze

Local CPython3.14, explicit LF release fixture:

```text
C:/Python314/python.exe -m pytest tests/unit/optics/test_native_bootstrap.py tests/unit/optics/test_native_source_policy.py -q
219 passed, 2 skipped in 95.04s
```

The two skips are actual POSIX checkout entry checks; they must pass remotely.
Observed actual exit0, local pytest PID33632 and its sampled child23996 absent
afterward; no scoped Python test/worker residual remained. Ruff changed modules
and bootstrap tests, plus diff checks passed. Remote revised validation is still
required and will be recorded separately, not inferred from the first candidate.

For historical context, first-candidate author remote six-file regression was
311passed/no skips/118.42s on robot_sim3.12 at the exact clean detached checkout.
Timeout/pytest PIDs2180034/2180059 and sampled injected-stall child2181777 exited;
scoped residual and GPU compute queries were empty. That result did not close
F1/F2. First local exploratory broad regression had25 Torch-hook import failures
because Torch is not installed locally, plus two Windows timeout-output test
failures subsequently repaired. No dependency was installed or contract relaxed.

## Scope

C2d0 is the CPU-only source-entry bootstrap, held-byte loader/registry, narrow
native AST v3/catalog v2, descriptive CPU report, regression and cleanup record.
It does not admit GPU imports/preparation, worker IPC, guarded lifecycle, arena
execution, R2/S/host delivery, C4 completion, a3b or benchmark/baseline results.
C2d integrated lifecycle, C2e/f real-device acceptance and later C4/B3 reviews
remain required. Every gate remains closed as before.
