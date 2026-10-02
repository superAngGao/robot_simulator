# C2c4 cleanup-ledger independent CPU/data/documentation review

Initiative: q54-optical-lab-p12-4c4c2c-render-cleanup-ledger
Stage: review
Author: codex
Version: v1
Date: 2026-10-03
Status: accepted
Related Files: cleanup_ledger.json; test_optical_render_migration.py; render-cleanup-ledger implementation-note v1; render-binding-migration review codex v2 C1; runtime-integration design v3 section 10
Owner Summary: Approve the exact three-file cleanup-ledger follow-up at f4dee5d. The prior non-blocking P3 C1 inventory gap is closed. Independent CPU validation passes; no runtime, source-policy or gate change is present. Cleanup execution and all later product gates remain outstanding.

## Identity, revision and verdict

Reviewer session: `/root/c4_c2c4_cleanup_ledger_review_v1`, fresh independent reviewer, not the candidate author. No additional reviewer or worker sessions were created. Author notes and prior pass counts were treated as claims, not validation evidence.

Verdict: **Approve**, solely for this bounded delta.

- Base: `6df7e953d0334e9793d7d135fb2ccd3028f3a40d`.
- Reviewed HEAD: `f4dee5d1f81f110d3f4f16df4f4e5a2f293fdefc`.
- HEAD tree: `d78366ea28a40bc5dcb2d4aaa21639d3dde1f636`.
- Branch: `implementation/c4-supervised-waits`.
- Checkout: `C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design`.
- Initial and final `git status --porcelain` were empty; final HEAD still matched the frozen candidate. No candidate file was edited or deleted.
- Actual delta: exactly three files, 93 additions, no deletions: the cleanup ledger, one added test in `tests/unit/benchmarks/test_optical_render_migration.py`, and the new implementation note.

Read `AGENTS.md`, `collab/README.md`, and `collab/REVIEW_PRINCIPLES.md` in full before task work. Initial PowerShell default decoding garbled Chinese; the principles were reread completely with UTF-8. Inspected the archived review's C1, accepted design v3 cleanup requirements, actual full delta, existing inventory validator, catalog/artifact paths, migration tests, extraction fixture reader and retained regression responsibilities.

## Findings and C1 disposition

No substantive findings, blocking defects, or required evidence gaps for this scope.

The prior P3 C1 is closed by concrete ledger coverage:

1. `canonical_resolved_render_contract` lists both production modules, all four actual versioned JSONL artifacts, and `test_render_parameters.py`. Both `cuda_selected` and `warp_selected` retain a single canonical owner, immutable inputs and selected-backend scalar/provenance regression, while excluding a permanent paired selector or losing-lane adapter. The four artifact paths agree with both the frozen catalog and independent on-disk enumeration.
2. `temporary_render_row_migration_snapshots` lists both v12/v13 snapshots and both consuming test modules. Repository-wide Python search finds actual snapshot reads only in `test_optical_render_migration.py` and `test_native_contract_extraction.py`; neither consumer is omitted.
3. Both terminal branches explicitly delete comparison snapshots/machinery only after equivalent canonical-parameter, closed-gate and moved-body identity regressions are retained. This matters because the migration test contains durable scalar/metadata rejection and descriptive-only roundtrip checks, while extraction tests protect AST/object identity, closed product-source bytes and unrelated CPU invariants. The entry says to remove snapshot reads, not remaining invariant coverage; `collapse` does not mean indiscriminate deletion of these test files.
4. Both entries have `must_resolve_by: P12.4e`. Historical collab review/evidence is explicitly preserved and a live v12 fallback is explicitly prohibited. The ledger remains `open_until_terminal_backend_selection` with `must_complete: true`; no cleanup-complete claim or backend selection is made.
5. Independent comparison confirms every pre-existing ledger entry and all top-level metadata are unchanged. Exact diff scope confirms no production module, gate, row/preimage fixture, product source manifest, protocol, runtime contract, collaboration policy or historical review changed. The note accurately describes a future obligation, not executed cleanup or new activation authority.

## Independent verification

Environment: local Windows PowerShell; `C:/Python314/python.exe`, Python 3.14.0, NumPy 2.5.2, pytest 9.1.1. `find_spec('torch')` and `find_spec('warp')` both returned `None`. No packages, assets or tools were installed. Tests used `-B` and disabled pytest cache; Ruff used `--no-cache`. Results below are fresh executions, not reused author output.

From the candidate checkout:

```text
git status --short
git rev-parse HEAD
git branch --show-current
git diff --stat 6df7e953d0334e9793d7d135fb2ccd3028f3a40d f4dee5d1f81f110d3f4f16df4f4e5a2f293fdefc
git diff --name-only 6df7e953d0334e9793d7d135fb2ccd3028f3a40d f4dee5d1f81f110d3f4f16df4f4e5a2f293fdefc
git diff 6df7e953d0334e9793d7d135fb2ccd3028f3a40d f4dee5d1f81f110d3f4f16df4f4e5a2f293fdefc
rg -n 'render_migration_v1[23]' --glob '*.py'
C:/Python314/python.exe -B -m pytest -q -p no:cacheprovider tests/unit/benchmarks/test_optical_render_migration.py tests/unit/benchmarks/test_optical_benchmark_contracts.py::test_inventory_and_cleanup_ledger_match_current_repository
C:/Python314/python.exe -B -m pytest -q -p no:cacheprovider tests/unit/optics/test_render_parameters.py tests/unit/optics/test_native_contract_extraction.py::test_single_definitions_preserve_all_bodies_and_alias_identities tests/unit/optics/test_native_contract_extraction.py::test_historical_gate_and_closed_product_source_boundary_are_preserved
C:/Python314/python.exe -B -m ruff check --no-cache tests/unit/benchmarks/test_optical_render_migration.py
git diff --check 6df7e953d0334e9793d7d135fb2ccd3028f3a40d f4dee5d1f81f110d3f4f16df4f4e5a2f293fdefc
C:/Python314/python.exe -B C:/Users/AG/新建文件夹/c4_assets/c4-c2c4-cleanup-ledger-probe-v1.py C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design
git rev-parse HEAD 'HEAD^{tree}'
git status --porcelain=v1 --untracked-files=all
git diff --name-status 6df7e953d0334e9793d7d135fb2ccd3028f3a40d f4dee5d1f81f110d3f4f16df4f4e5a2f293fdefc
```

Observed results:

- Claimed targeted suite independently reproduced: **47 passed in 0.65s**.
- Related durable parameter and extraction/closed-source checks: **124 passed in 2.16s**.
- Ruff: `All checks passed!`; diff whitespace check: exit 0, no output.
- Independent scratch probe: all old ledger entries/metadata preserved, all four actual artifacts covered, **17 negative controls rejected**. Controls independently remove each of the eleven new paths, change each of the four per-entry/per-backend actions, and move each deadline to P12.4f. They invoke the actual new test with in-memory JSON substitution, never editing the candidate.
- One auxiliary `git rev-parse HEAD HEAD^{tree}` invocation lacked PowerShell quoting around braces and failed because PowerShell transformed the argument. Quoted rerun above succeeded. This was a reviewer shell/tooling error, not a candidate failure; no candidate mutation occurred.

The scratch probe is retained outside the candidate at `C:/Users/AG/新建文件夹/c4_assets/c4-c2c4-cleanup-ledger-probe-v1.py`. This complete report is likewise outside the frozen candidate for coordinator recording.

## Limitations, residual obligations and closed scope

This review approves only the ledger inventory/test/documentation follow-up and closes C1. It does not reapprove or reopen core C2c4 native implementation; its independent approval remains bound to `4b7bf4cee0d1671e5d60c16c697d37caeec6d8a3` and its own evidence.

No remote, GPU or real-Go2 runtime validation was required or executed for the JSON/test/doc-only delta. The local CPU results are not native execution, GPU resource-lifecycle, source-admission activation, performance or backend-selection evidence. No new benchmark results or baselines were generated.

The new test validates inventory structure/path equality/actions/deadlines, not the eventual semantic quality of a future cleanup implementation. The human-readable retention conditions were independently inspected here; P12.4e must still perform and independently review cleanup with equivalent retained coverage. That future obligation is not a defect in this bounded maintenance change.

All product GPU PREPARE/admission, a3b, results/baselines, C2d/e/f, C4c3/c4, C4d/B3, host payloads, shipping and backend-selection gates remain closed or outstanding. Neither C2 nor C4 is declared complete.

## Final own-process and SSH hygiene

All local commands completed synchronously. The expanded 124-case pytest round was launched through a short Python `Popen` wrapper with a 60-second wait limit. It recorded child PID 5184, parent PID 20264, launch timestamp `2026-10-02T21:46:03.890554+00:00`, and exact test argv; it explicitly waited/reaped child 5184 with exit 0. No timeout fired. The initial short 47-case command and lint/probe invocations completed normally but did not record separate child PID/start ledgers; this operational evidence limitation is stated rather than inventing process provenance.

Final read-only `Get-CimInstance Win32_Process` checked those recorded PIDs and scoped Python/pytest/SSH command lines matching the review probe and the three exercised test modules. Observed **Scoped residual count: 0**. This is bounded process-state evidence, not a claim about all unrelated machine work. No process was killed, no shared resource was reset, and no unrelated session was touched.

The reviewer opened **zero SSH connections**, started **zero remote workers/supervisors or GPU jobs**, and created **zero additional agent sessions**. There is no reviewer-created remote/SSH cleanup obligation or unresolved observed local residual. Final candidate HEAD and clean worktree were rechecked after tests and probe.
