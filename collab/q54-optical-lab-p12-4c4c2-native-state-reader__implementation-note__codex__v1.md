# Captured input native-state reader — implementation candidate

Initiative: Q54 / P12.4 / C4c2
Stage: implementation-note
Author: codex
Version: v1
Date: 2026-10-10
Status: in_review
Related Files: native-state-reader design-request/review v1;
worker-intake gap-review v1; native_scope.py; native_scope_acceptance.py
Owner Summary: The production path now owns the input-stage observations that
previously existed only in a test observer. The duplicate test query/state
implementation is removed. CPU verification passes; exact-code preflight and
new real-device verification are still required. Complete C4 remains open.

## Exact candidate and scope

Base: `868225db0ac5794bd7089cd73357b5c920d0c0e9`.
Candidate: `4dcf73d78ef36dbfd40ac3025a228cf9be1b21cd`.
Frozen detached review checkout: `../c4_native_reader_code_review`.
The candidate was clean when frozen. This note records evidence separately.

The approved design is unchanged, committed LF SHA-256
`0b6dab55c496e730219dfbb253b7d9530dff40faa6c90676c71a2b90e0f41ca5`.
Design approval is archived in design-review v1, session
`01a124e2-d69d-72d3-a86c-b59248c84055`; it permits implementation only.

One captured `NativeStateReader` now supplies runtime stream/binding, CUDA
artifact create/exec, input upload/readback and eligible terminal-sync facts.
The source session retains the reader before its queries. Existing guards,
native helpers, custody schema and mandatory artifact joins are preserved.
Returned owners are retained before after-query/append/adoption failure.
CUDA artifact retention remains inside the original indexed contexts.

Shared helper readers are selected only for the corresponding input phase.
Later BVH callers keep their existing raw test diagnostics, using the same
production state query. C0 does not construct a reader. Tests retain fault
injection and diagnostics; they no longer construct successful scope facts.
The shared acceptance checker matches actual input guard occurrences and can
check integrated input coverage separately from sacrificial termination.

Native source policy/catalog release v18 contains 91 source files. The catalog
SHA-256 is `c056a01fc60ad951ac60e579c5b0cd8d047d9732e4d74faa3c5e8c8ec7aa6cb8`.
The reader's exact LF source pin confines the FFI exception; bootstrap pins and
the frozen integrated contract were updated together. C0 policy is unchanged.

## Executed author verification

Windows `C:/Python314/python.exe`; all commands below are terminal, exit 0.
These are CPU models/source checks, not real-device evidence.

- Affected scope/source/artifact suite: **352 passed, 4 skipped in 207.57s**.
  Files: `test_native_scope_observer.py`, `test_native_scope_integration.py`,
  `test_native_source_policy.py`, `test_native_sources.py`,
  `test_native_source_custody.py`, `test_guarded_native_readback.py`,
  `test_cuda_artifact_replay.py`, `test_cuda_artifact_runtime.py`, all under
  `tests/unit/optics/`; invoked with `-q --disable-warnings --maxfail=2`.
  `ROBOT_SIM_WARP_CONTEXT_SOURCE` pointed to the previously hash-checked
  `../c4_assets/worker_intake_gap_scratch_v1/pinned_warp_context.py`.
  Skips concern required real Go2 assets and actual POSIX release checks.
- Integrated worker/native manifest/C0 protocol/C0 worker regressions:
  **103 passed, 9 skipped in 67.69s**. Files:
  `tests/unit/optics/test_integrated_worker_lifecycle.py`,
  `tests/unit/optics/test_native_manifest.py`,
  `tests/unit/benchmarks/test_closed_worker_protocol.py`,
  `tests/unit/benchmarks/test_optical_closed_worker.py`; same pytest flags.
  Linux-only process cases remain required in their native environment.
- New integration tests also ran separately: **18 passed in 1.46s**.
  Actual guarded readback and artifact helper orchestration use modeled native
  leaves. They cover empty calls, omission/copied reader, later shared readback,
  event retention and real loader retention through exit/query/append failures.
- Standalone synthetic CPU coverage probe:
  `../c4_assets/native_scope_coverage_probe.py`. Later same-identity BVH helper
  does not demand an input receipt; missing or transplanted input receipt
  rejects. This does not establish native physical state or codec acceptance.
- Ruff checks passed for the new reader, source policy, integration tests and
  source-policy tests. Changed support files compile; `git diff --check` passed.
  No claim of a repository-wide lint pass is made.
- Release authoring validated all 91 normalized prospective LF sources before
  commit; the source regression suite then checked the committed release.

## Review and remaining work

Built-in fresh reviewer creation returned `agent thread limit reached`.
The owner-authorized equivalent fresh CLI session was started instead:
`01a12501-54b8-7220-a419-ead734cb7b14`, request SHA-256
`be04f54c083741dc307adcadb5c5576d9f03048fc8a38c2448e7eff8a1481b8d`.
It receives the frozen candidate and requirement/evidence packet without the
implementation conversation. Its result is pending at this note's creation.

No new reader/native library execution or GPU workload has run. Independent
exact-code/native-call preflight must precede bounded real setup/framed-worker
validation. Original CUDA/Warp campaign evidence retains its original meaning.
Final-sample physical-scope and numerical verification, full workload/failure
matrices, fixed-work/preallocated trace repair and whole independent C2f/C4
acceptance remain open. Formal performance, activation, result/baseline,
backend selection, R2/S/host delivery, a3b and shipping gates remain closed.
