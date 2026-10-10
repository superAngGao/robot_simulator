# Shared native query and upload failure boundaries

Initiative: Q54 / P12.4 / C4c2
Stage: implementation-note
Author: codex
Version: v3
Date: 2026-10-10
Status: in_review
Related Files: native-state-reader design-request/design-review v1;
code-preflight-review v1/v2; binding-fixture review v1
Owner Summary: The two v2 counterexamples are repaired at the existing shared
entry points. Committed affected CPU regressions pass381. Fresh independent
v3 preflight is running; no new GPU execution or speed comparison is claimed.

## Candidate and review disposition

Exact candidate: `df1a46ae5039fe8a25718c7ec60f52fa7c60c549`.
Prior reviewed candidate: `5d804f73b5ae00e89fc18455080b1e339d20dade`.
Frozen clean detached local checkout: `../c4_native_reader_code_review_v3`.
Clean primary-server checkout:
`/data2/shared/ang.gao/robot_sim_worker_df1a46a`.
Prior local and remote candidates remain preserved.

Preflight v2, session `01a1251b-f2f2-77f1-9b5c-2c815bedf4db`, returned Request
changes. Full report is archived unchanged; LF SHA-256
`9dea7939c12c86e1f42375172476408eee00db854ba0dcae87e3fdd82c4f78ac`.
It independently closed the original13 counterexamples, passed439 CPU tests
and checked91 source entries and16 unchanged installed dependency hashes.
Its four remaining failed assertions expose R2-followup and R3, not GPU faults.

The separate test-fixture delta at `eab674c6f9c7eb9161a4f731db51e69352561634`
received independent Approve from session
`01a12521-79bf-7e33-bbc8-523ba178ba88`; full report SHA-256
`2ff1bb4b6554126165bc06bfe4a93a7a7b82e001a8fb529a1b556e33e58d8c26`.
That reviewer reproduced the base fixture failure and passed all34 changed-file
tests, preserving16 test functions and71 assertions. Approval is test-only.
Both completed review CLI/launcher pairs returned exit0 and were checked absent.

## Consolidated repair

R3: `_state` keeps its existing admission and one query algorithm. Its body now
retains/poisons on every exception through the existing `_fail` authority.
Snapshot and diagnostic callers therefore share the same failure boundary.
Ordered raw prefixes and the original exception survive; caught getter failure
cannot permit another query, COMPLETE, extra sync or ordinary release.
No additional collector, native API, wait, ticket, registry or guardian exists.

R2-followup: upload-context consumption still burns its registration first.
Once that registration identifies the exact context, all field/adoption/source
checks run under its retained original controller authority. A substituted
`context.lifecycle` cannot redirect poisoning. Rejected context/source fields
fail the original armed operation before native scope/allocation/copy/query.
Restoring fields or constructing a replacement cannot revive that guard.
Unregistered/consumed objects have no newly invented authority; C0 behavior
continues through its original separate lifecycle.

New tests exercise both lanes and all five registered context fields. Actual
`_NativeScopeProbe._state` tests cover CUresult failure and a Python exception
from the Warp stream getter in normal and original terminal synchronization.
Actual later shared-helper wrappers cover successful raw diagnostics with no
input receipts, before-call query failure and after-return query failure with
the actual modeled return retained. These are CPU models, not GPU state proof.

Reader LF SHA-256:
`3a67a897834c7c118d9cac850ea1f3f68a8f0686daf318b0a0715d0ddd3b40f7`.
Catalog v18 LF SHA-256:
`644f4077e0e2420c0b742b4e0068a669b0ce35f939e6659c5fc9eeeb06d897f0`.
The author release-byte validator passed all91 source entries. Native query
symbols/ABI/library admission and approved design are unchanged.

## Executed verification and pending work

Windows `C:/Python314/python.exe`, pinned Warp source supplied through
`ROBOT_SIM_WARP_CONTEXT_SOURCE`:

- Final committed suite: `test_native_source_policy.py`,
  `test_native_scope_observer.py`, `test_native_scope_integration.py`,
  `test_native_upload_core.py`, `test_guarded_native_readback.py`,
  `test_guarded_runtime_binding.py`: **381 passed in19.94s**, no skips, exit0.
- Original v2 `test_independent_identity.py`, preserved unchanged in reviewer
  scratch and run against the repair: **17 passed in10.05s**. This includes
  all13 older cases and all4 newly failing v2 cases. No repro assertion relaxed.
- Earlier affected run passed203; after adding actual facade/helper coverage,
  the observer file passed37, and the final four getter cases passed4. These
  overlap the final381 and are not additional distinct coverage.
- Ruff passes reader and integration test; observer passes with only its
  pre-existing E501 lines excluded. Initial reader formatting/import failures
  were corrected before commit. `git diff --check` passes.
- Exact remote df1a46a, existing Python3.12 `robot_sim`, frozen Go2 assets:
  `test_native_sources.py::test_real_go2_cpu_materialization_has_no_unlisted_imports`
  and `::test_actual_posix_checkout_catalog_matches_reviewed_git_bytes`:
  **4 passed in17.57s**, no skips, exit0. Both lanes' CPU materialization covers
  both Go2 resolutions; no backend import or GPU execution. Driver/pytest
  PIDs2512201/2512219 were checked absent; checkout stayed clean. Local test
  handles are terminal and the scoped command scan found no author pytest.

Fresh independent exact-revision v3 preflight session:
`01a12529-1730-7533-a977-8fccfc754ade`.
Launcher/reviewer PIDs47388/37324, started2026-10-10T09:33:32Z.
Request SHA-256:
`b87159fe4904a18a94458cf70d842b2729b58decb5ab08d3917a12c592068179`.
It receives the original reports/repros, exact candidate and limits without the
implementation conversation. It must explicitly decide whether this revision
can proceed to bounded native validation. No verdict is presumed.

No author native worker/supervisor or GPU job was launched. Full C2f/C4 still
requires real both-lane input/failure validation, final physical/numerical
coverage including real Go2 and first/last/wrap indices, preallocated bounded
sample/ticket/trace work, fixed acquire counts, transport/ACK/cleanup/hang
matrix, bounded observed close and whole independent review. C4c3/c4 and
C4d/B3 remain ahead. Formal performance, activation, result/baseline, backend
selection, R2/S/host delivery, a3b and shipping gates remain closed.
