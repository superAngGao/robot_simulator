# Native reader admission and lifetime repair

Initiative: Q54 / P12.4 / C4c2
Stage: implementation-note
Author: codex
Version: v2
Date: 2026-10-10
Status: in_review
Related Files: native-state-reader design-request/design-review v1;
native-state-reader code-preflight-review v1; implementation-note v1
Owner Summary: R1/R2 have one consolidated repair using the existing lifecycle
and guardian ownership. CPU regressions pass; a fresh independent preflight is
running. No new native execution or performance result is claimed.

Candidate: `5d804f73b5ae00e89fc18455080b1e339d20dade`.
Previous reviewed candidate: `4dcf73d78ef36dbfd40ac3025a228cf9be1b21cd`.
The intervening commits only archive implementation/test/review evidence.
Original reader implementation base: `868225db0ac5794bd7089cd73357b5c920d0c0e9`.
The approved design remains unchanged, LF SHA-256
`0b6dab55c496e730219dfbb253b7d9530dff40faa6c90676c71a2b90e0f41ca5`.

## Repair and limits

R1: reader admission now checks existing lifecycle state before any getter.
Normal work requires the existing setup segment; emergency queries require
the original active target synchronization with retained custody and release
not authorized. Terminal failure, completed/failed synchronization, cleanup,
ordinary release and terminal intent cannot use a still-live emergency ticket
as query permission. Rejections poison through the existing authority.

R2: the controller checks its source link against the original source already
held by its guardian. Upload consumption validates the retained context source;
source-owned upload/readback entries require their input phase. Shared later
BVH helpers retain their separate phase behavior. The existing reader record
also retains original stream/binding identities; reader and binding must be
actual guardian owners. Caught admission/identity rejection stays failed.
No additional registry, guardian, synchronization, ticket or deadline is added.
The guardian owner scan is a setup admission check, not per-sample work.

The test fault hook no longer mutates the production reader's stream before
the real stream-return binding. Historical upload AST comparison now removes
only the bounded observation annotations while still comparing the original
native allocation/copy/adoption body. New CPU cases reject missing/substituted
source, wrong phase, copied reader/binding (including rewritten record), and
queries at actual terminal boundaries before the modeled getter. They also
check failed retries after catching and restoring corrupted fields.

Reader LF SHA-256:
`58f473ac75dc3031493689ca2df0be5752184292c4c175658de16c7150c260c1`.
All 91 prospective LF source entries passed the production AST validator.
Catalog v18 LF SHA-256:
`c73965d822269b82a48547ad301c2faf42b0d88e69ae2c4d920b4c5f84f6a86a`.

## Author verification

Windows `C:/Python314/python.exe`; pinned Warp context CPU model set through
`ROBOT_SIM_WARP_CONTEXT_SOURCE` to the existing exact-source evidence file.

- `test_native_scope_observer.py`, `test_native_scope_integration.py`,
  `test_native_upload_core.py`, `test_guarded_native_readback.py`:
  **155 passed in 9.39s**, no skips, exit 0. This is CPU model evidence.
- Ruff passed on the changed reader/integration test; changed remaining
  production modules and test facade compiled successfully. No whole-repo
  formatting claim is made.
- The affected source/lifecycle suite stopped at **347 passed, 4 skipped,
  2 failed in 207.92s** (`--maxfail=2`). The first four files, source policy,
  inventory, captured custody and guarded lifecycle, completed. The two failures
  were the first runtime-binding tests: their old unit source fixture omitted
  the controller source link. The first failure was independently reproduced
  unchanged at the previous frozen 4dcf73d candidate (1 failed in 0.61s).
  This group is not reported as passing. The integrated-worker file had not
  run when pytest stopped.

The author branch additionally repairs that binding-only CPU fixture at
`eab674c6f9c7eb9161a4f731db51e69352561634`, outside
the frozen preflight candidate: original source linkage/custody, actual reader
construction/adoption and the extra reader owner count. Native query/snapshot
leaves are explicitly omitted and emit no successful native-scope facts; this
fixture has no captured environment/input inventory and does not claim custody
replay proof. Its original binding identity/retention/failure assertions remain.
This test-only delta is not part of the pending exact-revision preflight and
requires its own review coverage. Follow-up execution of
`test_guarded_runtime_binding.py` and `test_integrated_worker_lifecycle.py`:
**82 passed in 68.06s**, no skips, exit 0. Together with the completed first
four files above, all selected CPU cases now have passing executions (the four
environment/platform skips remain skips); the initially failing run remains
recorded. This is not a claim of native correctness or whole-suite acceptance.

No native reader or framed GPU worker has run at this candidate. The previous
CUDA failure/Warp two-sample evidence is unchanged and is not acceptance.

## Fresh review and deployment

Frozen clean detached local checkout: `../c4_native_reader_code_review_v2`.
Built-in fresh agent creation again returned `agent thread limit reached`.
The owner-authorized equivalent fresh CLI session was therefore started:
`01a1251b-f2f2-77f1-9b5c-2c815bedf4db`. It does not inherit implementation
conversation and cannot edit the candidate. Request SHA-256:
`b8002ba934e9e254aa1ccc11b6d18eaeb804e7f91a8d9c203a071c9d3b304a28`.
Launcher/reviewer PIDs: 20732/42964; review is ongoing, no verdict yet.
Author pytest PIDs 43112 and 44364 and their direct descendants were checked
absent after terminal tool results. Both frozen review checkouts remain clean.
No author-owned native worker/supervisor was launched this round. The only
ongoing owned activity is the explicitly tracked independent review session.
The packet supplies the full v1 findings and original counterexamples and asks
for explicit exact-revision permission for bounded native validation only.

Primary host `tileops-dev` has a new clean detached checkout at
`/data2/shared/ang.gao/robot_sim_worker_5d804f7`, exact candidate HEAD verified.
The existing 4dcf73d checkout was verified clean and preserved. Transfer used
an incremental Git bundle; only Git/stdlib ran during this deployment.

After preflight approval, the existing framed-worker campaign can run both
lanes against this exact checkout. Real input scope/failure evidence, complete
final-sample physical and numerical proof, fixed-work trace/storage changes,
full C2f failure matrix and whole independent review remain required. Complete
C4, formal performance, product activation, result/baseline, backend selection,
R2/S/host delivery, a3b and shipping remain closed. This repair is not parent
acceptance or a speed comparison.
