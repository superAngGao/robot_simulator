# Warp preparation: complete actual-IDLE boundary checks

Initiative: Q54 / P12.4 / C4 remaining render integration
Stage: bounded code repair and native failure coverage
Author: Codex
Version: v4
Date: 2026-10-10
Status: draft
Related Files: warp-module-preparation code-review v3; test-evidence v2;
  design-request v2; runtime-integration design-request v3
Owner Summary: The aggregate issuer now rejects compromised scope before a
  resource API can invent a guard failure. Existing native fault selectors cover
  entry and issuance with healthy, unknown and incomplete custody. No controller
  policy, compiler, renderer or runtime abstraction was added.

Base f83edafa8c30d448127d230a5fb670c89da00194. Fresh independent reviewv3 closed
F1/F2 and reproduced F3: unknown scope after final DISARMED reached the actual
issuer's require_adopted call, recorded guard_failed with guard:null and changed
the client to FAILED. Quarantine/zero sync/retained roots already held; the defect
was state/evidence routing. The unchanged full report is archived as code-review
v3, SHA25659f58a531d869f112e936b60c1405aa41a64ecde9e08d57187ee629429e98264.
Its independent363 unit tests and10 historical repro checks passed; the new
issuance counterexample failed while its plain CPU rejection control passed.

The production delta extends the existing issuer precondition to require no
active context/reference and no compromised scope, before cache/owner validation.
It preserves original completed guards and the existing source-session IDLE
failure handler. The entry and aggregate now both reject unknown scope before
mutating resource APIs. Active/pending/failed guards still use their original
failure policy. No lifecycle state machine or emergency protocol is changed.

Repository regression reaches the actual issuer after three completed records,
checks the unknown-scope rejection, and checks incomplete-custody precedence on
a CPU issuance exception. It retains all records, no issued set/checkpoint,
original five fixture completions, IDLE/no failed guard, quarantine/zero sync.
The original independent two-case issuance repro now passes unchanged.

Author command: Python3.14 -B -m pytest -q -p no:cacheprovider with
test_warp_module_preparation.py plus outside-candidate original F2, F3,
test_author_warp_abi_repair.py and test_review_warp_compile_transition_v2.py.
Observed39 passed in10.04s, tool87162 exit0. Source refresh validates all84
prospective LF release sources; only warp_setup's catalog row changes, v15
catalog SHA25668ecaeb15bd365aaf46db99a142da45de2e8393848db944d915ca3d567ad0f16.

The existing native driver now names six IDLE faults: unknown_entry,
unknown_issuance, entry_raise, issuance_raise, incomplete_entry and
incomplete_issuance. Faults act on otherwise valid real input; issuance faults
follow all three real completed compiles and check retained record identities.
Incomplete cases deliberately invalidate stream guardian membership while the
actual stream stays held by the source-session graph, then trigger CPU rejection.
The production failure handler must discover incomplete custody. No native
resource is unloaded or allowed to continue after any case. The runner requires
exact original completions, zero fabricated guard_failed, one emergency sync only
for healthy exact-target CPU failures and full quarantine for unknown/incomplete.
These added selectors are not executed native evidence at this record cut.

Test-evidencev2 separately records predecessor positive/mutation cases and real
F2 entry rejection atf83edaf, matching raw archive and all15 owned PIDs absent.
Fresh independent review and native execution of this revision remain required.
Finish compilation acceptance and continue the parent graph/timeline/arena,
actual warmup/final readiness/barrier, numerical rendering/lifecycle, C4c3/c4,
C4d/B3, systematic whole-C4 refactor and fresh whole review. No product PREPARE,
performance/result/baseline or backend-selection gate is opened.
