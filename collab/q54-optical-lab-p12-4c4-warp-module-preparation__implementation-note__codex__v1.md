# Warp setup implementation in progress

Initiative: Q54 / P12.4 / C4 remaining render integration
Stage: partial implementation record, not acceptance
Author: Codex
Version: v1
Date: 2026-10-09
Status: draft
Related Files: warp-module-preparation design-request/design-review v2
Owner Summary: Implemented held-source linecache in the existing loader and
  authored the explicit header inventory. The private cache, guarded module
  preparation, ModuleExec set and actual-state failure routing still need
  implementation, then one integrated code review and native acceptance.

Base before implementation: 0356108a958f34c7d17064a92d2cc7a339b07d0b.
Design v2 at524f5d5 is independently approved; its complete approval report
is archived unchanged in design-review v2, SHA256
45a4e117b9722985be1ed951507cd8cf292b473449562be967ac007437008434.

## Changes and evidence

The existing captured loader now seeds linecache from its held source before
executing the module body/decorators. Its normal binding/audit path checks the
exact cache entry, origin and held text, including same-list mutation. A stale
preexisting cache is replaced only at initial execution; later drift rejects.
No additional loader or reflection/native exception was introduced. Updated
release pins use AST policyv16 and component catalogv15 (83 sources), digest
8ca067f6b143b6c0cd2fc5c047a70f55fa92250e723d26f2d3d35a1863413214.
Previous catalog files remain unchanged historical records.

Candidate profile tests/data/runtime_environments/183_warp_go2_author_v4.json
adds47 explicit data-file rows from the actual installed Warp wheel inventory,
without adding module execution permissions. Its canonical SHA256 is
02429134207218e87ca1305e6c89cc7b34948cda41499f1ac7399ac961ffbeeb.
The existing production RuntimeEnvironmentContract, run with remote Python
-I -S -B from the unchanged66240e4 checkout, parsed and verified all1057 listed
files with that expected digest. No Warp/Torch import occurred. Remote copy:
/data2/shared/ang.gao/c4-warp-module-environment-candidate-v1.json.
This is file validation, not compiler-input-closure acceptance or native success.
The initial Windows parse correctly rejected the Linux absolute-path profile;
the actual Linux verification above then completed exit0.

CPU commands use C:/Python314/python.exe -B -m pytest -q -p no:cacheprovider:

- test_native_bootstrap.py -k 'captured_inspect or linecache_drift or actual_source_entry':
  8 passed,76 deselected in23.12s; session89310 exit0. These use the actual cold
  production loader, including inspection before body execution, changed disk
  source, and clear/replaced/mutated cache negatives.
- test_native_bootstrap.py test_native_source_policy.py test_native_sources.py
  test_native_source_custody.py: session41730 exit1, 280 passed,13 skipped,
  27 failed in271.21s. One failure was the old policyv15 assertion;26 custody
  cases stopped at obsolete14-module counts before exercising their scenarios.
  The existing prelude already has15 modules including runtime_phases; setup
  has17. The baseline0356108 source also contains this nine-module prelude.
- Updated the policy assertion to16 and custody tests to exact expected module
  sets, including the audit fault's trigger. The first focused rerun passed the
  policy case but exposed an author-introduced indentation error in the test
  string; this was corrected before further execution.
- test_native_source_custody.py with -x: session75215 exit0, all26 cases passed
  in121.82s. The policy-version case passed in the preceding focused run.
  Thus the27 previous failures are addressed; the other280 passes and13 skips
  are from the broader run, not falsely reported as a fresh all-green rerun.

All named test/copy/probe/verification sessions reached terminal results.
No native compilation, module-load campaign, GPU query, service, install or
remote checkout modification occurred. The captured custody tests use their
existing CPU peers and wait/close paths. No live owned test job remains.

## Next integration work and limits

Implement the approved pre-init private cache/options, guarded optical imports
and three serial module loads, published ModuleExec retention, artifact/set
validation and healthy-IDLE CPU-failure disposition. The existing unconditional
SourceSession.fail wrapper cannot be copied for the new IDLE issuance boundary.
Then review that integrated code and run the required real Go2/dual-device/
failure/hang checks; these source tests are not a separate final milestone.

No code approval or native evidence is claimed here. Whole-C4 still requires
Warp timeline/arena, provisional graph/ten warmups, final readiness/recording/
seal/barrier, full renderer numerical/lifecycle validation, C4c3/c4, C4d/B3 and
systematic refactor with fresh whole review. Product PREPARE, benchmarks,
results/baselines and backend selection stay closed.
