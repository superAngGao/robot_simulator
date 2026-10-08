# Runtime custody implementation work in progress

Initiative: q54-optical-lab-p12-4c4c2d-runtime-custody
Stage: implementation and local CPU integration, incomplete
Author: codex, implementation coordinator
Version: v1
Date: 2026-10-08
Status: draft
Related Files: implementation-plan v3; gap-review v3; native_manifest.py;
runtime_custody.py; native_bootstrap.py; native_contract.py; fixed runtime driver
Owner Summary: The approved aggregate now has an initial implementation and
actual CPU import/guard integration. This is a work checkpoint, not code
approval, complete aggregate validation or whole-C2d3 acceptance.

## Boundary and approved prerequisite

Design candidate `ab24a7fe6619362875596fd8c3694c0756b5ef4c` received fresh independent
approval in reviewer session `/root/runtime_custody_schema_gap_v3`. The complete
unchanged report is archived at `c915c56`; original file SHA256 is
`8dc371c01d92ff591a02e90a84f0bcd16b36f0a213eae5301c3630255abbeb9a`.
The implementation starts from that evidence-only commit. Source catalog moves
to v12 and the source AST policy to v13; earlier catalogs remain unchanged.

This closes the already-approved missing descriptive import/runtime/storage/
lifecycle aggregate. It does not add native permission, execution readiness,
GPU queries or another upload path. C4 remains the continuous objective.

## Implemented so far

- Move/re-export the two existing manifest classes and codec through the
  stdlib-only native_manifest module. Preserve generic constructor behavior;
  the new from_payload methods restrict descriptive C2d3 fixed/replay parsing.
  Four retained real CPU manifests preserve their exact payload bytes/digests.
- Return actual held storage registry entries from the existing audit without
  a second physical getter traversal. The CUDA/Warp canonical dtype strings
  are shared literal metadata; the pinned remote Warp strings were checked by
  a CPU-only import, without init or a GPU call.
- Add an append-only SourceSession-held journal, exact-object tokens and frozen
  guardian adoption matches. Observe external/file/component occurrences,
  returned owners, binding consumption, prepared storage, validation and actual
  DISARMED phase completion. The parsed record grants no live authority.
- Cross-check the aggregate against the existing strict component parsers and
  existing lifecycle replay, independent source/interpreter/configuration/case
  inputs, and the unchanged custody artifact. Current and prior immutable cuts
  are distinct; broken current tails are not rewritten into passing histories.
- Add a separate cold -I -S -B launcher preflight using the existing bootstrap
  capture and alias AST check. The fixed sacrificial driver loads its upstream
  file, writes record bytes before the selection sidecar, and preserves raw
  failure diagnostics without calling normal auditing getters after failure.
- Source-only C2d2 custody remains available. C2d3 external configuration now
  requires configure_runtime_custody immediately after source binding and before
  setup imports. This allows the required independently validated baseline
  without making an environment a prerequisite for source-only C2d2 tests.

## Local checks completed during implementation

Environment: Windows, C:/Python314/python.exe, pytest plugin autoload disabled,
`-B -m pytest -q -p no:cacheprovider`, unique basetemps in sibling c4_assets.
These checks are bounded and partially overlapping; do not sum them as a gate.

- Manifest, existing native authority and selected extraction checks:
  74 passed, 4 deselected. The four exclusions were backend-import-dependent
  extraction checks, not reported as successful. All four retained manifest
  byte comparisons passed with no NumPy/Torch/Warp imports.
- Initial aggregate baseline/guard/selection checks: 21 passed. After adding
  actual ordered atomic-write fault windows: 23 passed. A misplaced test-tail
  insertion was corrected before that latter run; no failed run was hidden.
- First importer/native-global/binding run after observation hooks:
  187 failed, 57 passed, 3 skipped, 2 deselected. The first counterexample was
  the old _Owners test double lacking guard history. It now uses the existing
  real CPU lifecycle/Peer, preserving actual importlib bodies and native doubles.
  One readback ordering fixture also needed its returned manifest/report fields.
  Fresh importer result: 118 passed, 3 skipped. Fresh native-global/binding
  result: 126 passed, 2 actual-materialization cases deliberately deselected.
- Cold source custody and source policy: 194 passed in 124.24s.
- Four success cold import cases first verified aggregate issuance. The next
  full cold import/failure plus initial aggregate tests: 43 passed in 230.31s.
- Added assertions on actual terminal current-cut selection across the 20 cold
  import/disabled cases: 20 passed in 207.79s.
- Five cold native-loader-double failures: 5 passed in 66.16s. Actual return
  before adoption and failure after committed pending observation retain a
  valid current cut. Admission before call completion, failure after call
  completion and an interrupted observer commit select the exact previous
  immutable record. Actual objects/errors remain held and release is denied.

Scratch basetemps: runtime_custody_manifest_tests_v1,
runtime_custody_parser_tests_v1/v2, runtime_custody_observer_tests_v1/v2/v3,
runtime_custody_observer_first_failure_v1, runtime_custody_source_tests_v1,
runtime_custody_cold_parser_v1/v2, runtime_custody_cold_selection_v1,
runtime_custody_native_failure_v1. Results above are actual tool executions,
not cached historical GPU observations. All those pytest processes finished.

The only remote action during this implementation checkpoint was read-only:
primary183 remained clean detached a3d750b, and its existing interpreter printed
the Warp float32/int32/int64 descriptor strings. No new-schema GPU run, remote
source mutation, reset, package change or process signal occurred.

## Remaining work before acceptance

The implementation is not frozen for independent review. Finish the complete
cross-case/canonical-source, stream/storage/guardian, component-transplant,
guard/order and attachment mutation tests, including actual runtime/storage
observations. Exercise POSIX driver persistence, partial admission/observer and
all specified terminal/ACK/EOF/device failures. Correct any integration findings
without weakening the approved schema or existing component parsers.

Then run fresh new-schema both-lane asymmetric/replay/both-real-Go2 executions
and the full required native failure/hang/foreign/UNKNOWN/dual-GPU matrix,
preserving exact processes, source/profile identities and raw attempts. Obtain
a fresh independent complete review and fix/re-review its findings. Historical
v0 GPU evidence is not relabeled as execution of this implementation.

C2e/f, C4c3/c4, C4d/B3, product GPU preparation, a3b, result/baseline, host,
R2 and shipping remain closed. No performance or backend-choice conclusion
follows from these CPU checks; C2d3 and C4 are still unaccepted.
