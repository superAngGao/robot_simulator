# Committed source repair and alias-fixture regression

Initiative: q54-optical-lab-p12-4c4c2d-array-adoption-join
Stage: CPU evidence and bounded test-fixture repair, pending independent review
Author: codex, implementation coordinator
Version: v1
Date: 2026-10-09
Status: in_review
Related Files: array-adoption-join implementation-note v1/v2; runtime-adoption whole-code-review v3; tests/unit/optics/test_interpreter_alias.py
Owner Summary: The exact f1ca45a release now passes current-source capture,
but broader regression exposed19 stale alias fixtures. Preserve those failures
and connect the fixtures to the existing actual custody/guard path. No
production-code weakening or whole-C2d3 acceptance follows.

## Exact committed source validation

Code f1ca45aa0e6b178804828acedb2ad2539712a5ef is frozen before these checks.
Primary CPU tree robot_sim_c2d3_array_join_wip is clean at that commit;
the running GPU tree stays cleanfc86182. No GPU call/compiler/install occurs.

array_join_committed_source_probe_v1.py verifies all78 Git LF blobs equal the
actual Linux files and pass the production policy with catalog digests. The
current lifecycle passes; oldfc86182 lifecycle bytes reject under the new exact
pin. Actual capture/unchanged/loaded-module checks pass for both lanes:72Warp
sources and77CUDA sources. No NumPy/Torch/Warp import. PID3451610/parent3451607
are absent afterward. Its raw JSON is retained in sibling c4_assets.
An earlier inline probe reached the intended old-byte rejection but caught
ValueError instead of ClosedWorkerProtocolError, yielding exit1; the corrected
saved probe catches the actual specific exception and returns exit0. This is a
probe error, not a candidate failure or native execution.

Current policy digest7200096c15ceeb0d2b655a330464c3aa1c91264723a02df78990be2e960d18f6;
policy source4bc0fa13656dad549a29b46a754016000059eb5836ec33a45fab194336b7629c;
catalog raw63966da541cac1153e36254e8c09ae2a1136d38232745d631fa4a79c44dab6d3.
Only the already-reviewed regex owner's byte pin and dependent digests changed.

## Complete CPU results, including failures

Primary Python3.12.13, plugin autoload/cache disabled, actual retained Go2
assets, original v1 custody archive and pinned Warp context source configured:

- Expanded complete28-suite foundation: **1030passed/19failed/1skip,632.27s**.
  It includes closed-worker protocol, guarded waits/lifecycle, arena contracts,
  native source/byte policy/bootstrap/custody/real cold entry, import failure,
  alias/generated stdlib, aggregate/observer/readback and affected native input
  contracts. The sole skip tests unsupported-platform behavior covered locally
  on Windows. All19 failures are test_interpreter_alias's unbound controller
  prerequisite; no other suite failed. This is still a failed combined run.
- Separately isolated complete cold admission/global/replacement trio:
  **262passed/20.76s**, no skips. It is not collected with the CUDA arena test
  that imports real Torch, preserving the production cold-start assertion.
- Local committed5-suite source/policy/bootstrap/alias/custody-failure run:
  **292passed/19failed/13platform skips,295.19s**. Every failure is the same
  alias test family. Default child stderr encoding masks the actual exception
  as UnicodeDecodeError. Saved raw child bytes and decoded diagnostic show
  `source session has no exact custody binding` from the real observation
  append, before the intended alias mutation. PYTHONUTF8 on the parent does
  not alter isolated -I children's settings. No production verifier is changed
  to accommodate encoding or platform.

Archive locations: local c4_assets and primary /data2/shared/ang.gao.

| Archive | Bytes | SHA256 | Scoped residuals |
| --- | --- | --- | --- |
| c2d3-f1ca45a-foundation-cpu-v4.tar.gz | 42279171 | 27b72038eeb6a3bd8b9a4fcfe690573f9904564ed3705c5e1a1fd326b584a71e |302 observed owned process identities absent from OS/GPU|
| c2d3-f1ca45a-cold-importer-cpu-v2.tar.gz |60381|e95dfb4c56f015bf6878d268978112751afe9df9b60bd906a1241c5e400ebf1e|runner3445608/pytest3445624 absent from OS/GPU|

Foundation runner3445609/pytest3445626 exited with pytest1. Procfs polling
captures observed live descendants and may miss very short-lived children;
counts are not exhaustive process attestation. Logs retain all failures.

## Bounded alias test repair

The old fixture installs/verifies a source-only session, then imports and audits
multiprocessing without binding custody. The integrated observer now appends
alias facts through the exact production controller. Add that prerequisite in
the alias test's local wrapper over the unchanged actual cold driver: verify,
socketpair, bind_guarded_custody, existing guarded_peer.Peer, then a real setup
segment around the original assertion body. No fake observer/controller,
second protocol, new production callback or parser relaxation is added.

All original identity/publication/typed-fact mutation and poison assertions
remain. A marker is set only after those assertions complete; only then may
the wrapper catch the existing WorkerSetupProtocolError raised when an already
poisoned guard refuses normal completion. Unexpected body/assertion failures
still fail. The wrapper checks zero GPU leaves/sync, no backend import, actual
normal completion on the one healthy case, direct quarantine/no new ARM on
poisoned cases, and closes/joins the actual peer in finally.

Healthy CPU-only cancellation uses one terminal ARM to supervise release/exit,
as guarded_lifecycle.terminate already implements; it is not emergency GPU
synchronization. The initial wrapper wrongly expected zero terminal ARM for
this healthy case, producing31pass/1fail in155.12s. The corrected assertion
distinguishes healthy CPU cancellation from poisoned direct quarantine; the
healthy case rerun passes1/31deselected in7.95s. Both logs remain. The full
repaired Linux32-case suite is still pending at this commit, and this test-only
delta needs fresh independent review. Production equals f1ca45a exactly.

The original edba exchange/historical replay archive remains separately
attributed in implementation-notev2. The unchangedfc86182 complete85 GPU
matrix and approved12 terminal run are still pending; partial counts are not
passes. C2d3/C4 and every later compilation/readiness/optical/result/shipping
gate remain closed.
