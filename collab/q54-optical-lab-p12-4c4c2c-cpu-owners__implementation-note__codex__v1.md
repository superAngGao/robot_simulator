# C4c2c1: single production owners for existing CPU materializers

Initiative: q54-optical-lab-p12-4c4c2c-cpu-owners
Stage: implementation-note
Author: codex
Version: v1
Date: 2026-09-30
Status: in_review
Related Files: C4c2 runtime integration design v3 at 674af14; independent approval archived at c969296
Owner Summary: First bounded code slice of approved C2c, not completion of C2c or C4. Move existing definitions without changing algorithms or opening any runtime/result gate.

## Scope

Base: c969296, which only archives design approval of 674af14.

- `optical_worker.workloads` and `optical_worker.assets` are now the sole owners; benchmark modules explicitly re-export the same objects.
- `optics.mjcf_visual` owns the importer, both data types and every transitive importer helper; the original example re-exports them and retains its CLI/render/legacy preview-camera behavior.
- `optics.camera_presets` owns the three Lab camera functions; the tool module explicitly re-exports them.
- Canonical workload import uses production `native_input` and the production MJCF importer. No worker-to-benchmark/example/tool dependency is needed for these CPU definitions.
- The existing asset preflight negative test patches the actual production owner of its test constants. No production check is disabled.
- The test-only sacrificial source list adds the two new modules actually imported by its existing asymmetric workload path. This does not change the product source allowlist or admit a new runtime path.

This is ownership extraction only. Resolved render sequences, canonical camera artifact bytes, v13 gate/row/binding/schema extensions, asset read-byte/TOCTOU custody, final production import allowlist, worker driver adoption and admission remain later C2c/d work. The existing general importer still reads files normally; no new claim of immutable asset-read authority is made here. Historical scene/replay/fixture/case identities and gate files remain unchanged.

## Validation plan and evidence

The new tests cover every moved function/class, not selected public entries:

- exact new/legacy object identity and absence of duplicate legacy definitions;
- pre-extraction AST hashes from `git show c969296:<path>`, with only the exact approved MJCF import module relocation normalized;
- existing frozen non-asset scene/replay/fixed-fixture digests;
- absent Go2 input still fails closed;
- fresh-process import-attempt rejection for benchmark/example/tool/Torch/Warp, negative controls, and 120 CPU orbit-camera calls;
- no example CLI/render entry moved into the production importer.

Local Python 3.14 targeted new ownership/camera tests and lint are run before remote validation. An initial broader invocation also selected a3a runtime-hook tests: 101 passed, 11 failed because local Torch is absent (`cannot import runtime hook path 'torch.empty'`); those tests require the existing robot_sim environment and are not waived.

Exact remote results, real Go2 digest recomputation and process hygiene will be recorded in a separate evidence note before independent code review. No dependency installation, performance result, baseline or product activation is authorized by this slice. CPU AST/data parity does not substitute for the later full C2c resolved-sequence or integrated GPU lifecycle acceptance.

## Remaining scope and cleanup

C2c is incomplete: this slice only provides sole CPU definition owners. C2d/e/f, every R2/S/host path, C4d/B3, product GPU preparation/submission, a3b/result/baseline and C4 parent completion remain closed.

Compatibility imports intentionally remain while existing callers depend on them. P12.4e removes losing-lane/comparison-only aliases and tests once their callers are deleted; reusable Lab/example APIs continue to call selected production definitions. There is no second implementation to maintain.

Each author/reviewer test round checks owned processes and GPU residuals against its own launch ledger. Other projects' H200 jobs are not this run's cleanup targets. No unresolved owned stall may be silently carried into the next GPU round.
