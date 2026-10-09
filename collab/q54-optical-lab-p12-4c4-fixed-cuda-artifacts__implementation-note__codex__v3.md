# Fixed artifacts: factory routing and removal of superseded worker JIT

Initiative: Q54 / P12.4 / C4 scope convergence
Stage: implementation note, unfinished integrated candidate
Author: Codex
Version: v3
Date: 2026-10-09
Status: draft
Related Files: fixed-cuda-artifacts design-request/design-review v2;
  fixed-cuda-artifacts implementation-note v2; systematic-refactor implementation-note v1
Owner Summary: CUDA factories now resolve fixed modules, LBVH query/factory share
  one module, and live worker JIT/setuptools/custom compiler supervision are removed.
  The replacement still needs complete replay/source checks, an executable bounded
  acceptance packet and fresh independent preflight before real execution.

## Scope and authority

Implementation starts from clean `74a7a5b1aa03e30dd9dfe8c907b368333ca5965b`, branch
`implementation/c4-supervised-waits`. This is progress on the unchanged approved
design v2 (`3e31778`, SHA2564f7cca3d), not a new execution or product approval.
No reviewer was started for an incomplete implementation slice.

## Actual replacement and deletion

- Scene, LBVH and direct-light preparation resolve through one fixed recipe route.
  Captured calls require the actual registered CudaExtensionSet and exact runtime
  context; compatibility calls require an explicitly selected, verified deployment.
  The shared class/registry lives in the already captured `native_setup` module,
  allowing factories to check exact issuance without importing another bootstrap.
- LBVH sizing and factory construction use the same resolved module. Direct-light
  now has the private prepared-module constructor guard already used by scene/LBVH.
  Prepared-input factory helpers forward the captured extension set. Actual later
  C2e preparation scheduling and readiness remain unfinished and closed.
- `optics/cuda_extension_loading.py` only selects the fixed authority route.
  Non-captured standard-import machinery is in `cuda_extension_compatibility.py`,
  excluded from the captured source catalog. It verifies file/manifest bytes and
  loader/module/entry identity on repeated preparation, with no JIT fallback/cache.
  A failed native import is not retried as a fresh compatibility load.
- The three factory modules no longer import compiler tooling. Warp-only types and
  imports moved to the actual compatibility calls; native CUDA imports do not pull
  in Warp. Existing compatibility computations and acceleration identity remain.
- Deleted live `cuda_compilation.py`, `setuptools_admission.py` and
  `compiler_supervision.py`, bootstrap compiler roots/direct loader/setuptools
  topology hooks, and compiler-job callbacks from the ordinary render supervisor.
  Removed the superseded compiler and setuptools execution drivers/tests. These
  are deletions of replaced behavior, not proof of replacement native acceptance.
- Archived compilation/job readers and vocabulary moved under
  `optical_worker/historical/`. Their CPU tests use that explicit boundary.
  Generic input phases moved to `runtime_phases.py`. Runtime environment v6 remains
  readable for archived provenance and deployment input, but workers reject it
  before parsing; ordinary v5 parsing does not import historical compilation code.
  Historical profiles/catalogs/reports remain unchanged.
- Source policy v15 removes the setuptools/importlib/closure exceptions and the
  deleted direct loader's construction operations. Catalog v14 enumerates83
  prospective LF release sources, excluding all historical readers/live compiler
  machinery. Existing native policies were not relaxed to pass new source checks.

## Local verification so far

Environment is Windows, `C:/Python314/python.exe -B`; no backend or target job ran.
An initial factory/retention/source selection passed46 tests in7.36s. After removal,
the combined affected CPU selection passed **494, skipped9, in22.82s**, exit0:

```text
python -B -m pytest -q
  tests/unit/optics/test_cuda_extension_loading.py
  tests/unit/optics/test_cuda_extension_recipes.py
  tests/unit/optics/test_cuda_artifact_runtime.py
  tests/unit/optics/test_cuda_artifact_manifest.py
  tests/unit/optics/test_cuda_artifact_build_status.py
  tests/unit/optics/test_cuda_lbvh_native.py
  tests/unit/optics/test_cuda_scene_native.py
  tests/unit/optics/test_cuda_toolchain_inventory.py
  tests/unit/optics/test_runtime_custody.py
  tests/unit/optics/test_native_scope_observer.py
  tests/unit/optics/test_compiler_job.py
  tests/unit/optics/test_setuptools_contract.py
  tests/unit/optics/test_setuptools_replay.py
  tests/unit/optics/test_runtime_environment.py
  tests/unit/optics/test_guarded_wait_core.py
  tests/unit/optics/test_guarded_lifecycle.py
```

Compatibility tests use dummy artifact bytes and a modeled ordinary import;
they never load the dummy files. Factory query tests model Torch and the extension
but exercise the actual factory and sizing implementation. Their claims are
limited to resolution/identity/order, not GPU behavior. Skips are environment
limitations and do not count as native acceptance.

The first combined run had177 passes,9 skips,2 failures and15 setup errors. Its
historical setuptools unit fixture omitted the runtime schema version now read
by the shared v3-aware base parser. Both historical fixture roots now explicitly
declare v2; the production parser was not weakened. The final selection above
then passed. The historical reader itself retains v2 as its frozen input grammar.

Source preparation initially rejected the literal exec enum at a new call site;
the code now uses the existing descriptive `_NATIVE_EXEC_OPERATION`. The ordinary
compatibility loader was separated from captured source, so importlib/tooling
access was not added to captured factory permissions. All83 prospective LF sources
then passed the actual production source validator. Catalog v14 SHA256 is
`c915a489b67641bdbee7ea3b36379ffed79e574461365e396a1f362eae1bbcbe`.
This authoring check validates proposed LF release bytes; it is not an installed
source-custody run. Cold release tests still need a committed candidate.

`git diff --check` found two trailing blank lines after old helper/test removal;
these were removed and the source catalog refreshed. The subsequent check passed.
All local tests/commands completed. No SSH, systemd service, compiler, GPU/native
getter, fault campaign or reviewer was launched in this work.

## Remaining before preflight and parent completion

Run affected cold release/source checks against the committed candidate and fix
actual failures. Complete full v3 file-call/native-pair replay attacks and the
bounded installed-manager/build/loading acceptance commands, including real factory
binding under the approved preparation scope. Verify exact installed dependency
closure and the derived v5 launch selection. Then freeze the whole replacement
packet for fresh independent code preflight before real execution.

Replacement acceptance still does not complete C4. Warp compilation/cache custody,
timeline/arena, provisional graph, warmups/final readiness, C4c3/c4, C4d/B3 and the
whole-C4 refactor/review remain. Product PREPARE, benchmarks/results and backend
selection stay closed. No speed claim is made.
