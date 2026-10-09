# Fixed artifacts: deployment and ordinary loading work in progress

Initiative: Q54 / P12.4 / C4 scope convergence
Stage: implementation note, unfinished integrated candidate
Author: Codex
Version: v2
Date: 2026-10-09
Status: draft
Related Files: fixed-cuda-artifacts design-request/design-review v2;
  fixed-cuda-artifacts implementation-note v1; systematic-refactor implementation-note v1
Owner Summary: Deployment publication and ordinary runtime loading now have code,
  including immediate return retention and named entry/set adoption checks.
  Factory integration, deletion of the superseded live compiler machinery and
  source/profile closure remain unfinished. This is not an accepted code packet.

## Authority and starting state

The unchanged approved design is v2 at `3e31778766d7ea16ebe6b936dd00b9de0a0af5f0`,
SHA256 `4f7cca3d448f5fd243f2786571193790dbaf2fc7604e755b86a2328e1c4dcc7d`.
Its complete independent report is archived at `dcfa7a7`. This work starts from
`9c71caee0147fbdf67e70fd004c90c4b3db5f0ac` on `implementation/c4-supervised-waits`.
The deployment changes were already uncommitted when the loading work started;
they are preserved and included in this unfinished candidate. No new design,
execution approval or review verdict is claimed.

The preceding conversational turn only inspected status and answered the owner's
complexity concern; it made no implementation progress. The following work does.

## Actual changes

- The toolchain inventory producer moved to `tools/optical_cuda_toolchain_inventory.py`.
  The old compiler temporarily imports it; that alias must disappear with the old
  compiler, so runtime/build separation is not yet complete.
- `tools/optical_cuda_deploy.py` stages the fixed three recipes, invokes standard
  user services, checks manager main-exit facts and observed cgroup closure,
  validates the complete bundle before atomic publication, and preserves failed
  staging jobs. It has not been executed. The shared service argv and isolated
  build environment live in `optical_cuda_build_recipe.py`; setup disables user cfg.
- `tools/optical_cuda_artifacts.py` validates the canonical manifest, actual
  sources/files/footer cross-links and selected platform. It derives ordinary v5
  environment rows for one artifact root, its manifest and three runtime extensions.
  This is a cold launcher helper, not a compiler inventory scan inside a worker.
- `_ExternalSourceLoader` now uses the indexed native helpers only for the fixed
  artifact root. Create retains the returned module in its existing registry before
  either context exits or an observation/adoption can fail. File-call completion
  still belongs to the existing loader. Runtime initialization performs three literal
  imports after its runtime binding under the original guard3.
- A private session registry issues `CudaExtensionSet` only after ordinary file
  completion and actual named guardian adoption of all entries and the set. Resolution
  rechecks the original session/controller/runtime/admission/module/spec/delegate and
  entry identities. It is not yet consumed by the factories.
- Runtime custody v3 adds the six file-call/native-scope joins and one exact set
  binding event. The collector includes nested entry `token` references. CUDA v3
  runtime initialization cannot complete without binding. Existing v1/v2 meanings
  are retained; the historical synthetic native-scope fixture explicitly stays v2.
- The existing sacrificial native observer wraps the extracted artifact create/exec
  helpers; it does not invent observations from requested device arguments. None of
  these new native paths has executed in this candidate.

## Executed checks and their limits

Local interpreter: `C:/Python314/python.exe -B`, Windows. The command below exited0:

```text
python -B -m pytest -q
  tests/unit/optics/test_cuda_artifact_runtime.py
  tests/unit/optics/test_runtime_custody.py
  tests/unit/optics/test_cuda_artifact_manifest.py
  tests/unit/optics/test_cuda_artifact_build_status.py
  tests/unit/optics/test_cuda_extension_recipes.py::test_original_kernel_bytes_and_entry_abis_are_preserved
  tests/unit/optics/test_cuda_extension_recipes.py::test_cold_recipe_import_has_no_backend_or_compatibility_side_effect
```

Observed **90 passed in 9.11s**. The runtime tests check retention across context
exit/journal failures, exact context arguments, and entry/set substitution against
real CPU guardian adoptions. Their file/native-pair facts are explicitly synthetic;
they do not prove full v3 replay, actual extension creation, stream restoration on
the target or native failure handling. The manifest tests use dummy extension bytes
which are never imported. They do not prove real build or manager behavior.

The observer and historical codec selection exited0 with **27 passed, 110 skipped
in 2.20s**:

```text
python -B -m pytest -q
  tests/unit/optics/test_native_scope_observer.py
  tests/unit/optics/test_runtime_custody_actual_records.py
  tests/unit/optics/test_runtime_native_scope_codec.py
```

Skipped POSIX/retained-record cases are not renewed historical replay evidence.
The new runtime test first failed collection because it imported `_Replay` rather
than the actual `_RuntimeReplay`; correcting that test import produced19 passes,
then the combined90 above. No production relaxation was made for that mistake.

Earlier deployment work passed48 selected CPU checks in4.04s. Its first v5
composition fixture failed because literal Linux paths were interpreted on Windows;
the CPU fixture rebases its paths and the new root uses `as_posix()`. Production
path validation was not weakened. The later deployment dependency-profile module
check and resolved interpreter inventory path have only source/syntax inspection.

AST parsing of the five modified loading/replay/observer files passed. `git diff
--check` passed (Git reports normal Windows CRLF conversion warnings). A direct
current bootstrap source-policy check **rejected** with
`ClosedWorkerProtocolError: source-only loader definitions have drifted`.
This confirms the unfinished source closure is not native-executable. The combined
PowerShell inspection then ran `git status` and returned0; that shell status must
not be reported as a passing source-policy check.

Earlier read-only target inspection found manager ControlGroup
`/user.slice/user-1006.slice/user@1006.service` and confirmed installed setuptools
distutils handles `--no-user-cfg`. Remote `rg` was unavailable; a read-only grep
fallback completed. No code was copied to the target and no deployment was run.

## Remaining integrated work

1. Connect the issued set to scene/LBVH/direct-light preparation, with the same LBVH
   module for sizing and factory creation, and private direct-light preparation.
   Complete the ordinary fixed-deployment compatibility resolver without JIT fallback.
2. Delete superseded live worker JIT, setuptools admission, compiler supervision/job
   wiring and obsolete current driver/test branches. Move generic input phase constants
   out of the historical compilation vocabulary. Keep original evidence intact.
3. Update the exact source policy/catalog and runtime launch profile selection. Add
   full v3 ordinary file/native-pair replay and changed dependency closure checks.
   This candidate is deliberately not a runnable replacement yet.
4. Freeze the whole replacement packet and bounded commands for fresh independent
   code preflight before any systemd service, compilation, GPU/native or fault run.
   Do not restart the custom pidfd campaign or request slice acceptance.
5. After actual replacement acceptance, complete remaining Warp preparation,
   timeline/arena, warmups/final graph, C4c3/c4, C4d/B3 and whole-C4 refactor/review.

No service, build, GPU worker, fault campaign, product PREPARE or benchmark ran in
this work. All owned local commands/tests reached terminal status, and no reviewer
was started. C4 remains active and incomplete; no performance or backend-selection
claim follows from these CPU checks.
