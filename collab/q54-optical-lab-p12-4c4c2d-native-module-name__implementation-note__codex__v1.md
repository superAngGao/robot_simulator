# C2d3 exact native-name implementation and remaining publication failure

Initiative: q54-optical-lab-p12-4c4c2d-runtime-adoption
Stage: implementation-note
Author: codex
Version: v1
Date: 2026-10-08
Status: implemented
Related Files: native-module-name design-request/design-review/gap-scan v1; native-module-name implementation-gap-scan v1; runtime_environment.py; native_bootstrap.py; native source policy v10/catalog v9
Owner Summary: Bounded implementation is WIP, not C2d3 code-approved. The independent scan found no additional bounded implementation defect. Go2 passes the old native-name boundary but fails at an unlisted publication during final materialization audit. No successful guarded Go2 input, upload/readback, baseline or product activation is claimed.

## Authority, code and independent assessment

Design b05c1e1, exact digest
b0eb29a1d03e54606be7149c20e5a037f0bbafe5f59e3f522f9526afda62dadd,
was independently approved by /root/c4_c2d3_native_module_name_design_review_v1.
Its verbatim report SHA is
ac9e28ff7ea8925c388534de02defe68e22afd370dbcdbbaf981b5916e835739;
a4cc1a6 is the evidence archive, not implementation approval.

Implementation revisions:

- b11ff9e: one production intrinsic-name relation, create-return retention then
  raw-name check, full initialized exec-entry checks, post-exec/audit checks,
  exact delegate identity/path/name, negative tests, policy v10/catalog v9 and
  additional fixed-driver native metadata diagnostics. No dependency table change.
- 787da2e: fixes the omitted cold-bootstrap catalog selector and adds an AST
  consistency test against the production selector. This was an author defect,
  not an environment failure; the first remote failures are retained below.
- 672a247: caches a single pure immutable name index by complete canonical
  environment bytes. Physical files, modules, loader identities and liveness
  are not cached. Changed bytes and post-cache physical-file mutation reject.
- a50b227: separates exact local loader binding checks from full environment
  liveness traversal, avoiding a new whole-graph traversal per audited module.
  Audit still checks environment/files and then each actual module; exec still
  checks liveness before and after the native call. Tests count that distinction.

The sole name policy is in runtime_environment.py. Import/spec identity remains
qualified; the intrinsic string grants no alias or import. Delegate identity
uses `is`: an added negative test exposed that standard loader copies can compare
equal, so value-equal tuples were insufficient. Native return is retained before
fallible validation; failures remain sticky and retain the actual owner/error.

Independent session /root/c4_c2d3_native_name_implementation_gap_v1 examined
the exact base a4cc1a6 through a50b227. Full unchanged report SHA:
57fc60b1d72b73b57dd98845b0e58fda4cc5fb3242744d5903fd8f6685d73435.
It found no additional substantive defect in this bounded delta, independently
ran 339 CPU tests and 16 extra identity/sticky counterexamples, and inspected v7
failure artifacts. This was expressly a GAP SCAN, not code approval of C2d3 or
a CPU-subset substitute for the mandatory real matrix. See its command-error
record and limitations rather than treating all reviewer invocations as passes.

Final inspected implementation HEAD a50b227d38ac33e7e56109fb5b6807e485c98be4;
catalog v9 raw SHA c5a10ad522472d3e1733515833569c7e570e7d62a284f7d097edc68072ef8779.
Pins use staged LF Git bytes. Earlier catalogs and evidence remain unchanged.

## Tests, corrections and exact revision boundaries

Local CPython3.14, Windows:

- Early contract fixture missed explicit parent declarations: 1 failed,
  160 passed, 3 skips. Corrected the CPU fixture, not production validation.
- Added copied-delegate negative exposed tuple equality: 1 failed, 166 passed,
  3 skips. The implementation now compares exact held object identity.
- b11ff9e semantics: 167 passed, 3 Linux-condition skips; policy/source suite
  separately 204 passed, 4 platform skips. The latter did not cover cold entry.
- 787da2e policy consistency: 168 passed. After metadata cache: 168 passed,
  3 platform skips in the two contract/importer files.
- a50b227 new intrinsic-focused checks: 15 passed, 105 deselected. Ruff and
  git diff --check passed. Counts overlap; they are not added together.

Primary183 robot_sim CPython3.12.13, H200:

| Revision | Invocation | Observed result |
| --- | --- | --- |
| b11ff9e | nine-file CPU regression | 83 failed, 409 passed, 3 skipped / 42.26s; cold selector still v8 |
| 787da2e | same nine-file CPU regression | 493 passed, 3 skipped / 205.97s |
| 787da2e | cold Warp GPU suite | 6 passed / 198.78s |
| 787da2e | C0 protocol/setup regression | 30 passed / 13.30s |
| 672a247 | nine-file CPU regression | 494 passed, 3 skipped / 206.36s |
| a50b227 | five-file contract/importer/policy/source/driver regression | 382 passed, 2 skipped / 43.10s |

Nine-file set in tests/unit/optics: test_runtime_environment,
test_runtime_import_admission, test_native_bootstrap, test_generated_stdlib,
test_runtime_import_cold, test_runtime_import_failure_cold,
test_native_source_policy, test_native_sources, test_native_runtime_driver_process.
Five-file set is runtime_environment, runtime_import_admission,
native_source_policy, native_sources, native_runtime_driver_process.
The three remote skips are two unconfigured real-Go2 cases and the
non3.12/platform branch. Skips are not Go2 success. C0 set and cold Warp test
match the generated-stdlib implementation note; no test installs dependencies.

All remote pytest commands used -B, PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 and
-p no:cacheprovider. Outer bounds were 300s for nine-file, 120s for five-file/C0,
and 240s for GPU. Supervisor deadlines inside GPU tests stayed 30s/1000ms ACK.
The two intentional terminal hangs were killed/reaped by that supervisor, not
by the outer pytest bound. Some correctness runs overlapped; none is a paired
performance or memory measurement. Earlier revision totals do not certify a50.

GPU artifacts at /data2/shared/ang.gao/c2d3-787da2e-cold-warp-proof-v1,
test_actual_cold_canonical_warN/custody_artifact.json:

| N | SHA-256 |
| --- | --- |
| 0 | bdb9ff9d38573df453dc13b83c1064a0767d28e5bf51b7070db7de5b5abe596f |
| 1 | b5d798b40ad174912d39fa37c72d1ca8157ae2b8d1112e468d8bdaecbdcd407c |
| 2 | ecbfb501ace593cdfa30193458cddd893605ee1139e1ae765fdfffa16f9d9c7d |
| 3 | c88bbfbaafe0fd1026ddec76ed02cae2cc254b99693d6ea638227fce6bec665d |
| 4 | 46c84d09297ab22c80c8f5687798ebe1223c2b35fcf84f0c19ac91b3390d449a |
| 5 | 8e6caeaef25b79169256c8ecf5e6cb670c0f2976e7de31a13349decf4a48b886 |

## Guarded Go2 attempts: all remain failed

Unchanged frozen Go2 Warp table, real frozen 960x640 asset, target1. All use
120000ms operation /1000ms ACK; no deadline increase or in-flight renewal.
Outside invokers run_c2d3_go2_guarded_author_vN.py call the existing fixed
supervised driver, not a second importer/uploader. Their files and artifact
directories live under /data2/shared/ang.gao (local sources in c4_assets).

| N / revision | Parent / supervisor / worker | Actual terminal outcome |
| --- | --- | --- |
| 4 / b11ff9e | 3632424 / 3632438 / 3632442 | code74 before custody/ARM; catalog selector mismatch |
| 5 / 787da2e | 3634425 / 3634443 / 3634445 | original materialization deadline expired; worker killed/reaped -9 |
| 6 / 672a247 | 3643010 / 3643038 / 3643041 | original materialization deadline expired; worker killed/reaped -9 |
| 7 / a50b227 | 3651791 / 3651805 / 3651806 | code74 after unlisted-publication audit failure; no timeout/kill |

Artifact directories are c2d3-REV-go2-960-guarded-author-vN with the abbreviated
revision above. Invoker SHA-256, in order v4-v7:

```text
2b50056fb02b46128f439c801dd05cba0bb2fcb371ec534fba41fa4c2bc4eca1
e2f7b3791da9eb3ca1595b9bff15073c5fc44e0421a1e5a679b529a3f452aaca
dc954daf97a963b80a2f0c1a12ae73c5926a74e323f2417528881ac581c7eafa
6f47b55a4afeb1f18322022d946aedb1b9fe43e4c9c378f38037d3a2f0dd4053
```

custody_artifact.json SHA-256, in order v4-v7:

```text
d5be5159cc6dc0c7433e87d41e23428244e73554bb059b92fc9c1c5c6d5184e3
f460da4dde6842365a06b0e439a1baaa1df5e3b17c2a116d831e8994910fbb07
b6bac95e5bd7c7b82e40b40d982756de91e4cf3494dcd2df16317a96e2afd6b6
23ae27cdbafde33320e4eff48ea79ab7ea469a3ef08a27a613d10c3df2bdb4a0
```

During v5 diagnosis, a CPU-only query diagnostic measured 20 complete-table
name queries at approximately0.684s. That establishes repeated parsing cost,
not a product benchmark or proof that it was the only timeout cause. v6 still
timed out. The second change removes the added per-module whole-environment
traversal; v7 then finishes before its original deadline, but fails correctly.
No fictional worker final cleanup is inferred from killed v5/v6 prefix evidence.

v7 external_failure.json SHA
a05d85148a57ef557e6bc9e33e6a39d45b344a50d91029a38ba3bdbff157fe39.
It has4554 events. The exact uarray file records create/exec/completed once at
2322-2326, then PIL.ImageFile completes at4552, and audit fails at4553.
failure.json SHA e95ad9c6c1efbdc031f9c8a92aa2bb676f95a77c87fc9bef075f8e6309a62f9a.
Worker summary: failed setup sequence2, full quarantine, no release, zero
emergency sync/terminal ARM. Supervisor EXITED/code74, no expired deadline or
kill. No usable canonical input, runtime initialization, upload/readback or
final successful metadata snapshot was produced. The failure message does not
yet identify the unlisted key; passing the old boundary is not Go2 acceptance.

## New ordinary CPU diagnostic, not admission or a repaired design

Existing outside observer was revised only to report previously omitted
fileless global modules and actual native-boundary module-key deltas. It still
calls the sole canonical materializer with the real frozen asset, forbids
Torch/Warp imports, and grants no permission. Native boundary deltas can overlap
under nested calls and do not prove instruction-level creation or owner links.

audit_go2_cpu_dependencies_v5.py SHA
6bb9d9b694d75d307b3f0310188a9105432a9d15a6293079398b90318c15f900.
Executed robot_sim -I -S -B under90s bound, PID3658000/3657998; exited0.
Artifact c2d3-go2-960-cpu-dependencies-v5.json SHA
17f559fb58dfffc9d7fee34a82e42363cadf86bc3e403d55cd478f128b7584d1.
913 reported modules,4759 observed events; canonical manifest remains
ac3d950309917168d0bb5758e9916adcffcbb17bd68f6611e4fe0f3bdfed3a12.

Extra keys are __main__, the already-covered generated stdlib module, and
cython_runtime / _cython_3_2_5 / _cython_3_2_8. The three Cython modules have no
file/spec origin. Observed native return boundaries include NumPy random
bit_generator/_bounded_integers and scipy._lib._ccallback_c. They are candidates
for the v7 failure, not yet a captured identification or proof of a valid new
creator relation. Do not turn this observation into an automatic allowlist.
Next: capture the actual rejected/retained objects, inspect creation/ownership,
then independently gap-review any required bounded design change.

## Hygiene and closed scope

At2026-10-08T03:47:14+08, scoped ps on183 showed no ang.gao Python/pytest/worker/
supervisor process; all author test tool handles had terminated. NVML process
listing showed only the pre-existing MPS server2238410. Nothing was signaled
outside supervisor-owned test children. No101 fallback, environment install,
GPU reset or broad cleanup. Both candidates were clean a50b227 before this
evidence-only archive. This does not diagnose the historical101 hang.

C2d3 remains WIP. Final versioned runtime-custody schema, complete both-lane
environment/canonical/Go2 upload-readback and failure/hang/dual-device matrices,
and independent whole-stage review remain mandatory. C2e/f, C4c3/c4, C4d/B3,
product GPU preparation, a3b, result/baseline, host payload and shipping remain
closed. No C4 completion is claimed, and no shared README/private instructions
or frozen activation gate was modified.
