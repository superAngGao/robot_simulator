# Canonical precision migration and real R0 frame comparisons

Initiative: Q54 / P12.4 / C4 numerical input and intersection semantics
Stage: implementation checkpoint; full amendment remains incomplete
Author: codex
Version: v3
Date: 2026-10-10
Status: implemented
Related Files: input-precision design-request/review v4; implementation-note v1/v2; native_manifest.py; workload_contract.py; native_sources/v17.json
Owner Summary: Canonical identities now describe the indexed double query representation. Real native R0 frames are compared against every pixel and all ten channels of the unchanged original CPU oracle. This is author evidence, not independent amendment approval, performance acceptance or whole-C4 completion.

## Candidate and identity migration

Numerical/source candidate: `3c1a08fd0495b6bba30d3c1666aac88faa71327b`,
following renderer/readback checkpoint `fcb9acceec92bf900809a03d5eba0dbf90df8a14`.
Local branch: implementation/c4-input-precision. Remote native checkout:
`/data2/shared/ang.gao/robot_sim_input_precision_wip`. Every native launch
requires that exact clean revision. The later fixture/test/document delta in
this checkpoint does not change those executed source bytes or their catalog.

The original CPU oracle remains at clean
`812ca29030a45d802addf10ca5811a4354ce0eb9` in the separate intersection-precision
checkout. Original assets remain the frozen Go2 directory. No oracle,
tolerance, logical case, camera/light sample or replay sample was changed.

The actual original and new workload builders materialized all five cases.
`c4-workload-precision-diff-v1.json` compares their saved arrays, records all
before/after descriptors and enumerates the 22293 newly retained Go2 source
keys. It verifies original source-key retention, unique new keys, regenerated
dense ordinals, preserved source values after the old representation's rounding,
unchanged metadata/material/light/replay values and outward bounds containing
the reconstructed world vertices. This does not claim that old and new geometry
arrays, topology or primitive counts are equal. Diff SHA256:
`8da70006f37b297f14e9c5ceeb58639861e6abdf6b007f906bd278a4a2e67a33`.

The subsequent canonical verification at 3c1a08f uses the actual current
materializer, official fixed-transform identities and source assets. It
reproduces every frozen v2 render-sequence artifact and checks all 120 camera
and lighting frames against retained v1 records. Four current native-input v4
manifests and the Go2 1920 replay manifest are retained under
`tests/unit/optics/data/native_manifests/v4/`; their exact byte hashes are in
`test_native_manifest.py`. Old manifests remain historical negative fixtures.
Summary SHA256:
`0168eb461d507d741e0ff11a4d716ff56cca39725a8d13d1424866835e7b8326`.

Current scene/fixture hashes are owned by workload_contract.py; sequence hashes
by render_sequence_contract.py. Gate/result v15, row semantics v10 and C3
binding v8 describe the changed representation. The gate has SHA256
`a46d22540ee79d280f3c0e1d3a7482cc86f098eafbdb41462deea68a7062390c`.
Historical cases.py f32 fields remain unchanged logical source-case declarations;
the versioned gate's native_representation explicitly owns the new native
dtypes. The migration test verifies the exact allowed v13-to-v15 delta and
preservation of all other gate fields, including activation=false.

Source policy/catalog advances to v17 with the existing read-only ctypes
exception extended to c_double. All 87 actual LF source files pass validation.
Catalog SHA256:
`aa36b5ca48077cf375333752149876f60edbc0636e7c539604e0d2e77c1b2727`.
No additional AST exception or new control-plane mechanism is introduced.

Input migration archive, present with matching hashes locally in c4_assets and
remotely in /data2/shared/ang.gao:
`c4-precision-input-migration-evidence-v1.tar.gz`, SHA256
`53bb71320078301a92c3d22579c01e73c0f93d067bcd9ef8fd918ce14fad1217`.
It includes actual before/after arrays, derivations, canonical manifests and
the exact derivation/comparison/verification scripts.

## CUDA compilation and native execution

The existing isolated deployment tool freshly compiled all three recipes at
3c1a08f, with the original pinned dependency profile and build budgets. All
three services exited normally, status0; deployment verified each owned cgroup
absent. Recipes: scene v3, unchanged LBVH v3, renderer v10. Deployment:
`/data2/shared/ang.gao/c4-precision-cuda-artifacts-v1`.
Manifest SHA256:
`814db2bc607baea4c3e143712cf777373156e499528d5ba9d08f7711d07b84e4`.
The existing cold environment composer binds that deployment to the independently
pinned platform and original CUDA Go2 base. Environment SHA256:
`ee33bb5e3ddf4cde9dbbbc96f4fc9faa7252e0e87e71b4e0a950fd27fb7776ee`.
Build archive `c4-precision-cuda-build-evidence-v1.tar.gz`, matching local/remote
SHA256 `75d1cf298dd84fecf11557d03e9db6c1db25d41ccadd59fed3f059ba0fcf904a`.

Native runs use the existing supervised `run_case` route with
`runtime_render_warmup_geometry`, original device1 and Go2 assets, independently
owned native allocations, ten actual warmups, final-frame readback and original
terminal disposal. Source is not changed underneath these runs. Readback now
includes actual double origins/directions and float32 intensity in addition to
the previous channels. These are untimed setup diagnostics, not samples.

The one-shot comparison script verifies every CPU chunk hash and contiguous
full-frame coverage, native result/frame/shape/dtype/readback hashes and exact
channel coverage. It compares the retained original full CPU frame10 against
the new native frame10; it does not regenerate the oracle with new code.
Original integer equality and numerical tolerances remain unchanged. Detailed
per-channel maximum errors and first-failure lists are saved in the comparison
JSON files; an empty failure list is not asserted to mean bitwise CPU equality.

All four runs completed successfully. Every channel has zero pixels outside
the original comparison tolerance; hit/instance/material equality is exact.

| Lane | Resolution | Pixels compared | Channels | Failed pixels in any channel | Parent / supervisor / worker, all absent |
| --- | --- | --- | --- | --- | --- |
| Warp | 960x640 | 614400 | 10 | 0 | 3856524 / 3856581 / 3856584 |
| Warp | 1920x1080 | 2073600 | 10 | 0 | 3862821 / 3862900 / 3862902 |
| CUDA | 960x640 | 614400 | 10 | 0 | 3865166 / 3865320 / 3865329 |
| CUDA | 1920x1080 | 2073600 | 10 | 0 | 3873096 / 3873215 / 3873216 |

The raw run directories are `c4-precision-{warp,cuda}-{960,1920}-geometry-v1`
under /data2/shared/ang.gao; comparison files use the same stem followed by
`-cpu-comparison-v1.json`. RGB and intensity also match byte-for-byte between
lanes at each resolution. Position/normal arrays have some cross-lane bit
differences while each satisfies the original CPU tolerance. Maximum original
CPU RGB absolute error is3.71758e-7 at960 and4.14053e-7 at1920; range error
is below9.536e-7. This closes these particular original full-frame failures,
not all required frames or the R1 matrix.

Raw native evidence archive:
`c4-precision-native-r0-evidence-v1.tar.gz`, 219374896 bytes, SHA256
`c4bcd8cb18e96ee65829fefd0d24e517d0b5de845d58720c0966d237bdf390a9`.
It includes all four completed run directories, full channel readbacks,
comparison JSON files, the final one-shot launch/comparison scripts and the
exact owned-PID absence observation. No benchmark duration is inferred from
the setup/custody wall time.
The downloaded local archive has the same digest and185 unique members.

## Tests and remaining scope

Final coherent local command covers native_manifest, native_source_policy,
native_sources, render_parameters, prepared_fixed_render_inputs,
native_render_preparation, native_warmup_support, resolved_camera_contract and
optical_render_migration: 614 passed, 4 skipped in87.77s. A preceding run had
610 passes and one stale cleanup-ledger expectation; that expectation was
updated to include the explicitly retained historical manifests. The final
command above includes that correction and actual Go2 replay fixtures.
Separate host scalar/intersection suite: 137 passed in10.96s. Windows Python
3.14 runs are not Warp/CUDA device tests. Runtime-custody actual-record tests
without configured evidence paths are skipped, not claimed as validation.
Ruff passes native-manifest and render-migration tests. The actual-record test
file still has eight pre-existing import/line-length findings; a HEAD replay
has those same eight plus the overlong old manifest-path line repaired here.

Actual-record replay at97ac4b4604cf0861a52c201027e3c2d41418cf78 configured all
four completed native directories: 248 passed,16 skipped,12 failed in554.94s.
All12 failures were test-construction KeyError exceptions, before the production
parser: the shared `_reindex` helper recognized `*_token` fields but omitted
the existing plain `token` field of CUDA extension entry records. The real
unchanged records had already passed their fixture's production parse and
custody attachment checks. This was not a native binding failure.

Test-only repair8d7defda83fa90469620874453d3deda551cf493 recognizes both global
token spellings while retaining the local-token/manifest/signature exclusions.
The new positive check verifies reindexing each original actual record changes
no field. Fresh targeted replay covers all24 admission-progression mutations
across both lanes/resolutions plus the four preservation checks:28 passed,
252 deselected in35.84s. It includes all12 original failures. No full-suite
rerun at8d7defd or combined new-revision280-pass claim is made. The16 earlier
conditional skips do not prove scenarios requiring component occurrences or
zero-size arrays absent from the particular actual records.
No production source, catalog, guard, lifecycle or admission rule changed.

Exact one-shot launchers `replay_precision_runtime_custody_v1.py` and v2, and
raw logs `c4-precision-runtime-custody-replay-v1.log` and v2, are retained locally
in c4_assets and remotely in /data2/shared/ang.gao. Log SHA256 values:

- v1: `036d9ec1ae2da1d7d6a47a0e8a8b07af958f25f2b7685621089e0883ed2b6e7e`.
- v2: `e9d1c311dc384cf3766d6a9371e4192408260cc39ace6e9c1638545a6ff21998`.

Both replay subprocesses returned and were reaped; the failed v1 parent/pytest
PIDs3885522/3885527 and successful v2 parent3899258 are absent. Final scoped
ps check also reconfirms all12 native parent/supervisor/worker PIDs absent.
All author tool sessions are terminal. The only observed remaining GPU process
is the pre-existing unrelated MPS server2238410, which was not touched.

Required continuation is unchanged: complete original CPU R1 comparisons,
remaining required profiles/shadows/frames/sentinels/diagnostics and affected
failure/compatibility cases, then fresh independent full-amendment review.
Final graph/arena/barrier integration, complete lifecycle matrices, C4c3/c4,
C4d/B3 and the owner-requested systematic whole-C4 refactor/review remain due.
No timing result, selected backend, product activation or later gate is issued.
