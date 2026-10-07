# C2d3 generated stdlib implementation and next real Go2 gap

Initiative: q54-optical-lab-p12-4c4c2d-runtime-adoption
Stage: implementation-note
Author: codex
Version: v1
Date: 2026-10-08
Status: implemented
Related Files: generated-stdlib design-request/review v1; native_bootstrap.py; native source policy v9/catalog v8; test_generated_stdlib.py; test_native_bootstrap.py; native_runtime_process.py
Owner Summary: Bounded generated-stdlib implementation is WIP, not independently code-approved. Its production-entrypoint tests and synthetic GPU regression pass. Actual Go2 now reaches a later SciPy native metadata incompatibility and fails safely; no whole C2d3 acceptance or product gate is claimed.

## Design authority and revisions

Exact design791a39c / digest0c90f2d331e79898edfd5182d6036b2c1594d207bb12d8b259ba5b31106956a9
was independently approved by /root/c4_c2d3_generated_stdlib_design_review_v1.
Verbatim report SHA2cf3534415211c8f84567a545616fd80d859689377461b073c655c75a2b22da2
archived unchanged at110d1fd. Approval permits the bounded implementation only.
The review explicitly assigns standard SourceFileLoader/pyc behavior to trusted
immutable interpreter deployment; source SHA is not executed-bytecode attestation.

f26cccd implements the exact descriptor/name/path/source/override/audit contract,
source-policy v9 and catalog v8, plus cold/fault tests. e87f516 and3840778 repair
test-only fixture initialization/cleanup ordering. f97186c adds bounded failure
diagnostics to the existing fixed sacrificial driver, not a product interface.
Current tested production source is unchanged across those last three revisions.

Bootstrap SHA acec0d7788660a50d3debe53113c50bd897453cce17e0cea9f37ce3262590659.
Catalog v8 raw SHA e71e429f20bcb5c30ee20d1c7d5ba7ba40f8813f7646a8cae6240b7f60f6fcdb.
Pins derive from exact staged LF Git bytes; historical catalogs remain unchanged.
The reviewed schema-v3 Go2 environment/table is unchanged; no new import name,
site root, namespace, alias or publication permission was inserted into it.

## Implemented mechanics

- CPython3.12 POSIX only: snapshot intrinsic executable/version/ABI/platform/
  multiarch/stdlib-directory facts, derive one exact name, and require one regular
  non-symlink source directly in the canonical cold stdlib directory. Other
  interpreters receive no special generated-name permission.
- Reject any _PYTHON_SYSCONFIGDATA_NAME presence before startup and at topology
  checks. Descriptor construction/verification uses no CUDA query or sysconfig
  helper that could select a module from the environment.
- Sole finder resolves that exact full name through its already held PathFinder
  binding and exact directory; validates source bytes, nonpackage status and
  actual SourceFileLoader name/path/origin. No prefix, reload or dotted-child
  permission; no external grammar/root relaxation.
- One classification function serves interpreter dispatch and external ledger
  reconciliation. Source and external audits explicitly verify generated-source
  integrity and actual module/spec/loader metadata, retaining the first valid
  boundary's exact module identity. A cached later query cannot hide mutation.
- Existing cold preload rejection remains strict. Failures use the same sticky
  controller/custody path; no new sync, watchdog, release or executable result.

The new fault tests use disposable CPU source files and real bootstrap methods,
not copies of the implementation. Their simulated startup intrinsics are openly
test-only and do not certify the real deployment. Real sysconfig lazy/cached
queries run separately on183. Native admission unit tests still use CPU owner
doubles, not claimed guardian/GPU proof.

## Tests and failed-harness correction

Local Python3.14: importer/policy set260 passed in5.05s. Selected cold tests9
passed/7 Linux-contract skips in4.39s. Later importer-only93 passed/3 Linux skips
in2.66s. These are separate overlapping invocations, not an additive total.
Ruff and git diff --check passed for changed Python files.

Primary183 robot_sim CPython3.12.13, H200:

1. f26cccd generated-focused run:25 passed,1 unsupported-platform skip,2 failures
   in29.01s. The fixture captured an existing module in admission.baseline, then
   removed/reimported it; production correctly detected changed identity. Fix:
   establish fixture cold state BEFORE baseline capture, and keep test cleanup
   identity independent of monkeypatch registry undo. Production checks were
   not relaxed to pass the fixture.
2. 3840778 rerun:27 passed,1 unsupported-platform skip,159 deselected in28.89s.
   Covers real lazy load, cached repeats, late env override, copied/replaced or
   removed module, changed metadata/intrinsics, source byte/missing/symlink
   drift, wrong spec/loader/name/package, exact-prefix denial and external audit.
3. Eight-file CPU regression at3840778:461 passed,3 skipped in189.75s, exit0.
   Two skips are unconfigured real-Go2 tests; one is the non3.12/Windows branch
   exercised locally. No skip is promoted to Go2 success.
4. f97186c fixed runtime-driver process tests:4 passed in12.91s, exit0. Checks
   original bounded diagnostics without changing terminal custody/exit behavior.
5. f97186c C0 regressions:30 passed in11.09s, exit0, no skips.
6. f97186c actual cold Warp GPU suite:6 passed in161.35s, exit0, no skips.

All remote tests used python -B -m pytest, PYTEST_DISABLE_PLUGIN_AUTOLOAD=1,
-p no:cacheprovider, and explicit outer bounds. These bounds were120s for
focused/C0 tests,90s for driver tests,300s for eight-file CPU regression,240s for
the synthetic GPU suite. They are test-failure safeguards, not product/performance
gates. Actual native calls are supervised separately by the unchanged guarded
protocol; the intentional hang tests succeeded because that supervisor killed
and reaped the worker, not because outer timeout expired.

Eight-file set: tests/unit/optics/{test_runtime_environment,
test_runtime_import_admission,test_native_bootstrap,test_generated_stdlib,
test_runtime_import_cold,test_runtime_import_failure_cold,
test_native_source_policy,test_native_sources}.py.
C0 set: tests/unit/benchmarks/{test_optical_closed_worker,
test_closed_worker_protocol}.py and tests/gpu/test_optical_closed_worker_setup.py.
GPU set: tests/gpu/test_native_runtime_cold_warp.py with
ROBOT_SIM_TEST_C2D3_COLD_WARP=1 and fresh basetemp below.

## Actual GPU custody artifacts

Root `/data2/shared/ang.gao/c2d3-f97186c-cold-warp-proof-v1`, paths
`test_actual_cold_canonical_warN/custody_artifact.json`. Parent3597138,
outer timeout3597137. C0 regressions ran concurrently; this is correctness/fault
validation, not a performance baseline or comparative memory measurement.

| N | supervisor/worker | fault/case | SHA-256 |
| --- | --- | --- | --- |
| 0 | 3597228/3597229 | success/asymmetric | defdab8cbae8db669d13e07ead9727e089366596aab6cffcbf4ac1e4a4fe9cbd |
| 1 | 3597871/3597873 | success/dynamic | 70407b3218ddfe09aa1482054598e193de0b2b7920856681a41e43532602a215 |
| 2 | 3598295/3598296 | sync-raise/asymmetric | 03073933a1bab5db9b2c84926b0c8918ed77f81dcf5c436264cd8b79c6c4d21c |
| 3 | 3598741/3598742 | sync-raise/dynamic | b63a78fbfbda73f98ba7d587db80610dae4c82149f5c9fb5b088f53584714f87 |
| 4 | 3599386/3599387 | sync-hang/asymmetric | 682b1a50696dd9be71bdea479c1aead0c570de6dbaabe9168d11fbe9d17942c8 |
| 5 | 3600458/3600459 | sync-hang/dynamic | bb45ff362147a409a7d3532c2ee2b936deb1cdd1db9943e4ced958747bdf0770 |

Normal exits0; raised sync exits74; intentionally hung sync exits-9 after the
original30s deadline. Killed workers have prefix evidence, not fictional final
cleanup. Each actual child/supervisor exit was observed/reaped.

## Actual Go2 attempts remain failures

Case go2_static_960x640_v1, frozen real asset, unchanged independently reviewed
Warp author table. Setup operation budget120000ms/ACK1000ms remained unchanged
from the prior attempt; no timeout or in-flight budget increase occurred.

Attempt v2 at3840778 started02:58:39+08, parent3587771/supervisor3587786/worker3587787.
Outside fixed invoker run_c2d3_go2_guarded_author_v2.py SHA
5d6ab235e385776be2f150ab2b6ec6e78f7767d2e4845b6c3cd405f2dab7321d.
Directory `/data2/shared/ang.gao/c2d3-3840778-go2-960-guarded-author-v2`:

| File | SHA-256 |
| --- | --- |
| request.json | 172c13878bfc114f14d32159586fa9beccc4bfa374b58edc648a99550bd0f7e6 |
| custody_artifact.json | 8cc196649ccc360f2e9d96905c8e537b501b66044c12e5a67752c63d7120b127 |
| external_failure.json | 1bc9d86067f550629f48ca5a895f259c38df23a2ea2b788e3179f09facdcb8b5 |
| failure.json | 2e85fc9cecc90e479051e0647ab417a952068fc8d74c67ad8fd74cf722fc5696 |
| supervisor_stdout.json | 45647d8cf9598348bc981f39eccf796b8522d379344b243b60f6e83cbcb90f0b |

It passed the old event614 position, then failed at2326 in native
scipy._lib._uarray._uarray. No canonical input upload/readback was reached.
The prior test driver retained only the wrapped error, so f97186c adds at most
8 cause/context entries and8 retained failure metadata observations. These are
bounded diagnostics, not admission rules or parsed runtime-custody authority.
No production error is swallowed or renamed by that change.

Attempt v3 atf97186c started03:03:13+08, parent3594945/supervisor3594988/worker3594989.
Outside invoker run_c2d3_go2_guarded_author_v3.py SHA
6ea3d4db9008bc5591ca1bc175150d90aa32b36d83999906f779a7213663a7ee.
Directory `/data2/shared/ang.gao/c2d3-f97186c-go2-960-guarded-author-v3`:

| File | SHA-256 |
| --- | --- |
| request.json | 67a00e3e3ba09d76cc1a4a223485369dcfea949a6f313d5ce59f61949c8ac9fd |
| custody_artifact.json | c28e0ee245812a335b7b95bf7680f7a92a45b3cd311201829e669f61127a6007 |
| external_failure.json | 85ef3338a51beadca0ca05307dcb24a4e28d25222cdb5c99ed0ca81247cf5b5c |
| failure.json | 55fab24bf3a637d363b37a0c8288f14c44307585d64fa8bae75841e793c9aed6 |
| supervisor_stdout.json | 586badbade5867619c23b38c03c11c200fdc5e1910a1cb358a4e2f66592fd68e |

Actual first retained failure: external module metadata drift. The native object
has __name__='uarray._uarray' while the import key/spec.name is
'scipy._lib._uarray._uarray'. Package/file remain expected; exact_spec,
exact_loader and spec_loader_is_wrapper are true. A short ordinary CPU import
of that SciPy module (no Warp/Torch) observed the same values. A read-only scan
of the historical v4 complete CPU dependency trace found this sole mismatch
among exec_returned file modules. These ordinary facts do not replace captured
execution evidence or prove a future metadata policy safe.

Both attempts: worker74, supervisor EXITED/reaped, full quarantine with UNKNOWN
history, no sync/terminal ARM/release, failed sequence2, two earlier completed
segments. This is a safe failure, not Go2 success or whole failure-matrix proof.
The new metadata incompatibility is submitted to a fresh independent gap scan;
no generic name relaxation or module renaming is implemented.

## Round-end hygiene and remaining work

At03:07:55+08 on183, no ang.gao Python/pytest/supervisor remained. All tool test
handles reached terminal exits. GPU target1 had no author compute process.
Pre-existing MPS2238410/GPU0 and user232071 PID3560091/GPU2 were untouched.
Local/remote candidates are clean f97186c; git diff/show checks pass. No101
fallback, dependency install, GPU reset or broad signal. Historical101 hang
root cause is not claimed solved by these checks.

Whole C2d3 still needs complete real Go2/CUDA environments and upload/readback,
full native/partial/setup fault/hang/dual-device/reap matrix, versioned runtime-
custody schema and independent full code review. Current code is WIP. C2e/f,
C4c3/c4,C4d/B3,product GPU preparation,a3b,result/baseline,host/shipping remain
closed. C4 remains incomplete; no reviewed CPU subpiece opens parent gates.
