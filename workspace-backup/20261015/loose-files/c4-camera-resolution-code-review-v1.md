# Independent code review: C4c2 asymmetric camera contract

Initiative: q54-optical-lab-p12-4c4c2-camera-resolution
Stage: review
Author: codex
Version: v1-code
Date: 2026-09-29
Status: accepted
Related Files: optical_worker/camera_contract.py; tests/unit/optics/test_resolved_camera_contract.py; benchmarks/results/q54_p12_4/cleanup_ledger.json; collab/q54-optical-lab-p12-4c4c2-camera-resolution__design-request__codex__v1.md
Owner Summary: Approve the exact candidate's bounded CPU descriptive camera contract. Independent Windows/Linux tests and adversarial payload checks passed. This is not runtime adoption, actual native ABI binding, GPU parity or parent-stage approval; all later gates remain closed.

## Identity and isolation

Reviewer session: `/root/c4_camera_resolution_code_review_v1`, a fresh independent Codex reviewer supplied a bounded packet without the implementation conversation. I did not author the candidate. This is context/role isolation, not a claim of another vendor or security boundary.

Reviewed base and independently confirmed merge-base: `848b5a2abf24f15021b1dcab4c859beefce031ed`.

Reviewed HEAD: `1bcb622af82d820c7247dced82b58c459b89233c`.

Local checkout: `C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design`, branch `implementation/c4-supervised-waits`. Initial and final `git status --porcelain=v1` were empty. Candidate remained at the same HEAD. Actual diff is five files: the new contract, its tests, one cleanup-ledger entry, implementation note and archived prior design-review report. No implementation file was modified by this reviewer.

Design SHA-256 independently checked: `811027332afcbfd37b23a77c177b9e46169c3cd6db7668b5bffce8759b85a1b6`.

Archived design-review SHA-256 independently checked: `ed0015fad94a2309b004254b495ef3c107bdd4c1b0803da989f7699951d25e29`. Its conclusions were claims to inspect, not substitute code approval.

Linux checkout: `/data3/ang.gao/robot_sim_c4_camera_code_1bcb622`, detached at the same full HEAD, clean before and after review executions. Used owner-authorized fallback `ang.gao@192.168.80.101` after the packet's reported primary-host timeout; I did not independently repeat that timeout.

Read AGENTS.md, collab/README.md and collab/REVIEW_PRINCIPLES.md completely, followed the isolation/frozen-candidate rules, and inspected the full accepted design and actual candidate diff.

## Verdict

**Approve.** No substantive findings or unresolved blocking evidence gaps for this exact, CPU-only implementation scope.

## Inspected implementation and independently verified behavior

1. **Single definition and real source identity.** `_frozen_payload()` obtains a fresh canonical case, whose SHA property hashes its actual `asdict` payload rather than returning a caller-supplied digest. It requires the frozen case hash and symbolic rotation, derives intrinsics/translation/range/role from that case and uses the sole production hex rotation table. It then checks the complete resolved payload hash. There is no runtime trigonometric resolver, pose override or unknown-case fallback. Independent reconstruction from the design's rational matrix and literal source values matched every payload field and SHA-256 `04ab7bc2571b8396f06d36b7baea66224868001894c4ce7a5ea12e4faecda1ef` on Windows and Linux.

2. **Payload types and exact values.** `_require_exact` checks exact Python type before value equality, exact string keys and field set, list length/order, and every leaf. It rejects bool/int/float ambiguity, subclasses, null/nonfinite values, noncanonical hex, missing/extra fields, changed dimensions/matrix/metadata and a supplied hash field. The independent probe rejected 370 mutations on each host, extending the candidate tests to every scalar position, string-subclass keys/values, list/int subclasses and equality-bomb objects. No untrusted equality method ran. Dict ordering is correctly not significant; matrix/intrinsic list ordering is significant.

3. **Immutability and repeated operation.** Frozen slots and exact primitive/tuple checks keep supported construction immutable. `from_payload` constructs from independently generated canonical data after validation, not from caller-owned lists. Each export rechecks source and record values; payload exports are fresh. Probe mutations of all exported lists left the record unchanged; tuple-subclass construction and subclass deserialization rejected. Deliberate `object.__setattr__` damage to every stored field was rejected by `to_payload`, `scalar_values` and `sha256`. A fresh resolution remained correct afterward. This proves supported-path descriptive integrity, not hostile same-process Python memory isolation. There are no allocations with external ownership, asynchronous state transitions, device handles or release paths in this addition.

4. **Numerical convention and all rays.** Inspected `SpatialTransform.from_rpy`, `OpticalPinholeCameraSpec` and actual `build_pinhole_camera_rays`: Rz Ry Rx, camera-to-world rotation, integer pixel coordinates and row-major flattening agree with the design. Newly executed Fraction tests independently compose axes, check exact orthogonality and every binary64 hex value, then check positive determinant. Every one of the 21 actual host ray directions matches an independent scalar-loop formula, including boundary indices 0/6/7/13/14/20; origins repeat correctly and directions are unique. Transpose, identity, reversed-axis composition and column-major pixel order fail equality. The matrix transpose also covers column-major interpretation of the 3x3 matrix.

5. **Actual scene, not a replacement fixture.** Tests use the existing canonical packed-scene/fixed-fixture builders, registry, CPU BVH direct-light executor and brute-force CPU reference. Both hosts reproduced hit indices `[4,11,16,17]`, material ordinals `[5,12,16,18]`, BVH/brute hit/range agreement, finite hit outputs and the existing infinity/NaN/zero miss contract. This passed shadows on/off and binary32-quantized inputs. Scene hash `ab3b06cd48e0315ce5d33c0f81a880e71fe5df7369c805a2ae85f5c0197b9e35` and fixture hash `4fcece950264cb2ebb78cf1ec06ad52f5a151b3d2cf2970314f1cae5a2aa7907` are enforced by the tests; existing all-case/content tests passed against unchanged gate records.

6. **Import and activation boundary.** The new module imports only stdlib and `.cases`; the package's existing import path remains CPU-safe. A fresh isolated interpreter's import-attempt hook, including a deliberate forbidden-import negative control, passed without NumPy/Torch/Warp/benchmark/example/tool/renderer imports during contract import/resolution. Repository references show no production runtime consumer added. Actual diff leaves case definitions, materializers, ray generation, protocol, gate and runtime contract unchanged. The new record issues no readiness, C3/C4 authority, GPU preparation or result capability. Existing closed-worker and result-contract regressions passed. The cleanup ledger explicitly retains one common canonical definition while consumed and requires removal if no selected runtime/regression consumer remains, with P12.4e resolution.

## Executed verification

Inspection used `git status --porcelain=v1`, `git rev-parse HEAD`, `git branch --show-current`, `git merge-base`, `git diff --stat/--name-status/--check`, targeted `rg`/Get-Content and Get-FileHash. `git diff --check` passed. Targeted `git diff --quiet` confirmed source case/protocol/workload/gate/materializer/ray/spatial/executor paths unchanged.

Windows environment: Python 3.14.0, NumPy 2.5.2, pytest 9.1.1. Commands from the local checkout:

```text
C:/Python314/python.exe -B -m pytest -q -p no:cacheprovider tests/unit/optics/test_resolved_camera_contract.py tests/unit/optics/test_direct_light_executor.py tests/unit/optics/test_scene_contract.py tests/unit/optics/test_native_contract_extraction.py
C:/Python314/python.exe -B -m ruff check --no-cache optical_worker/camera_contract.py tests/unit/optics/test_resolved_camera_contract.py
C:/Python314/python.exe -B -m ruff format --check --no-cache optical_worker/camera_contract.py tests/unit/optics/test_resolved_camera_contract.py
C:/Python314/python.exe -B C:/Users/AG/新建文件夹/c4-camera-code-independent-probe-v1.py C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design
```

Observed: **122 passed in 5.13s**; Ruff all checks passed; two files already formatted; independent probe passed with 370 payload rejections and the exact expected digest.

Linux environment: conda `robot_sim`, Python 3.12.13, NumPy 2.5.1, pytest 9.1.1. Commands were transported with `ssh -o BatchMode=yes -o ConnectTimeout=8 -i C:/Users/AG/.ssh/tileops_ed25519 ang.gao@192.168.80.101`; the test command executed after `cd /data3/ang.gao/robot_sim_c4_camera_code_1bcb622` and successful exact-HEAD/clean-tree checks:

```text
/data3/ang.gao/anaconda3/bin/conda run --no-capture-output -n robot_sim python -B -m pytest -q -p no:cacheprovider tests/unit/optics/test_resolved_camera_contract.py tests/unit/optics/test_direct_light_executor.py tests/unit/optics/test_scene_contract.py tests/unit/optics/test_native_contract_extraction.py tests/unit/benchmarks/test_optical_benchmark_contracts.py tests/unit/benchmarks/test_closed_worker_protocol.py tests/unit/benchmarks/test_optical_closed_worker.py
scp -q -o BatchMode=yes -o ConnectTimeout=8 -i C:/Users/AG/.ssh/tileops_ed25519 C:/Users/AG/新建文件夹/c4-camera-code-independent-probe-v1.py ang.gao@192.168.80.101:/tmp/c4-camera-code-independent-probe-v1.py
/data3/ang.gao/anaconda3/bin/conda run --no-capture-output -n robot_sim python -B /tmp/c4-camera-code-independent-probe-v1.py /data3/ang.gao/robot_sim_c4_camera_code_1bcb622
```

Observed: **183 passed in 13.11s**, no skips; independent probe passed with the same 370 rejections and digest. Probe SHA-256 matched on both hosts: `093c9f975ad3cbe8fffaa403cb4cab8ca839242fd948c81cc1406fe7580c2396`.

Only review writes were the external scratch probe, its `/tmp` copy and this report. `-B`, disabled pytest cache and no-cache lint options were used; final tracked/untracked status was clean on both candidate checkouts. No installs, pushes, source changes, material deletion, timing runs or GPU submissions occurred. An initially guessed root REVIEW_PRINCIPLES path and runtime-manifest filename were discovery misses, corrected to `collab/REVIEW_PRINCIPLES.md` and `closed_worker_runtime_contract.json`; they were not code/test failures. All reported test results are this review's new executions, not author outputs.

## Exact approved scope and limitations

Approved at the stated HEAD: the single asymmetric 7x3 stdlib-only descriptive resolved-camera definition/resolver/validator/scalar accessor; its CPU verification tests; and bounded cleanup/evidence changes. No unresolved substantive finding or mandatory CPU acceptance gap remains.

This approval does not certify native binary32 ABI arguments, GPU ray arithmetic, runtime materializer import closure, native GPU execution, lighting coverage beyond the stated fixture diagnostics, performance or full R0/R1 integration. CPU quantization checks are not GPU parity evidence. Those checks are intentionally outside this slice, not waived.

Still closed: C4c2 runtime adoption/integration; C4c3/C4c4/C4d/B3; product GPU preparation/submission; host-payload sealing; shipping; a3b/results/baselines; backend selection and parent C4 completion. Existing source case/content identities, rows, thresholds and activation flags remain unchanged; no old result gains camera binding retroactively.

Before runtime adoption, the separate complete integration design must bind this resolved-camera digest AND actual scalar/ABI values to case/row, prepared plan, render evidence and receipt; enforce wrong-matrix/correct-digest, wrong-row and altered-ABI pre-enqueue failures; resolve every remaining mandatory camera; and retain all previously required readiness, timeline, ownership, supervised-failure and real-GPU validation obligations. Design approval and this bounded code approval do not open that next stage.
