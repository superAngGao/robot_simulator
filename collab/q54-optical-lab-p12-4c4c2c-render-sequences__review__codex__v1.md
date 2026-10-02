# C2c3 CPU render-sequence independent code review

Initiative: q54-optical-lab-p12-4c4c2c-render-sequences
Stage: review
Author: codex
Version: v1
Date: 2026-10-03
Status: accepted
Related Files: runtime-integration design codex v3 section 3; render-sequences implementation-note codex v1; optical_worker/render_parameters.py; optical_worker/render_sequence_contract.py; optical_worker/workloads.py
Owner Summary: Approve the exact C2c3 CPU single-materialization, four frozen sequence artifacts and descriptive bound scalar-record delta at c8b86e70d717c2787152b967c08dbb4cc1440a87. Independent real-Go2 reproduction and 288 CPU regressions pass. This is not approval of complete C2c, native ABI adoption, worker activation or C4 completion.

## Identity, isolation and verdict

Reviewer session: `/root/c4_c2c3_sequence_code_review_v1`, fresh independently spawned session; reviewer did not author or change the candidate.

Verdict: **Approve**, bounded to the reviewed delta.

- Base: `b0a6f82e4541e8c9334795e54e04f419766a5d7c`.
- Source-derivation commit: `bb85f027b35120d0dd3875f3765960d41b7f947e`.
- Candidate HEAD: `c8b86e70d717c2787152b967c08dbb4cc1440a87`.
- Local candidate: `C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design`.
- Remote isolated candidate: `/data3/ang.gao/robot_sim_c2c3_c8b86e7` on `192.168.80.101`.
- Both candidate working trees were clean when checked before/after review. No tracked candidate files were edited. Scratch scripts and this report are outside the candidate.
- Accepted design: `collab/q54-optical-lab-p12-4c4c2-runtime-integration__design-request__codex__v3.md`, unchanged from `674af14eb55e34e3a249b2ee58b3e6ede54621a9`; Git-blob SHA-256 `a76bbe8963064876895ba56eeebe72221d941af97f0c84722189b505d6536067`.

Read `AGENTS.md`, `collab/README.md` and `collab/REVIEW_PRINCIPLES.md` fully. Inspected the accepted section 3/3.1 requirements and implementation note as claims, the complete source delta, new tests and their relevant production call chains. No author test count, prior approval or author-generated result was used as independent approval evidence.

## Findings

No substantive findings or unresolved blocking evidence gaps were found within this exact CPU-only slice.

### High-risk behavior inspected and verified

1. `build_canonical_packed_scene` delegates to one `_materialize_canonical_scene`. For Go2, the same locally re-preflighted, captured byte set reaches the existing importer; its exact registry reaches the packer and its actual float64 bounds reach camera resolution. The independent real-asset probe observed precisely one importer call and one packer call per `resolve_render_source` call, checked registry/result identity, and verified the bounds are not float32-rounded substitutes. Existing scene and fixed-fixture identities remain intact.
2. Synthetic world bounds reuse `_derive_fixed_fixture_arrays`; dynamic uses canonical replay array frame zero and the packed instance transform. The tests establish z=.25 and exact lifted float32 .2 extents, not local z=0 or later replay offsets. Dynamic retains one fixed front camera for 120 indexed frames; asymmetric retains its existing exact numeric camera for one frame.
3. Go2 camera values come from the existing `build_lab_video_camera` with frozen adapter-owned dimensions, camera-orbit mode, 120-frame policy and 30-fps metadata. The probe independently compared intrinsics, origin, row-major rotation and range for every one of the 120 frames at both resolutions against that existing builder using the actual imported bounds. No new lane-local camera formula was introduced.
4. `resolve_render_source` reproduces the source first, then compares exact canonical bytes to the frozen artifact. It does not silently substitute a frozen result for a changed source/formula. The sequence loader checks canonical JSON, schema/field shapes, finite canonical binary64 strings, identity, exact frame count/order and the frozen digest. Extra/missing JSONL framing, corruption and noncanonical bytes reject. `.gitattributes` preserves artifact LF framing.
5. The bound scalar record uses the design's exact 24-value order and little-endian binary32 bytes, with one `struct.pack` conversion from the binary64 value. Tests exercise nearest-even ties, signed zero, profile/shadow/pass-mode combinations, sample metadata, modulo/wrap indices and no warmup offset. Reconstructed/rehashed wrong records and reordered scalars reject against the supplied sequence/request/role table. These public records remain descriptive: they do not authenticate that a GPU consumed the values or that a caller supplied an authoritative physical scene.
6. Public materialize/load/bind preparation paths reject timed-region use before source/file access. Malformed type/shape/value requests, missing Go2 evidence and S sequence requests fail closed in the tested paths. There is no product worker caller, gate migration or native execution authority introduced in this delta.
7. The historical AST baseline is unchanged. The normalization is restricted to the exact wrapper delegation and enumerated import/bounds-return statements. An independent in-memory negative-control probe changed the selected registry, changed float64 bound extraction to float32 rounding, and changed the returned bounds to `None`; all three were rejected, while unchanged source matched the historical baseline. No candidate file was modified for these controls.

## Executed verification

### Environment and provenance

- Local static/AST checks: `C:/Python314/python.exe` (Python 3.14), available NumPy/pytest/Ruff; no local trimesh or GPU runtime was required.
- Mandatory real-asset reproduction and CPU regression execution: `/data3/ang.gao/anaconda3/envs/robot_sim/bin/python`, Python 3.12.13, on authorized fallback host 101. The handoff reported primary 183 transport unreachable; this reviewer used the supplied isolated fallback checkout and independently checked its revision/cleanliness. No installs, asset downloads, GPU preparation or GPU tests were performed by this reviewer.
- Asset workspace: `/data3/ang.gao/c4_asset_frozen`; actual Git checkout is its `out/external/mujoco_menagerie` subdirectory. Real model path is that workspace plus `optical_worker.assets.DEFAULT_GO2_MODEL_DIR`.
- Independently observed asset revision: `da76818e269b82289eba39808e2fb91d679d6994`.
- Fresh production preflight aggregate SHA-256: `479dcbdb52c64ca86243d6b1d3b7167bb7d3820af1e60e355168b3c17ddf3ed0`.
- Fresh production preflight manifest SHA-256: `5bdac372dd79897e203a1295e5c3e7cb524b1eb018e08f371db603bbfbcee5d5`.

SSH calls used:

```text
ssh -n -o BatchMode=yes -o ConnectTimeout=8 -i C:/Users/AG/.ssh/tileops_ed25519 ang.gao@192.168.80.101 <command>
```

Initial read-only attempt to run `git -C /data3/ang.gao/c4_asset_frozen rev-parse HEAD` failed because the containing asset workspace is not the Git root. Correcting the query to the actual checkout subdirectory above succeeded. This was a path-inspection error, not a candidate or asset validation failure.

### Independent real-asset probe

Scratch source: `C:/Users/AG/新建文件夹/c4_assets/c4-c2c3-independent-probe-v1.py`, copied by `scp` to `/tmp/c4-c2c3-independent-probe-v1.py`. Run from the clean remote candidate:

```text
timeout --kill-after=5s 180s /data3/ang.gao/anaconda3/envs/robot_sim/bin/python -B -u /tmp/c4-c2c3-independent-probe-v1.py
```

Result: exit 0. All four calls used `resolve_render_source`, thereby performing fresh materialization and exact frozen-byte comparison. Import attempts for `benchmarks`, `examples`, `tools`, `torch` and `warp` were blocked throughout the probe. No such import occurred; subsequent deliberate negative-control imports all tripped the blocker.

Freshly reproduced sequence hashes (record bytes, excluding JSONL LF):

| Case | Frames | SHA-256 |
| --- | ---: | --- |
| asymmetric_synthetic_7x3_v1 | 1 | `59e28d8b3e218d3079bf32ab535d97f4e8a3557a6819c2aadafb295ec53f3a24` |
| synthetic_dynamic_rigid_960x640_v1 | 120 | `5d4c2c54e5cae75c65d1e273bf512cb74ea09fb780dd4db500a1a6cedebf8a70` |
| go2_static_960x640_v1 | 120 | `bd8378f0a57bd3a95d501728c9e7c9409e4c4a7e02d7dccf94d5892045b1ccdc` |
| go2_static_1920x1080_v1 | 120 | `d8519b3977b6d0a144884166ffa943747a389ee61f574ac5cb84e1ae9920250e` |

Both Go2 sequences used these actual imported bounds:

```text
min: -0x1.88426329f4d91p-2, -0x1.4bcd78c436fc2p-3, -0x1.3020a02fd1ef7p-6
max:  0x1.53eccc46950fcp-2,  0x1.4bcd78c436fc2p-3,  0x1.6f722e1ac57e2p-2
```

Rebuilt frozen Go2 fixed-fixture hashes also matched:

- 960x640: `fce32d13427918022c4eaee45485ed6fbfaf5e03b13d1390a0216e138e483182`.
- 1920x1080: `8354993fb682d5a13a2dff62d1d7521afd7d3d46d20b876e15732b4c38415db0`.

### Independent CPU regressions

Scratch bounded runner: `c4-c2c3-independent-tests-v1.py` in the same local scratch directory and remote `/tmp`. It starts pytest, records PID/parent/time, waits at most 180 seconds, and reaps its own child on timeout; an outer 210-second timeout also bounds the remote command.

```text
timeout --kill-after=5s 210s /data3/ang.gao/anaconda3/envs/robot_sim/bin/python -B -u /tmp/c4-c2c3-independent-tests-v1.py

/data3/ang.gao/anaconda3/envs/robot_sim/bin/python -B -m pytest -q -p no:cacheprovider tests/unit/optics/test_render_parameters.py tests/unit/optics/test_cpu_materialization_ownership.py tests/unit/optics/test_resolved_camera_contract.py tests/unit/optics/test_captured_asset_bytes.py tests/unit/benchmarks/test_optical_benchmark_core.py tests/unit/benchmarks/test_optical_benchmark_contracts.py tests/unit/benchmarks/test_native_input_observations.py
```

Observed result: **288 passed in 6.28s**, exit 0, no skips. This includes actual trimesh parsing in the custody regression on the remote environment; it does not replace the separate real-Go2 probe above with that test's synthetic fixture.

### Static checks and AST mutation controls

From the local candidate:

```text
C:/Python314/python.exe -m ruff check optical_worker/render_parameters.py optical_worker/render_sequence_contract.py optical_worker/workloads.py tests/unit/optics/test_render_parameters.py tests/unit/optics/test_cpu_materialization_ownership.py
git diff --check b0a6f82 HEAD
C:/Python314/python.exe -B C:/Users/AG/新建文件夹/c4_assets/c4-c2c3-independent-ast-v1.py
```

Ruff: all checks passed. Diff check: no errors. Independent AST probe: all three in-memory negative controls rejected; unchanged source passed. Its direct local execution completed normally in approximately 1.5 seconds.

## Process hygiene and side effects

- Real-asset probe: remote parent/timeout PID `1846077`, start `2026-10-03 04:01:47` host-local; Python PID `1846103`, parent `1846077`, start epoch `1790971307.7283347`. Tool session `36802` completed with exit 0.
- Test round: remote parent/timeout PID `1847470`, start `2026-10-03 04:02:39` host-local; runner `1847492`, parent `1847470`, start epoch `1790971359.972424`; pytest `1847493`, parent `1847492`, start epoch `1790971359.9729407`. Pytest exit epoch `1790971367.2460902`, status 0. Tool session `6065` completed with exit 0.
- Local AST probe: Python PID `22652`, parent `21932`, start epoch `1790971438.935801`, normal exit. Subsequent exact-PID query found it absent.
- Baseline GPU query during concurrent author validation listed unrelated PIDs `1838786` and `1843842`; they were reported to the author, not killed or modified. The author later identified them as its validation processes. No author test output is counted as this review's evidence.
- Final independent `ps -o pid,ppid,lstart,args -p 1846077,1846103,1847470,1847492,1847493,1838786,1843842` produced only the header: those processes were absent.
- Final independent `nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader` was empty. Local `Get-CimInstance Win32_Process` queries found no remaining `ssh.exe` and no AST-probe PID. Both long-running SSH tool sessions were explicitly reaped.
- Final remote/local `git status --porcelain` was empty and remote HEAD still matched the candidate. No broad kill, GPU reset, source edit or cleanup of shared user files was performed. Normal test temporary files/caches may exist outside the tracked candidate; scratch scripts were retained as reproducible evidence.

These are scoped observations, not a claim that no unrelated process can ever hang. No unresolved reviewer-owned residual was observed.

## Approval boundary and limitations

Approved only: this delta's CPU materialization/bounds handoff, common canonical sequence data and four actual frozen artifacts, source-to-artifact checking, descriptive 24-scalar records, and associated compatibility/validation tests.

Not delivered or approved here: whole C2c completion; v13 gate/result and v8 row-semantic migration or C3 binding version adoption; real native Warp descriptor/CUDA ABI adoption and actual execution receipt validation; final import admission; C2d/e/f; R2/S/host/depth-two; C4d/B3; product GPU preparation/submission; a3b; result/baseline authority; backend selection; C4 completion. These remain required later tasks/gates, not waived evidence gaps inside an allegedly completed integration.

No GPU execution, dual-device correctness, physical parameter binding, setup/readiness authority or worker lifecycle acceptance was attempted or inferred. Exact sequence reproduction was demonstrated in the frozen Linux robot_sim environment; behavior on other libm/NumPy platforms is not asserted. The implemented exact-byte comparison fails closed if such a platform produces different values.

The supplied role table and public descriptive records can be reconstructed by callers. Their validation establishes consistency with supplied source/request data, not authenticity of native storage or submission. Later integration must bind the actual scene and native arguments through the approved physical-ownership/ABI path; it must not promote these hashes into readiness authority.
