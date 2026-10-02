# Independent C2c2 asset-byte custody code review

Initiative: q54-optical-lab-p12-4c4c2c-asset-custody
Stage: review
Author: codex
Version: v1
Date: 2026-10-03
Status: accepted
Related Files: runtime-integration design v3 section 3 at 674af14; C2c1 approval 4cac1be; candidate implementation note v1; optical_worker/assets.py; optical_worker/workloads.py; optics/mjcf_visual.py
Owner Summary: Approve the exact C2c2 CPU asset-byte custody delta. Independent real-Go2 parsing, legacy parity, frozen identities and 191 selected regression tests passed. This is not complete C2c, runtime activation, result authority, or C4 approval.

Reviewer session: `/root/c4_c2c2_asset_custody_review_v1`, fresh independent review agent without implementation-history fork. The reviewer did not author or modify the candidate.

Base: `6b882f6b87e16189c334e1e41df50c86f5b1abf5`.
Reviewed HEAD: `536b815ee65db351c502922ce8b1cf9ddf3c7fd0`.

Local candidate: `C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design`.
Remote candidate: `/data3/ang.gao/robot_sim_c2c2_536b815` on `ang.gao@192.168.80.101`.
Both candidates reported the exact HEAD and clean tracked/untracked status at the review boundaries. Final local `git diff --check 6b882f6 HEAD` passed. Only the six declared files differ from base. No candidate edits were made.

## Verdict

**Approve**, limited to the asset-byte custody implementation at the exact reviewed HEAD.

No substantive blocking or non-blocking code findings were identified in this delta. The approval is based on inspected implementation and independent executions, not the implementation note, author test summaries, or prior approval.

## Scope and inspected behavior

Read `AGENTS.md`, `collab/README.md` and `collab/REVIEW_PRINCIPLES.md` in full. Reviewed the accepted runtime-integration design's materialization requirements, predecessor boundary, candidate implementation note, all changed code/tests and their relevant call chains.

1. `capture_case_asset` performs a fresh production preflight. The optional earlier evidence is an exact-type consistency constraint, not authorization. Fresh evidence must agree with it before capture. Each captured immutable byte string must match the freshly validated length and SHA-256. Manifest bytes are checked independently. A final full preflight must equal the initial one before a record is returned.
2. The returned mapping is a proxy over a private dictionary whose values are immutable bytes. The public dataclass is descriptive CPU data, not an unforgeable capability. No supported canonical materialization path accepts such a record in lieu of executing capture itself.
3. `_read_capture_file` checks observed relative-path symlinks and containment; opens with available no-follow/nonblocking flags; rejects nonregular descriptors; checks descriptor identity/change metadata and final path identity. Actual content hashes, rather than metadata alone, determine byte identity. Windows descriptor/path ctime differences are narrowly accounted for.
4. `build_canonical_packed_scene` retains its frozen-source checks, calls capture itself, and passes `captured.files` into the production importer. Both the scene and fixed fixture remain verified against frozen workload identities. No parent-supplied labels can select different bytes in this path.
5. The importer copies the mapping structure, rejects mutable values and non-normalized/escaping keys, parses captured XML from bytes and captured OBJ from `BytesIO`, and fails on missing/escaping referenced files. The captured branch cannot select the legacy path-based branch. OBJ receives `process=False`, fixed `file_type="obj"` and an explicit empty resolver.
6. Independently inspected the installed Trimesh resolver implementation: resolver creation is conditional on `resolver is None`; `{}` is retained. OBJ material resolution indexes that resolver and does not acquire a filesystem/network resolver as a fallback. The real captured importer also ran with `builtins.open` and `Path.open` denied.
7. The legacy path-based importer remains the same shared importer, not a copied mathematical implementation. AST baseline data is unchanged. The new test projection requires exact signature/statements, exactly one match per I/O substitution, and otherwise retains the historical mathematical AST comparison. Existing object-identity tests still pass.
8. The changed file set contains no gate, protocol, result schema, activation, CUDA/Warp execution, timing, source admission or native lifecycle changes. No new mathematical baseline or frozen scene/replay/fixture identity was introduced.

## Independent verification

Remote environment used: Python `3.12.13` from `/data3/ang.gao/anaconda3/envs/robot_sim/bin/python`, installed real NumPy/Trimesh/pytest, frozen real asset root `/data3/ang.gao/c4_asset_frozen`, model `out/external/mujoco_menagerie/unitree_go2`. No install, download, surrogate asset, asset modification or frozen-constant patch was used in real-Go2 acceptance.

Transport used for remote commands:

```text
ssh -n -o BatchMode=yes -o ConnectTimeout=8 -i C:/Users/AG/.ssh/tileops_ed25519 ang.gao@192.168.80.101 <command>
```

The handoff reported primary 183 transport unavailable and explicitly authorized this fallback. This reviewer used the isolated clean candidate, never the dirty common worktree `/data3/ang.gao/robot_sim_c4c1b2`.

### Selected unit/regression tests

Final independently executed command, in the remote candidate:

```text
/data3/ang.gao/anaconda3/envs/robot_sim/bin/python -B -m pytest -q -p no:cacheprovider tests/unit/optics/test_captured_asset_bytes.py tests/unit/optics/test_cpu_materialization_ownership.py tests/unit/optics/test_resolved_camera_contract.py tests/unit/benchmarks/test_optical_benchmark_a3a.py tests/unit/benchmarks/test_optical_benchmark_core.py tests/unit/benchmarks/test_optical_benchmark_contracts.py
```

Result: **191 passed in 5.99s**, zero skips, exit 0. This includes the real-parser and POSIX symlink/FIFO custody checks, legacy identity/AST checks, resolved camera regressions, asset preflight and benchmark authority/contract tests. Mechanics fixtures patch frozen constants only inside their own tests; they were not used to assert real-Go2 correctness.

Earlier independent runs were 38 passed in 4.12s (custody/ownership) and 153 passed in 4.58s (the remaining listed files). These overlap the final 191-test run and are not additional unique coverage.

The final run was launched by a bounded reviewer runner:

```text
timeout 200s /data3/ang.gao/anaconda3/envs/robot_sim/bin/python -B /tmp/c2c2_review_cpu_tests_v1.py
```

Its retained local source is `C:/Users/AG/新建文件夹/c4_assets/c2c2_review_cpu_tests_v1.py`; subprocess timeout is 180 seconds with owned-process termination/reaping only if exceeded. No timeout or termination was needed.

### Real-Go2 independent probe

Reviewer-authored script, retained at `C:/Users/AG/新建文件夹/c4_assets/c2c2_independent_probe_v1.py` and copied to `/tmp/c2c2_independent_probe_v1.py`, ran as:

```text
timeout 300s /data3/ang.gao/anaconda3/envs/robot_sim/bin/python -B /tmp/c2c2_independent_probe_v1.py
```

Result: exit 0, within the bound. The script independently:

- Captured the actual frozen Go2 through the production capture entrypoint.
- Parsed the legacy path and captured bytes; during the latter, denied `builtins.open` and `Path.open`.
- Compared actual bounds and every applicable geometry array (`vertices_local`, `triangles`, `normal_local`, `point_local`) exactly, with matching geometry keys and **398,432 triangles**.
- Built both canonical packed scenes and fixed fixtures through the changed production builder, and independently asserted their hashes against the frozen workload contract.
- Built and checked the 1080p replay hash.
- Installed an import finder rejecting `benchmarks`, `examples`, `tools`, `torch` and `warp`; confirmed none entered the process modules. No GPU preparation occurred in this probe.

Observed identities:

| Case | Scene SHA-256 | Fixed fixture SHA-256 |
| --- | --- | --- |
| go2_static_960x640_v1 | `745f01f3d526681c3a350086d55211f92d4a1e5f5b22bcbbc9c43f7dd18f072f` | `fce32d13427918022c4eaee45485ed6fbfaf5e03b13d1390a0216e138e483182` |
| go2_static_1920x1080_v1 | `57e86366429979a2460a0ceb01f00749c7e4be7449bbc3acc1577ed5ce94b923` | `8354993fb682d5a13a2dff62d1d7521afd7d3d46d20b876e15732b4c38415db0` |

1080p replay: `2b1ecfb401476a9737400b401a72377c880bc9e289370f2d3bd5600cc287a216`.

## Process hygiene and side effects

Remote owned process ledger (epoch times are script-emitted wall times):

| Round | Identity and start | Completion |
| --- | --- | --- |
| Real Go2 | timeout PID 1797201; Python PID 1797226, parent 1797201; `ps` start 2026-10-03 03:25:54 remote host time; epoch 1790969154.8488197 | Python PASS at 1790969235.1711905; SSH/exec exit 0 |
| 153-test run | timeout PID 1802303; runner 1802324; pytest 1802325; epoch 1790969403.6318605/1790969403.6323183 | pytest exit 0 at 1790969409.1032634; SSH/exec exit 0 |
| Final 191-test run | timeout PID 1808771; runner 1808792; pytest 1808793; epoch 1790969682.0795405/1790969682.0800583 | pytest exit 0 at 1790969689.0035608; SSH/exec exit 0 |

The first 38-test invocation completed synchronously with exit 0 without printing its PID; the same coverage was rerun in the final fully ledgered invocation.

After the 153-test round, scoped `ps` showed its three owned PIDs absent and no GPU compute applications; the real probe also had exited. After the final round, repeated:

```text
ps -o pid,ppid,lstart,args -p 1797119,1797201,1797226,1802303,1802324,1802325,1808771,1808792,1808793
nvidia-smi --query-compute-apps=pid,process_name,gpu_uuid --format=csv,noheader
```

Only the `ps` heading was returned; all listed processes were absent. GPU compute-app query returned no entries. Local `Get-CimInstance Win32_Process` filtered to `ssh.exe` commands naming either reviewer script returned no entries. Both long-lived tool sessions completed with exit 0; no outstanding owned test/SSH/worker/supervisor handle remains. No unrelated process was killed and no GPU reset was performed. This is scoped observed cleanup, not a claim that every possible system process or hang is ruled out.

Scratch sources were created outside the candidate and copied to remote `/tmp`; they remain as reproducible evidence. Pytest uses disposable fixtures and cacheprovider was disabled. No candidate or real-asset writes were performed, and final local/remote Git status stayed clean.

One auxiliary inline package-version diagnostic hit a shell-quoting `NameError` after printing Python 3.12.13. This was a diagnostic-command error, not a candidate failure; it did not affect any test/probe. The final status/residual check was repeated separately and exited 0.

## Limitations and remaining gates

- This is the approved design's closed trusted-process boundary, not hostile Python or hostile-filesystem confinement. The snapshot proves the bytes owned at capture; it does not certify future disk state or an atomic repository-wide snapshot. Existing path-based preflight is not represented as a general bounded-I/O service or hostile-filesystem sandbox.
- Local Python 3.14 lacks required parser/runtime dependencies; mandatory real parsing and POSIX checks were therefore executed remotely with zero skips. No approval relies on local dependency skips.
- The reviewer independently ran the selected 191 tests and real-Go2 probe. The author's separately reported 516-test suite and author probe were not counted as independent executions or used in place of these checks. GPU lifecycle/dual-device tests were not rerun because this delta does not change or admit GPU execution.
- Resolved 120-camera artifacts, complete bound native scalar records, v13 row/schema/binding migration, and final worker source allowlists remain unfinished C2c work. No claim of complete C2c is made.
- C2d/e/f, integrated worker activation, R2/S/host/depth-two, C4d/B3, product GPU preparation/submission, a3b, result/baseline authority, and C4 completion remain closed and unapproved by this review.

No implementation repair is requested for the reviewed C2c2 delta. Subsequent operative code/design/test changes require their own independent delta review under the shared protocol; evidence-only archival does not broaden this verdict.
