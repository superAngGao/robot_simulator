# C4c2c1 CPU definition ownership — independent code review v1

Initiative: q54-optical-lab-p12-4c4c2c-cpu-owners
Stage: review
Author: codex (independent reviewer)
Version: v1
Date: 2026-09-30
Status: in_review
Related Files: runtime-integration design-request codex v3 at 674af14; its approval archived at c969296; CPU-owners implementation-note codex v1
Owner Summary: Request changes. The production extraction preserves the inspected algorithms and identities, including independently recomputed real-Go2 content and camera parity. One blocking P2 test portability defect causes all four AST-baseline test cases to fail in the required Python 3.12 robot_sim environment. No runtime or product gate is approved by this review.

## Identity, isolation, and exact scope

- Reviewer session: `/root/c4_c2c1_cpu_owners_review_v1`, fresh independent code-review agent; not the implementation author. No child reviewer was spawned.
- Local candidate: `C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design`.
- Base and actual Git parent: `c9692964ddb9a2226c6d6b076b991c6c22922d59`.
- Reviewed HEAD: `234268aa1c967b2365d7dd08f34cd215d24f306a`.
- Remote candidate: `ang.gao@192.168.80.101:/data3/ang.gao/robot_sim_c4_c2c1_234268a`, same exact HEAD and parent.
- Both working trees were clean when independently checked; local and remote end-of-round checks remained clean. The candidate was not edited. Scratch source and this report are outside the candidate under `c4_assets`; the remote scratch copy is `/tmp/c2c1_independent_reviewer_probe.py`.
- Read `AGENTS.md`, `collab/README.md`, and `collab/REVIEW_PRINCIPLES.md` completely, plus the approved v3 design and current implementation note. The initial attempted root-level `REVIEW_PRINCIPLES.md` read failed because the file is under `collab`; the correct file was subsequently read in full. Large combined outputs were reread in bounded sections where truncated.
- Scope is ONLY C2c1: extraction of existing CPU definitions into sole production owners with explicit compatibility re-exports. This is not a review claiming full C2c delivery.

## Verdict

**Request changes** for F1 below. No substantive production-algorithm or gate-opening defect was found in the inspected delta. Passing direct parity checks does not waive the broken committed regression test in the required environment.

## Findings

### F1 — P2 blocking: AST fixture hashes depend on the interpreter's debug-dump format

Location: `tests/unit/optics/test_cpu_materialization_ownership.py:83`–84 and `tests/unit/optics/data/cpu_materialization_extraction_baseline.json`.

Trigger: execute the new ownership tests in the prescribed remote robot_sim environment, Python 3.12.13. The test hashes `ast.dump(node, include_attributes=False)` and compares that string's SHA-256 with fixtures generated under local Python 3.14.

Observed consequence: all four parameterizations of `test_moved_ast_matches_frozen_pre_extraction_definition` fail, despite the moved bodies matching their actual Git-parent definitions. My independent remote targeted run returned **4 failed, 108 passed in 3.34s**, exit 1. The failures cover workloads, assets, camera presets, and the MJCF importer. Example: `_asymmetric_registry` gives `65389fd11c9dbb32f84cf98875dc76eb24fb6ed2a1c9b45213ad704914b9ee9b` on Python 3.12, while the fixture expects `e6b6e0bf43f012ce42fac62681f59317e2a53b4c5d7b7b51b504e48a85df6001`.

Cause established independently: local `C:/Python314/Lib/ast.py` defines `dump(..., show_empty=False)` and omits empty-list fields. Remote Python 3.12's `ast.py` has no `show_empty` option and formats those fields. Thus a debug representation, not a stable structural encoding, has become the cross-environment identity contract. My probe verified the original source SHA-256s directly against `git show HEAD^:<path>` in both environments: the source bytes agree, while all 36 stored definition-dump hashes match locally and disagree remotely. Direct same-interpreter old/new AST equality passes for all 31 moved definitions in both environments.

Why current checks miss it locally: baseline generation and execution use the same Python 3.14 dump defaults. The committed tests do not define an interpreter-independent serialization. This is a candidate test-code/fixture defect, not missing Torch/Warp, absent assets, or a production numerical regression.

Proposed direction: freeze a deterministic structural serialization of AST node types, field names, and values, preserving semantically relevant empty lists/None and other values, or an equivalently explicit version-portable representation. Regenerate the fixture from the unchanged historical base source, not from the moved implementation. Retain the tightly scoped importer-module normalization; do not blanket-remove imports or weaken mathematical-body checks.

Required repair proof: independently verify fixture provenance against c969296, run the complete ownership/camera and a3a targeted selection on both Python 3.14 and robot_sim Python 3.12 with zero AST failures, and show that an actual algorithm/argument/import mutation still changes the structural identity or fails comparison. Keep candidate frozen for a fresh re-review of the repaired revision.

## Inspected code, data flow, and boundary checks

1. Actual `git diff c969296..234268a` contains 13 files: four new production owners, four legacy compatibility surfaces, the implementation note, the ownership test and baseline, one existing test import, and two added entries in the test-only source set. No worker driver, production source-admission contract, protocol/schema, activation setting, gate, result, baseline, case catalog, or frozen content contract file was changed.
2. Independently enumerated definitions from the new modules, without trusting the test's selected-name list. Verified exact old/new AST equality for **31 moved definitions**: assets 9; workloads 5; importer 14; camera 3. The only normalized body change is the precise `examples.mujoco_menagerie_robot_preview` to `optics.mjcf_visual` import relocation. Also verified that the five retained example functions match the Git parent and that removed/moved versus retained definitions partition the original definition set exactly. No second legacy implementation remains.
3. Checked all four fixture source hashes and all 36 baseline definition names/source line spans against Git-parent source. The baseline is genuinely derived from that source; F1 concerns its serialization portability, not an invented baseline.
4. Compatibility tests assert exact object identity and production `__module__` for every moved definition. The old assets/workloads/camera modules are explicit imports. The example retains its CLI, render, and legacy preview-camera behavior. Imported constants are explicitly retained where used by the old assets/workloads API. The change from `.native` to production `.native_input` resolves to the same existing types because the former benchmark module already re-exports them.
5. The MJCF importer retains all transitive body/material/light/mesh/math helpers and imports production registry types and CPU frame/spatial types. No example rendering or CLI entry was moved into it. Its optional trimesh behavior and ordinary file reads are unchanged. Read-byte/TOCTOU custody remains excluded from C2c1; this review does not certify immutable asset custody.
6. Independently executed camera comparisons against the actual pre-extraction camera source obtained from Git, not only the alias or new fixture: exact scalar and array equality for 120 orbit frames plus indices -1, 120, and 121; eight fixed-bounds cases including zero extent and front/side/top/fallback view; fixed-view dispatch; and reserved pose-sequence rejection. These passed on Python 3.14 and 3.12. This is preservation of the existing builder behavior, not implementation of the deferred worker's modulo-index/render-sequence contract.
7. Real-Go2 preflight, packed scenes, fixed fixtures, replay, and all 120 camera values were independently recomputed on 101 with real assets. A deny-import finder was installed before production workload/assets/importer loading, rejecting `benchmarks`, `examples`, `tools`, `torch`, and `warp`. No forbidden import attempt or loaded forbidden root occurred through both real-Go2 paths. The committed fresh-process closure test additionally exercises explicit negative controls and the asymmetric fixed fixture.
8. The existing negative asset-preflight test now patches `optical_worker.assets`, where the actual function's globals live. Its checks were not disabled. The entire a3a file passed in my remote selection; the only failures in that selection were F1.
9. `tests/support/sacrificial_sources.py` adds only `optical_worker/assets.py` and `optical_worker/workloads.py`. I traced the existing sacrificial asymmetric case through the benchmark compatibility import into these modules. The source ledger still verifies checked source bytes; no bypass or production admission expansion was added. The unchanged product worker still terminates/rejects GPU PREPARE in `_reject_gpu_preparation`.

## Executed independent verification

Local environment: Windows PowerShell, `C:/Python314/python.exe`, Python 3.14.0. Remote environment: fallback 101, `/data3/ang.gao/anaconda3/envs/robot_sim/bin/python`, Python 3.12.13. The handoff reports primary transport unavailable; I used only the explicitly authorized fallback and did not alter either host's environment or install dependencies.

Commands were run from the candidate directory unless otherwise stated:

```text
git status --short
git rev-parse HEAD
git diff --stat c969296..234268aa1c967b2365d7dd08f34cd215d24f306a
git diff c969296..234268a -- <changed legacy modules and existing tests>
git diff --check c969296..234268a
C:/Python314/python.exe -B ../c4_assets/c2c1_reviewer_probe.py
C:/Python314/python.exe -B -m pytest -p no:cacheprovider -q tests/unit/optics/test_cpu_materialization_ownership.py tests/unit/optics/test_resolved_camera_contract.py
C:/Python314/python.exe -m ruff check optical_worker/assets.py optical_worker/workloads.py optics/camera_presets.py optics/mjcf_visual.py benchmarks/optical/assets.py benchmarks/optical/workloads.py examples/mujoco_menagerie_robot_preview.py tools/optical_pipeline_lab/camera_presets.py tests/support/sacrificial_sources.py tests/unit/benchmarks/test_optical_benchmark_a3a.py tests/unit/optics/test_cpu_materialization_ownership.py
```

Results: clean exact candidate; `git diff --check` clean; local ownership/camera **83 passed in 2.67s**; Ruff **All checks passed**. The independent probe, retained under `c4_assets/c2c1_reviewer_probe.py`, reads Git source with subprocess, validates fixture provenance, compares all moved/retained bodies, and directly compares camera outputs. Its first run had a reviewer-script-only line-span assertion that did not account for a dataclass decorator being recorded as the first line. I corrected that scratch assertion to include decorator lines and reran; it passed. No candidate fix was made or inferred from this scratch error.

Remote transport for all calls:

```text
ssh -n -o BatchMode=yes -o ConnectTimeout=8 -i C:/Users/AG/.ssh/tileops_ed25519 ang.gao@192.168.80.101 <command>
```

The independent scratch probe was copied with `scp -o BatchMode=yes -o ConnectTimeout=8 -i C:/Users/AG/.ssh/tileops_ed25519 ../c4_assets/c2c1_reviewer_probe.py ang.gao@192.168.80.101:/tmp/c2c1_independent_reviewer_probe.py`. Remote commands:

```text
cd /data3/ang.gao/robot_sim_c4_c2c1_234268a && /data3/ang.gao/anaconda3/envs/robot_sim/bin/python -B /tmp/c2c1_independent_reviewer_probe.py --assets /data3/ang.gao/c4_asset_frozen
cd /data3/ang.gao/robot_sim_c4_c2c1_234268a && timeout 60 /data3/ang.gao/anaconda3/envs/robot_sim/bin/python -B -m pytest -p no:cacheprovider -q --tb=short tests/unit/optics/test_cpu_materialization_ownership.py tests/unit/optics/test_resolved_camera_contract.py tests/unit/benchmarks/test_optical_benchmark_a3a.py
```

Remote pytest: **4 failed, 108 passed in 3.34s**, exit 1, F1 only. Remote independent probe: exit 0, complete. Its printed PASS means the direct Git-parent and runtime parity checks passed; it explicitly records the 36 committed baseline-digest mismatches and does NOT treat those mismatches as acceptable.

Independent real asset root: `/data3/ang.gao/c4_asset_frozen/out/external/mujoco_menagerie/unitree_go2`, default `q54_asset_manifest.json`; no surrogate or download. Observed identities:

| Object | Independently observed SHA-256 |
| --- | --- |
| Go2 asset aggregate, both resolutions | `479dcbdb52c64ca86243d6b1d3b7167bb7d3820af1e60e355168b3c17ddf3ed0` |
| 960x640 scene | `745f01f3d526681c3a350086d55211f92d4a1e5f5b22bcbbc9c43f7dd18f072f` |
| 960x640 fixed fixture | `fce32d13427918022c4eaee45485ed6fbfaf5e03b13d1390a0216e138e483182` |
| 1920x1080 scene | `57e86366429979a2460a0ceb01f00749c7e4be7449bbc3acc1577ed5ce94b923` |
| 1920x1080 fixed fixture | `8354993fb682d5a13a2dff62d1d7521afd7d3d46d20b876e15732b4c38415db0` |
| 1920x1080 replay | `2b1ecfb401476a9737400b401a72377c880bc9e289370f2d3bd5600cc287a216` |

All agree with the unchanged frozen workload contract. Non-asset scene/replay/fixture identities and absent-Go2 rejection passed through the independently executed committed tests.

I read the author's broader remote log tail at `/data3/ang.gao/c2c1_234268a_validation_logs/pytest.log`: it reports **4 failed, 488 passed in 112.95s**, with the same four AST failures. This is author-run supplementary evidence, not my independent GPU execution. I did not rerun a heavy GPU suite while that suite was active. Local missing Torch/Warp described in the handoff is an environment limitation; the previously reported 11 missing-Torch failures are not waived or represented as a local pass. My remote a3a run completed without those failures.

## Round-end process hygiene and limitations

- Reviewer local probe ledger: initial scratch run PID 38292, parent PowerShell 31420; corrected successful run PID 23424, parent PowerShell 34940. The initial process exited on its assertion; the corrected process printed completion. Local pytest and Ruff tool invocations completed; no reviewer GPU worker/supervisor was launched.
- Remote independent probe: PID 2377731, parent 2377719, observed Python 3.12 and exact HEAD in its own output; completed with `reviewer probe PASS pid 2377731`, SSH tool session 12975 returned exit 0.
- Remote targeted pytest had a 60-second outer timeout, returned exit 1 normally in 3.34s, not timeout status. Its short-lived OS PID was not captured before exit; similarly, the local pytest child PID was not individually captured. Tool completion plus subsequent scoped process scans are the available exit evidence for these short invocations. This is an explicit launch-ledger limitation, not a claim of complete system-wide process accounting.
- Checked `ps -p 2377731,2377719 -o pid,ppid,lstart,etime,pcpu,stat,args`: only header, both absent. Then checked all ang.gao process command lines for the independent probe, ownership/camera selection, and optical sacrificial worker/supervisor names; no matches remained. The first remotely quoted grep alternation was misparsed by the Windows/SSH shell boundary and produced command-not-found errors; it was replaced with remote `ps` piped to local PowerShell `Select-String`, which completed cleanly. No process was killed.
- Local `Get-CimInstance Win32_Process` scan for the four recorded PIDs, their children, and python/ssh/scp commands matching c2c1 or the selected tests returned no entries. It does not prove the absence of arbitrary detached processes with unrelated names/parentage.
- `nvidia-smi --query-compute-apps=pid,gpu_uuid,process_name,used_memory --format=csv,noheader` showed only unrelated PID 2000951, `/home/lin.li/ChainLoRA/chainlora-conda/bin/python`, 4910 MiB on `GPU-22a6a96e-63d9-b348-e2db-ff3e4813a989`. No reviewer GPU work or residual was observed. That unrelated job was untouched; no GPU reset or cleanup of another user's work occurred.
- End-of-round local and remote Git status remained clean. Scratch files were retained for reproduction. Tests used `-B` and disabled pytest's cache provider where invoked; the inspection did not rely on previous test output or cached test results. No unresolved owned hang was observed.

## Closed gates and remaining work

F1 must be fixed and independently re-reviewed before C2c1 approval. Even a subsequent C2c1 approval would approve only these CPU owners and compatibility behavior. Full C2c resolved sequences, canonical camera artifacts, v13 row/schema/binding extensions, read-byte asset custody, final production import allowlists, and worker adoption remain incomplete and unapproved here.

C2d/e/f, integrated Warp/CUDA lifecycle acceptance, GPU product preparation/submission, R2/S/host/depth-two delivery, C4d/B3, a3b/result/baseline authority, backend selection, and C4 parent completion remain closed. CPU comparisons and author-run GPU regression logs do not substitute for their later real-device acceptance. Existing compatibility aliases remain intentional single-object imports; their later cleanup does not authorize maintaining duplicate implementations.
