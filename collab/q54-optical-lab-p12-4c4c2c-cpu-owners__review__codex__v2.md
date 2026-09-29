# C4c2c1 CPU definition ownership — independent code re-review v2

Initiative: q54-optical-lab-p12-4c4c2c-cpu-owners
Stage: review
Author: codex (independent reviewer)
Version: v2
Date: 2026-09-30
Status: accepted
Related Files: runtime-integration design codex v3 at 674af14; approval archived at c969296; CPU owners implementation-note v1, complete review v1, review-followup v1
Owner Summary: Approve exact 4cac1be for C2c1 CPU definition ownership only. F1 is independently closed by reproducing the original Python 3.12 failures, verifying the historical fixture provenance, and executing the repaired tests. Full C2c and all runtime/product/result activation gates remain closed.

## Identity and frozen scope

- Reviewer session: `/root/c4_c2c1_cpu_owners_review_v2`, fresh independent code-review agent, not the author. No child agent was spawned.
- Candidate: `C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design`, branch `implementation/c4-supervised-waits`.
- Extraction base: `c9692964ddb9a2226c6d6b076b991c6c22922d59`.
- Original implementation: `234268aa1c967b2365d7dd08f34cd215d24f306a`.
- Reviewed repaired HEAD: `4cac1be29efde9912c3ea1ebcf0781ef37423db5`.
- Remote candidate: `ang.gao@192.168.80.101:/data3/ang.gao/robot_sim_c4_c2c1_4cac1be`, independently checked at the same exact HEAD.
- Local and remote tracked/untracked trees were clean at initial and final checks. No candidate source, test, document or gate was edited. Scratch probes and this report are outside the candidate. Normal tool caches are not evidence of test success; Python checks used `-B`, and pytest cache provider was disabled.
- Read AGENTS.md, collab/README.md and collab/REVIEW_PRINCIPLES.md fully, the approved v3 design, complete first code review and current implementation/follow-up notes. Initial root-level REVIEW_PRINCIPLES lookup failed; the correct collab file was then read fully as UTF-8. Truncated combined output was reread in bounded sections for relevant instruction/design content.

## Verdict and findings

**Approve**, limited to C2c1's sole production CPU definition owners and their compatibility behavior at the exact repaired HEAD. No substantive blocking or non-blocking finding remains in this review.

### F1 (prior P2): independently closed

The original test/fixture defect is real. I loaded the actual original test source and original JSON fixture using `git show 234268a:<path>` and executed all four original parametrized AST test bodies against the unchanged production extraction. Python 3.14 produced zero failures; Python 3.12 produced the original four failures, for workloads, assets, camera presets and the importer. Separately, all 36 original ast.dump hashes matched in 3.14 and all 36 mismatched in 3.12. This is reproduction of the original test, not an assertion that a similar toy test would fail.

For the repair, I independently read each original source with `git show c969296:<path>`. All four source-byte SHA-256s, all 36 definition-name entries, decorator-aware first/last line spans, and all 36 repaired structural hashes agree with the committed fixture on both Python 3.14 and 3.12. Original source hashes and spans are unchanged relative to the old fixture. The fixture was not merely checked against the new production definitions.

`tests/support/structural_ast.py` encodes explicit node types, ordered field names and values as deterministic compact JSON. Empty lists and None remain represented; only empty function/class type_params are omitted. Nonempty type parameters remain significant, unsupported bytes/complex constants raise TypeError, and nonfinite JSON numbers cannot silently pass. Location attributes are excluded as intended. The exact importer module/name/alias/level normalization in the test remains tightly restricted, not blanket import deletion.

The committed algorithm/argument-order/default/import drift tests passed. Additional scratch negative probes mutated the actual extracted build_canonical_packed_scene import module, alias, relative-import level and default in memory; every mutation changed its structural hash. A nonempty generic type parameter also changed identity; unsupported constants failed explicitly. These checks preserve semantic sensitivity while removing ast.dump formatting dependence. No production code changed in the repair.

## Production ownership and boundary checks

1. The full base-to-HEAD delta is confined to four new CPU owners, their four legacy compatibility surfaces, supporting ownership tests/fixture/helper, the existing a3a test's owner import, two test-only sacrificial source additions, and review/implementation records. The original-to-repaired delta contains only the helper, fixture, ownership tests and two review records. No worker driver, manifest policy, protocol, frozen workload/case contract, result/gate artifact, collab/README or CLAUDE change is present.
2. Independently enumerated new top-level definitions rather than trusting the fixture's selected-name list. Same-interpreter old/new AST equality passes for all 31 moved definitions: assets 9, workloads 5, MJCF importer 14, camera 3. The sole normalized body change is the precise importer relocation. The original definition sets partition exactly into moved and retained definitions; all five retained example functions also match the historical source. Module-level constant assignments in assets and workloads are unchanged.
3. Executed exact legacy/new object-identity and production __module__ tests for every moved definition. Old modules explicitly re-export the real objects; there is no second implementation. Existing benchmark native/case/workload-contract modules already re-export the production objects, so relocating their imports does not introduce parallel types/contracts. The example keeps its CLI/render and legacy preview-camera functions, not a worker-only copied importer.
4. Inspected actual import dependencies and retained transitive MJCF helpers. The importer uses production registry/frame/spatial types. Fresh-process closure tests reject benchmark/example/tool/Torch/Warp import attempts and exercise negative controls. My real-Go2 run additionally installed a deny-import finder before loading production assets/workloads/importer, successfully materialized both resolutions, and observed neither forbidden import attempts nor loaded forbidden roots.
5. Recomputed exact camera scalar/array parity against Git historical camera source for 120 orbit indices plus -1, 120 and 121, eight bounds/view cases including zero extent, fixed-view dispatch and reserved pose rejection. These passed in both interpreters. Additionally compared all 120 cameras from the actual imported Go2 bounds remotely. This preserves the existing builder; it does not implement or approve the future worker modulo-index/sequence adapter.
6. The changed a3a test now patches the actual production owner where the function globals reside. Its negative check is not disabled; the entire a3a file passed remotely. The sacrificial list adds only optical_worker/assets.py and workloads.py to the existing test path. Product source admission is unchanged. Traced the unchanged worker PREPARE branch to `_reject_gpu_preparation`: it terminates/fails closed and reports no measured sample GPU submission.
7. Asset preflight/body logic is structurally unchanged. Real Go2 inputs were used, not a surrogate or download. Read-byte/TOCTOU custody and final production source allowlists are explicitly outside this ownership-only slice and are not certified here.

## Independent execution and observed results

Local: Windows PowerShell, `C:/Python314/python.exe`, Python 3.14.0. Remote: authorized fallback 101, `/data3/ang.gao/anaconda3/envs/robot_sim/bin/python`, Python 3.12.13. The packet established primary transport unavailability; I did not install dependencies or mutate either environment. Local find_spec confirms Torch and Warp absent, so local a3a runtime-hook coverage was not claimed.

Read-only checks included git status/rev-parse/branch, full and repair-only diff path/stat inspection, `git diff --check c969296 HEAD` (clean), actual source/alias/manifest/worker reads, and the archived first report's SHA-256 (`f3082f40284707a1984109d9e81844e0a978b655c2a6e7cc08b7e09229bbca6b`, matches the follow-up).

Executed from the local candidate:

```text
C:/Python314/python.exe -B ../c4_assets/c2c1_v2_review_probe.py
C:/Python314/python.exe -B -m pytest -p no:cacheprovider -q tests/unit/optics/test_cpu_materialization_ownership.py tests/unit/optics/test_resolved_camera_contract.py
C:/Python314/python.exe -m ruff check optical_worker/assets.py optical_worker/workloads.py optics/camera_presets.py optics/mjcf_visual.py benchmarks/optical/assets.py benchmarks/optical/workloads.py examples/mujoco_menagerie_robot_preview.py tools/optical_pipeline_lab/camera_presets.py tests/support/sacrificial_sources.py tests/support/structural_ast.py tests/unit/benchmarks/test_optical_benchmark_a3a.py tests/unit/optics/test_cpu_materialization_ownership.py
```

Results: probe passed; **88 passed in 2.49s**; Ruff **All checks passed**. The external v2 probe contains strict provenance, original-repro and mutation checks. It reuses the fully inspected prior external parity probe with only the reviewed HEAD/base and repaired serializer adapted in memory; no candidate assertions are weakened. The prior probe's printed `parent` label in that adapted output denotes extraction base c969296, not repaired HEAD's immediate parent.

Transport for remote calls:

```text
ssh -n -o BatchMode=yes -o ConnectTimeout=8 -i C:/Users/AG/.ssh/tileops_ed25519 ang.gao@192.168.80.101 <command>
```

Copied only external probes using scp with the same identity/options to `/tmp/c2c1_v2_review_probe.py` and `/tmp/c2c1_reviewer_probe.py`. Remote commands from the clean repaired candidate:

```text
timeout 60 /data3/ang.gao/anaconda3/envs/robot_sim/bin/python -B -m pytest -p no:cacheprovider -q --tb=short tests/unit/optics/test_cpu_materialization_ownership.py tests/unit/optics/test_resolved_camera_contract.py tests/unit/benchmarks/test_optical_benchmark_a3a.py
timeout 240 /data3/ang.gao/anaconda3/envs/robot_sim/bin/python -B /tmp/c2c1_v2_review_probe.py --assets /data3/ang.gao/c4_asset_frozen
```

Results: **117 passed in 3.36s**, exit 0; full real-asset/provenance/camera probe completed with exit 0. An earlier full asset probe used only timeout 60, printed successful 960x640 results, then ended nonzero before completing. Tool session 79360 reported exit 1, without a Python traceback or captured raw timeout status. I do not treat that partial attempt as a pass or confidently assign its raw shell termination code. Its recorded PIDs were confirmed absent before the longer bounded rerun, which completed all checks without candidate changes. The pytest command chained after that initial attempt did not execute; the successful pytest selection was a separate invocation above.

Real asset root: `/data3/ang.gao/c4_asset_frozen/out/external/mujoco_menagerie/unitree_go2`, manifest q54_asset_manifest.json. Independently observed:

| Identity | SHA-256 |
| --- | --- |
| Asset aggregate, both resolutions | `479dcbdb52c64ca86243d6b1d3b7167bb7d3820af1e60e355168b3c17ddf3ed0` |
| 960x640 scene | `745f01f3d526681c3a350086d55211f92d4a1e5f5b22bcbbc9c43f7dd18f072f` |
| 960x640 fixed fixture | `fce32d13427918022c4eaee45485ed6fbfaf5e03b13d1390a0216e138e483182` |
| 1920x1080 scene | `57e86366429979a2460a0ceb01f00749c7e4be7449bbc3acc1577ed5ce94b923` |
| 1920x1080 fixed fixture | `8354993fb682d5a13a2dff62d1d7521afd7d3d46d20b876e15732b4c38415db0` |
| 1920x1080 replay | `2b1ecfb401476a9737400b401a72377c880bc9e289370f2d3bd5600cc287a216` |

All match unchanged frozen content contracts. Committed tests independently covered nonasset scenes/replay/fixtures and absent-Go2 rejection.

Supplementary, not my GPU execution: I read the author's repaired pytest.log, metadata.json and summary.json under `/data3/ang.gao/c2c1_4cac1be_validation_logs`. They record exact HEAD, clean before/after, **497 passed in 112.86s**, monitor returncode 0, watchdog_reason null. That selection includes benchmark regressions and selected real Warp/CUDA tests. I did not duplicate the heavy GPU suite, and do not substitute those author logs for this round's independent CPU/asset checks or any later mandatory integrated-device acceptance.

## Round-end process hygiene and limitations

- Local successful probe PID 30416, parent PowerShell 35124, printed completion. Final Get-CimInstance scan for these PIDs and their children returned no entries. Local pytest and Ruff invocations returned normally; their short-lived child PIDs were not individually captured.
- Initial remote bounded probe PID 2424481, timeout parent 2424480: both absent in explicit ps before rerun and again at round end.
- Successful rerun PID 2432097, timeout parent 2432076: live ps captured start `Wed Sep 30 04:42:20 2026`, matching command, parentage and 32-second elapsed state; probe printed PASS/completion and tool session 9482 returned exit 0. Final `ps -p 2424481,2424480,2432097,2432076 -o pid,ppid,lstart,etime,pcpu,stat,args` returned header only.
- Remote pytest returned exit 0 in 3.36s; its short-lived OS PID was not captured. Final ang.gao process scan for the v2 probe, ownership test selection, optical worker/supervisor names and repaired checkout path had no matches. This is scoped exit evidence, not complete system-wide accounting of arbitrary detached children. No reviewer GPU worker/supervisor was launched.
- Final nvidia-smi compute-app query showed only unrelated PID 2000951, `/home/lin.li/ChainLoRA/chainlora-conda/bin/python`, 5230 MiB on GPU-22a6a96e-63d9-b348-e2db-ff3e4813a989. No reviewer GPU residual observed. No unrelated process was killed and no GPU reset performed.
- Local and remote final Git status remained clean at exact reviewed HEAD. No unresolved owned stall was observed. Scratch probes remain available for reproduction outside the candidate.

## Explicit closed gates and residual scope

This approves C2c1 only: moving the existing CPU definitions to sole production owners and retaining exact compatibility objects/behavior, including the repaired portable extraction regression.

Full C2c remains incomplete: resolved render sequences, canonical camera artifacts, v13 row/schema/binding extensions, immutable asset-read custody, final product import allowlists and worker adoption remain unapproved. C2d/e/f, full integrated Warp/CUDA lifecycle acceptance, separate activation, product GPU preparation/submission, R2/S/host/depth-two delivery, C4d/B3, a3b/results/baseline authority, backend selection and C4 parent completion all remain closed. CPU ownership approval is not runtime or product activation.

Compatibility imports are intentional single-object imports, not duplicate implementations. Their eventual bounded cleanup remains as specified in the accepted design/implementation note. This report introduces no new feature, performance threshold, activation permission or broader approval.
