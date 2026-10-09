# Warp compilation prerequisite — independent native acceptance review

Initiative: Q54 / P12.4 / C4 Warp module preparation
Stage: bounded native acceptance review
Author: Codex, independent reviewer session /root/warp_compilation_native_acceptance_v1
Version: v1
Date: 2026-10-10
Status: accepted (bounded compilation prerequisite only)
Related Files: warp-module-preparation design-request v2, design-review v2, code-review v4, test-evidence v1–v7; warp-module-fault-support code-review v2; warp-admission-support code-review v1; runtime-integration design-request v3 sections 4, 5, 8, 9; remaining-setup gap-scan v1 G1
Owner Summary: Approve the bounded Warp compilation prerequisite. Independent raw-evidence replay, artifact inspection, source inspection and fresh CPU boundary checks support the required compilation acceptance. This accepts neither newer output-storage code nor complete C2d/G2/C4, rendering, readiness, warmup, final barrier or product activation.

## Verdict and exact revision

**Verdict: Approve**, for the existing guarded Warp compiler, frozen source/private-cache policy, actual ModuleExec custody, module-set admission and corresponding native prerequisite verification. No substantive finding or unresolved mandatory evidence gap remains within this bounded supplement. The approval is based on checked code and raw evidence, not prior verdicts or author summary counts.

Production compilation implementation is `beedab6143e702a1a50ef0e2686fd0138e194376`. Native support checkpoints are `552d1b78f90d3b358710174bed2c58b6ccdd571f` and `4fe6b3ae71ac6aa1586cad5f4f7ac66c6717a2ec`. The last is the exact native-reviewed source checkpoint and remains clean, detached, in `/data2/shared/ang.gao/robot_sim_warp_modules_wip` on `tileops-dev`. I independently checked the cumulative diff: only support tests and records changed after beedab6; `git diff --name-only beedab6 4fe6b3a -- optical_worker optics tools` is empty. Thus these runs exercise the same production compilation code, with separately reviewed support observations and assertions.

The shared local repository initially had clean HEAD `12c161969a6393722510cb3c373e2a0e3c25a637`. During this review its coordinator archived a different review and evidence at `8379f3423fbb89720bce0e3d1ccc7335b6ee639c`. I inspected that delta: two collaboration reports plus PROGRESS, no source changes; the tree was then clean. At report cut the coordinator had begun separately authorized next-storage support work: final local status has modified `tests/support/cuda_artifact_runtime_acceptance.py`, `tests/support/native_runtime_process.py` and `tests/unit/optics/test_prepared_fixed_render_inputs.py`, with HEAD still8379f342. None of those in-progress bytes was used or reviewed here. All source inspection/testing here used `git show 4fe6b3a:path`, the remote frozen source, or a scratch export of 4fe6b3a. Final remote status remains clean at4fe6b3a. The newer allocation/storage implementation at12c1619 is **not exercised or accepted by this native evidence**.

Supplement v2 SHA256 independently matches `194b3c25130df77954f5b2de2b556da81009b98f94efb861848ae06566a41a5b`. I read repository AGENTS.md, collab/README.md and REVIEW_PRINCIPLES.md, the supplement and its design review, full compilation code-review v4, both support reviews, relevant parent sections and G1. Historical author evidence is context; only the identified current-production runs below count. This is a fresh independent session; I authored no candidate code, changed no candidate file and delegated no work.

## Requirement-to-evidence assessment

| Supplement obligation | Independently checked evidence and conclusion |
| --- | --- |
| Captured source wins over stale linecache and changed disk bytes; later mutation rejects | Actual captured loader publishes held UTF-8 lines with mtime=None before source execution (`native_bootstrap.py:401,455`) and audits identity/content. Fresh production-loader tests in both lanes prove stale-entry replacement and inspect consumption after disk mutation; four later-cache mutations reject. Pinned Warp codegen uses `inspect.getsourcelines/getsource`, not an alternate reader. Real v15/v16 compile the captured optical modules; artifact source hashes match committed sources. Native v36 rejects subsequent linecache mutation. This combined proof establishes the CPU source-selection contract and actual native compilation; CPU tests alone are not credited as a CUDA compiler test. |
| One private fresh cache and pinned ordinary compiler | `configure_warp_cache` retains the exclusive mkdtemp path before fallible setup, checks nonsymlink/empty/0700, and configures before init/decorators. `compile_module` rejects existing execs, failed builds and existing module-cache directory before its leaf. Three fresh CPU production-boundary cases verify those rejections. No caller cache-adoption route exists. Native v15 and v16 have distinct retained roots and cold module compilation. |
| Exact compiler inputs/options and sm90 output | All 47 literal header rows match local installed-source copies and the selected profile. A fresh read-only remote check matched 56 installed compiler/header/source/binary files. Inspected build/context/codegen/native flags select in-process NVRTC, CUBIN sm90, serial explicit-module loading, block256, release/default optimization3, backward enabled, fast_math false, fuse_fp true, PCH false and no extra CUDA include directory. Six actual CUBINs independently have ELF64, CUDA machine190, flags0x5a055a (sm90); all18 CUBIN/meta/generated-source artifacts match recorded sizes/hashes. |
| Cold real Go2 at960x640 and1920x1080, target1/ambient0, owner-preserving reuse | v16 and v15 respectively. Each has three compile guards and three same-session reuse calls, unchanged executable identities, exact target context/stream during load and restored ambient state outside. Raw observations come from pinned driver/runtime getters under the original ticket; they are not target-label assertions. Both workers exit0 after one terminal sync and completed release. |
| Replacement/configuration/artifact/backend/target/stream/ABI/record rejection | v22–v36 mutate after all three real compiles and reach the actual composed production binding/source/module-set admission. All reject at their specified error, retain quarantine, exit74 and enter no later native leaf/new guard/sync. v25 exercises unsupported llvm_cuda; v29 redirects the actual CUBIN outside its root. Actual issuance to two distinct CPU sessions followed by foreign-owner admission also rejects, without a leaf. |
| Guarded import before execution; original-guard load/publication/adoption failures | v17 observes the actual captured module adopted with loader state started while guard6 is ARMED, then prevents its body. v18 fails before public load under guard7; v19 raises after actual executable publication but before public return; v20 fails returned-owner adoption after production session retention. Partial-root files, live identity checks and trace replay prove retained Python module/module/execs mapping, published executable where applicable, original error, no completed compile record/set and no next leaf/guard/sync. |
| Hang uses original deadline and honest prefix | v21 stalls after real executable publication under guard7. Original120000ms budget expires, supervisor kills/reaps worker with return -9, seven normal completions/eight ARMs. The available worker record is only the earlier six-completion durable prefix. No worker.json or failure.json terminal record is invented; no restoration or completed quarantine claim is made for the killed hang. |
| Healthy-IDLE entry and post-DISARMED issuance, unknown/incomplete precedence | v9–v14 cover all six cells. Actual IDLE/no active guard and original completed history are preserved. Healthy complete exact-target entry/issuance gets exactly one existing emergency sync and completed release; unknown/incomplete gets quarantine without sync or invented guard_failed. Issuance has three already-held records after ten normal completions. |
| Regress inputs, fixed CUDA preparation and source admission | v37 Warp Go2 input960, v38 CUDA fixed artifacts960, v39 CUDA fixed artifacts1920 all at4fe. Production custody, runtime record, selection and attachment validation pass; phase counts6/7/7 and one terminal sync/release each. Fresh captured-source tests cover both lanes. These are prerequisite regressions, not optical-output or later graph acceptance. |

The source consumption assessment is deliberately compositional. No new native campaign compiled deliberately changed on-disk Python during this review. The actual common loader/inspect tests establish the selection mechanism, source inspection connects the pinned real Warp codegen to that mechanism, and the native cold records/artifacts establish real compilation of those captured definitions. Repeating a GPU compile solely to retest standard linecache selection is not necessary to prove this CPU boundary.

Likewise, v32's selector name `foreign_set` means mutation of the already-issued set's session projection. It is not evidence of a second native session. My extra CPU test uses the actual production issuer twice for separate sessions and passes the first registered owner to the second actual validator; it rejects at the registry identity check. This supplies the distinct-session identity counterexample without claiming another native compiler or foreign-GPU execution. v34/v35 target/stream substitutions reject as CPU binding corruption before GPU use; they are not native foreign-device fault injections.

## Raw native matrix independently replayed

All paths are `/data2/shared/ang.gao/c4-warp-module-author-vN`. C/A below are supervisor normal completions/ARMs; sync is worker terminal emergency attempts. Every non-hang artifact was checked against the corresponding raw failure or binding observations, not merely the table.

| vN | Boundary/case | Revision | C/A | Worker exit | Sync / outcome |
| --- | --- | --- | --- | --- | --- |
| 9 | unknown issuance | beedab6 | 10/10 | 74 | 0, quarantine |
| 10 | healthy entry failure | beedab6 | 6/7 | 74 | 1, released |
| 11 | healthy issuance failure | beedab6 | 10/11 | 74 | 1, released |
| 12 | incomplete entry | beedab6 | 6/6 | 74 | 0, quarantine |
| 13 | incomplete issuance | beedab6 | 10/10 | 74 | 0, quarantine |
| 14 | unknown entry | beedab6 | 6/6 | 74 | 0, quarantine |
| 15 | Go2 1920 positive + reuse | beedab6 | 11/12 | 0 | 1, released |
| 16 | Go2 960 positive + reuse | beedab6 | 11/12 | 0 | 1, released |
| 17 | import raise | 552d1b7 | 6/7 | 74 | 0, quarantine |
| 18 | before-load raise | 552d1b7 | 7/8 | 74 | 0, quarantine |
| 19 | published-before-return raise | 552d1b7 | 7/8 | 74 | 0, quarantine |
| 20 | executable adoption raise | 552d1b7 | 7/8 | 74 | 0, quarantine |
| 21 | load hang | 552d1b7 | 7/8 | -9 | prefix0; original-deadline kill |
| 22–24 | ABI dtype, rank, entry hash | 4fe6b3a | 10/11 each | 74 each | 0, quarantine |
| 25–29 | config, options, cache permissions, artifact bytes, artifact redirect | 4fe6b3a | 10/11 each | 74 each | 0, quarantine |
| 30–33 | executable mapping, copied set, session field, copied record | 4fe6b3a | 10/11 each | 74 each | 0, quarantine |
| 34–36 | target, stream, source cache | 4fe6b3a | 10/11 each | 74 each | 0, quarantine |
| 37 | Warp input960 regression | 4fe6b3a | 6/7 | 0 | 1, released |
| 38–39 | CUDA fixed artifacts960/1920 | 4fe6b3a | 7/8 each | 0 each | 1, released |

Prior compilation F1 is closed by immutable ABI/entry snapshots plus native22–24; F2 and F3 are closed by the pure CPU healthy-IDLE prechecks and native9–14 disposition/history proof. Support diagnostic exception preservation is visible in the inspected nested diagnostic try/bare re-raise, and native17–20 preserve the exact intended original errors. I do not infer exception identity solely from serialized strings.

The positive caches are `/tmp/robot-sim-warp-a_rl7e1h` (v15) and `/tmp/robot-sim-warp-iufhvmk9` (v16). Remote paths remain nonsymlink, private0700 and contain all nine matching artifacts each. Binary sizes for scene/BVH/render are195960/67192/751272 bytes in each run; distinct binary hashes between workers are recorded per file, not assumed to be reproducible-build hashes. All six ELF flags are0x5a055a. Generated sources include WP_NO_CRT and builtin.h; inspected CRT/tile/PNanoVDB conditionals avoid the host-only include paths for NVRTC. Installed support remains within the accepted trusted dependency/native constructor boundary.

## Independent commands and results

Scratch root: `C:/Users/AG/新建文件夹/c4_assets/warp_native_acceptance_v1`; scripts are in its parent. No candidate files were written. Tools used Windows PowerShell/Python3.14.0 locally and `/home/ang.gao/miniforge3/envs/robot_sim/bin/python`3.12.13 remotely. Native evidence is from Warp1.12/Torch2.9.1+cu128, H200 target1/ambient0. Selected Warp environment digest is `02429134207218e87ca1305e6c89cc7b34948cda41499f1ac7399ac961ffbeeb`; CUDA is `a0b6cf8e0feaad6e6bd4ea75849bce623ff91f73ef3626f97f4ae9f6d785d3ce`.

1. `git status --porcelain=v1`, `git rev-parse HEAD`, cumulative path/stat inspection, focused `git show` and UTF-8 source reads. `git -c core.autocrlf=false archive 4fe6b3a` supplies the final scratch snapshot. Cumulative `git diff --check` reports only a preexisting trailing blank line in the archived code-review v4 document; no production whitespace failure was found. This record-format observation is nonblocking and was not edited.
2. `C:/Python314/python.exe -B .../warp_native_review_inventory_v1.py`: hashes and safely extracts original archives and exports frozen source. Independently matched local and remote archive digests below.
3. `Get-Content -Raw .../warp_native_review_replay_v1.py | ssh -o BatchMode=yes -o ConnectTimeout=8 tileops-dev '/home/ang.gao/miniforge3/envs/robot_sim/bin/python -I -S -B -'`: exit0, **31 raw cases passed**. This read-only replay calls actual `validate_custody_artifact`, `RuntimeCustodyRecord.from_payload`, selection and attachment validators. It also AST-extracts and executes the exact frozen runner's success/fault/evidence-checking tail. Launch, writes and live-process checks are excluded from that tail; separate inspection establishes process absence. Saved `offline_replay.log` contains every case/revision/result.
4. `C:/Python314/python.exe -B .../warp_native_review_artifacts_v1.py`: exit0;47 header copies and18 artifact sizes/hashes match, six native ELF64 CUDA sm90 checks pass, captured source hashes equal committed bytes.
5. Same SSH stdin route with `warp_native_review_remote_inspect_v1.py`: exit0;56 installed source/header/compiler-support binary hashes match profile, both retained caches match all18 artifacts, prior owned processes absent. Saved `remote_inspection.log`. No Warp/Torch import, native getter, compilation or device operation occurs in these scripts; nvidia-smi is used only for the scoped process census.
6. Final LF-source CPU verification through `warp_native_review_cpu_final_v1.py`, exit0:

   `python -B -m pytest -q -p no:cacheprovider tests/unit/optics/test_native_bootstrap.py -k 'captured_inspect or captured_source_audit' --basetemp=../cpu_source_final`

   **6 passed,78 deselected in18.02s**.

   `python -B -m pytest -q -p no:cacheprovider tests/unit/optics/test_warp_module_preparation.py <assets>/test_warp_preexisting_admission.py <assets>/test_warp_native_acceptance_foreign_v1.py --basetemp=../cpu_boundaries_final`

   **31 passed in6.83s**. These are CPU tests with doubles for native dependencies; the actual shared loader, issuer/identity registry, production boundary checks and lifecycle run. They are not native compiler/scope substitutes. The supplied preexisting-state repro independently retains SHA256 `ce6adc5d0039132add895b8961a1d75ae343750ecb9be296bf561209f8cd1e4f`.

Initial inspection mistakes were corrected without candidate changes: parent-directory AGENTS lookup used the wrong directory; one broad output was truncated and reread in focused ranges; v9/v10 were standalone archives rather than members of archivev2; local Linux-profile parsing correctly rejected Windows path semantics, so unmodified validation moved to read-only Linux replay; initial Windows Git archive converted LF to CRLF, so it was re-exported with core.autocrlf=false and committed hashes plus final tests rechecked. A PowerShell Get-FileHash argument mistake and an installed-source subdirectory lookup were corrected. None was a production test failure or weakened validator.

| Archive | Independently matched SHA256 |
| --- | --- |
| author-v9.tar.gz | add2066ba7d3e62e9bd870134f3cd7966164f70aa87d2dae582410999f137188 |
| author-v10.tar.gz | 3a2922c2d248a88c020419aa9fd46c10615693aaee92fe3ac0359f36a6b87231 |
| author-evidence-v3.tar.gz | 0cdea96fa3467aea46fc0ff9a5d3b1d7dc079e8bfb72dc3d57fa827a0c75329c |
| author-evidence-v4.tar.gz | 91d08111f6434ca734b202efcb4ce4370e9c311c205dce0c6657510317393c65 |
| author-evidence-v5.tar.gz | 6ac564235e6171b727fac86246052fa2847ce45489160ad7d65e78a54b68c354 |
| author-evidence-v6.tar.gz | bdcd27b71a0c9c00cf527c6aa0e09268ab44443c7c5b27803d6f4467fe3f8e5d |

Names above share prefix `c4-warp-module-`. v3 contains v11–16 and positive caches; v4 contains v17–21; v5 contains v22–36; v6 contains v37–39. Archivev2 was inspected only as historical context. The archives are not exhaustive copies of failure caches, which remain remotely retained.

## Process hygiene, limits and gates

I inspected archive scripts' post-exit `/proc` assertions, original parent/worker identity files and supervisor terminal observations. Independently checked **68 distinct author parent/supervisor/worker PIDs for v9–39 absent**, plus reviewer replay PID3172232/parent3172231, giving70 IDs in `remote_inspection.log`. This includes the final seven regression PIDs3149177,3149401,3149403,3152172,3152181,3160310,3160311. A later exact `ps` found inspection PID3177229/parent3177228 absent (ps exit1 with header only means no matching process). The GPU-process census at that inspection showed only unrelated MPS2238410, untouched. Before report cut the coordinator reported a later census also containing `/usr/bin/python3` PID3180691. My read-only exact `ps` confirmed parent3180687, started Oct10 02:03:27, running a separate TileOps `benchmarks/ops/bench_bmm.py` runner; it is not a reviewer or author compilation process. It was untouched. That final check again found all four named reviewer remote PIDs absent. The earlier census is not represented as a current global absence claim.

Final local CPU wrapper41668 (PowerShell parent43276) reaped pytest43156 and41448, both exit0. Exact Win32_Process queries subsequently returned none of those four IDs. Source test subprocess helpers explicitly communicate/wait, kill/reap their own child on timeout, and the guard-peer fixtures close in finally. Earlier preliminary CPU invocations also reached terminal exit0; their individual child PIDs were not retrospectively reconstructed. The final repeat both validates the corrected byte-exact export and records the relevant PID ledger in `cpu_final_ledger.json`. Every yielded reviewer tool/test session has completed; no reviewer native worker/supervisor/compiler was launched or left running. No remote file mutation, process signal, cache deletion, service change, package installation or GPU reset occurred. PID absence is scoped observation, not a claim about every possible host process or every future hang.

Native evidence is author-produced and independently replayed/inspected here; I did not rerun a GPU campaign. Partial-root custody is established by the actual production retention graph, fault placement, after-failure identity observations and lifecycle traces; it does not claim ownership of an unpublished native handle inside a trusted constructor. Driver-internal caches and trusted native exception safety remain the design's explicit boundary. Failure restoration is established for the observed before-load/after-publication/adoption exits, not arbitrary internal native faults, and not for the killed hang. Dynamic internal compiler tracing and optical kernel execution are not claimed.

Compilation prerequisite acceptance does not issue R_setup/R_final or any prepared token and does not authorize product PREPARE. The later one-slot graph/NativeOpticalArena/timeline integration, recording/entry-hook initialization, ten real warmups, final readiness/factories/seal/barrier, CPU-oracle optical rendering/lifecycle matrices, C4c3/c4, C4d/B3, systematic refactor, losing-lane cleanup after backend decision and fresh whole review remain incomplete. Benchmark/performance/results/baselines and backend selection stay closed. New storage code requires its own native evidence; these older input/compiler runs cannot satisfy that obligation.
