# Provisional render graph: bounded native evidence review v1

Initiative: Q54 / P12.4 / C4 render integration
Stage: independent bounded native-evidence acceptance review
Author: codex
Version: v1
Date: 2026-10-10
Status: accepted
Related Files: AGENTS.md; collab/README.md; collab/REVIEW_PRINCIPLES.md; runtime-integration design-request codex v3 and approved review v3; provisional-render-graph code-review codex v2; render-graph-native-support code-review codex v1; provisional-graph-native-repair code-review codex v1
Owner Summary: Approve only the eight frozen positive Go2 provisional graph preparation cases at exact native source revision 6300531a300d68dd558d155979e0ef86e3e78dfb. Original native artifacts pass independent replay and integrity checks. They establish recording/binding and event initialization, with zero optical execution/warmups. Full graph failure acceptance and every later C4 gate remain closed.

Reviewer session: `/root/render_graph_native_acceptance_v1`, fresh independent Codex session, task packet only; no implementation authorship and no delegated agents.

Verdict: **Approve**, exclusively for the positive native-evidence scope stated here. No substantive blocking or non-blocking finding in this bounded scope. No whole-graph or whole-C4 acceptance is granted.

## Exact source, archive and review isolation

Native candidate: clean detached `6300531a300d68dd558d155979e0ef86e3e78dfb` at `tileops-dev:/data2/shared/ang.gao/robot_sim_render_graph_repair_wip`. Initial and final remote HEAD/status checks matched; no native candidate edits, commits, new worker jobs or GPU jobs occurred. Local documentation checkout was `C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design`, HEAD `3a3cd3af583cf28a9cbc3addeab360867de6770a`. That local documentation HEAD is not the native source revision.

Frozen archive: `C:/Users/AG/新建文件夹/c4_assets/c4-render-graph-native-evidence-v1.tar.gz`, independently hashed locally and remotely as `585cebd270937e04593f19e8cb0797c7d4cf9f3908ca5a7ae28249dcc86bcb78`. The archive contains 241 regular files. Every archived regular file was compared byte-for-byte with its original remote counterpart, including all eight raw directories, scripts, batch record and both environment profiles: all matched. Author summary and archive script were inspected as claims, not executed as the review or substituted for raw replay.

Scratch extraction, exact-source snapshot and review programs/results are under `C:/Users/AG/新建文件夹/c4_assets/render_graph_native_acceptance_v1`. The source snapshot was generated from exact 6300531 with `git -c core.autocrlf=false archive`; source catalog LF SHA256 is `5fe8701d31b6953444e9c666620a5217ead719c744ad566a08d9d0b22c358504`. CPU replay ultimately imported the unchanged exact remote candidate directly, with `-I -S -B`.

Read the collaboration protocol and complete review principles, accepted design sections 4, 5/5.1/5.2, 8/8.1/8.2 and 9, approved design review, and relevant predecessor code reviews. Inspected actual source entry, graph preparation, resource custody, native recording/event path, support observer and validators. Prior approval conclusions and old test totals were not reused as evidence of these runs.

## Native cases accepted

All cases use `direct_light_full`, shadows enabled, target GPU ordinal 1 and ambient ordinal 0; one independent single-lane worker per case. Native environment: CPython 3.12.13, Warp 1.12.0 or Torch 2.9.1+cu128, NVIDIA H200, driver 595.71.05. Target UUID is `GPU-3a3112ff-837c-3b87-38d1-cb203d656589`.

| Raw directory suffix | Lane | Go2 resolution | Diagnostics | Normal completions | Supervisor / worker PID |
| --- | --- | --- | --- | --- | --- |
| v2 | Warp | 960x640 | off | 13 | 3361452 / 3361453 |
| v3 | CUDA | 960x640 | off | 9 | 3366249 / 3366250 |
| v4 | Warp | 1920x1080 | off | 13 | 3373542 / 3373549 |
| v5 | CUDA | 1920x1080 | off | 9 | 3380621 / 3380623 |
| v6 | Warp | 960x640 | on | 13 | 3389232 / 3389235 |
| v7 | CUDA | 960x640 | on | 9 | 3394406 / 3394408 |
| v8 | Warp | 1920x1080 | on | 13 | 3403872 / 3403874 |
| v9 | CUDA | 1920x1080 | on | 9 | 3408001 / 3408002 |

Each case binds ten distinct frame bundles with ordered frame IDs 1..10. Each Warp case has 50 distinct retained recorded commands, five per frame, using the retained compiled render ModuleExec. Each CUDA case has three issued native entry observations per frame and the original renderer/context binding. Diagnostics off has one initialized terminal event; diagnostics on has seven events, consisting of the terminal event and three diagnostic pairs. Graph guard descriptors show exactly those event create/record counts, one final event wait, and no graph allocation or warmup launch.

Every case has warmup_frames_completed=0, sample_submission_count=0 and final_execution_readiness_issued=false. The complete worker descriptor traces also contain no warmup_launch. These are provisional construction frames, not ten completed warmups or ten rendered frames.

## Independent verification and evidence chain

1. Executed exact-source `validate_custody_artifact` on every original combined artifact, checking equality with the separately archived worker and supervisor-output artifact. Executed `validate_runtime_custody_selection`, `RuntimeCustodyRecord.from_payload` with the original launch inputs and lane-specific environment, and `validate_runtime_custody_attachment`. All eight return `worker_prefix_matched`; actual runtime journals, components, object custody and prepared storage/input validation are replayed, not merely their summary hashes. Each input validation report covers 42 arrays. Request/batch/result/lane/case/revision/catalog/configuration and parent/supervisor/worker relationships were cross-checked.

2. Executed exact-source `_validate_fixed_storage` and `_validate_render_graph` on every original raw binding and full worker trace. Verified supported resolution, diagnostics, exact target allocation scope, output dimensions, original prepared/view identity, frame/event/command custody and ambient restoration. Warp snapshots restore default device 0 and no current context; inside graph adoption only primary context 1 is active and the held target stream is current. CUDA restores logical device 0 with raw device 1, matching its frozen lazy-restoration contract; inside graph adoption both logical/raw device are 1. This is not a claim that CUDA raw device is restored to 0.

3. Independently checked Warp module observations: three compiled modules, six first/reuse calls, identical module/executable identities across reuse, three restored outer scopes, and every graph command executable matching the retained render module's executable. Checked CUDA factory scope, original issued factory-binding observation, target/compute capability and zero execution authority. The production observer calls the real SourceSession graph entry and original preparation function. It checks guardian-held partial owner lists before observing/adopting returns, then validates actual Warp packed bundle parameters or CUDA transaction stage ABI and issued entry/context identity. It restores its observation wrappers in finally; it does not manufacture native commands/events.

4. Inspected `SourceSession.prepare_fixed_render_graph`, `native_setup._prepare_fixed_render_graph`, `_warp_command_recording`, event preparation, the compiled-owner checks and native-driver dispatch. Production creates truthful warmup0 readiness under the original guard, binds canonical frozen parameters and original storage, retains resources before subsequent leaves, validates every frame, records completion initialization last and waits on that event. Warp uses `record_tape=False, record_cmd=True`. Independently read pinned Warp context.py (SHA256 `eb099d908c50effc3cacbf7ecc408be4187ca9a6a6f72e1e85c70676961524ee`): the forward CUDA record_cmd branch returns a Launch before native kernel launch. CUDA creates/binds stages; no stage enqueue is invoked by this path. Event initialization is real device work but is not optical completion.

5. Ran fresh cold CPU source preflight once per lane through the original `capture_runtime_custody_inputs`/isolated preflight. Entire results exactly matched archived upstream inputs, including executable/source alias identities. Reproduced source snapshot hashes: Warp `707c87b12ec89328ab01f94787ddacfedebe277cb5e7931b38a23ccc76a6a3c8`; CUDA `72b4e8a45928935d68ce3f81852c9349b1a21c9a591b29b5bbbb6b544956201a`. All eight archived records use the respective snapshot. Ran `RuntimeEnvironmentContract.verify_files` against both current pinned profiles: Warp `02429134207218e87ca1305e6c89cc7b34948cda41499f1ac7399ac961ffbeeb`, CUDA `a0b6cf8e0feaad6e6bd4ea75849bce623ff91f73ef3626f97f4ae9f6d785d3ce`. Both passed. The integrity process explicitly verified it imported no torch, warp, numpy or multiprocessing; no native initialization was performed by review.

6. Applied 11 relevant alterations independently to copies for each original case: fabricated warmup credit, changed expected executable/entry, changed event identity, foreign primary-context scope, changed graph guard, missing frame, removed event-wait descriptor, forged terminal sync count, forged supervisor normal count, cross-case trusted input and changed source revision. All **88 alterations rejected** by the corresponding exact-source validators. Mutated copies existed only in reviewer memory; no original evidence was modified. Representative errors include `recorded Warp command replaced its compiled executable`, `CUDA stage captured another executable entry`, `summary mismatch`, `summary does not equal replayed state`, `completed case differs from trusted launch`, and `runtime custody differs from trusted launch inputs`.

7. Strict terminal checks passed for all eight: 13/9 normal completions, one additional emergency arm, exactly one successful terminal target synchronization, completed release, guardian RELEASED, worker exit 0, supervisor EXITED/complete with observed return code 0, no pending ACK, no kill request, no expired deadline, no stderr, and no failure/fault/ordinary-atexit marker. The worker emergency ticket remains ARMED at terminal intent; no emergency normal-completion credit is fabricated. Full terminal traces show normal_disarmed, custody retention, terminal arm, sync success, release authorization, cleanup/release and terminal intent. Supervisor evidence supplies observed exit separately. This single terminal disposal synchronization is not the later final setup barrier.

These checks validate archived observations within the approved trusted closed-worker model. JSON object IDs and hashes alone are not unforgeable attestation of a live GPU object. The inference of native behavior depends on the verified source path, original live observer assertions, raw lifecycle/journals and supervisor observations together. This review did not rerun native construction or claim reviewer-generated native evidence.

## Reproduction commands and outcomes

From the review scratch directory:

```text
C:/Python314/python.exe -B run_remote_replay.py
```

This local transport sends the preserved `remote_replay_stdin.py` to:

```text
ssh tileops-dev /home/ang.gao/miniforge3/envs/robot_sim/bin/python -I -S -B -
```

Result: exit 0; all eight original cases and all 88 rejection probes passed. The retained `remote_replay_output.txt` includes case results, errors for every altered copy, timestamps and reviewer PID. Pure replay portion took about 26.5 seconds. The same SSH/interpreter command with `remote_integrity_stdin.py` as stdin completed exit 0 and produced `remote_integrity_output.txt`. `hygiene_stdin.py` and `hygiene_output.txt` preserve the final source/process/GPU query. No author campaign or archive-creation script was executed.

Initial reviewer-only attempts are disclosed: Windows git archive inherited CRLF conversion, so it was regenerated with core.autocrlf=false before byte-sensitive use. Linux environment paths then correctly failed the Windows Path validator, so validation moved to the pinned Linux interpreter. One initial remote diagnostic used an incorrect reviewer assertion that validated_arrays was an integer; the artifact contains a 42-element list, and the probe was corrected to check its length. None of these was a native candidate failure. A broad output truncation was followed by targeted source/JSON reads. No failed probe is counted as a pass.

## Process hygiene and side effects

Original batch parent is 3361273. Archived identity/parent files and raw supervisor records corroborate the eight pairs above. The packet reports the author's outer batch tool handle terminal exit 0; the reviewer cannot independently recover that historical tool handle and does not treat the packet assertion as native evidence. The batch/result records, raw observed worker exits and fresh PID checks are the independently examined evidence.

At `2026-10-09T19:50:55.008515Z` (2026-10-10 03:50:55 +08:00), read-only /proc checks found the batch parent and all sixteen recorded supervisor/worker PIDs absent. Reviewer remote CPU processes 3436366 (first diagnostic, parent3436364), 3436808 (successful replay, parent3436807), and 3437944 (integrity, parent3437943) were also absent. Local SSH transport PIDs 46992, 7092 and 48068 were absent on the final Windows query. Replay tool session 11732 ended exit 0; other substantive checks were synchronous terminal calls. Short-lived CPU preflight children and read-only utility processes were not all individually sampled. No review worker/supervisor/GPU process was started, no signal or reset was issued, and no unresolved owned residual was observed.

PID/start identity caveat: archived identity files contain PID and parent PID, not OS start ticks. Fresh absence proves those numbers are no longer present; it does not retroactively provide complete start-time attestation or rule out all PID reuse. It is combined with archived supervisor-observed exits and the case launch chain, not interpreted as a global no-process/no-hang proof.

Archived per-case GPU lists contain no recorded graph worker PID. The fresh GPU list showed the ambient MPS process2238410 and unrelated target1 process3432063. Scoped identity inspection identified the latter as TileOPs full/nightly pytest, parent3432041, start ticks243959205; /proc cwd access was denied. It was left untouched and is not a graph residual or a reviewer process. The shared host is not claimed idle.

Review side effect: one process-identity transcript, `gpu_other_identity.txt`, was accidentally created as an untracked file in the local documentation checkout. It was immediately moved intact into the outside-tree review scratch directory with Move-Item. No tracked file was edited; final local status was clean at unchanged 3a3cd3a. The remote candidate and original archive/artifacts were never modified. All remaining review artifacts are outside the candidate.

## Limits and closed gates

Approval covers only the eight positive real-Go2 cases listed above, original source revision6300531, profile direct_light_full, shadows on, provisional R0 graph binding/recording, native event initialization and successful terminal disposal under the existing supervisor. It closes the positive native bring-up evidence for that specific slice.

It does **not** close graph failure/partial-return/retention/hang matrices, guardian sentinel failures, fault-device restoration, cleanup-hang deadline tests, all profiles/shadow modes, six R0/two R1 row acceptance, numerical optical correctness, warmup execution authority or ten observed warmups. No optical image or measured performance was produced by this campaign. R_final/final origin, new final factories/graph, coverage seal/final setup barrier, timeline/arena/sample/token/reclaim integration and full C4 validation/refactor/fresh whole review remain required.

Product/native preparation activation, sample submission/IPC, host delivery/external holds/depth two, R2/S/physics publication, result authority, backend comparison/selection, a3b, baselines, shipping/default activation and whole-C4 completion remain closed. There is no optional code repair demanded by this bounded verdict; broader mandatory acceptance is not waived or relabelled as a non-blocking pass.

The coordinator may archive this complete report unchanged under the shared evidence-only recording exception. That recording must preserve the exact source/archive binding, verdict, scope and closed gates.
