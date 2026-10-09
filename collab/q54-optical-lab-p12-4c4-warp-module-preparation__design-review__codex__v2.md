# Independent Warp module preparation design re-review v2

Initiative: Q54 / P12.4 / C4 remaining render integration
Stage: independent design re-review
Author: Codex independent reviewer /root/warp_module_design_v2
Version: v2
Date: 2026-10-09 (Asia/Shanghai)
Status: accepted — Approve, design only
Related Files: warp-module-preparation design-request v2 and design-review v1; runtime-integration design-request v3; remaining-setup gap-scan v1; fixed-cuda-artifacts acceptance-review v1
Owner Summary: F1 is resolved. The revised design specifies a separate guarded import segment, keeps module validation and adoption within each compile guard, and preserves the parent's actual-state disposal for CPU failures while healthy IDLE. The existing Warp compiler, private cache and retained ModuleExec route can proceed to bounded implementation and its independent review. This is not C4 or native acceptance.

## Identity, candidate and verdict

Fresh independent session `/root/warp_module_design_v2` received the review packet and repository access without the implementation conversation. I did not author or edit the candidate and did not spawn agents. I read AGENTS.md, collab/README.md and collab/REVIEW_PRINCIPLES.md fully; the Chinese principles were reread explicitly as UTF-8 after the initial shell decoding produced mojibake.

- Repository: `C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design`.
- Branch: `implementation/c4-supervised-waits`.
- Original base: `84ed0e761f1ba1348c8955e6368346a37da0eb74`.
- Prior reviewed candidate: `4f1534ea476e7813b26c46e82d7abfb5f7a0b698`.
- Exact reviewed HEAD: `524f5d5faf3b4813ff812361da1b1a51e34ac93f`.
- Complete candidate: `collab/q54-optical-lab-p12-4c4-warp-module-preparation__design-request__codex__v2.md`, 239 lines.
- Candidate SHA256: `194b3c25130df77954f5b2de2b556da81009b98f94efb861848ae06566a41a5b`.
- Archived prior report: `collab/q54-optical-lab-p12-4c4-warp-module-preparation__design-review__codex__v1.md`.
- Prior report SHA256: `08b431eb6bc5350f2d0879580838947c98c3f5050821385d9ed476166d01be6e`.
- Base-to-HEAD delta: three documentation files, 538 insertions. Prior-candidate-to-HEAD delta: v2 request plus unchanged v1 report, 329 insertions. No production change.
- Initial and repeated final `git status --porcelain=v1` were empty; final HEAD remained the exact revision above. `git diff --check 84ed0e761f1ba1348c8955e6368346a37da0eb74 HEAD` passed without output.
- Read-only SSH independently observed clean remote `tileops-dev:/data2/shared/ang.gao/robot_sim_cuda_compilation_wip` at `66240e4fedc24ad8a5109690e3b1e78b450e2f62`.

**Verdict: Approve.** No substantive blocking or non-blocking finding remains for this complete design supplement at the exact candidate digest. F1 is independently closed as a design defect. Its prescribed production and native boundary tests remain implementation acceptance obligations, not tests claimed to have run here.

## F1 closure and state-flow assessment

The original counterexample was an optical import after input_validated with no live setup segment, followed by a blanket failed-guard rule that also covered CPU exceptions after successful DISARMED. The actual captured loader at `optical_worker/native_bootstrap.py:397–425` requires `_require_setup()` before non-core captured execution. The actual controller at `optical_worker/guarded_lifecycle.py:171–238` requires a live setup ticket, rejects nesting, and completes only successful normal segments.

V2 lines 124–137 now place the initial CPU entry checks in healthy IDLE, then require an ordinary import segment with ARMED before captured execution, scope marking and retained definitions. Successful COMPLETE/DISARMED must precede the first compile segment. This is executable through the existing loader/controller API without an outer nested guard or a bypass.

Lines 141–160 place literal identity/options/source/target checks, preexisting-executable rejection, owner adoption, load, returned-executable retention, artifact checks and per-module CPU fact construction inside each ordinary compile guard. Successful DISARMED is required before treating those facts as successful compilation evidence. The new sequence removes the ambiguous checks between compile guards.

Lines 162–173 explicitly put aggregate issuance after the final DISARMED in healthy IDLE with constituent owners already held. Lines 175–187 distinguish active/pending/failed guarded work from these outside-segment CPU failures. The former preserves the original failed segment with no COMPLETE, replacement emergency guard, sync or ordinary unload. The latter applies parent v3 section 8 from actual sticky scope, custody and live guard state. Unknown/foreign scope or incomplete custody still takes precedence; a genuinely eligible complete exact-target IDLE failure can use the parent's single bounded emergency disposition. No completed guard is relabelled failed or resurrected.

This matches the parent's ordered disposition table and sections 8.1/8.2. The existing controller's `_observe_public_operation` distinguishes pre-guard rejection from active failure, `_segment_context` handles failed armed work, and `terminate` selects disposition from its actual state. V2 lines 224–229 add the missing import/entry/aggregate-issuance boundary coverage and explicitly require actual guard history, retained roots and terminal process outcome. The existing in-guard failure matrix remains in lines 220–223. No new lifecycle policy is necessary to implement this clarification.

During final review the coordinator pointed out the current source-session wrapper as an implementation concern; I independently inspected it. `native_bootstrap.py:2363–2394` (`_SourceSession.fail`) unconditionally calls `controller._fail_guard("operation", error)` and passes `emergency_target_synchronize=None`. `_fail_guard` at `guarded_lifecycle.py:133–146` can record a failed guard even from IDLE. Therefore the new compilation entry cannot simply copy an existing broad `except: self.fail(error)` wrapper for its healthy-IDLE checks/aggregate issuance. It must retain source-integrity failures' terminal policy while routing ordinary new-phase CPU failures from their actual state, with the exact held runtime synchronization operation only when the parent admits it. Otherwise the required post-DISARMED test would falsely record a guard failure and quarantine. V2 lines 127–128, 165–167 and 182–187 explicitly forbid that behavior and lines 224–229 require production boundary coverage, so this is a concrete required implementation adaptation, not an unresolved design policy or a claim that the existing source wrapper is already sufficient. Implementation review must inspect that routing and regression coverage for source-integrity failure precedence.

## Whole-design assessment

I inspected required parent sections 4, 5, 5.1, 5.2, 6 and 8, the accepted gap scan's G1–G4 and dependency order, and the fixed CUDA acceptance's bounded scope. The supplement remains consistent with two readiness identities, ten actual warmups, new final factories, one coverage seal followed by the sole final barrier, and existing NativeOpticalArena reuse. It neither interprets current input validation as final readiness nor claims that compilation completes C4.

The source-backed route is feasible:

- Installed `build.py:40–115` calls the existing native in-process compiler and explicit-device module loader. `context.py:7740–7880` supports an actual Module argument and serial max_workers=0. The public wrapper returns no ModuleExec, so a separate compiler process manager or invented returned owner would be unjustified.
- `context.py:2935–3042` publishes executable objects in `module.execs[(device.context, block_dim)]`; `ModuleExec.__del__` at 2287 unloads the native handle. Holding the module and its original execs mapping before load, then retaining the actual published executable before further checks, covers publication-before-wrapper-return failure. The design correctly retains the trusted constructor exception-safety boundary for native handles never published to Python.
- `context.py:2710–2933` uses module/hash/architecture-derived paths and a sibling build directory beneath the selected cache root. A fresh private root, serial route and checks of actual source/binary/metadata/options/module identities support a bounded provenance contract. A later worker cannot adopt a previous worker's cache. Driver-internal caching remains honestly outside the enumerated cache claim.
- `config.py`, `context.py:2429–2448`, and native `warp.cu:3753–3993` support the specified default options, explicit CUBIN preference, NVRTC compilation and PCH disablement. The implementation must still freeze the complete literal options and prove actual sm_90 output and selected compiler inputs; setting a preference is not artifact proof. This is already required by lines 71–76 and 109–120.
- `codegen.py:1171–1180` uses inspect source extraction. Seeding held source in linecache before captured execution is correctly ordered before decorators consume source; mtime=None prevents ordinary checkcache replacement. The contract audits entry identity and content, including mutation of the existing mutable lines list, and applies to referenced captured repository functions. It does not require another loader or broad reflection permission.
- The three existing optical modules contain the named kernels/functions. `WarpDirectLightRenderCommandFactory.prepare` at `optics/warp_native_render.py:2584–2590` performs a load that can reuse the retained module executable, while the existing recorded-launch snapshot around line 3014 already records command/module_exec/hooks/ABI identity. The proposed set and recording checks can extend those mechanisms; no parallel renderer or kernel implementation is needed.
- Existing runtime initialization at `native_bootstrap.py:2725–2815` has a concrete place to configure the cache/options after Warp import and before init. Captured phase lists, linecache integration, literal inventory, session authority and new observations are still implementation work, not falsely claimed existing features.

The selected `tests/data/runtime_environments/183_warp_go2_author_v3.json` currently lists only `warp/native/cuda_util.cpp`, `warp/native/warp.cpp`, and `warp/native/warp.cu` in that native-source area. I independently matched all 47 copied native headers against the supplied offline inventory. This confirms authoring inputs are available, not that the exact runtime compiler closure has been admitted. The design explicitly requires the reviewed file rows and loaded-library selection before native acceptance. No broad native-directory permission or speculative attestation infrastructure is needed.

No additional maintainability issue was found: one existing compiler route, one session-owned private cache, one retained actual module set, and the existing loader/guardian/registry/renderer/arena provide a small implementation direction. Cache archival/deletion has an explicit post-exit boundary. The whole-C4 systematic refactor and losing-lane cleanup remain their existing later obligations.

## Executed verification and limitations

Executed read-only Git identity/status/diff/stat/diff-check commands, UTF-8 source reads and rg searches over the candidate, parent requirements, captured loader/controller, runtime initialization, optical kernel/factory/recorded-command paths and installed Warp compiler/cache/config/source/native sources. V1-to-v2 `git diff --no-index` showed only the stated sequencing/failure/boundary-test changes plus version/context updates. Its difference exit and CRLF warnings are expected; files were not rewritten.

Local hashes were independently matched to read-only remote `sha256sum` via `ssh -o BatchMode=yes -o ConnectTimeout=8 tileops-dev`. No remote interpreter/backend was invoked. Both local and installed sources matched:

| Source | SHA256 |
| --- | --- |
| _src/context.py | eb099d908c50effc3cacbf7ecc408be4187ca9a6a6f72e1e85c70676961524ee |
| _src/build.py | 5993e58b44a916fc12f35afde1cededc59ab1db7df7a7eecd75551191d9197df |
| _src/codegen.py | c5538f63618a64af90cf6e0eb629df0d47000a8fe7e57844a0c4b8fd6b41bbf2 |
| config.py | e4b95ce936aa6d57aa11f5046879fe04ebdb8793107c88137c9ba0bd6b0b6cf5 |
| native/warp.cu | 7c7e6e7a02db7b285ca1800d0ec52e28d2cf8a704435d925c70a8c5ccd8f176b |

Offline authoring inventory `C:/Users/AG/新建文件夹/c4_assets/c4-warp-compilation-wheel-inventory-v1.json` independently hashes to `9371a533fb0e5b273609cb60ab2a603ba9f0f6b09fbea8749f8ad36cd6c00ab6`. A PowerShell JSON/read/hash loop reported `Inventoried native headers matched: 47`.

After reading its source, independently executed:

`C:/Python314/python.exe -I -S -B C:/Users/AG/新建文件夹/c4_assets/warp_setup_source/linecache_probe.py`

Terminal exit 0, Python 3.14.0. It confirmed that the held function executes held bytes while unseeded inspect reads different existing disk source; seeded mtime=None source survives a disk edit; entry identity alone misses same-list mutation while held-content comparison detects it. Held source SHA256 was `add539def0e86f1aff171d864de11520416864731eb265aa372d800250de94e7`. The probe imported no Warp/Torch/backend, removed only its exact temporary source and empty directory, and left the candidate unchanged. This establishes Python source-selection behavior, not actual Warp codegen or CUDA compilation.

Some broad displays were truncated; focused source ranges and structured inventory reads were used for the relevant facts. An exploratory rg for native profile rows in `runtime_environment.py` returned no matches (and exit 1); the corrected JSON profile inspection found the three actual rows above. These are inspection/tool-output issues, not candidate failures.

No pytest suite, native import/init/getter, compilation, GPU query, runtime fault campaign, service, installation, signal, cache cleanup or process takeover was executed. Actual target1/ambient0 behavior, source-to-generated-code correspondence, both Go2 resolutions, cold cache creation and owner-preserving reuse, entry initialization, compiler native failures/hang, and the newly required guard boundary/disposal matrix remain mandatory implementation/native evidence. CPU mocks and this source probe cannot replace them.

Every reviewer tool call reached a terminal result; no yielded exec session remains. Only short-lived shell/Git/read/hash/SSH utilities and the CPU source probe were launched. No worker, supervisor, compiler or GPU job was started or signalled. Process hygiene is bounded to those completed calls and the probe's exact cleanup; I did not enumerate or attest unrelated OS processes. The packet's no-live-owned-job statement was not treated as a global process census. This report is the sole reviewer-created persistent file and is outside the candidate.

## Approval scope and remaining gates

Approval binds the complete v2 design at HEAD `524f5d5faf3b4813ff812361da1b1a51e34ac93f` and SHA256 `194b3c25130df77954f5b2de2b556da81009b98f94efb861848ae06566a41a5b`. It permits bounded implementation of the existing guarded Warp compiler/private cache/frozen source/actual ModuleExec custody contract, followed by independent code review and native acceptance. The unchanged report can be archived under the evidence-only exception; operative changes require review.

No implementation, native compilation, optical output, one-slot graph/timeline/arena, ten-warmup execution, R_final/final recording, seal/barrier, CPU-oracle render correctness or full lifecycle matrix is accepted by this report. Whole C4 remains required, including C4c3/c4, C4d/B3, systematic refactor and fresh whole review. Product PREPARE, benchmark/results, baseline publication, backend selection and later parent gates remain closed. There is no performance claim or permission to substitute a subset milestone for entire C4 completion.
