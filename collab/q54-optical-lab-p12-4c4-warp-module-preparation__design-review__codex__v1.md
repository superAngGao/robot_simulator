# Independent Warp module preparation design review v1

Initiative: Q54 / P12.4 / C4 remaining render integration
Stage: independent design review
Author: Codex independent reviewer /root/warp_module_design_v1
Version: v1
Date: 2026-10-09 (Asia/Shanghai)
Status: in_review — Request changes
Related Files: warp-module-preparation design-request v1; runtime-integration design-request v3; remaining-setup gap-scan v1; fixed-cuda-artifacts acceptance-review v1
Owner Summary: The proposed existing Warp compiler/private-cache/ModuleExec route is feasible and appropriately bounded. Clarify guard coverage around imports and failures outside the three compile guards before implementation. No new compiler infrastructure or change to the parent's warmup/seal policy is needed.

## Identity and verdict

Fresh independent reviewer session `/root/warp_module_design_v1` received the review packet and repository access without the implementation conversation. I did not author or change the candidate, delegate implementation, or spawn additional agents. Read AGENTS.md, collab/README.md and collab/REVIEW_PRINCIPLES.md fully; the Chinese principles were reread explicitly as UTF-8 after an initial console-encoding problem.

- Repository: `C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design`.
- Branch: `implementation/c4-supervised-waits`.
- Exact base: `84ed0e761f1ba1348c8955e6368346a37da0eb74`.
- Exact reviewed HEAD: `4f1534ea476e7813b26c46e82d7abfb5f7a0b698`.
- Candidate: `collab/q54-optical-lab-p12-4c4-warp-module-preparation__design-request__codex__v1.md`.
- Candidate SHA256: `fbe13f38f4211890bdcc7c2b158aa5e8511166727363d9d0b5c830d73628ae3b`.
- Base-to-HEAD delta: only the candidate document, 209 inserted lines. `git diff --check 84ed0e7 HEAD` passed. Initial and final observed porcelain status were empty.
- Read-only SSH independently observed clean remote `tileops-dev:/data2/shared/ang.gao/robot_sim_cuda_compilation_wip` at `66240e4fedc24ad8a5109690e3b1e78b450e2f62`.

**Verdict: Request changes.** One blocking design clarification, F1 below. The proposed compiler/cache/source/ownership direction has no other substantive finding from this review. This verdict does not reopen the accepted parent's two-readiness, ten-warmup, one-seal or final-barrier policy.

## F1 — P2, blocking: specify guard boundaries for imports and idle-state failures

Location: candidate lines 118–146, 148–163; parent v3 sections 4 and 8; actual `optical_worker/native_bootstrap.py:397–425` and `optical_worker/guarded_lifecycle.py:171–176,194–238,449–477`.

The stated ordered entry imports the three captured optical modules at lines 121–122, before the per-module setup guards begin at line 133. The existing captured loader calls `controller._require_setup()` before executing each non-core captured module. Thus a literal implementation of this sequence after `input_validated`, with the previous input guard completed, rejects at the import boundary. Parent section 4 also requires compilation/import/recording to be guarded. An outer guard around the whole entry would conflict with the specified nonnested per-module guards. A short ordinary import segment is sufficient, but the design should say so instead of leaving the author to invent that boundary.

Separately, the module/option checks in step 1 precede each compile guard, and the module-set authority is constructed only after all three successful guards have completed. Both can fail while the actual client is healthy IDLE. The blanket rule at lines 160–163 nevertheless says **any** failure terminates under the original failed guard, with no new emergency guard. There is no such failed guard in these cases. For example, a CPU exception while joining/issuing the final module-set record after the third DISARMED acknowledgement leaves all three compile guards successfully completed. The previous guard cannot truthfully be relabelled failed or resurrected. The parent already routes this case from actual scope/coverage and guard state, permitting its bounded emergency disposition when the healthy-IDLE exact-target conditions hold.

Consequence: two incompatible implementations can claim compliance: bypass/restructure import supervision and manufacture failed-guard state, or follow the existing loader and parent disposal rules while contradicting the supplement's universal prohibition. This is a design ambiguity affecting normal construction and failure custody, not a demonstrated executed native failure.

Why the required matrix misses it: the listed faults cover compile/load inside an original guard, publication-before-return, returned-owner adoption and hang. It does not distinguish import before the first compile guard, pre-guard validation failures between successful modules, or authority issuance after the final DISARMED acknowledgement.

Minimal correction: explicitly place the captured optical imports in an ordinary nonnested setup segment, and identify which prechecks/adoptions/fact issuance are inside or outside each segment. Limit the original-failed-guard/no-new-sync rule to active/pending/failed guarded work. For failures while genuinely healthy IDLE, refer to the unchanged parent section 8 actual-state disposition; unknown scope/incomplete custody still quarantines without repair synchronization. Preserve original exceptions and retained roots in both cases. Do not add a guard manager, restart a deadline, or create an emergency route after a failed armed segment.

Proof after repair: the revised sequence must be executable through the existing captured loader and guard API. Implementation verification should cover (1) imports with ARMED before captured execution and successful nonnested completion; (2) the existing in-guard failure matrix with no COMPLETE/new ARM/sync; and (3) an injected pre-guard or post-DISARMED CPU/issuance failure, whose actual guard state, parent-selected disposal, retained roots and terminal process exit are recorded without inventing a failed normal guard. This is additional boundary coverage, not another infrastructure stage.

## Source-backed assessment

The inspected installed sources support the central proposal:

- `build.py:40–115` reads generated CUDA and calls the in-process `wp_cuda_compile_program`, then loads through the explicit device context. `context.py:7740–7880` accepts the actual Warp Module and serial loading; the public wrapper returns no executable owner. There is no reason to add a subprocess compiler manager for this route.
- `context.py:2935–3042` reuses `execs[(context, block_dim)]` when the module hash matches, skips repeated failed builds, and publishes a newly constructed ModuleExec in that mapping. `ModuleExec.__del__` at 2287 unloads the native module. Holding the module **and original mapping** before entry, then the returned published executable before further fallible work, addresses the wrapper-raises-after-publication case. The candidate correctly does not claim custody of an unpublished native handle or override trusted constructor exception safety.
- `context.py:2710–2933` uses the expected module-name/hash/architecture paths, writes source, binary and metadata in a sibling build directory beneath the selected cache root, then moves/copies them into the cache. A fresh private root and serial compilation make successful first-load provenance tractable without a cache scanner or copied-binary admission route. Same-session reuse must continue to validate the exact retained executable, not merely a file digest.
- `config.py`, `context.py:2429–2448,3789–3866` and `warp.cu:3753–4089` substantiate the selected defaults/options, CUBIN architecture selection and NVRTC path. Implementation must still freeze/check the promised complete option contract, native support/header inventory and actual sm_90 output; the config preference alone is not proof of a CUBIN. The proposal already requires actual artifacts and target checks. Disabling PCH is a bounded simplification; no additional compiler subsystem is necessary.
- `codegen.py:1024–1060,1171–1180` obtains source during decorator/Adjoint construction using inspect. Seeding linecache **before captured module execution**, with mtime=None and held content, addresses this at the correct point. Auditing both cache-entry binding and mutable lines content is necessary and already specified. It must not be delayed until `load_module`.
- The existing three optical modules contain the scene-update, BVH refit and renderer definitions. Existing render factory `prepare` at `optics/warp_native_render.py:2584–2590` calls load_module again; installed `Module.load` can reuse the retained executable. Existing recorded-command snapshots already bind module_exec/hooks/ABI identities. The proposed pre-record validation and post-record comparison should extend these existing checks instead of creating another factory or renderer. New variants remain prohibited after warmup under parent v3.
- Actual source-session, adoption and registry mechanisms exist in `native_bootstrap.py`; the new phase/import closure and selected runtime-environment file rows remain implementation work. The candidate accurately identifies missing native headers rather than treating the current three native-source rows as complete compiler input closure. A literal reviewed inventory is sufficient; broad runtime directory admission or a new attestation system would be unnecessary.

No additional substantive maintainability finding: one private cache per worker, one standard compiler route, one session-held module set, existing guards/registry and the existing arena are appropriately small choices. Cache retention has an explicit post-exit archive/deletion boundary. The eventual whole-C4 systematic refactor remains required; this supplement is not C4 completion.

## Independent verification and limitations

Inspected required parent sections 4, 5, 5.1, 5.2, 6 and 8, the remaining-setup gap scan, and fixed-artifact acceptance scope. Reviewed source loader/adoption/runtime initialization and guard success/failure transitions; actual optical decorators/factory loading/recorded executable checks; installed Warp compiler/cache/config/source extraction/ModuleExec and native NVRTC source. Source reads used `rg` and bounded UTF-8 Get-Content. Some broad outputs were truncated; relevant source ranges were reread separately. An exploratory rg included native filenames in the repository instead of the installed-source-copy directory and returned missing-file errors; corrected reads succeeded. These are inspection/tool errors, not candidate failures.

Local copied installed files were hashed, then independently matched against read-only SSH `sha256sum` under `/home/ang.gao/miniforge3/envs/robot_sim/lib/python3.12/site-packages/warp`:

| File | SHA256 |
| --- | --- |
| _src/context.py | eb099d908c50effc3cacbf7ecc408be4187ca9a6a6f72e1e85c70676961524ee |
| _src/build.py | 5993e58b44a916fc12f35afde1cededc59ab1db7df7a7eecd75551191d9197df |
| _src/codegen.py | c5538f63618a64af90cf6e0eb629df0d47000a8fe7e57844a0c4b8fd6b41bbf2 |
| config.py | e4b95ce936aa6d57aa11f5046879fe04ebdb8793107c88137c9ba0bd6b0b6cf5 |
| native/cuda_util.cpp | 21d9dfe57b861ada6b98d0fa2ccfb3b8ed0b14972ed840eb04d0ad7e5f70a89f |
| native/warp.cu | 7c7e6e7a02db7b285ca1800d0ec52e28d2cf8a704435d925c70a8c5ccd8f176b |

The first three exactly match the candidate's declared pins. SSH command was `ssh -o BatchMode=yes -o ConnectTimeout=8 tileops-dev` with read-only git and sha256sum commands; it exited 0. No remote interpreter/backend was invoked by this reviewer.

Independently read and reran the coordinator-provided outside-candidate CPU probe:

`C:/Python314/python.exe -I -S -B C:/Users/AG/新建文件夹/c4_assets/warp_setup_source/linecache_probe.py`

Exit 0, CPython 3.14.0. The executed function used held bytes while unseeded inspect read the different existing disk source; mtime=None seeding made inspect retain held text after a disk edit. Mutation of the same lines list escaped identity-only checking but failed the content comparison. No Warp/Torch import occurred in that script. The exact scratch source and empty temporary directory were removed by the probe. This is a Python source-selection reproduction, not Warp compilation proof. The coordinator reports separately running it under remote 3.12.13; that report is not counted as my independent execution.

The coordinator supplied an offline wheel inventory during review. I independently hashed local `c4_assets/c4-warp-compilation-wheel-inventory-v1.json` as `9371a533fb0e5b273609cb60ab2a603ba9f0f6b09fbea8749f8ad36cd6c00ab6`. I did not treat that authoring inventory as reviewed runtime admission or complete compiler closure. Exact selected rows/support-library selection belong to implementation review as the proposal states.

No pytest suite, native import/init/getter, compilation, GPU query, fault campaign, new service, installation, signal or runtime cache operation was executed. No implementation or optical-output claim is established. Real target1/ambient0 restoration, cold compilation at both resolutions, source-to-generated-code correspondence, entry initialization, native partial failures/hang, and real process disposal remain required implementation/native acceptance evidence.

All reviewer commands completed synchronously with terminal results; no yielded exec session remains. The reviewer launched only short-lived local shell/Python/git and read-only SSH utilities, no test/worker/supervisor/compiler/GPU job. No process was signalled or taken over. Process hygiene is bounded to those completed commands and the CPU probe's exact cleanup; I did not enumerate or attest unrelated OS processes. The supplied packet says no live owned test/build job existed, but this review does not convert that statement into an independent global process census. Candidate remained unchanged and clean.

## Scope and remaining gates

No design approval is issued until F1 is clarified and the revised exact candidate receives independent review. The unchanged report may be archived verbatim under the evidence-only protocol exception. Fixing F1 needs only explicit sequencing and preservation of the already accepted parent failure rules.

Even later design approval permits only bounded implementation. Compile acceptance alone cannot establish the one-slot Warp graph/timeline/arena, ten real warmups, R_final/final command recording, sole seal/barrier, CPU-oracle render correctness or complete lifecycle matrices. C4c3/c4, C4d/B3, systematic whole-C4 refactor/review, product PREPARE, benchmark/results, baseline, backend selection and other parent gates remain closed.
