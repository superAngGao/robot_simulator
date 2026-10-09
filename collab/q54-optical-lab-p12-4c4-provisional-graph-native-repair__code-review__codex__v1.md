# Provisional graph native bring-up repair independent review v1

Initiative: Q54 / P12.4 / C4 render integration
Stage: independent code review of native bring-up repair
Author: codex
Version: v1
Date: 2026-10-10
Status: accepted
Related Files: AGENTS.md; collab/README.md; collab/REVIEW_PRINCIPLES.md; runtime-integration design-request codex v3 sections 4, 5, 5.1, 8, 9; provisional-render-graph code-review codex v2 and implementation-note codex v3; render-graph-native-support code-review codex v1
Owner Summary: Approve the bounded import-closure and output-profile repair at the exact revision below. No substantive finding. This does not establish native graph recording/event acceptance or open any later C4 gate.

Reviewer session: /root/provisional_graph_native_repair_v1, fresh independent review; no implementation edits or subagents.
Repository: C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design
Base: f1ff683dd8ef01064b65e2b2f796486b17185622
Reviewed HEAD: 6300531a300d68dd558d155979e0ef86e3e78dfb
Actual tree status: clean at initial, intermediate and final verification; HEAD unchanged.
Verdict: **Approve**, exclusively for the bounded code repair described here.

## Scope and findings

No substantive finding. No blocking repair or optional code follow-up is required for this delta.

Read the collaboration protocol and complete review principles, the requested accepted-design sections, both predecessor reviews and the new implementation note. Inspected all seven changed files and traced actual source admission, captured loader execution, session graph entry, graph preparation, original Warp factory sealing and binding validation. Prior verdicts and author test counts were treated as context, not current verification.

Approval covers the two literal Warp graph-phase dependencies and use of the existing output-profile normalizer. It covers the associated tests and regenerated source pins. It does not approve native recording, successful graph acceptance, rendering, warmup execution authority, ten observed warmups, R_final/final origin, final factories/graph, coverage seal/final barrier, R1/arena integration, whole-C4 validation/refactor/review, sample/IPC/product activation or performance/results/baselines.

## Code and dependency checks

1. At native_bootstrap.py:167 and :2395, the added map admits exactly optics.execution and optics.scene, only for lane warp in preparing_fixed_render_graph and fixed_render_graph_ready. An independent comparison executed the base revision's actual admitted_sources method extracted from its AST against the candidate method for all 48 combinations of two lanes and 24 recognized phases. Only those two Warp phases gained those two names; no path changed or disappeared, and all CUDA admissions remained equal. Earlier CPU, compilation and storage gates remain unchanged.

2. The original call chain is prepare_fixed_render_graph -> _prepare_fixed_render_graph -> factory.seal_camera -> _seal -> _WarpRenderBinding.__post_init__ -> _validate. The last method imports OpticalOutputProfile from optics.execution before validating the binding and before _warp_command_recording opens. execution directly imports optics.scene, output_contract, registry and sensing.optical; scene imports registry, physics.publish and physics.spatial. The latter dependencies already belong to the admitted closure. These two newly admitted files already belonged to both lanes' source catalog; no repository source path was added to the catalog.

3. In a separate fresh Python process, the actual candidate _CapturedSourceFinder and _CapturedSourceLoader executed optics.execution and its transitive repository imports from held HEAD Git bytes. All 32 captured modules completed, including execution, scene, sensing and physics dependencies. For this diagnostic, topology/liveness/controller methods were explicit doubles, and a routing finder delegated non-repository imports to the ordinary interpreter. This independently checks executable repository dependency closure, not the frozen external environment or native source-custody acceptance. It complements the committed twelve finder-resolution tests, which check held source identity and rejection but do not execute module bodies.

4. native_setup.py:276-279 normalizes the already validated profile string exactly once on the Warp branch and passes the resulting enum to all ten seal_camera calls at :316. SourceSession still accepts only exact strings in direct_light_full/rgb_preview/render_only. The original _WarpRenderBinding._validate at warp_native_render.py:2787 requires the exact OpticalOutputProfile enum and one of those same values. normalize_output_profile returns that existing enum without introducing a duplicate definition. The captured-loader diagnostic independently checked exact enum type and unchanged value for all three profiles. Frozen render parameters and graph descriptive metadata retain their original string, while CUDA behavior is unchanged.

5. Normalization and imports occur inside the existing armed setup segment and device/stream scope, before factory construction. Captured loaders still retain each module before executing its body. Any exception still reaches the original sticky session/guard failure path. There is no new native operation, retry, synchronization, event, rendering algorithm, executable variant, resource registry or warmup credit.

6. Catalog comparison found the same 84 paths and identical lane lists. Only native_bootstrap.py, native_setup.py and native_source_policy.py source digests changed. Every catalog digest matches the corresponding HEAD Git blob. Canonical LF catalog SHA256 is 5fe8701d31b6953444e9c666620a5217ead719c744ad566a08d9d0b22c358504. The Windows checkout's CRLF file hash differs, as expected; release fixtures use LF. The policy digest changes because bootstrap pins change. After excluding the two bootstrap pin assignments, old/new native_source_policy ASTs are identical: no policy rule or exception was widened. The policy version is unchanged.

## Independently inspected failed-native evidence

Read the local archive without extracting or executing its scripts. Its SHA256 is 243685222f6487ace68a7786e0a6f6dffc1a9d7458bb68c88269f11c28a797af, matching the adjacent JSON and review packet. Read the raw request, failure, worker, supervisor, identity, parent and scripts; ran the production validate_custody_artifact on the archived custody_artifact.json successfully.

The request names clean candidate 50143c6a92b2c1d21dd52dae512b43a3cdd3cef9 and go2_static_960x640_v1. The first Warp case failed with SourceBootstrapError, message "repository import outside CPU core or reload", in fixed_render_graph. The graph trace contains factory creation/public module loading, provisional BVH-view retention and upload-context retention, then guard failure; there is no command-recording scope. Supervisor replay shows 13 arms, 12 normal completions, worker return code 74, no expired deadline and no kill request. Worker evidence shows failed guard sequence 12, full quarantine, release not authorized and zero emergency synchronization attempts.

The batch script runs eight cases sequentially through run_case and stops on an exception; the archive contains only the first case and no success result. Thus neither later cases nor any native graph success is inferred. The generic exception does not identify an import name. Locating the rejected execution import is a static call-flow diagnosis consistent with the recorded boundary; the next profile-type mismatch is independently established from source, not an observed second native failure.

Archived process identity is parent 3328284 -> supervisor 3328515 -> worker 3328520. The adjacent evidence and archived creation script record all three absent before archive creation; the raw supervisor records observed worker exit. This review did not independently query live remote /proc or GPU state, so archived PID absence is reported as archived evidence rather than a fresh remote observation.

## Executed verification

Environment: Windows PowerShell, C:/Python314/python.exe, -B, pytest cache provider disabled. Commands ran from the frozen repository. No dependency installation, SSH operation or GPU campaign was performed.

1. `git status --short`, `git rev-parse HEAD`, `git log --oneline -8`, `git diff --stat f1ff683 HEAD`, full seven-file diff, targeted source/document reads and `git diff --check f1ff683dd8ef01064b65e2b2f796486b17185622 HEAD`: clean candidate, exact revision, no whitespace errors. Initial broad output truncation was followed by targeted reads of the relevant design and code sections.
2. `C:/Python314/python.exe -B -m pytest -q -rs -p no:cacheprovider tests/unit/optics/test_native_render_preparation.py tests/unit/optics/test_prepared_fixed_render_inputs.py tests/unit/optics/test_native_render_graph_support.py`: **131 passed in 27.81s**, exit 0, tool session 18031.
3. `C:/Python314/python.exe -B -m pytest -q -rs -p no:cacheprovider tests/unit/optics/test_warp_render_command_contract.py tests/unit/optics/test_warp_render_buffer_contract.py`: **76 passed in 0.58s**, exit 0.
4. `C:/Python314/python.exe -B -m pytest -q -rs -p no:cacheprovider tests/unit/optics/test_native_source_policy.py tests/unit/optics/test_native_sources.py tests/unit/optics/test_native_bootstrap.py`: **281 passed, 13 skipped in 173.16s**, exit 0, tool session 18054. This is a fresh execution at the reviewed candidate.
5. `Get-FileHash -Algorithm SHA256 ../c4_assets/c4-render-graph-failure-evidence-v1.tar.gz`, `tar -tzf` and bounded `tar -xOf` reads: verified archive identity and contents. Inline `C:/Python314/python.exe -B -` diagnostics loaded the archive using tarfile/json, called validate_custody_artifact, and inspected bounded trace/summary fields. Replay passed.
6. Additional inline `C:/Python314/python.exe -B -` diagnostics: compared catalog structures and all HEAD source digests using git show and hashlib; compared policy AST excluding the two bootstrap pin assignments; executed the base admission method extracted via ast and compared all 48 phase/lane results; executed the real captured dependency import and checked three normalized profile values as described above. All assertions passed. These were process-local diagnostics, not modifications to candidate files.

The 13 skips are two unavailable real-Go2 asset cases, two POSIX release-byte cases, two actual POSIX entry cases, and seven generated-module cases requiring pinned CPython 3.12 POSIX. Skips are not passes. Source fixtures create temporary LF release checkouts and isolated child interpreters outside the candidate. Their temporary commit identities are not candidate revisions.

## Limits and remaining acceptance

The integrated Warp fixture still substitutes factory construction/sealing, with the new assertion checking the original enum requirement. The separate command/buffer contracts and captured import diagnostic improve local coverage but do not establish real packed recording, actual ModuleExec reuse, native event initialization/waits, dual-GPU scope/restoration, or native failure/partial-retention/hang behavior. The source/bootstrap tests also retain their explicit host-dependent skips.

A new source-frozen native run at the repaired revision is required before native graph acceptance can be claimed. All required normal/fault native graph cases, optical rendering, observed warmups and subsequent C4 work remain incomplete. This code approval does not waive those checkpoints. The unrelated storage-only batch 81866 at 3d6692b and the coordinator's separate next-work worktree were not queried or modified.

## Process hygiene and artifacts

Observed source/bootstrap pytest PID 9604, parent 38748, creation 2026-10-10 03:02:59 +08:00, with the exact command above. Sampled its isolated source-test child PIDs 47916, 30456 and 44808 with matching parentage/command context. Its tool handle completed exit 0. The focused test process, short Warp-contract process and short inline diagnostics completed before their OS PIDs were sampled; their tool results were terminal. This is a bounded launch ledger, not exhaustive per-child attestation.

Final scoped Win32_Process query at 2026-10-10T03:06:04.7805602+08:00 found no PID 9604, no child with parent 9604, and none of parent 38748 or sampled children 47916/30456/44808. All reviewer test handles were terminal. The bootstrap helper uses kill plus communicate for its own sacrificial child on timeout and checks exit; this review performed no manual process signal. Unrelated pytest PID 30512, parent 38908, was observed during an intermediate query and left untouched. No claim is made that the entire shared host has no Python processes.

No reviewer native worker/supervisor/SSH/GPU process was launched; GPU-process polling was unnecessary for this CPU-only round. No remote cleanup, shared-GPU reset or unrelated process termination occurred. No observed reviewer-owned residual remains.

The only reviewer-written artifact is this outside-tree report at C:/Users/AG/新建文件夹/c4_assets/provisional_graph_native_repair_review_v1.md. Tracked candidate files were not edited. The reviewer session remains active only to return this report to the coordinator.
