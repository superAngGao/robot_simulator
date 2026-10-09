# Independent bounded release review: intersection precision v4

Initiative: Q54 / P12.4 / C4 intersection precision
Stage: bounded independent code/release review
Author: codex
Version: v4
Date: 2026-10-10
Status: accepted
Related Files: AGENTS.md; collab/README.md; collab/REVIEW_PRINCIPLES.md; runtime-integration design-request/review codex v3; intersection-precision code-review v3; intersection-precision release-followup codex v4; optical_worker/native_source_policy.py; optical_worker/data/native_sources/v16.json
Owner Summary: Approve the exact release-pin/catalog delta. Independent verification gives 232 passed and 4 declared platform/asset skips, both fixed cold preflights pass on the local LF fixture, and the existing deployment verifier accepts the archived artifacts against unchanged candidate build-source bytes. Preserve original a42dbe0 build provenance. Native rendering, optical correctness, performance, integrated C4 acceptance and later gates remain outstanding.

Reviewer session: `/root/intersection_release_review_v4`, a fresh independent Codex session supplied the bounded review packet and repository, without the implementation conversation. I did not author or edit the candidate, spawn agents, install dependencies, access remote hosts, compile native code or launch GPU work.

## Verdict and revision

**Approve**, solely for the three-file delta from `a42dbe0154e7c97dcd23a1de2514636551079d74` to `812ca29030a45d802addf10ca5811a4354ce0eb9`. No substantive blocking or non-blocking finding remains in this bounded review.

- Candidate: `C:/Users/AG/新建文件夹/c4_intersection_precision`.
- Branch: `implementation/c4-intersection-precision`.
- Reviewed tree: `e6894e90698e41c052b6f3805bc595074c4d78a6`.
- Initial and final tracked/untracked status: clean. HEAD remained unchanged. Candidate-only `git diff --check` passed.
- Delta: one existing renderer whole-file source pin, catalog metadata, and the 66-line release-followup note; 68 additions and 2 removals. No numerical source, test, runtime mechanism, recipe, inventory or operative workflow change.

I read AGENTS.md and both shared review documents, the accepted runtime-integration v3 source-admission and mandatory-validation boundaries, its independent review, the preceding numerical review and the author follow-up. The prior report freshly hashes to `3402d0b1491eae97eb7b65a12f4edfb9069707f9aaeb1612bc2c8b57ead15219`; its source-level numerical approval and mandatory native continuation retain their original meanings. This review does not independently re-approve numerical arithmetic or widen that approval.

## Findings and inspected invariants

There are no substantive findings. The high-risk checks were exact byte admission, the narrow ctypes exception, cold bootstrap composition, and honest artifact provenance/reuse.

An independent script compared committed Git blobs, rather than trusting author digests or Windows newline normalization. All 86 literal catalog entries hash correctly and pass the actual unchanged `validate_native_source_bytes`. Parsed old/new catalogs have identical schemas, other metadata and per-lane inventories. Exactly three source digest rows change: `optics/warp_native_render.py`, `optics/_cuda_direct_light_source.py`, and `optical_worker/native_source_policy.py`; the calculated AST policy digest changes accordingly.

The complete policy file is byte-identical to the base after replacing only the old renderer digest with the new one. Forbidden import/name/attribute rules, bootstrap pins, permitted ctypes attributes and the restricted snapshot function pin are unchanged. The numerical files themselves are byte-identical to the approved base. The snapshot function's physical UTF-8 lines still hash to `725c7f315fdba4ad6df78e31a149865fd97e26da3c95df24678dbadba0459199`. No new ctypes use or general native ingress is admitted.

Fresh verified identities:

| Item | SHA-256 |
| --- | --- |
| Renderer whole file | `e19dfc74327111fa70ce7d06aa9a4379a9ce50d8932664a0a2dedef53895886f` |
| CUDA direct-light source | `c67ac69850b5efbefd02396098592c7410ab826c36933866e8d11cecccc948c6` |
| Policy source file | `7a7e337a3085a8a72424b95e6b1d42e796722068717ff28d6597ed221e8c5143` |
| Calculated AST policy | `32fa77913657b9b37916663de24e51fcd5a0135d11dfea36dd35260cccae584f` |
| Release catalog | `43b90be45c9149c8395cd644b74ff421efce8111eaae00b43763e24297effe86` |

The base catalog's two numerical hashes do not match its already-updated numerical source. Independently loading the base policy and checking the current renderer with its correct caller-supplied digest reproduces `ctypes exception source is not the reviewed release`. The candidate policy accepts the same renderer. Existing negative tests still reject a rehashed renderer with an escaped ctypes reference, added native call, writable mutation, changed snapshot body, appended comment, changed newlines, or wrong source path. Thus the repair closes both stale layers without replacing fail-closed validation with a permissive path.

## Artifact evidence and reuse

The local archive freshly hashes to `c90fc21deb24556f1021c32be5f74dbc9550e765e5d5bf6c2aaf29195a948c34`. Its original manifest hashes to `8497e8410d17addb6e6e4739465bc6e636afc775ca181541c039023c2b1fab6d`, and the local derived environment JSON to `c027ece426cb83961efb1887d9d6f563b0aa4b837e65d3045d4e9171ea753d67`. All match the packet.

I inspected the archive member inventory, validated the manifest with production `validate_manifest`, checked its nine source rows against candidate Git bytes, and checked its 18 build log/status/input members. The manifest records three normal successful service exits with status0, distinct invocation identities, and absent cgroups. These are inspected historical records, not my observation of live remote processes.

I then copied only the manifest and referenced artifact/evidence members into external scratch and materialized the nine candidate source files from Git there. Actual production `verify_deployment` passed, including the three binary hashes, exact recipe inputs, logs and terminal footers. No native binary was imported or executed. For this offline call `expected_platform` was taken from the validated manifest; this demonstrates artifact/source consistency, not independent compatibility with a live host ABI. The remote pre-launch verifier must still use its trusted environment as usual.

All nine `SOURCE_FILES`, including CUDA source, recipe definitions and deployment tools, are byte-identical between base and candidate. The existing verifier checks those bytes and recipe identities; it does not require current repository HEAD to equal the recorded build revision. Reuse of the fixed deployment is therefore consistent with the existing contract. Its source revision must remain `a42dbe0154e7c97dcd23a1de2514636551079d74`; neither rewriting it to the release commit nor claiming a new build is justified or needed. Record the worker release revision separately as `812ca29030a45d802addf10ca5811a4354ce0eb9`.

The archive's first-launch failure record explicitly labels itself an author observation, identifies source digest failure during `capture_runtime_custody_inputs` before supervisor creation, and lists only `gpu_mapping.txt` in the failed output. It does not pretend to contain a worker render artifact. I inspected that record but did not independently observe the historical remote launch. Preserve that failed directory when retrying.

## Fresh verification

Environment: local Windows PowerShell, installed `C:/Python314/python.exe`, Python3.14.0, no installs. Commands used `-B`; pytest cache provider was disabled and test/scratch outputs stayed outside the candidate.

```text
C:/Python314/python.exe -B -m pytest -x -q -p no:cacheprovider --basetemp=C:/Users/AG/AppData/Local/Temp/c4_intersection_release_review_v4 tests/unit/optics/test_native_source_policy.py tests/unit/optics/test_native_sources.py tests/unit/optics/test_native_source_custody.py
```

**232 passed, 4 skipped in 172.10s**, exit0. The skips are the two real-Go2 materialization cases without a configured asset path and the two actual-POSIX-checkout byte checks on Windows. They are explicit limitations, not native acceptance passes. Coverage includes both lane captures, actual isolated import custody, source/catalog drift rejection and all 13 custody success/fault cases per lane. This is a new execution, not reuse of the author's results.

The following external drivers were executed against the candidate, all exit0:

```text
C:/Python314/python.exe -B C:/Users/AG/新建文件夹/c4_assets/intersection_release_review_v4_runner.py C:/Users/AG/新建文件夹/c4_intersection_precision
C:/Python314/python.exe -B C:/Users/AG/新建文件夹/c4_assets/intersection_release_review_v4_audit.py C:/Users/AG/新建文件夹/c4_intersection_precision
C:/Python314/python.exe -B C:/Users/AG/新建文件夹/c4_assets/intersection_release_review_v4_cold.py C:/Users/AG/新建文件夹/c4_intersection_precision
```

The cold driver separately invokes the existing `tests/support/runtime_custody_preflight.py` with `-I -S -B`, once per lane. Both return valid custody payloads with the reviewed catalog hash, exit0 and no stderr. Its production bootstrap installs and verifies the actual captured sources; its final assertion rejects imported NumPy, Torch, Warp or multiprocessing. This local test uses the suite's explicit LF fixture, Git revision `e86b4ac657b331ec7362824a9e4c697ce9844f1f`, not a false claim of executing the unnormalized Windows candidate as a POSIX release. The case label does not imply Go2 materialization or rendering.

During review the author supplied `c4-intersection-release-preflight-v4.json`, freshly hashed here as `a81ec5ec3abd2c271780a6cf604c7ba38c5286d9fc144a222ea882cb8b65b476`. I inspected its two Linux Python3.12.13 preflight records: both bind exact candidate revision812ca29 and catalog43b90be4, with the expected lane distinction. It also records successful remote deployment verification with original build revisiona42dbe0. That is author execution evidence; I did not rerun Linux checks or infer live host facts from its JSON fields alone.

Saved independent evidence: `intersection_release_review_v4_suite.log`, `intersection_release_review_v4_processes.json`, `intersection_release_review_v4_audit.json`, `intersection_release_review_v4_cold.json`, and `intersection_release_review_v4_hygiene.json` under `c4_assets`. Read-only inspection used Git status/revision/diff/check, rg, UTF-8 reads and file hashes. An initial wrong-path review-principles read, unsupported PowerShell wildcard rg arguments, a non-ASCII stdin-encoding probe and unquoted Git `HEAD^{tree}` shell parsing failed; subsequent explicit UTF-8/path/script calls obtained the intended data. These were inspection/tool invocation issues, not candidate failures. Truncated combined output was followed by focused reads or saved JSON inspection.

## Limits and scope preservation

This approval covers binding the already-reviewed numerical source into the existing release mechanism and permits the previously authorized native verification continuation. It does not approve native output or create a new artifact policy. No numerical suite or historical integration matrix was repeated because its code/tests are unchanged by this delta.

Native source-frozen renderer execution, Warp JIT, supervised real Go2960/1920 complete geometry/RGB readbacks, lane-specific CPU diagnostics and equal-boundary performance measurements remain outstanding. The preceding review's required six RGB outliers, normal-only pixels1116499/1116503, complete residual retention, unchanged tolerances, profile/shadow coverage and lifecycle observations remain required. CUDA build success and these CPU checks cannot close the original optical discrepancy.

All parent boundaries remain: full C4 integration/fault acceptance, systematic refactor/whole review, native numerical/performance acceptance, result/baseline authority, backend selection, default/shipping activation, host payload/external holds/depth two, R2/S/physics publication, full-image IPC, reusable shutdown/joint lifecycle and parent completion are not opened here. The accepted parent design and prior approvals keep their exact scopes.

## Process hygiene

Suite runner Python9132, parent34440, synchronously waited pytest Python11680 from `2026-10-10T05:53:28.795015+08:00` to `05:56:21.593375+08:00`, exit0; tool session25314 ended exit0. Cold driver Python444, parent42148, waited Warp preflight39928 and CUDA preflight1232, both exit0 between05:55:47.878654 and05:55:52.889677. These are recorded launch/end times, not OS creation-time claims. The audit calls and their short Git children completed; final audit tool session75160 ended exit0. The suite's internal cold children were synchronously waited by its existing helpers; their individual PIDs were not separately sampled.

At `2026-10-10T05:56:30.5676416+08:00`, checker shell15796 queried Win32_Process for recorded runner/test/preflight IDs and Python command lines matching this review's exact script or temporary paths, excluding itself. It found zero residual processes. No process was killed and no unresolved owned hang was observed. No reviewer CLI, SSH, worker, supervisor, remote service or GPU process was launched by this review; remote/GPU process hygiene is not claimed by these local checks.

Only external scripts, logs/JSON, explicit fixture/scratch copies and this report were created. No candidate edit or commit was made. The unchanged complete report may be recorded verbatim under collab through the evidence-only exception; that record cannot change the verdict, approved revision, provenance, remaining validation or closed gates.
