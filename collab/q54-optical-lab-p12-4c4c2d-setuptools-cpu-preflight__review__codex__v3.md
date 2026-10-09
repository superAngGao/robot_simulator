# Independent private-directory repair and CPU execution preflight

Initiative: Q54 / P12.4 / C4c2 / CUDA G1
Stage: independent bounded code/execution preflight, dependency-only
Author: codex, independent reviewer /root/setuptools_cpu_preflight_v3
Version: v3
Date: 2026-10-09 (Asia/Shanghai)
Status: accepted — bounded execution preflight only
Related Files: setuptools-cpu-preflight review-request v1/v2/v3, review v1/v2, test-evidence v1; setuptools-admission design-request/design-review v2, implementation-note v3/v4/v5, test-evidence v2/v3; compilation-admission design-request/design-review v3
Owner Summary: Approve one new exact dependency-only execution at0177a86, stopping at RESTORED. The actual Linux shared filesystem independently reproduces inherited2700, and the revised helper creates exact0700 without changing shared-parent permissions or reusing old runs. Nine runner/profile tests and all1,791 production file checks pass. No actual dependency initialization, compilation or higher acceptance is claimed.

## Verdict and exact execution scope

**Verdict: Approve.** No substantive blocking or non-blocking finding remains for this bounded directory-creation repair and readiness for one new dependency-only run. This approves the exact frozen candidate `0177a869aec966da2e112d5f92120892aa2fb1fa`, not the whole implementation and not a successful dependency result.

The single approved command, with the target still clean at that exact revision and the following directory still absent, is:

```text
/home/ang.gao/miniforge3/envs/robot_sim/bin/python -I -S -B /data2/shared/ang.gao/robot_sim_cuda_compilation_wip/tests/support/setuptools_dependency_process.py parent /data2/shared/ang.gao/robot_sim_cuda_compilation_wip /data2/shared/ang.gao/setuptools-cpu-0177a869aec966da2e112d5f92120892aa2fb1fa-v1
```

The command uses the real source session, source verification, custody binding, external admission, ordinary production dependency loaders, initialization owner, guardian and guard transport. Its input/device/toolchain prerequisites remain the explicit descriptive outer stand-ins in the inspected test runner. It may observe RESTORED, the same-object alias, real resource reads and version83.0.0. It must not emit or claim COMPLETE, cpp_extension completion, validated compilation evidence, native scope, readiness or optical results. A failed run consumes this one execution authorization and must be retained as failed evidence; this report does not approve an automatic retry or modification during execution.

**Neither driver `parent()` nor `child()` was executed in this review.** No installed setuptools, _distutils_hack, distutils or backend package was imported. The runner module was imported only for inert constants and permitted helper calls. The proposed execution result remains unknown.

Complete dependency positive/failure integration, the negative dependency matrix, complete v2 compilation evidence/replay, renewed exact compiler/native preflight, actual three builds, process/native failure and residual proofs, whole-checkpoint approval, later C4 work and the mandatory systematic refactor remain outstanding. Compilation, native/device execution, timing/performance/backend choice, product/result/baseline authority and full C4 acceptance remain closed.

## Independence, revision and tree

This fresh review session received the task packet without the implementation conversation. I did not author, modify or fix the candidate and did not delegate this review. I read the project collaboration/review principles, all three preflight requests, prior reports and original R1/R2 repros, accepted design requirements/reviews, and relevant implementation/evidence records. Prior conclusions and author test counts were treated as claims; the current delta and affected mechanics were independently checked.

- Local repository: `C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design`; branch `implementation/c4-supervised-waits`.
- Original implementation base, confirmed by merge-base: `52fb36a8960596ddb2ac3decd8d129188e4b1eb4`.
- Previously approved code: `311f16f1cfec641957b1a688454bb834006ccbce`.
- Evidence-only archive: `a9dfc3f7f551eafece6f8313d1e7d15d187b7b30`; inspected commit contains only the unchanged v2 report.
- Reviewed HEAD: `0177a869aec966da2e112d5f92120892aa2fb1fa`.
- Reviewed tree: `4bd1a71f067805eaa18380a7dfb813f8f45cbcd9`.
- Primary target: `tileops-dev`, `/data2/shared/ang.gao/robot_sim_cuda_compilation_wip`, detached at the same HEAD/tree. No fallback host was used.
- Target Python: `/home/ang.gao/miniforge3/envs/robot_sim/bin/python3.12`, CPython3.12.13, Anaconda, GCC14.3.0. The command's `bin/python` resolves to this same executable.
- Initial and final local and target porcelain status were empty. HEAD/tree remained unchanged. `git diff --check 311f16f1cfec641957b1a688454bb834006ccbce HEAD` passed.

The base-to-candidate diff has six paths: PROGRESS, v3 request, unchanged archived v2 report, failed-run evidence v1, runner and runner tests. The only operative code change is `_create_run_directory` plus its parent call; the sole new regression exercises real Linux setgid inheritance. Production bootstrap, source/policy/catalog, environment/profile, initialization, parsers and compiler code have no delta from the previously reviewed code. The progress/evidence text preserves the failed-run outcome and later gates.

## Findings and original failure closure

**No substantive findings.** High-risk areas checked were fresh-path exclusivity, inherited permissions, descriptor ownership and closure, symlink/path identity, failure propagation, child-entry ordering, exact file pins, continuing source-audit allowance, owned CPU group cleanup, outer stand-in boundaries and the RESTORED stop.

The raw failure archive independently hashes to `f3ac562ac4b9848346813494038a699439a734242f583cf1181d7b4b44c23b9a`. Its request names311f16f and the exact target root; parent evidence records child597340, returncode1, timed_out=false and waited=true. stdout is empty. stderr identifies the child's exact0700 assertion before identity writing, source bootstrap and dependency initialization. The archive records the run directory as2700 and contains only request.json, parent.json, stdout.log and stderr.log. This is a runner/filesystem startup failure, not a dependency or toolchain failure.

Current target facts independently match: shared parent mode2775, uid1006, gid985; old run mode2700. The target archive has the same digest, and every recorded original run file's bytes and mode still match it. No chmod or retry of that run occurred. PID597340 is absent.

At `tests/support/setuptools_dependency_process.py:32`, the helper rejects nonabsolute/noncanonical paths before creation. `mkdir(mode=0700)` retains exclusive creation: an existing file or directory raises, without alteration. It opens the newly created directory using O_DIRECTORY|O_NOFOLLOW, verifies directory type and current UID through fstat, and changes only that held descriptor to0700. The following lstat must match its device/inode and exact0700 mode. The finally closes the descriptor on both success and failure. An error leaves a failed fresh artifact; it does not proceed to the driver or make an existing run reusable.

Parent integration at line217 calls the helper before Git identity checks, Peer creation, request publication or child launch. The child still checks canonical directory, owner and exact0700 at entry. No check was weakened, shared-parent mode altered, general filesystem permission added, or new package import introduced. The use of fchmod makes the correction specific to the opened owned directory. This is the fixed trusted deployment boundary, not an assertion of protection against arbitrary hostile concurrent filesystem writers.

The candidate regression first proves actual mkdir0700 inherits2700 under a real setgid parent, then invokes the actual helper and requires0700, unchanged parent2700, and refusal to reuse an existing directory while preserving its marker. The independent probe additionally used a temporary sibling under the actual `/data2/shared/ang.gao` filesystem. It observed inherited2700, successful0700 correction, current owner, unchanged shared-parent identity/mode/UID/GID, existing-file/directory refusal, relative/noncanonical/symlink-ancestor refusal, and preservation of the existing marker. An injected fchmod exception propagated, left the failed directory2700 and did not leak an FD; the normal path also had stable FD count. All reviewer filesystem scratch was removed afterward.

Earlier R1/R2 remain closed for this bounded preflight on current evidence: Git profile bytes equal the target file and runner pin; production schema/file verification passes all1,791 files. The runner suite exercises actual cold source install/verification, custody binding and repeated production audits under the existing narrow Git hook, plus wrong command/revision/cwd/env and forbidden backend/process/library denials. Both Linux group tests run: direct leader alive and leader already exited with a descendant remaining. The helper stops the owned group and waits its direct child. Those tests establish descendant cessation, allowing OS-reaper zombies; they do not grant compiler subreaper/wait ownership.

I read the full fixed runner flow. Its dependency path and assertions still stop at RESTORED, require actual alias/version/resource facts, preserve source/admission/retention audits and ordinary records, reject backend presence and a completed initialization action, then use the existing HOST_OWNER terminal path with no native synchronization callback. No `_Compilation` or fake cpp_extension is introduced. Parent timeout and final cleanup still use the private CPU group helper. The new directory function runs before any of those resources exist.

## Executed verification

Read-only inspection used UTF-8 Get-Content, rg, Git status/rev-parse/merge-base/show/diff, Get-FileHash, and local isolated stdlib tarfile reading without extraction. Local Python was `C:/Python314/python.exe -I -S -B`; target execution was isolated Python3.12.13 with `-I -S -B`.

The new independent launcher was sent from the workspace on stdin:

```powershell
Get-Content -Raw -Encoding UTF8 c4_assets/setuptools_cpu_preflight_v3_independent_tests.py | ssh tileops-dev /home/ang.gao/miniforge3/envs/robot_sim/bin/python3.12 -I -S -B - | Tee-Object -FilePath c4_assets/setuptools_cpu_preflight_v3_independent_tests.log
Get-Content -Raw -Encoding UTF8 c4_assets/setuptools_cpu_preflight_v3_final_facts.py | ssh tileops-dev /home/ang.gao/miniforge3/envs/robot_sim/bin/python3.12 -I -S -B - | Tee-Object -FilePath c4_assets/setuptools_cpu_preflight_v3_final_facts.log
```

The first script verifies revision/clean status and installs a reviewer import-denial hook for setuptools/_distutils_hack/distutils and the backend roots. It explicitly inserts only repository and robot_sim site-packages paths to supply pytest, with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`. No site startup or .pth processing runs. Pytest arguments are `-q -p no:cacheprovider tests/unit/optics/test_setuptools_dependency_runner.py tests/unit/optics/test_compilation_environment_profile.py`.

| Independent check | Observed result |
| --- | --- |
| Profile Git blob/target bytes/runner pin and catalog digest | Passed |
| Production RuntimeEnvironmentContract schema and verify_files | Passed,1,791 files |
| Actual shared-filesystem helper and failure/path/FD checks | Passed |
| Runner and profile tests, including cold binding and both Linux groups | 9 passed in5.54s; pytest exit0 |
| Reserved modules before/after | Absent; final list[] |
| Final local/target revision/tree/status | Same frozen clean candidate |
| Original target archive and recorded run bytes/modes | Unchanged |
| Reviewer634313, author615971, failed child597340 | All absent in /proc at final check |
| New approved execution directory | Absent |
| Reviewer temporary sibling directories | Removed |

The author log independently matches SHA256 `cf63bf4b1c4331ac7fa1c6b78eced21e4b5ff63b88eddb6f1bfe05fffdce8401` and says9 passed in5.69s. That historical output is not counted as this review's test execution. The earlier v2 suite results likewise remain prior evidence; they were not rebranded as fresh tests. Nine current tests plus new filesystem/provenance checks are proportional to the two-file operative delta.

Review tooling issues only: the initial root REVIEW_PRINCIPLES.md lookup failed and was corrected to collab/REVIEW_PRINCIPLES.md; initial default encoding was corrected to UTF-8. Unquoted PowerShell `HEAD^{tree}` was misparsed and was rerun quoted. A remote stat format lost shell quoting; the successful isolated Python stat checks supplied the actual UID/GID/modes instead. Large reads were followed by bounded section reads. These were inspection-command issues; no candidate repair resulted. Both new independent scripts completed successfully on their first execution.

## Artifact identities, hygiene and limits

All new reviewer artifacts are outside the tracked tree under `C:/Users/AG/新建文件夹/c4_assets`:

| Artifact | SHA256 |
| --- | --- |
| setuptools_cpu_preflight_v3_independent_tests.py | 1c6ea39918c1a89ca3defea840ad880e858e73512daf6a4610c7622fbcb09fc9 |
| setuptools_cpu_preflight_v3_independent_tests.log | 39115428ce44388d0532cef6c9f8d33138c78da1f9a49e796a654a50b09afe11 |
| setuptools_cpu_preflight_v3_final_facts.py | 105ca61bcbf07d02e9c73bc4f94de3173abc8962fd9713b9d7c9b61f526eaa69 |
| setuptools_cpu_preflight_v3_final_facts.log | ae536f22636537a821a3af5c66b43a4eb2d159f3ac274b85b0f4aeb2d8ae94e0 |
| Current target runner bytes | 5c6db68e82c99b020dd5be6b3b827d667132046dab7c2fa7ca774fe7cd26a8aa |
| Profile Git blob and target file | 0cb6288345fc3a2644e61bb01dc6e5897ee58d52eb71050c31b2e76f8ea60753 |
| Source catalog v13 | 17387af6b8188c40707da7e1835f639e87d1464ffa664b7cdcdb301eedd9d399 |
| Accepted setuptools design v2, local bytes | e8343ed7c9217a47ad45da5f055801cf43537a67f8059240453ed14078e27db8 |
| Accepted compilation design v3, local bytes | 028b52c466ef46433235d4ec2495cd47d00200822cc64dfaea95614fd5bf26a2 |
| Preserved preflight review v2, local bytes | 9791a877970573ca6bbbbc671cb452b4db8959e317e52ced4469920212975c7c |

No candidate write, checkout, reset, dependency installation, shared-parent chmod, old-run mutation, compiler/GPU process, subreaper, native observer or persistent service was performed. Temporary source-test repositories and stdlib process-group children are bounded test side effects. Only owned test groups were signaled; the tests verify direct-child wait and descendant cessation. This is scoped process hygiene, not exhaustive host/GPU attestation.

Actual installed initialization and the complete parent/child positive or timeout path intentionally remain unexecuted. Static inspection, helper probes, file verification and source-only tests cannot establish every future dependency import/resource call succeeds. The runner audit hook is a bounded restriction for a trusted Linux deployment, not a hostile-code sandbox or general compiler-process owner. No Windows runner compatibility or concurrent hostile filesystem guarantee is granted. These unchanged limits do not block the one explicitly bounded CPU execution; its result must be examined as new evidence.

Approval binds the exact revision and single-run scope above. Any operative code, test, design or workflow change requires review of its delta. This complete report may be archived verbatim as evidence without changing its verdict, scope or later gates. Its whole-file SHA256 is recorded in the adjacent sidecar and handoff, avoiding a self-referential digest.
