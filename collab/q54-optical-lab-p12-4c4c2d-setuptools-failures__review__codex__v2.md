# Independent setuptools failure-matrix re-preflight

Initiative: Q54 / P12.4 / C4c2 / CUDA G1
Stage: fresh independent bounded code and execution re-preflight
Author: codex, independent reviewer /root/setuptools_failures_preflight_v2
Version: v2
Date: 2026-10-09 (Asia/Shanghai)
Status: accepted — Approve
Related Files: setuptools-failures review-request v1/v2 and review v1; accepted setuptools-admission design v2; compilation-admission design v3
Owner Summary: Approve the repaired CPU driver at the exact frozen revision for one positive control and the 32 fixed isolated negatives. R1 and R2 are closed by source inspection and independent target CPU regressions. This is permission for the finite dependency campaign, not acceptance of its unexecuted outcomes or any compiler/native/device scope.

## Verdict and identity

**Approve.** No unresolved substantive code finding or required preflight evidence gap remains in this bounded scope. Both previous blockers are repaired without changing production lifecycle behavior. Actual installed dependency execution has not occurred during this review and is not represented by these fixtures.

I am a fresh independent reviewer, did not author or edit the candidate, and did not delegate this review. I read AGENTS.md, collab/README.md and collab/REVIEW_PRINCIPLES.md in full; the v1/v2 requests, original review and original probe; the accepted dependency contract and relevant compilation requirements; the candidate delta and full affected driver; and the production alias, source failure, loader restoration, retention and guard paths. Author summaries were checked against code and independently obtained results.

| Identity | Value |
| --- | --- |
| Local repository | `C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design` |
| Local branch | `implementation/c4-supervised-waits` |
| Original bounded base | `7794c321478d3994ff45bffeb5802ba10d614107` |
| Original reviewed candidate | `44a57da4d80c1372f12ccc2478be981e9b5afee8` |
| Unchanged v1 report archive | `82adad5668a77b800d7f08adfe3d32f3d730a0f3` |
| Approved candidate, local and target | `427c90ae6b69360b5cd16178e027677bf3ae09a0` |
| Approved tree | `47902dc48df77845b1067c25abed8b47363b0385` |
| Target | primary `tileops-dev`, `/data2/shared/ang.gao/robot_sim_cuda_compilation_wip`, detached HEAD |
| Initial/final local and target status | Clean, including untracked files; frozen HEAD unchanged |
| Driver working bytes, local and target SHA256 | `b0f5a9e1e321c9880336e153119a70381697cd2440c8a0a1e8a29b20558f9740` |

The delta from the report archive consists only of `tests/support/setuptools_dependency_process.py`, `tests/unit/optics/test_setuptools_dependency_runner.py`, `tests/unit/optics/test_setuptools_admission.py` and review-request v2. Production code, profile/catalog pins, installed dependency files and accepted contracts are unchanged. `git diff --check 44a57da4d80c1372f12ccc2478be981e9b5afee8 HEAD` passed.

Verified predecessor SHA256 values:

- Unchanged v1 report: `9d7892c6bce0cf9ac83dbce1482c5f125135945facd4dceff05a9616a2f02a90`.
- Unchanged original probe: `4b165676acd12d5b484a40d40629ca3fabe0a75d69176925b0ca02875ad547a6`.
- Accepted setuptools design v2: `e8343ed7c9217a47ad45da5f055801cf43537a67f8059240453ed14078e27db8`.
- Accepted compilation design v3: `028b52c466ef46433235d4ec2495cd47d00200822cc64dfaea95614fd5bf26a2`.

## R1 disposition — closed

Original defect: the negative observer required healthy cancellation, RELEASED custody and completed release even though a source failure inside guard6 poisons the guard and requires quarantine.

`_DependencyFault.observe_failure` now calls `_assert_negative_terminal` and separately verifies that the actual initialization owner and compile root remain in the real guardian. The validator requires quarantined_process_exit, TRANSFERRED_TO_QUARANTINE, not_authorized release and full_quarantine custody. It requires a failed setup guard at sequence6 after exactly six normal completions, zero setup operations/device leaves, zero synchronization and terminal-arm attempts, and no dirty or unknown device scope. It cannot accept the healthy positive's released terminal. No production failure or release path was relaxed.

Independent adaptation of the original controller/Peer probe ran on target Python3.12.13 and observed:

| Fact | Healthy control | Failed guard6 |
| --- | --- | --- |
| disposition | cancelled_before_submission_exit | quarantined_process_exit |
| guardian_state | RELEASED | TRANSFERRED_TO_QUARANTINE |
| release_status | completed | not_authorized |
| custody | retained | full_quarantine |
| guarded_segment_failed | false | true |
| normal_complete_count | 7 | 6 |
| terminal_arm_attempt_count | 1 | 0 |
| setup_operation_count / entered_leaf_count | 0 / 0 | 0 / 0 |
| emergency_sync_attempt_count | 0 | 0 |

The real validator accepted the failed control and explicitly rejected the healthy control. The failed control retained the actual adopted object. This exercises the production controller and real Peer transport with explicit outer CPU fixtures and the controller operations used by `_SourceSession.fail`; it does not claim installed source execution. Static inspection also verified the unchanged `_SourceSession.fail` / `_ExternalAdmission.fail` recursion commits poison and retains the graph before the one terminal operation. The candidate's corresponding new regression passed independently.

The v2 request correctly supersedes v1's erroneous negative host-only cancellation wording. Child hard exit, parent wait/owned-group cleanup and post-run PID absence remain required for actual quarantine evidence.

## R2 disposition — closed

Original defect: partial_create/partial_exec raised only after the protected alias operation had returned. Shim unwind then recorded a missing-publication error before the injected exception.

Both selectors now wrap `owner.check_source_metadata`, retain the real check, and raise inside the real alias operation's try/except after its delegate acts:

- partial_create raises after the intermediate metadata check, after the returned object is retained and before `alias_created` is emitted.
- partial_exec counts only non-intermediate checks in ALIAS_CREATED and raises after the second check. The first is before the real exec delegate and the second is after it, before ALIAS_EXECUTED and `alias_executed`.

I checked the full call sequence, including `_check_binding`: the pinned delegate exec is a no-op, and no extra non-intermediate metadata check is introduced between these two calls by this exact path. The validator now also requires the inappropriate success action absent; partial_create still requires a preceding alias_spec and partial_exec a preceding alias_created. A missed injection remains a harness failure.

An external v2 probe was derived from the unchanged original repro. It uses the same real owner, `_AliasLoader`, `module_from_spec`, descriptive source/delegate records and shim removal sequence, with repaired-behavior assertions and delegate-call accounting. It adds the explicit outer fixture's no-op `retain_import_boundary` because successful repair now reaches the production FAILED branch of `leave_shim`; this method was absent from the old fixture, whose bad-path expectation never reached that branch. This no-op supplies no source, module, terminal or success evidence.

Target results were:

```
repaired_first_error partial_create injected:partial_create injected_identity True
repaired_first_error partial_exec injected:partial_exec injected_identity True
```

Both first errors and propagated errors are the exact injected object. Create executed once; exec executed once only for partial_exec. The actual returned source remains in the observation/compile-root retention graph, state stays FAILED and every installed hook has a restoration attempt. The candidate's adapted regressions additionally check restored hook identity, exactly one failed action, no later successful initialization action, and no inappropriate alias-created/executed success. All passed independently.

I re-read the installed `_distutils_hack` source as text: its create method changes the retained source's name and returns it, its exec method is a no-op, and shim.__exit__ calls `_remove_shim()` without forwarding the active exception. The unchanged ordinary source loader owns `finish_file` restoration from its finally. The repair therefore addresses the original exception-order counterexample instead of merely changing the expected error. The original probe/report files were not edited; their old-bad-behavior assertions were not relabeled as fixed tests.

## Affected invariants and additional inspection

No additional substantive candidate finding was established. I reviewed all 32 fixed selector groups and the parent/child path, not only the two repaired branches:

- Reserved preloads reject at begin; the owner is recovered from session custody even when begin raises. Path mutations, repeated/early/foreign-thread shim calls and changed shim/code/closure bindings exercise actual checks.
- Alias spec/closure/module/loader and missing publication faults still target the existing owner. Unlisted child is attempted only after RESTORED. Resource and metadata faults use actual fixed callbacks, including caught denial and changed resource identity.
- Before/after assignment faults still count real restoration assignments exactly once. Restoration-before must preserve the failed binding and cannot claim successful restoration. Late publication retains the actual source after poison.
- Negative validation retains first-error identity for injected errors, verifies source/session/admission poison and subsequent live denial, rejects absent boundaries, requires one failed action and no later success/COMPLETE, and verifies retained owner/root custody.
- The positive branch is unchanged: same source/alias object, version83.0.0, real resource/version observations, closed handles, original hooks/methods, singleton finder/path, source/admission/retention audits, RESTORED and no COMPLETE.
- Parent/child selectors and child request fields remain closed. There is one cold child per invocation. Foreign-thread injection has a bounded10s joined thread. Parent wait150s, owned-group SIGKILL and leader wait10s, and Peer join2s are unchanged.
- The child audit hook still admits only the two fixed read-only Git commands and denies backend imports, arbitrary subprocesses, fork/spawn and ctypes library loading. This is a trusted fixed harness, not a hostile-code sandbox or compiler process-tree proof.

The orchestration script was also inspected outside the frozen candidate. Its first presented version skipped child-PID checks if parent.json was missing. I notified the coordinator; the author repaired it without changing the candidate. Final local/target bytes match SHA256 `988e1372bf407d4b0fe2e387351f26455caa266fdca9036d08646af6b9570705`. It now reads both identity.json and parent.json, checks every recorded child PID, records unknown identity explicitly and stops before another case if identity is unknown or any recorded parent/child remains. It checks ppid against the waited parent and agreement between both child identities. This issue is closed in the final inspected script.

The script checks the exact clean revision, creates the new campaign through the driver's0700 creation helper, enumerates the fixed33 cases, runs each exact parent command once serially, preserves parent stdout/stderr and campaign.json, and has no additional kill policy. It was inspected but not executed during preflight. Any stopped/incomplete campaign must remain archived; this approval does not authorize an automatic retry or reuse of its directory.

## Independent verification

Target interpreter: `/home/ang.gao/miniforge3/envs/robot_sim/bin/python -I -S -B`, Python3.12.13, Linux. Repository and robot_sim site-packages were added explicitly for pytest; `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`, `-p no:cacheprovider`, `--noconftest`. Reserved roots were absent before and after: torch, warp, numpy, scipy, trimesh, setuptools, _distutils_hack, distutils and backports. Temporary fixture ModuleType publications are undone by monkeypatch.

Exact independent execution:

```
ssh -o BatchMode=yes -o ConnectTimeout=15 tileops-dev '/home/ang.gao/miniforge3/envs/robot_sim/bin/python -I -S -B /data2/shared/ang.gao/setuptools-failures-427c90ae6b69360b5cd16178e027677bf3ae09a0-review-runner-v2.py /data2/shared/ang.gao/robot_sim_cuda_compilation_wip /home/ang.gao/miniforge3/envs/robot_sim/lib/python3.12/site-packages /data2/shared/ang.gao/setuptools-failures-427c90ae6b69360b5cd16178e027677bf3ae09a0-review-probe-v2.py'
```

The runner executed `tests/unit/optics/test_setuptools_dependency_runner.py`, `tests/unit/optics/test_setuptools_admission.py` and the external adapted probe. Result: **37 passed in6.61s**, pytest exit0. This includes the actual Linux setgid-directory and owned-process-group cleanup tests and both independent control/repair pairs. Cleanup tests permit OS-reaper zombies for grandchildren and do not prove compiler subreaper ownership. Reviewer runner PID857306 was separately checked absent afterward.

The runner verified all **1,791** profile file rows by size/SHA256 without executing those files, and checked clean HEAD before/after. Released profile SHA256 is `0cb6288345fc3a2644e61bb01dc6e5897ee58d52eb71050c31b2e76f8ea60753`; catalog SHA256 is `17387af6b8188c40707da7e1835f639e87d1464ffa664b7cdcdb301eedd9d399`.

The author target log was separately read and hashed:36 passed in6.80s, SHA256 `b697c99fe06e4bfad00c4b6962fbd71e2d7fa52afe6f12ee67079191d20264ef`; PID849585 was absent. This is corroborating author evidence. The reported local29-pass/4-skip run was not independently rerun; the independent target run exercised the Linux cases directly. Historical v1/0177a86 actual positive evidence was not treated as a new execution or as evidence for the new negatives.

Other verification: revision/tree/branch/status reads; exact delta and diff-check; SHA256 comparisons; direct production/source reads; profile pin and all-file checks. One initial read guessed REVIEW_PRINCIPLES.md at repository root and was corrected to its actual collab path. One extraneous read-only rg of /dev/null failed on Windows. Neither affected candidate or evidence. No candidate mutation or environment/toolchain failure occurred.

External review artifacts under `C:/Users/AG/新建文件夹/c4_assets`:

| File | SHA256 |
| --- | --- |
| setuptools_failures_preflight_probe_v2.py | `32f4477c6ac93f1d671159866624597319dd3dfae78d8c3ec538d327365395ac` |
| setuptools_failures_preflight_runner_v2.py | `9a154bc058aada3c7b6b65cdb68da434dca3789d1e97d783950728b462fc052a` |
| setuptools_failures_preflight_target_v2.log | `9b418bdd4b086be738035af36f2b1396be665f8c3eca9f822846ce384d4f412f` |
| run_setuptools_failure_matrix_427c90a.py | `988e1372bf407d4b0fe2e387351f26455caa266fdca9036d08646af6b9570705` |

Target scratch runner/probe bytes match their local hashes. The log hash binds the local PowerShell Tee-Object transcript. This report's final digest is supplied separately to avoid self-reference.

## Exact approved execution and remaining gates

At candidate `427c90ae6b69360b5cd16178e027677bf3ae09a0` only, approve one serial invocation per following case:

```
positive
preloaded_setuptools preloaded__distutils_hack preloaded_distutils preloaded_backports
extra_path copied_path vendor_first repeated_begin early_shim repeated_shim
foreign_thread foreign_shim function_code wrapper_closure
alias_spec alias_closure alias_module alias_loader missing_alias unlisted_child
resource_traversal resource_caught resource_changed metadata_path metadata_enumeration
partial_install_before partial_install_after partial_create partial_exec
restoration_before restoration_after late_publication
```

Each command is exactly:

```
/home/ang.gao/miniforge3/envs/robot_sim/bin/python -I -S -B /data2/shared/ang.gao/robot_sim_cuda_compilation_wip/tests/support/setuptools_dependency_process.py parent /data2/shared/ang.gao/robot_sim_cuda_compilation_wip /data2/shared/ang.gao/setuptools-failures-427c90ae6b69360b5cd16178e027677bf3ae09a0-v1/<case> <case>
```

The inspected SHA-bound wrapper may issue those commands once via:

```
/home/ang.gao/miniforge3/envs/robot_sim/bin/python -I -S -B /data2/shared/ang.gao/run_setuptools_failure_matrix_427c90a.py
```

The campaign and each case directory must be fresh and exactly0700. Preserve all raw outcomes, including failed/incomplete ones; inspect terminal/resource/error facts and every recorded parent/child PID afterward, including the orchestrator PID. Do not reuse old directories, alter old artifacts, run a changed-code retry or silently substitute selectors. Fresh actual outcomes still require subsequent checkpoint review; preflight does not accept them in advance.

No installed setuptools case, backend import, compiler/Ninja, native observer, GPU/device work, subreaper operation, installation, reset, broad kill, shared-parent chmod or old-artifact modification occurred during review. No campaign directory was created. Only source/stdlib/CPU fixtures, bounded owned-process tests, file verification and new external evidence were used.

These fixtures cannot prove that every selector reaches its intended boundary through the installed dependency's full execution; that is the authorized campaign's task. They cannot prove complete compilation-record joins, actual cold builds, native ownership, process-tree quiescence, device correctness, readiness or performance.

Still closed: all new compiler/backend/native execution until its own exact code/source/catalog/runner/observer/ABI preflight; full actual v2 compilation-record integration and mutation joins; compiler-tree OS/subreaper proof; three real cold builds and process/device/failure matrices; COMPLETE/readiness/performance/product activation; parent C2d/C4 final acceptance. A fresh whole-checkpoint review and the owner's systematic whole-C4 refactor after functional and real validation remain mandatory.

The author may archive this complete report unchanged as review-only evidence, preserving the exact approved parent revision, finite execution scope, finding dispositions and all limitations. A report-only archive does not expand the approval.
