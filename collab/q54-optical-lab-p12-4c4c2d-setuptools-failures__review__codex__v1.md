# Independent fixed setuptools failure-matrix preflight

Initiative: Q54 / P12.4 / C4c2 / CUDA G1
Stage: independent bounded code and execution preflight
Author: codex, independent reviewer /root/setuptools_failures_preflight_v1
Version: v1
Date: 2026-10-09 (Asia/Shanghai)
Status: in_review — Request changes
Related Files: setuptools-failures review-request v1; setuptools-admission design-request/design-review v2; compilation-admission design-request v3; setuptools-cpu-preflight test-evidence v2
Owner Summary: Request changes. The negative harness requires the healthy positive's RELEASED terminal, although failures inside guard6 correctly quarantine. Its partial_create and partial_exec injections also occur outside the real owner's error-catching boundary, so shim unwinding records a different first error. Both defects were reproduced without importing the installed dependency. No new dependency execution is approved.

## Verdict and frozen identity

**Request changes.** Two blocking findings remain. The proposed one-positive-plus-32-negative campaign is not ready at this revision. This review authorizes no invocation of that campaign, including no renewed positive run. Repair the test driver, add bounded source/CPU regressions against these counterexamples, freeze a new candidate, and obtain independent re-review. Do not change the production failure lifecycle to satisfy the harness.

This was a fresh independent reviewer session. I did not author or edit the candidate and did not delegate the review. I read AGENTS.md, collab/README.md and collab/REVIEW_PRINCIPLES.md, the exact review packet, accepted setuptools requirements and relevant compilation requirements, the operative diff, the full driver and initialization helper, affected source/loader/failure/guard call paths, and existing source-only tests. Author summaries and prior approvals were not used as proof of the new faults.

- Local repository: `C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design`.
- Local branch: `implementation/c4-supervised-waits`.
- Full base: `7794c321478d3994ff45bffeb5802ba10d614107`.
- Reviewed HEAD, local and target: `44a57da4d80c1372f12ccc2478be981e9b5afee8`.
- HEAD tree: `84e4d45f5f3c50125a450065567ddbd0a9e60361`.
- Target: primary `tileops-dev`, `/data2/shared/ang.gao/robot_sim_cuda_compilation_wip`, detached HEAD. No fallback host was used.
- Initial/final local and target tracked/untracked status: clean. HEAD remained frozen.
- Delta: `tests/support/setuptools_dependency_process.py` plus the review-request document; no production source, source catalog, environment profile or installed dependency change.
- Driver working-byte SHA256, equal locally and on target: `37225a6d2053f9807c44db34d73ccffc00a4dad360e8ced16666e53690833434`.
- Accepted setuptools design v2 verified SHA256: `e8343ed7c9217a47ad45da5f055801cf43537a67f8059240453ed14078e27db8`.
- Accepted compilation design v3 verified SHA256: `028b52c466ef46433235d4ec2495cd47d00200822cc64dfaea95614fd5bf26a2`.

## R1 — Blocking: every negative requires the wrong terminal disposition

Location: `tests/support/setuptools_dependency_process.py:273` and `:274`; failure entry at `:497`. Relevant production paths: `optical_worker/native_bootstrap.py` `_SourceSession.fail`, `_ExternalAdmission.fail`; `optical_worker/guarded_lifecycle.py` `_fail_guard`, `_begin_termination`, `terminate`.

Trigger: a selector produces its intended rejection during active guard6. The source/admission failure latches poison and calls the real controller's `_fail_guard("operation", error)` before termination. That makes the guarded replay failed. `terminate` then retains/quarantines the graph; it does not authorize ordinary release simply because this particular run has no device work.

The new harness instead requires `cancelled_before_submission_exit`, guardian `RELEASED`, and release `completed`. Those are the old positive's healthy cancellation facts, reached after its normal setup segment successfully returns. They are incompatible with the failure path under test. Thus a correctly denied negative reaches `observe_failure`, fails this assertion, writes failure.json and exits74 instead of producing the requested negative proof. The packet's phrase “actual host-only cancellation” repeats the same incorrect expectation and must be corrected with the driver.

Independent reproduction used the real guarded controller, real Peer/client/core transport and actual HOST_OWNER adoption, with six completed setup segments followed by guard6. It compared healthy cancellation against the exact controller calls made by SourceSession.fail. Target Python3.12.13 observed:

| Fact | Healthy positive control | Active-guard rejection |
| --- | --- | --- |
| disposition | cancelled_before_submission_exit | quarantined_process_exit |
| guardian_state | RELEASED | TRANSFERRED_TO_QUARANTINE |
| release_status | completed | not_authorized |
| custody | retained | full_quarantine |
| guarded_segment_failed | false | true |
| terminal_arm_attempt_count | 1 | 0 |
| setup_operation_count / entered_leaf_count | 0 / 0 | 0 / 0 |
| emergency_sync_attempt_count | 0 | 0 |

The failed control also verified the actual adopted object remained in guardian custody. This is real controller behavior with an explicit outer fixture, not installed dependency execution or a fabricated serialized terminal.

Existing checks miss it because existing runner tests exercise source restrictions/directory/group cleanup, and initialization helper fixtures use an outer session that records poison without exercising this complete source-to-controller failure bridge. The old successful dependency archive proves only healthy cancellation.

Repair direction: require the actual failed-guard quarantine facts, zero device work/synchronization, retained graph and no ordinary release; retain exact owned-process exit/cleanup proof outside the child. Preserve the positive's existing healthy cancellation semantics. Add a source/CPU regression joining an actual production failure to the negative validator and an explicit healthy positive control. Do not remove terminal validation or weaken production quarantine.

## R2 — Blocking: partial create/exec injections are outside the owner's failure boundary

Location: driver `:183`–`:192` (`partial_create`) and `:202`–`:205` (`partial_exec`); first-error/identity requirements at `:266`–`:270`. Relevant production behavior: `_SetuptoolsInitialization.alias_create`, `alias_exec`, `leave_shim`, `_AliasLoader`, and the source-inspected installed `_distutils_hack.shim.__exit__`.

Trigger: either original alias operation succeeds and returns, then the new test wrapper raises its injected exception. The owner's own try/except has already returned. It has not called `fail` for that exception. On create failure the alias was never published; on exec failure importlib removes the failed publication. Installed `shim.__exit__` calls `_remove_shim()` without forwarding the active exception, before the outer ordinary source loader catches it.

Consequently real `leave_shim` sees ALIAS_CREATED or ALIAS_EXECUTED, not FAILED, and finds the required alias publication missing. It raises and records `ValueError("setuptools alias publication order")` first. The injected RuntimeError is only the exception context. The driver requires the first error to contain `injected:partial_create`/`injected:partial_exec` and be the exact injected object, so both cases fail independently of R1.

The independent CPU fixture invokes the real driver `_DependencyFault.install`, real initialization `alias_create`/`alias_exec`/`leave_shim`, real `_AliasLoader`, real guardian/adoption/retention and stdlib `module_from_spec`. It supplies descriptive module/delegate/source records and mirrors the inspected shim exit plus importlib's failed-publication deletion; it does not import or execute installed setuptools. Both Python3.14 local and target Python3.12.13 produced:

```
first_error partial_create setuptools alias publication order injected_in_context True
first_error partial_exec setuptools alias publication order injected_in_context True
```

The probe asserts exception-context identity, that owner.errors[0] is the publication error rather than the injected object, final FAILED state, and restoration attempts for all installed hooks. These are counterexamples to the harness's promised first-error assertions, not evidence that the installed negative matrix has run. Static inspection of the actual pinned shim and production source-loader finally ordering establishes applicability of the fixture's unwind sequence.

Existing helper tests cover hook installation/restoration and resource denials but do not execute these new driver selectors across alias/shim unwinding. The driver currently tests an exception after a completed operation while naming it a partial operation; changing only its expected error would not establish the promised partial-create/exec contract.

Repair direction: inject after the real delegate has acted, inside the production owner's protected operation and before the successful alias-created/alias-executed action. Preserve the actual source object and real delegate call. Prove that the owner's real exception handler latches the injected object before shim unwind, retains the returned source, attempts restoration once, emits one failed action and no inappropriate successful action after the partial cut. Add source-only selector regressions and then obtain exact candidate preflight before real dependency execution.

## Other inspected behavior and finite scope

The selector dictionary has exactly32 distinct names and the packet enumerates those same32, plus positive. There is no arbitrary import name, input payload or callback selector supplied by the command line. Parent and child validate the selected case; child request keys are exact. Each invocation owns one cold child. The after-loaded assertion rejects a missed injection; the sole deliberately post-load case is unlisted_child, which requires RESTORED before attempting the undeclared child.

I inspected all selector groups: four reserved preloads; extra/copied/vendor-first path; repeated begin and early/repeated/foreign-thread shim; foreign shim/function/wrapper closure; alias spec/closure/module/loader and missing publication; unlisted child; traversal/caught/changed resource; metadata path/enumeration; before/after installation and restoration; partial create/exec; late publication. Most target existing real callback checks or actual assignment boundaries. R2 identifies the two faulty placement choices. Preloads use foreign ModuleType objects only to request rejection before installed dependency entry, not as successful packages.

Failure validation otherwise checks named first error and injected identity, sticky owner/session/admission poison, the first error in source failures, actual subsequent denial calls, one failed initialization event and no later success/COMPLETE, restored hook identities or the deliberately still-failing restoration-before binding, exactly one restoration assignment per installed hook in assignment cases, retained actual source for partial/late-publication cases, and actual late publication. These assertions do not compensate for R1/R2.

The new foreign-thread selector has one daemon test thread and a bounded10s join, checked dead before accepted evidence. It does not create another importer or process owner. Existing bounds remain parent wait150s, owned-group SIGKILL plus leader wait10s, and Peer join2s. The allowed child subprocesses remain only the two exact read-only Git commands; backend imports, compiler/general subprocesses, fork/spawn and ctypes library loading remain denied. This is a trusted fixed CPU harness, not a hostile-code sandbox or compiler process-tree proof.

The positive branch's same-object alias, version83.0.0, closed real resource handles/hash, singleton finder/path, original hook/method checks, source/admission/retention audit and RESTORED/no-COMPLETE assertions are unchanged. Profile and catalog pins remain unchanged. No extra runtime permission or import row was added.

Fresh campaign directory naming with full approved HEAD, exact0700, one case per fresh subdirectory, serial execution, archive-all-outcomes and parent/child PID absence remain execution obligations after a later approval. Existing directory creation handles inherited setgid on the newly owned descriptor, never by chmod of the shared parent. No old run directory was reused in this review.

## Verification, commands and results

Primary environment: `/home/ang.gao/miniforge3/envs/robot_sim/bin/python -I -S -B`, Python3.12.13, Linux; explicit repository and site-packages sys.path entries for pytest; `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`, `-p no:cacheprovider`, `--noconftest`. Reserved roots were absent before and after: torch, warp, numpy, scipy, trimesh, setuptools, _distutils_hack, distutils, backports. Fixture publication of temporary ModuleType names is undone by pytest monkeypatch; no installed package executes.

The complete command and assertion driver is preserved in `c4_assets/setuptools_failures_preflight_runner_v1.py`. Exact target invocation:

```
ssh -o BatchMode=yes -o ConnectTimeout=15 tileops-dev '/home/ang.gao/miniforge3/envs/robot_sim/bin/python -I -S -B /data2/shared/ang.gao/setuptools-failures-44a57da4d80c1372f12ccc2478be981e9b5afee8-review-runner-v1.py /data2/shared/ang.gao/robot_sim_cuda_compilation_wip /home/ang.gao/miniforge3/envs/robot_sim/lib/python3.12/site-packages /data2/shared/ang.gao/setuptools-failures-44a57da4d80c1372f12ccc2478be981e9b5afee8-review-probe-v1.py'
```

Independent results: **33 passed in6.48s**:29 existing runner/helper cases plus4 independent counterexample/control cases. This ran the Linux directory inheritance and exact owned-process-group tests. Those cleanup tests prove leader wait and descendant termination and permit OS-reaper zombies; they are not compiler subreaper acceptance. Reviewer runner PID783551 was independently checked absent afterward. Both candidate trees remained clean. All1,791 target profile file rows matched size/SHA256 without loading their contents as packages. Released profile SHA256 is `0cb6288345fc3a2644e61bb01dc6e5897ee58d52eb71050c31b2e76f8ea60753`; catalog SHA256 is `17387af6b8188c40707da7e1835f639e87d1464ffa664b7cdcdb301eedd9d399`.

Local independent scratch reproduction: Python3.14.0, `-I -S -B`, explicit repository and `C:/Users/AG/AppData/Roaming/Python/Python314/site-packages`; four cases passed in2.49s. First scratch attempt had reviewer fixture mistakes: guessed terminal enum spellings and a POSIX path string inconsistent with Windows Path normalization. Those affected only the external scratch probe, were corrected to observed production values/Path semantics, and both local and target probes then passed. They were not candidate or toolchain failures. Two initial reads used nonexistent guessed test paths; discovery located the actual files. A PowerShell unquoted `HEAD^{tree}` read-only command was parsed incorrectly and rerun with quoted revision. No implementation edit resulted from these tooling mistakes.

Other read-only verification: `git rev-parse HEAD`, `git rev-parse 7794c32`, `git rev-parse 'HEAD^{tree}'`, branch/status checks, `git diff --stat 7794c32 HEAD`, `git diff --check 7794c32 HEAD` (passed), targeted source reads and SHA256 checks. Installed `_distutils_hack/__init__.py` and override.py were read over SSH, not imported.

Author target log was independently read and its supplied digest checked:32 passed in6.48s,1791 files,32 selectors; SHA256 `b521ae22acb67ba3ddaf667b91aa86732117d47fa1c36b3fc9b1f7b7324a1fff`. PID732242 was absent. This is corroborating source-only evidence, not independent execution of the matrix.

The prior archive `setuptools-cpu-0177a86-restored-v1.tar.gz` was SHA256-verified as `18ce11bb63b7f12cdde7939a743cedb820d4b104e4bc0a0f32c1148db7df0599` and its JSON read directly without extraction. It describes revision0177a869aec966da2e112d5f92120892aa2fb1fa, RESTORED, version83.0.0, one metadata/resource read, and healthy RELEASED cancellation. It is useful historical positive evidence only and grants no retry permission.

Review artifacts outside tracked files:

| File under c4_assets | SHA256 |
| --- | --- |
| setuptools_failures_preflight_probe_v1.py | 4b165676acd12d5b484a40d40629ca3fabe0a75d69176925b0ca02875ad547a6 |
| setuptools_failures_preflight_runner_v1.py | 78f3748ad2c66c53f7c10320f242cd221bc6041f8f6a1dc0faf091fd97334ade |
| setuptools_failures_preflight_target_v1.log | 19de01224e9ba48e906ec0d374419ddb2cd7abb7aea51955f7550146b2c4cf49 |

Target scratch scripts use the full candidate HEAD in their names as shown in the invocation; their bytes match local hashes. The log is the local PowerShell Tee-Object transcript; its hash binds those recorded bytes. This report's final hash is supplied separately to avoid a self-referential digest.

## Limits and remaining gates

No installed setuptools matrix case, real dependency positive, Torch/Warp/NumPy/SciPy/trimesh import, compiler/Ninja, native observer, GPU/device route, installation, reset, broad kill, shared-parent chmod or old-artifact mutation occurred. Only bounded source/stdlib/CPU fixtures, explicit owned-process cleanup tests, file reads/hashes, and new scratch evidence were used. No campaign directories were created. No successful real negative result is claimed.

No additional substantive finding was established beyond R1/R2. Static inspection and CPU fixtures cannot establish that every new selector hits its intended boundary through all actual installed source behavior; that is precisely the subsequent separately reviewed campaign's job. The current request cannot execute successfully as promised, and acceptance of future negative evidence is not implied by a repaired preflight.

Still required after repair/re-review: exact actual dependency positive and32 negative evidence with raw outcomes/process hygiene; full actual v2 compilation-record integration and mutation joins; new exact compiler/source/catalog/observer/ABI preflight; three real cold builds and compiler-tree OS proof; required process/device/failure matrices; fresh whole-checkpoint review. COMPLETE, compilation/native/readiness/performance authority, parent C2d/C4 acceptance and product activation remain closed. The owner's whole-C4 systematic refactor remains mandatory after functional and real validation before final acceptance.

The author may archive this complete report unchanged as review-only evidence. That archival step must preserve the verdict, both original counterexamples, exact candidate and denied execution scope.
