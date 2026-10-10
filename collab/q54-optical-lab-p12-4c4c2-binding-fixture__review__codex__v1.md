Initiative: Q54 / P12.4 / C4c2 bounded runtime-binding test fixture
Stage: independent code review, TEST-FIXTURE ONLY
Author: codex
Version: v1
Date: 2026-10-10
Status: accepted
Related Files: tests/unit/optics/test_guarded_runtime_binding.py; collab/q54-optical-lab-p12-4c4c2-native-state-reader__design-request__codex__v1.md
Owner Summary: Approve the exact test-only delta. Independently reproduced the base fixture's source-custody failure and ran the candidate file: 34 passed. No native-reader production, native execution, or whole C2f/C4 approval is granted.

## Identity and frozen scope

Verdict: **Approve**.
Reviewer session ID: `01a12521-79bf-7e33-bbc8-523ba178ba88` (CODEX_THREAD_ID).
Reviewer: fresh independent Codex review session; did not author the candidate and received no implementation conversation. The owner packet and author test claims were treated as claims, not verdict evidence. No agents were spawned.
Evidence run ID: `3f6fa166-52a0-42ca-85cf-20baf0a77332`.
Candidate: `C:\Users\AG\新建文件夹\c4_binding_fixture_review`.
Base: `5d804f73b5ae00e89fc18455080b1e339d20dade`.
HEAD: `eab674c6f9c7eb9161a4f731db51e69352561634`.
HEAD parent equals the stated base. Detached HEAD; `git branch --show-current` and `git symbolic-ref -q HEAD` produced no branch. Initial and final porcelain status were empty; final ignored-file status was also empty. No candidate edits, resets, installs, signals, remote access, or native/GPU execution were performed.
Diff: exactly one file, `tests/unit/optics/test_guarded_runtime_binding.py`, 17 insertions and 2 deletions. Production files are identical to base.

Read AGENTS.md, collab/README.md and collab/REVIEW_PRINCIPLES.md completely. The initial PowerShell default decoding garbled non-ASCII text; README and REVIEW_PRINCIPLES were reread completely with explicit UTF-8 before review. Read the complete specified design. Its LF-normalized SHA256 is exactly `0b6dab55c496e730219dfbb253b7d9530dff40faa6c90676c71a2b90e0f41ca5`. Raw Windows checkout SHA256 is `784eb2e632d689ebae487bd1c6b05931d646a0943cb976147f3ef031a43071d6`; this is the CRLF representation, not a semantic design difference. The packet supplies the design approval; this review neither reissues that approval nor relies on its in-file status as approval evidence.

## Findings

No substantive findings. No blocking evidence gap for this bounded fixture repair, and no required non-blocking follow-up.

The inspected risks and conclusions are:

1. **Original source custody is modeled, not bypassed.** Fixture lines 57 and 88 establish `controller._native_source_session` and adopt the same session under `captured_source_session`. These match the production binding path at native_bootstrap.py:2487-2539 and guardian lookup at guarded_lifecycle.py:191-205. The real `bound_controller()` remains in use and checks controller, guardian, original configuration, channel, lane and source identity. The old fixture omitted the new link and failed there before exercising its assertions.

2. **Real reader construction/adoption and identity progression remain active.** Initialization at native_bootstrap.py:2836-2958 still constructs `NativeStateReader` inside the actual armed stream leaf, retains/adopts it, retains and journals the returned simulated stream, calls real `stream_returned`, mints/adopts/registers the real runtime binding, and calls real `binding_returned`. NativeStateReader.__init__, `_ticket`, stream/binding methods and `require_reader` (native_scope.py:21-119 and 511-541) are not replaced. The two omitted reader leaves call the real `_ticket()`, preserving live guard, owner-thread, sticky-failure and exact-reader checks. The environment placeholder and initial `_native_scope_record = None` supply the minimum constructor inputs without creating an admitted environment inventory.

3. **No second successful native fact producer.** Fixture lines 61-66 replace native query initialization and snapshot production only. `omitted_snapshot` checks the real ticket and returns a local `(journal length, operation)` pairing value. It does not query native state, construct a native-state payload, append `native_scope`, or publish a successful receipt. Production snapshot's fact construction/append is at native_scope.py:433-456 and is deliberately not exercised. Deferred CUDA/Triton inspection is omitted because the simulated module has no inventoried source/import state; its no-op returns no success fact. The reader may retain this `None` return in its raw diagnostic list, which is not native-state evidence. The fixture comments and module docstring clearly limit this to binding/state unit behavior and retain cold/GPU validation as a separate gate.

4. **The checkpoint omission is limited and honest.** The placeholder environment would make `_runtime_phase_completed` call checkpoint (native_bootstrap.py:2611-2621); the real checkpoint requires captured environment/input inventories and custody replay (2299-2305), which this pre-existing unit fixture never created. The fixture therefore omits checkpoint instead of inventing a valid custody record. The parent-factory test separately overrides checkpoint with a call counter and still asserts one successful publication checkpoint, none on partial failure, and no repeat completion. `_publish_optical_stage` and `_session_prepare_cuda_factories` were inspected at native_setup.py:193-233. Complete replay correctness is outside this fixture's evidence.

5. **No original assertion weakened and no newly vacuous test.** An AST comparison found the same 16 test functions and 71 assert statements at base and HEAD; only `test_file_loader_owned_native_root_is_not_adopted_twice` has a changed test-function AST. That change is the exact owner delta, from 3 to 4: target, real reader, stream and binding, with the pre-adopted native root still not duplicated. The second no-growth assertion is unchanged. All original checks for exact binding fields, single use, forged copies, seven object-field drifts, six actual-resource/configuration drifts, upload/readback rejection before helper calls, checkpoint ordering, partial factory retention, unknown-scope quarantine, zero fallback synchronization, returned-stream liveness after adoption failure, materialization limits and simulated copy scope ordering remain active. Real binding validation/consumption was inspected at native_bootstrap.py:3148-3211; upload/readback ordering at 3213-3307. These checks are not replaced by the new mocks. Positive paths still call the production entry points and assert completion/identity; negative paths still assert rejection and relevant final state.

6. **Cleanup remains exercised.** The unchanged fixture terminates its controller when needed, removes runtime/upload registrations and calls Peer.close in finally. The monkeypatch fixture restores class methods, environment/module substitutions and custody registration after fixture cleanup. Peer.close joins its thread, checks it is stopped, closes both sockets and asserts no peer errors. All 34 test cases completed without teardown errors. The eligible terminal-sync source lookup/snapshot call flow at guarded_lifecycle.py:742-772 was inspected; its native snapshot leaves are likewise omitted by this fixture and cannot establish physical synchronization or release safety for a real backend.

## Independent verification and commands

Environment: local Windows, `C:/Python314/python.exe`, Python 3.14.0 (64-bit), pytest 9.1.1, pluggy 1.6.0. All test execution used a newly started interpreter, `-B`, `PYTHONDONTWRITEBYTECODE=1`, `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`, repository PYTHONPATH, and TEMP/TMP under the authorized scratch directory. Pytest cacheprovider was disabled; no old pass outputs/caches were used as proof. Candidate output additionally used `PYTHONIOENCODING=utf-8`.

Read-only revision/static commands included:

- `git rev-parse HEAD`, `git rev-parse HEAD^`, `git log -1 --format=fuller`, `git branch --show-current`, `git symbolic-ref -q HEAD`.
- `git status --porcelain=v1 --untracked-files=all` before and after; `git status --short --ignored` after: empty.
- `git diff --stat BASE HEAD`, `git diff --no-ext-diff BASE HEAD -- tests/unit/optics/test_guarded_runtime_binding.py`: exact stated scope; full delta saved as reviewed_delta.diff.
- `git diff --check BASE HEAD`: passed, no whitespace errors.
- `rg --files -g AGENTS.md -g conftest.py -g pytest.ini -g pyproject.toml` and `git ls-files '*conftest.py' '*AGENTS.md'`: root AGENTS.md only, no tracked conftest.
- `Get-Content -Raw -Encoding UTF8` for instructions, design, complete candidate test, pyproject.toml and CPU peer support; numbered PowerShell reads and `rg -n` for the production call flow cited above. Inspected package imports for CPU-only entry behavior. Did not inspect another agent's private context.
- `Get-FileHash -Algorithm SHA256` for the design, plus Python hashlib over bytes with CRLF replaced by LF: digests recorded above.
- Python ast.parse/ast.dump comparison of base `git show` bytes and candidate test bytes: 16/16 test functions, no additions/removals, 71/71 assertions, only the owner-count test function changed.

Baseline reproduction:

`git show 5d804f73b5ae00e89fc18455080b1e339d20dade:tests/unit/optics/test_guarded_runtime_binding.py` was captured byte-for-byte to scratch/test_guarded_runtime_binding_base.py. This did not switch or modify the checkout. Because production is identical at base and HEAD, this uses the base test fixture against the exact base production code.

Executed from the candidate cwd:

```text
C:\Python314\python.exe -B -m pytest -p no:cacheprovider -vv --tb=short --basetemp=C:\Users\AG\新建文件夹\c4_assets\binding_fixture_review_scratch_v1\pytest_tmp_base_repro C:\Users\AG\新建文件夹\c4_assets\binding_fixture_review_scratch_v1\test_guarded_runtime_binding_base.py::test_initialization_mints_exact_adopted_single_use_binding
```

Observed: **1 failed in 1.53s**, exit 1, the expected `SourceBootstrapError: source custody identity changed`, reached from initialize_native_runtime:2845 through bound_controller:2491. Log: base_repro.log. Pytest rootdir for this external scratch test was the common parent directory; imports used the explicitly supplied candidate PYTHONPATH. No full base suite or old 4dcf73d candidate was executed.

A scratch harness output-decoding error occurred after that child had already exited and been reaped: its first attempt read the Windows-encoded baseline log as UTF-8 and raised UnicodeDecodeError. This was a reporting-tool failure, not an additional pytest failure or candidate defect. The complete baseline log remained available and was read with PowerShell's Windows decoding. The scratch runner was corrected to request UTF-8 output and run only the candidate; the baseline was not rerun. The retained run_review_tests.py represents that corrected candidate invocation. execution.json records only the candidate run; baseline process/outcome are recorded here and in its log.

Candidate command, from the candidate cwd:

```text
C:\Python314\python.exe -B -m pytest -p no:cacheprovider -vv --tb=short --basetemp=C:\Users\AG\新建文件夹\c4_assets\binding_fixture_review_scratch_v1\pytest_tmp_candidate tests/unit/optics/test_guarded_runtime_binding.py
```

Observed: **34 passed in 3.80s**, exit 0, including all parameterizations and teardown. Log: candidate.log. Child wall time including interpreter startup: 4.585s. The author-reported combined 82-pass result was not reused as independent evidence or independently rerun; this review ran the entire changed test file as requested.

## Owned process closure and side effects

Both pytest children were waited/reaped using subprocess.Popen.wait: baseline PID 40236 exited 1, candidate PID 39124 exited 0. The corrected wrapper PID 6028 exited 0; the initial wrapper exited 1 after its reporting-only decoding error. No background worker or native child was requested by this test file; its peer uses an in-process thread.

After execution, `Get-CimInstance Win32_Process` was filtered for the known baseline/candidate/wrapper PIDs and Python command lines containing binding_fixture_review_scratch_v1 or test_guarded_runtime_binding. It returned an empty array, saved in process_closure.json. No owned matching processes remained. No signals or forced termination were used. This checks closure of this CPU review's processes, not native integrated-worker process closure.

All reviewer-created files are inside `C:\Users\AG\新建文件夹\c4_assets\binding_fixture_review_scratch_v1`: this report, scratch runner/base reproduction, test logs, execution metadata, reviewed delta, process-closure evidence and final Git status. No tracked, untracked or ignored candidate side effects were observed. Final HEAD remained eab674c6f9c7eb9161a4f731db51e69352561634.

## Limits and exact approval

Approved only: the 17-insertion/2-deletion runtime-binding CPU test-fixture delta in tests/unit/optics/test_guarded_runtime_binding.py at eab674c6f9c7eb9161a4f731db51e69352561634 relative to 5d804f73b5ae00e89fc18455080b1e339d20dade. The fixture accurately exercises local binding/guard/retention behavior under simulated CUDA leaves and accurately omits native-scope facts and full custody replay.

Not executed or approved: native-reader ABI/read-only/cache/cold-query preflight; native library loading; Torch/Warp/GPU runs; physical device/context/stream restoration; complete artifact inventory or six-pair replay; real environment/source custody validation; native getter/receipt failure matrix; integrated worker/native acceptance or full C2f workload/failure/numerical/physical-scope/closure matrix. These are outside this test-only delta or explicitly prohibited for this session. The separate production-base preflight has no presumed verdict here. The declaration that reader behavior has its own integration tests is not a claim this reviewer ran or approved them.

No whole C2f/C4, C4c3/c4/C4d/B3, final-sample physical proof, product activation, shipping, R2/S/host delivery, result/baseline, a3b, performance or backend decision gate is opened. No assertion that mocks prove native behavior is made. Review is complete; stop after recording this report.
