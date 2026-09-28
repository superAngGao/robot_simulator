# C4c1c fixed sacrificial GPU validation — independent review

Initiative: q54-optical-lab-p12-4c4c1c-gpu-validation
Stage: review
Author: codex (independent reviewer)
Version: v4
Date: 2026-09-28
Status: in_review
Related Files: C4c1 ABI design v1/v2/v3; supervision review v3; recording review v2; GPU-validation reviews v2/v3
Owner Summary: Request changes. Prior F1/F2 repairs remain sound on inspected code and newly executed checks, but the exact C4c1c acceptance scope still lacks same-bundle repeated sentinel validation, real recorded-leaf contamination instrumentation, and execution-time stream-restoration proof. Passing the existing matrix cannot discharge these missing assertions. No checkpoint or product gate is approved.

## Identity and scope

- Fresh independent session: `/root/c4_sacrificial_gpu_review_v4`; no implementation conversation inherited, no candidate authorship and no subagents.
- Local candidate: `C:/Users/AG/新建文件夹/p12_4_c4_sacrificial_gpu`, branch `implementation/c4-sacrificial-gpu`.
- Original base: `7de02e89c3e868e7554e5b1c385cfbbcf06570e1`.
- Code candidate: `9db8146f9c818dd580ce73d886f7afe4c5d22d73`.
- Reviewed HEAD: `55365c528cbdb5208804be9e46220e4bdf6a7335`; tree `aa11be6cebb9727165deafc28a1da713b1bd71ed`.
- Independent Linux execution: `/data3/ang.gao/robot_sim_c4_gpu_verify_55365c5`, authorized fallback `ang.gao@192.168.80.101`, detached HEAD matching the local candidate. Both trees were clean on entry. The old dirty development checkout was not used or changed.
- The code-candidate-to-HEAD delta contains only the preserved v3 report and its separate test-evidence record. No executable, design or operative workflow delta is hidden in that evidence-only commit.
- Base-to-HEAD changes 18 files, 2,584 insertions / 80 deletions. The production delta is the shared CPU host-observation comparison extraction and the `device_to_host_copy` setup-leaf descriptor kind. Other executable changes are test support, tests and fixed test-only supervisors. Product protocol, allowlist and activation files are unchanged.

Read `AGENTS.md`, `collab/README.md`, and `collab/REVIEW_PRINCIPLES.md` in full (reread Chinese with UTF-8 after initial PowerShell decoding). Read design v2/v3 and the still-applicable v1 contracts, supervision v3, recording v2, both prior GPU-validation reviews, implementation note and follow-up. Historical reports/author results are claims, not substitutes for current execution. Design hashes: v2 `106ad2380480acd18e4733ea23a98735df837ae1e79d797c9001c8a641011e30`; v3 `291d4248633a9398eb4466b5e3a0ec514e04b810c5f6575886a4fab2804e1626`.

## Verdict: Request changes

The new findings below are concrete acceptance/verification gaps, not demonstrated numerical corruption, use-after-free or actual stream drift. They block approval of the precise complete C4c1c scope requested. No existing accepted document exempts them: v2 preserves the non-conflicting v1 contracts, while v3 changes lifecycle containment, not these correctness/instrumentation requirements. The coordinator confirmed that there is no approved scope waiver; this report does not narrow the contract to fit the existing tests.

### F3 — P2 / blocking: repeated sentinels never exercise the same recorded bundle

Locations: `tests/support/sacrificial_warp_child.py:150-168`, `:195-221`; `tests/gpu/test_warp_shade_microfixtures.py:125-180`, `:205-239`. Requirement: design v1 section 14.4, retained by v2.

The fixed child runs ordinary `wp.launch` warmup, writes a single `77` sentinel to non-ray buffers, then executes the recorded command tuple exactly once before drain, inspection and release. Its setup deliberately skips origins/directions when poisoning outputs. The two-sentinel microfixtures do execute production kernels repeatedly, but they are direct kernel launches, not a second execution of the same sealed recorded bundle. Separate `run_case` invocations construct new processes/storage/bundles and cannot prove reuse of the existing command bindings.

Trigger/consequence: a recorded-owner/binding/reuse-specific defect that appears only on the second invocation, or a ray output left dependent on prior contents, is not challenged by this validation. Current tests can pass without establishing the required same-bundle deterministic reuse property. This is not a claim that such a runtime defect was observed.

Direction/proof: add fixed supervised validation that retains the same exact recorded bundle and physical storage across two distinct full sentinel patterns, then checks both executions against the independent CPU oracle and against each other, including physical ray outputs for camera bindings, all four shade variants and the zero-ray scalar resets. Keep both invocations within a deliberately defined guardian/watchdog lifetime; do not reopen a released/quarantined guardian or bypass the monotonic lifecycle to make the test pass. If this requires a lifecycle clarification, freeze that clarification before changing the safety authority.

### F4 — P2 / blocking: recorded execution has no runtime contamination guard

Locations: `tests/support/sacrificial_warp_child.py:195-221`; `tests/gpu/test_warp_sacrificial_render.py`. Requirement: design v1 section 14.4 and acceptance item 7.

The actual leaf calls `command.launch(stream=driver.stream)` and checks the ambient device. There are no runtime hooks surrounding that leaf that reject allocation, host materialization/conversion, interop, module load, event operations or synchronization. The older recording and buffer tests patch forbidden calls while validating or recording; they expressly prevent `Launch.launch` and therefore cannot establish the behavior of an executed recorded command. Final numerical parity does not detect hidden work that leaves the same answer.

Trigger/consequence: introducing a hidden allocation/copy/wait/module load inside actual recorded execution would not be rejected by this new matrix merely because the device result remains correct. The source looks like a narrow call, but source inspection and successful output are not the required runtime contamination proof.

Direction/proof: install scoped, fixed instrumentation around the real recorded-launch interval in the sacrificial child, covering the enumerated forbidden operation families while allowing only the frozen recorded launches. Demonstrate detection with safe fixed negative controls for injected forbidden operations, and restore hooks before supervised drain/inspection/setup work that legitimately performs those operations. Preserve source-manifest admission, owner retention and process-terminal fault handling; do not expose callback/module inputs over IPC.

### F5 — P2 / blocking: execution-time ambient stream restoration is not tested

Locations: `tests/support/sacrificial_warp_child.py:90`, `:210-211`, `:310`; `tests/gpu/test_warp_sacrificial_render.py:117-128`. Requirement: design v1 section 14.3.

The two-device test selects target 1 and checks target evidence plus ambient device ordinal 0. The child saves only `wp.get_device()` and asserts that device after launches. It neither freezes nondefault ambient streams on both involved devices nor checks their identities after recorded execution. Existing recording tests check stream restoration during preparation/recording only and forbid the execution under review.

Trigger/consequence: changing a device's current stream while restoring the current device can still pass these assertions. The test name's restoration claim is therefore broader than what it observes. No actual stream-restoration failure has been demonstrated.

Direction/proof: capture the exact current stream objects on both involved devices, retain the required owners, and compare device/current-stream identities before and after successful recorded execution and relevant injected failure paths where observation is safe. Include a nondefault target current stream so restoring a default is not accidentally equivalent; use an explicit ambient-stream fixture consistent with the target-only setup rule. If creating a nondefault foreign-device fixture requires extending that setup rule, resolve it in a scoped reviewed lifecycle clarification rather than concealing foreign dirty scope. Execute with real target 1 / ambient 0; a skip or an ordinal-only check is not proof. Ordinary cleanup after an unsafe failure must remain prohibited.

## Prior findings

### F1 — source-cache identity: closed on current evidence

The admitted loader reads source bytes, checks that exact byte buffer against the manifest, and compiles it without calling the cache-reading parent `get_code`. Both fixed entry scripts create exclusive fresh cache namespaces before their first repository helper import and disable cache writes. The source ledger still admits only the fixed repository closure; installed runtime dependencies remain the declared trusted boundary.

The original external repro was rerun unchanged with local isolated/no-site Python. It printed `SOURCE: VALUE = 'new'`, `EXECUTED_VALUE: new`, and `VERIFIED_LEDGER: ('cache_probe.py',)`, then exited 1 at its intentional old `assert cache_probe.VALUE == 'old'`. That exit is expected repair evidence, not a failing candidate test. Fresh Linux source tests independently cover timestamp/unchecked-hash collisions for ordinary modules and package initializers, changed source before compilation, wrong loader identity and bootstrap ordering.

### F2 — explicit readback ownership: closed on current code and real-GPU evidence

`readback_input` allocates and adopts each NumPy destination, builds and adopts its CPU Warp view, checks both exact owner identities, and only then enters the explicit target-stream copy leaf. All observations remain retained across subsequent leaves. An S-ACK-supervised stream wait precedes the single shared CPU comparison implementation. The ordinary public validator still supplies its lazy materializer to that same comparison; no duplicate dtype/shape/content/hash policy or IPC callback was introduced.

Inspected the installed Warp 1.12.0 source directly: `types.py:3233-3237` aliases CPU NumPy storage with `copy=False`, retaining its array reference; `context.py:8708-8850` only allocates cross-device staging for noncontiguous branches, not the fixed contiguous arrays used here. This checks the actual installed dependency, not only a version label. The five fixed readback faults target storage adoption, view adoption, pre-copy, copy-submit-then-raise and pre-return failure; their fresh GPU results are recorded below.

## Additional inspected invariants

- Guardian creation precedes CUDA-capable imports; setup owners transfer before the next leaf; unknown-owner exceptions preserve capsules and cannot become ordinary release after synchronization.
- Frozen source execution, actual import ledger, exact prepared/readiness authority, same production math and command owners; no product import of sacrificial helpers.
- Final S quiescence precedes seal; pre-call launch latch and diagnostic return count remain separate; later pre-latch failure still accounts for prior submissions.
- Shared supervisor extraction retains deadline-before-ACK, bounded canonical frames, exact session/role/sequence checks, S-only disarm, terminal V/E deadlines, parent-loss kill and no restart.
- V guards render synchronization, copies into already-retained destinations and the final wait; inspection follows successful V. V/S failure does not retry E; journal failures hard-exit instead of ordinary unwinding.
- Independent CPU registry/BVH/executor/rays, exact integer fields and frozen float tolerances; three profiles/four shade variants, 31x13 asymmetric camera, multiple IDs/materials, hit/miss and shadow cases. Counter checks use fixed invariants and hand-derived small fixtures rather than inventing equality with GPU traversal output.
- Gate inspection: `gpu_preparation_enabled` remains false; `EXPECTED_NATIVE_RESULT_ACTIVATION.result_capable` remains false; base diff does not change product A/K, allowlist, dispatcher, result or baseline activation files.

## Fresh verification

SSH used `ssh -o BatchMode=yes -o ConnectTimeout=8 -i C:/Users/AG/.ssh/tileops_ed25519 ang.gao@192.168.80.101`. Commands use the clean remote candidate with `/data3/ang.gao/anaconda3/bin/conda run --no-capture-output -n robot_sim`; GPU commands additionally set `env CUDA_HOME=/data3/ang.gao/anaconda3/envs/robot_sim`.

Observed remote environment: Python 3.12.13, Torch 2.9.1+cu128, Warp 1.12.0, NVCC 12.8.93. Torch reports ordinal 0 NVIDIA H200 NVL, 1 and 2 NVIDIA A100-SXM4-40GB; no assumption about `nvidia-smi` ordering is used. Installed Warp resolves under `/home/ang.gao/anaconda3/envs/robot_sim/lib/python3.12/site-packages/warp`. Local static/repro interpreter is Windows Python 3.14.

1. Local/remote `git status --porcelain=v1`, revision/branch checks; local `git diff --stat`, code-parent-to-HEAD diff, executable diff and `git diff --check 7de02e89c3e868e7554e5b1c385cfbbcf06570e1 HEAD`: clean, matching identities and no diff whitespace errors.
2. Local `python -m ruff check` and `python -m ruff format --check` over all 12 changed Python paths returned by `git diff --name-only BASE HEAD -- '*.py'`: all checks passed, 12 already formatted. No files formatted or edited.
3. Fresh independent Linux command:

   ```text
   python -m pytest -q -p no:cacheprovider tests/unit/benchmarks/test_native_input_observations.py tests/unit/optics/test_sacrificial_sources.py tests/unit/optics/test_sacrificial_lifecycle.py tests/unit/optics/test_sacrificial_watchdog.py tests/unit/optics/test_warp_render_command_contract.py tests/unit/optics/test_warp_fixed_shade_abi.py tests/unit/optics/test_warp_native_kernel_source.py tests/unit/benchmarks/test_native_readiness_authority.py tests/unit/benchmarks/test_closed_worker_protocol.py tests/unit/benchmarks/test_optical_closed_worker.py
   ```

   **181 passed in 11.99s**, no skips. This includes actual POSIX S/V/E tests and clean-tree C0 integration; it is not inherited v3/author output.
4. `python -I -S C:/Users/AG/AppData/Local/Temp/c4-review-v2-cache-fixture-20260928/check.py`: expected exit 1 with current-source execution and intentional old-value assertion failure, detailed above.

5. After the author released the GPUs, a separate reviewer invocation ran:

   ```text
   python -m pytest -q -p no:cacheprovider tests/gpu/test_warp_sacrificial_render.py::test_recorded_camera_matches_independent_cpu_oracle tests/gpu/test_warp_sacrificial_render.py::test_recorded_render_targets_second_real_gpu_and_restores_ambient tests/gpu/test_warp_sacrificial_render.py::test_every_command_partial_submission_lifetime tests/gpu/test_warp_sacrificial_render.py::test_input_readback_resources_remain_in_guardian_across_failures 'tests/gpu/test_warp_sacrificial_render.py::test_setup_and_drain_failures_have_process_terminal_disposition[validation_raise]' 'tests/gpu/test_warp_sacrificial_render.py::test_setup_and_drain_failures_have_process_terminal_disposition[validation_hang]' tests/gpu/test_warp_shade_microfixtures.py tests/gpu/test_warp_render_recording.py tests/gpu/test_warp_render_buffers.py tests/gpu/test_optical_gpu_runtime.py
   ```

   **165 passed in 351.92s (0:05:51)**, exit 0, no skips. This selection independently exercises 12 camera/profile/shadow/diagnostic CPU-oracle cases, one real two-device case, every pre-latch/post-latch/submit-then-raise position (15 cases), all five repaired input-readback fault boundaries, validation-drain raise/hang, and the direct-kernel hand-derived fixture, plus recording/buffer/compatibility runtime regressions. The real two-device test executed, not skipped. These passes close F2's original real-device evidence gap for the corrected explicit readback path; they do not close F3/F4/F5.

The complete 79-case new GPU matrix was not independently repeated in this round. Once the three missing acceptance requirements were established, the coordinator explicitly allowed risk-proportional independent GPU selection for a Request-changes verdict. The author's separate full matrix output is not counted as independent evidence. In particular, the real-GPU setup/import/compile/record/warmup faults, missing/wrong ACK cases, parent loss, journal failure and resident/empty recorded cases outside this selection are not claimed as newly rerun here. Their source was inspected; POSIX CPU supervision was freshly executed, and the selected real-GPU cases above prove only their own covered paths. Revised-candidate review must execute the complete required matrix and the new acceptance tests.

The existing runtime regression triggered actual NVCC/Ninja compilation of `robot_sim_cuda_direct_light_v8` under `/data3/ang.gao/.cache/torch_extensions_robot_sim_cuda128`; read-only `ps` inspection confirmed compiler activity during the quiet interval. Normal Warp/Torch caches were neither purged nor treated as previous test output. This is not a cold-cache performance or source-recompilation equivalence claim. No packages were installed.

Tooling notes: remote `rg` was absent, so installed-source searches used `grep`/`sed`; this was not a candidate failure. A local wildcard supplied as a literal `rg` pathname was rejected, and the relevant source/requirements were inspected using explicit paths. Combined long reads that truncated were narrowed; no conclusion relies on omitted output.

6. Final post-suite local and remote `git status --porcelain=v1` were empty; HEAD and tree remained exactly `55365c528cbdb5208804be9e46220e4bdf6a7335` / `aa11be6cebb9727165deafc28a1da713b1bd71ed`. No candidate file changed. Pytest may leave its normal external temporary manifest/journal/supervisor evidence and ignored Python/JIT caches; cacheprovider was disabled. Nothing was deleted. Fresh independent test total in this report is **346 passed** (181 CPU/POSIX plus 165 selected GPU/regression), with no skips in either invocation; it is not a full repository or full new GPU-matrix count.

## Limits and disposition

The review is not a general Python/native sandbox audit. It assumes the frozen trusted test program and installed runtime, not hostile in-process code, concurrent source rewriting or arbitrary allocator/process corruption. Ordinary dependency/JIT caches may be used where the contract permits them; old logs are not proof of new executions. No package installation, push, merge, product gate change, baseline/result artifact or candidate edit was performed. Reviewer report lives outside the frozen candidate.

Fix F3/F4/F5 within the retained contract, freeze the new candidate and request a fresh independent re-review. Passing existing CPU/GPU tests does not replace these absent assertions. Full C4c1c/C4c1/C4 acceptance remains unapproved. C4c2–C4c4, C4d/B3, product GPU preparation, a3b, result/baseline access, host payload/shipping and backend selection remain closed. This report does not revoke earlier narrowly bound checkpoints or extend them.
