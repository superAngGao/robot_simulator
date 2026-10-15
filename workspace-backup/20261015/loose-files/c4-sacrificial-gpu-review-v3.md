# C4c1c fixed sacrificial GPU validation — independent re-review

Initiative: q54-optical-lab-p12-4c4c1c-gpu-validation
Stage: review
Author: codex (independent reviewer)
Version: v3
Date: 2026-09-28
Status: in_review
Related Files: C4c1 ABI design requests v2/v3; supervision review v3; recording review v2; GPU-validation review v2 and review-followup v3
Owner Summary: Blocked. The prior F1/F2 code defects are addressed in the inspected revision and no additional substantive code finding is raised. Required clean Linux/POSIX and real-GPU evidence remains unavailable; both authorized SSH routes independently timed out. This is not C4c1c acceptance or approval of any product/later-stage gate.

## Identity and scope

- Fresh independent reviewer session: `/root/c4_sacrificial_gpu_review_v3`. The reviewer did not author this candidate or inherit the implementation conversation. No subagents were used.
- Candidate: `C:/Users/AG/新建文件夹/p12_4_c4_sacrificial_gpu`.
- Original checkpoint base: `7de02e89c3e868e7554e5b1c385cfbbcf06570e1`.
- Previously reviewed HEAD: `f2829d26233892bf9a17ead4484f48457477af37`.
- Reviewed HEAD: `9db8146f9c818dd580ce73d886f7afe4c5d22d73`.
- Reviewed tree: `2b68fcb160fae4c0bd34062aa1792b591573142d`.
- Branch: `implementation/c4-sacrificial-gpu`.
- `git status --porcelain=v1` was empty on entry and after tests/inspection. No candidate source, tests, documents, or tracked evidence were changed. Pytest cacheprovider was disabled; normal ignored Python/Ruff caches may have been created.
- Original-base diff: 16 files, 2,378 insertions / 80 deletions. Prior-reviewed-to-current diff: 12 files, 540 insertions / 9 deletions. The latter includes preserved review/session records and the follow-up note, not only executable fixes.
- Actual production-code delta is the shared CPU observation-comparison extraction in `benchmarks/optical/native.py` and one setup descriptor kind in `optical_worker/lifecycle.py`. Remaining executable changes are tests, fixed support code and standalone test supervisors. The product protocol, allowlist and activation source are unchanged.

Read `AGENTS.md`, `collab/README.md` and `collab/REVIEW_PRINCIPLES.md` completely. The Chinese principles were reread with UTF-8 after the default PowerShell decoding was garbled. Read the v2/v3 requirements, supervision v3, recording v2, prior complete GPU-validation v2 report, implementation note v1 and follow-up v3. Author results and prior approvals were treated as historical claims, not execution evidence for this revision.

Design SHA-256 values observed locally:

- v2: `106ad2380480acd18e4733ea23a98735df837ae1e79d797c9001c8a641011e30`.
- v3: `291d4248633a9398eb4466b5e3a0ec514e04b810c5f6575886a4fab2804e1626`.

## Verdict: Blocked

No new substantive code finding is raised. F1 and F2 are closed as code defects on the evidence described below. This does not certify the repaired path's real-device behavior: its mandatory Linux/POSIX/GPU validation remains an explicit blocking evidence gap. No new checkpoint is approved by this report.

The unavailable hosts are an environment failure, not a demonstrated candidate code failure. Local CPU tests, source inspection and GPU collection cannot satisfy the device ownership, numerical or real watchdog requirements.

## Prior findings and independent verification

### F1 — verified source versus stale bytecode: code defect closed

Locations: `tests/support/sacrificial_sources.py:145-194`, `tests/support/sacrificial_warp_child.py:30-35`, `tools/optical_sacrificial_warp_supervisor.py:16-20`.

The previous finder validated the source origin but returned the normal cache-reading loader. The repaired finder replaces the admitted repository module loader. `VerifiedRepositorySourceLoader.get_code()` checks module identity, reads a byte buffer, verifies that buffer against the manifest digest, and compiles that same buffer. It does not invoke `SourceFileLoader.get_code()` and therefore does not consume a timestamp or unchecked-hash bytecode body. Normal package specs retain their package search locations while using the verified source loader.

Before their first repository helper import, both fixed entry scripts create a new cache namespace using `mkdir(exist_ok=False)`, set `sys.pycache_prefix`, and disable bytecode writes. An existing namespace causes failure, not fallback to a prior cache. This covers the ledger/lifecycle helpers needed before ledger installation. The scripts themselves are passed as source-file entrypoints to isolated, no-site Python. This reasoning is within the documented frozen, trusted-test-code boundary, not protection from a hostile interpreter, concurrent source rewriting or arbitrary in-process hooks.

Fresh execution of the unchanged original external reproduction:

```text
SOURCE: VALUE = 'new'
EXECUTED_VALUE: new
VERIFIED_LEDGER: ('cache_probe.py',)
AssertionError at assert cache_probe.VALUE == 'old'
```

The standalone command exits 1 because the old reproduction deliberately asserts the stale value. It demonstrates repaired current-source execution, not a candidate test failure. The original fixture and check script were not edited.

An additional reviewer-authored, fixed constant-only probe uses that same external old/new fixture. Ordinary import still executes `old`; deleting only its module-table entry and selecting a newly created empty prefix executes `new`, leaving the prefix empty with cache writes disabled. It exits 0. This independently confirms that the original stale cache still exists and that the bootstrap cache policy defeats it on the local interpreter; it is not a claim to have run the actual POSIX supervisor on Windows.

The committed tests freshly passed for ordinary modules and package initializers under both timestamp-colliding and unchecked-hash stale caches, source changes between loader construction and compilation, wrong loader module identity, and AST cache-before-repository-import ordering. Full actual Linux bootstrap/import-ledger execution remains required with the GPU matrix.

### F2 — implicit input readback ownership: code defect closed; runtime proof pending

Locations: `tests/support/sacrificial_warp_case.py:127-158`, `:313-318`; `benchmarks/optical/native.py:1071-1138`; `optical_worker/lifecycle.py:63`.

Traced the original `setup_wait -> validate_prepared_native_input -> _native_array_to_numpy -> Warp .numpy()` path and the replacement. The fixed driver no longer invokes the implicit materializer for input validation. For each manifest array it now:

1. allocates its NumPy host destination and adopts the exact object;
2. creates the explicit CPU Warp view and adopts that exact object;
3. checks both identities in guardian custody before the copy leaf;
4. declares the target-device copy leaf and copies on the frozen stream;
5. retains every destination/view while subsequent copies are submitted;
6. obtains the S ACK before stream synchronization; and
7. compares only after that wait returns.

The fixed manifest uses the matching scalar dtype and C-contiguous allocations. Upstream Warp 1.12's CPU `copy=False`, non-pinned array path aliases NumPy storage and retains its array reference, rather than allocating a device/readback destination. Its contiguous cross-device copy path uses the supplied buffers; the temporary staging allocations are in noncontiguous branches not used here. Inspected the official versioned [array implementation](https://raw.githubusercontent.com/NVIDIA/warp/v1.12.0/warp/_src/types.py) and [copy implementation](https://raw.githubusercontent.com/NVIDIA/warp/v1.12.0/warp/_src/context.py). These are source-level dependency checks, not inspection of an installed remote runtime.

Readback storage/view objects are now covered by the weakref observers as well as the guardian's strong references. Copy-submit-then-raise is handled by the setup leaf failure path, with already-adopted destinations retained while the target-only failure takes supervised E. Failure after copy and before host return also retains the destination independently of the observation tuple. An S arming/wait failure quarantines and hard-exits without a second sync. Existing owner-producing GPU leaf failures still set unknown ownership, retain reachable exception/partial-owner capsules and prohibit upgrading to ordinary release.

The new five-case GPU matrix checks failure after host storage adoption, after CPU view adoption, before copy, after copy submission, and before returning the host observation. Its code requires retained records, no recorded launch or inspection, one emergency sync and no premature covered finalizer. These tests were collected but **not executed** here; no claim about actual Warp finalization or successful device drain follows from static inspection.

The production comparison is not duplicated. Both the public ordinary validator's lazy materialization iterator and the fixed driver's ready observations call `_validate_native_input_host_observations`. Manifest identity, array-name set, ordered names, dtype, shape, C-contiguity, equal-NaN content equality, content hash and report construction remain in that one loop. Strict zip rejects missing/extra observations. The public wrapper retains its host-materialization instrumentation restriction; the private helper performs CPU comparison and does not itself establish readback or readiness authority. Exact storage/readiness registration is unchanged, and no observation or callback parameter was added to IPC.

Eight new CPU tests freshly verify ordinary/explicit report equality, no implicit materializer invocation by the explicit path, and missing/extra/name/value/dtype/shape/stride rejection. The one `device_to_host_copy` descriptor addition is shared by validation/replay of the existing CPU lifecycle contract; it introduces no product request or activation switch.

## Original flow and affected invariants inspected

- Inspected the original-base file surface, the original supervision-loop extraction, and the complete repair executable diff. The shared supervisor retains monotonic deadline commitment before ACK, bounded frames, session/role/sequence validation, S-only disarm, V/E terminal deadlines, parent-loss kill and no restart. No caller-selected executable/module/callback is introduced.
- Traced guardian-before-CUDA-import, target and stream owners, individual uploads, ready event, explicit input readback, compile/readiness/scene construction, recording interception and restoration, and adoption of raw returned command owners. Unknown-owner failures remain process-terminal; no repair clears that state.
- Checked setup quiescence before seal, per-command pre-call latch and separate return count, partial-submit failure disposition, preallocated output readback storage, V-before-drain, no V-to-E retry, inspection only after the guarded drain, and hard exit on journal encoding/write failure.
- Checked the CPU oracle construction from canonical registry, independent CPU BVH/executor and host camera rays; direct field comparisons, exact integer checks and fixed tolerances; three profiles/four shadow-diagnostic combinations; 31x13 rotated/off-center/unequal-focal camera; and 21-ID/material coverage assertions. These assertions have not been proven against GPU output on this revision.
- Checked resident/empty command-count and reset cases, counter dtype/shape/nonnegative/stack/decomposition/bound rules, and hand-derived shade microfixtures. The microfixture calls production kernels directly, not the recorded bundle; the recorded bundle remains confined to the fixed child.
- Product searches found no sacrificial support imports under `optics`, `optical_worker` or `benchmarks`. The closed-worker contract still has `gpu_preparation_enabled: false`; `EXPECTED_NATIVE_RESULT_ACTIVATION` still has `result_capable: False`. Gate/protocol/allowlist files have no base-to-HEAD delta. Sharing a CPU comparison helper is not host shipping or product GPU preparation.

## Fresh commands and results

Working directory was the candidate unless stated otherwise. Observed local environment: Windows PowerShell, Python **3.14.0**, no importable Warp or Torch (`find_spec` returned `None` for both). No dependencies were installed.

1. `git status --porcelain=v1`, `git branch --show-current`, `git rev-parse HEAD 'HEAD^{tree}'`, and original-base/prior-review `git diff --stat` / executable diffs: clean identities and bounded scope recorded above. `git diff --check 7de02e89c3e868e7554e5b1c385cfbbcf06570e1 HEAD`: no diagnostics.
2. `python -m ruff check` and `python -m ruff format --check` over every changed Python path returned by `git diff --name-only 7de02e89c3e868e7554e5b1c385cfbbcf06570e1 HEAD -- '*.py'`: **All checks passed; 12 files already formatted**. The exact file list follows:

   ```text
   benchmarks/optical/native.py
   optical_worker/lifecycle.py
   tests/gpu/test_warp_sacrificial_render.py
   tests/gpu/test_warp_shade_microfixtures.py
   tests/support/sacrificial_sources.py
   tests/support/sacrificial_supervisor.py
   tests/support/sacrificial_warp_case.py
   tests/support/sacrificial_warp_child.py
   tests/unit/benchmarks/test_native_input_observations.py
   tests/unit/optics/test_sacrificial_sources.py
   tools/optical_sacrificial_test_supervisor.py
   tools/optical_sacrificial_warp_supervisor.py
   ```

3. Fresh focused CPU suite:

   ```text
   python -m pytest -q -p no:cacheprovider tests/unit/benchmarks/test_native_input_observations.py tests/unit/optics/test_sacrificial_sources.py tests/unit/optics/test_sacrificial_lifecycle.py tests/unit/optics/test_sacrificial_watchdog.py tests/unit/optics/test_warp_render_command_contract.py tests/unit/optics/test_warp_fixed_shade_abi.py tests/unit/optics/test_warp_native_kernel_source.py
   ```

   **130 passed, 21 skipped in 8.52s**. The skipped cases require POSIX supervision.
4. `python -m pytest -q -p no:cacheprovider tests/unit/benchmarks/test_native_readiness_authority.py tests/unit/benchmarks/test_closed_worker_protocol.py tests/unit/benchmarks/test_optical_closed_worker.py`: **22 passed, 8 skipped in 2.35s**. These platform skips do not establish actual closed-worker execution.
5. `python -m pytest --collect-only -q -p no:cacheprovider tests/gpu/test_warp_sacrificial_render.py tests/gpu/test_warp_shade_microfixtures.py`: **79 tests collected in 0.27s**. No GPU test execution is claimed.
6. `python -I -S C:/Users/AG/AppData/Local/Temp/c4-review-v2-cache-fixture-20260928/check.py`: expected exit **1**, with new-source/new-executed/accepted-ledger observations and the intentional old-value assertion failure above. Executed twice with the same observations; the final invocation was standalone so its exit status was explicit.
7. `python -I -S C:/Users/AG/AppData/Local/Temp/c4-review-v3-bootstrap-check.py`: exit **0**, ordinary cache `old`, fresh prefix `new`, no cache writes. The fixed probe was created with `apply_patch`, outside the candidate. It ran twice and created only empty namespaces under `C:/Users/AG/AppData/Local/Temp/c4-review-v3-bootstrap-6xjuqsel/child-source-cache` and `C:/Users/AG/AppData/Local/Temp/c4-review-v3-bootstrap-93i55dpa/child-source-cache`.
8. `ssh -o BatchMode=yes -o ConnectTimeout=5 tileops-dev true`: exit **1**, timeout at **183.166.183.33:40031**. `ssh -o BatchMode=yes -o ConnectTimeout=5 -i C:/Users/AG/.ssh/tileops_ed25519 ang.gao@192.168.80.101 true`: exit **1**, timeout at **192.168.80.101:22**. No remote command ran. No remote revision or environment was independently established in this round.
9. Read-only official Warp v1.12 source inspection, `rg` scope/entrypoint searches, design `Get-FileHash`, final Git status/identity checks: results described above.

Reviewer tooling limitations/errors were resolved without changing the candidate: the initial unquoted PowerShell `HEAD^{tree}` expression was misparsed and was rerun quoted; combined reads that truncated output were narrowed/reread; an initially guessed standalone activation JSON path did not exist, and the actual activation definition was located in `benchmarks/optical/schema.py`. These are not candidate failures. The full repository suite was not run; author-reported broader Torch-dependent failures were not relabeled or inherited as local evidence.

## Blocking evidence gap and required next verification

The packet's intended runtime is clean Linux/POSIX with `/data3/ang.gao/anaconda3/bin/conda run --no-capture-output -n robot_sim`, Python 3.12, Warp 1.12, Torch 2.9.1+cu128 and NVCC 12.8. These environment versions remain packet information, not current remote observations. The dirty `/data3/ang.gao/robot_sim_c4_sacrificial_gpu_work` draft is not this frozen candidate and was not used as substitute proof.

Once an authorized route is reachable, first independently establish the clean candidate revision/tree and environment. Then freshly execute at minimum:

```sh
python -m pytest tests/gpu/test_warp_sacrificial_render.py tests/gpu/test_warp_shade_microfixtures.py tests/unit/optics/test_sacrificial_sources.py tests/unit/benchmarks/test_native_input_observations.py -q
python -m pytest tests/unit/optics/test_sacrificial_lifecycle.py tests/unit/optics/test_sacrificial_watchdog.py tests/unit/benchmarks/test_closed_worker_protocol.py tests/unit/benchmarks/test_optical_closed_worker.py -q
```

Rerun affected recording/buffer/ABI/kernel/readiness regressions, including `tests/gpu/test_warp_render_recording.py`, `tests/gpu/test_warp_render_buffers.py`, `tests/gpu/test_optical_gpu_runtime.py`, the three focused ABI/kernel unit files listed above, and `tests/unit/benchmarks/test_native_readiness_authority.py`. Recheck exact source/import coverage and cache behavior in the real fixed bootstrap.

Outstanding required properties include the 31x13 CPU-oracle matrix, actual readback owner/view custody and all five new failure cases, unknown-owner quarantine/no early finalizer behavior, every partial-submit position, real S/V/E hangs and wrong/missing ACKs, parent-loss and journal-failure cases, 56 shade microfixture runs and repeated empty reset, clean C4c0 integration and actual two-device ambient restoration. A dual-device skip is not a pass. Required GPU proof must not be replaced by local mocks, AST tests, collection, Windows skips, old output or earlier dirty-draft results.

## Disposition and closed gates

The code-level repairs do not require further changes on the evidence available here, but the checkpoint cannot advance until the missing required runtime evidence is obtained and independently assessed. Do not weaken ownership, source identity or device acceptance requirements to bypass the environment blocker. Recording this report verbatim in a separate evidence-only commit does not extend its verdict.

Full C4/C4c1/C4c1c acceptance, C4c2–C4c4, C4d/B3, product GPU preparation, a3b, result/baseline access, host payload/shipping and backend selection remain closed. No push, merge, installation, product activation, baseline/result artifact or candidate code change was performed. The only reviewer-created files/directories are this external report and the harmless external bootstrap probe/empty cache namespaces; nothing was deleted.
