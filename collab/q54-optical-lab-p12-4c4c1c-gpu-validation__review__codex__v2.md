# C4c1c fixed sacrificial GPU validation — independent code review

Initiative: q54-optical-lab-p12-4c4c1c-gpu-validation
Stage: review
Author: codex (independent reviewer)
Version: v2
Date: 2026-09-28
Status: in_review
Related Files: C4c1 ABI design requests v2/v3; supervision review v3; recording review v2; GPU-validation implementation note v1
Owner Summary: Request changes. Two code findings remain: source digests do not bind executed cached Python code, and input validation bypasses incremental readback-owner adoption. Required final Linux/GPU evidence is independently blocked by both SSH transports timing out. No product or later-stage gate is approved.

## Identity, scope and verdict

- Reviewer session: `/root/c4_sacrificial_gpu_review_v2`, a fresh independent reviewer that did not author this candidate. No subagents were used. The prior v1 session produced no completed verdict; its reported test results were not inherited as evidence.
- Candidate: `C:/Users/AG/新建文件夹/p12_4_c4_sacrificial_gpu`.
- Base: `7de02e89c3e868e7554e5b1c385cfbbcf06570e1`.
- HEAD: `f2829d26233892bf9a17ead4484f48457477af37`.
- Tree: `1f7f0c7ee174479b3e22da2a1c55be853e8c11ae`.
- Branch: `implementation/c4-sacrificial-gpu`.
- `git status --porcelain` was empty at entry and after verification. No candidate code, tests, documents or tracked evidence were changed. Tests may have created normal ignored Python bytecode; pytest cacheprovider was disabled in reviewer test commands. Scratch files and this report are outside the candidate.
- Actual base-to-HEAD scope: 10 files, 1,844 insertions / 77 deletions. One implementation note, test support/fixtures/tests, the new fixed Warp supervisor, and extraction of the existing CPU supervision event loop. No production implementation, product allowlist or gate changes.

**Verdict: Request changes.** F1 and F2 below are code findings, not consequences of unavailable servers. Separately, mandatory Linux/POSIX and real-GPU validation is incomplete. Correcting these findings alone would not establish full C4c1c acceptance without fresh required runtime evidence on the revised frozen candidate.

Read `AGENTS.md`, `collab/README.md`, and `collab/REVIEW_PRINCIPLES.md` completely; reread the Chinese principles with explicit UTF-8 after the default PowerShell decoding was garbled. Read the full v2/v3 design requirements, the supervision v3 and recording v2 reports, and implementation note v1. Earlier reports and author outputs were treated as historical claims, not new execution. The parent supplied an adjacent-audit pointer to `input_validation`; the finding below follows independent inspection of that call chain and the versioned Warp implementation.

## Findings

### F1 — P2 / blocking: validated source bytes are not necessarily the Python code executed

Locations: `tests/support/sacrificial_sources.py:145-149` (`RepositoryImportLedger.find_spec`), `tests/support/sacrificial_warp_child.py:30`, and the supervisor/child bootstrap imports that precede ledger installation.

The finder asks `PathFinder` for a normal module spec, hashes its `.py` origin, and returns the same spec/loader. The standard source loader may execute an existing valid `.pyc` instead of compiling the bytes that were hashed. `sys.dont_write_bytecode = True` suppresses cache writes, not cache reads. The post-import ledger rechecks `__file__`, which still names the accepted source, so it cannot detect the difference. The supervisor likewise imports its helpers before enforcing its source manifest.

Trigger: a normal stale timestamp cache whose source size and integer-second modification time match the current file. Equal-length edits within the timestamp resolution, or restored file timestamps, suffice; no hostile plugin, arbitrary-code input, native bypass or mutation of the frozen checkout is necessary. The frozen manifest can identify current source while execution uses a different historical helper/kernel/lifecycle body. Consequently the source closure and guardian-before-import evidence cannot certify the revision actually executed.

Independent harmless reproduction used the candidate's unchanged `RepositoryImportLedger` class loaded directly from its source path, and narrowed only its source inventory to one external fixture module. The module contains a single fixed string assignment; it performs no I/O or other side effects. Compile `VALUE = 'old'`, then replace it with equal-length `VALUE = 'new'` while retaining the old timestamp, construct the manifest from the new bytes, install the ledger and import. On Python 3.14 with `-I -S` and `dont_write_bytecode=True`, output was:

```text
SOURCE: VALUE = 'new'
EXECUTED_VALUE: old
VERIFIED_LEDGER: ('cache_probe.py',)
```

The reproduction files remain at `C:/Users/AG/AppData/Local/Temp/c4-review-v2-cache-fixture-20260928/{check.py,cache_probe.py,__pycache__/...}`. This is a focused test of the actual finder, not a claim that the full GPU child was executed or that its existing cache was stale. The candidate itself was not modified or furnished with a foreign cache.

Why existing checks miss it: tests mutate manifest entries and reject unlisted modules, but never distinguish source bytes from code returned by the import loader. AST ordering checks examine current source only. Clean Git status does not establish cache consistency.

Direction: make repository execution consume the same verified source bytes (including bootstrap support code), or establish an equally strong cache-validation policy that cannot accept timestamp-colliding or unchecked stale bytecode. Merely adding `-B`, deleting a cache once during review, or rehashing `__file__` does not close the contract.

Required proof: an external fixed old/new cache fixture must execute the verified current body or reject before executing the stale body; cover bootstrap and later imports, package initializers, and timestamp/unchecked-hash cache cases as applicable. Preserve the fixed trusted-test-code boundary; this does not request a general Python sandbox.

### F2 — P2 / blocking: input-validation readback owners bypass the setup ownership protocol

Locations: `tests/support/sacrificial_warp_case.py:126-141` and `:278-282`; downstream `benchmarks/optical/native.py:1085-1098` and `:1326-1329`.

`build_case()` wraps `validate_prepared_native_input(source, prepared)` in one `setup_wait("input_validation", ...)`. That wrapper declares a non-owner-producing `event_wait` leaf and transfers no returned owners. The validator loops through every input array, calling `_native_array_to_numpy`, whose Warp branch calls `value.numpy()`. Only the final validation report is adopted afterwards.

Warp 1.12.0's actual implementation makes this more than a wait: `array.numpy()` enters the device null stream, calls `to("cpu")`, and converts the returned allocation to a NumPy view. `to()` calls `clone()` for a device change; `clone()` allocates with `empty_like()` and then calls `warp.copy()` before returning. Thus a readback destination exists before another GPU-capable leaf without any guardian transfer. Successive validator iterations replace their temporary host results; those owners never join the graph. See the official versioned [Warp array implementation](https://raw.githubusercontent.com/NVIDIA/warp/v1.12.0/warp/_src/types.py) (`numpy`/`to`) and [Warp clone implementation](https://raw.githubusercontent.com/NVIDIA/warp/v1.12.0/warp/_src/context.py) (`clone`). These are upstream v1.12.0 sources, not an independent observation of an installed remote runtime.

Trigger: the normal nonempty input-validation path already exercises the unregistered allocation/copy sequence. An exception within allocation/copy or between successive materializations also has no explicit partial-owner transfer and the descriptor does not identify the helper as owner-producing.

Consequence: the claimed register-before-next-GPU-leaf invariant and complete setup-owner graph are false for this admitted helper. Design v3 section 3.2 explicitly rejects relying on locals/module state as an ownership protocol. The S ACK does bound the call, but deadline supervision is not owner adoption. Importantly, the existing `setup_wait` exception path quarantines and hard-exits; I did **not** demonstrate a GPU use-after-free, an ordinary-release authorization on this exception path, or failure of the supervisor to kill a hung child. This finding is the concrete ownership/proof gap, not an unsupported memory-corruption claim.

Why existing checks miss it: `watch_owner` observes only explicit `wp.array` owners returned through selected driver leaves. It never sees the allocations hidden in `.numpy()`. `assert_owner_retention` checks only that same registered list. The fault matrix injects upload/allocation/compile/record/warmup faults, not failures at these readback allocation/copy/transfer boundaries. The report-only return makes the outer leaf appear owner-free.

Direction: route materialization through a scoped, incrementally adopting builder, preferably with retained explicit host destinations/views and controlled copies, while retaining the production validation semantics as the single source of truth. If a helper cannot expose all partial owners, classify that fact explicitly as unknown ownership and retain reachable capsules until process-terminal disposition; do not describe the helper as a no-owner wait. Avoid duplicating native-input validation rules in a new test-only implementation.

Required proof: observe actual readback destination owners and their exact guardian custody before each copy/next GPU leaf. Add safe fixed fault cases around allocation, copy and transfer, including partial-return failure. Show no covered finalizer before the appropriate supervised disposition and no upgrade from unknown ownership to ordinary release. Rerun actual input validation and the full GPU setup/failure matrix on the repaired frozen revision.

## Inspected control/data flow and satisfactory boundaries

- Shared `supervise_child` is a direct extraction of the previously reviewed CPU event loop, with configurable startup timeout and fault role. Monotonic supervisor deadlines are committed by `SupervisorGuardState.accept` before ACK writes. Exact session/role/integer-sequence checks, S-only disarm and V/E terminal arming remain present. Parent-liveness loss kills the one-shot child. No new caller-selected executable, module or callback crosses IPC.
- The child constructs the CPU guardian before exposing site-packages/importing its GPU case builder in current source. Initial source hashes and later import admission cover a large explicit repository closure. F1 limits what this proves about actual executed code.
- Explicit upload/array/event/stream/record leaves transfer their returned owners before the next driver leaf. Owner-producing exceptions use partial exception capsules and unknown-owner disposition. Recording intercepts `wp.launch` only during sealing and restores the original function in `finally`; returned raw commands are adopted. F2 identifies a helper not covered by this pattern.
- The final S barrier precedes guardian seal. `before_launch(index)` executes before each fixed recorded launch, and only a normal later path increments `launch_returned`. Submit-then-raise faults preserve attempted-versus-confirmed distinction. Prior submissions make later pre-latch failures post-submit cases.
- V is armed before the validation callback; the callback first synchronizes the target device, copies into preallocated/retained host destinations and synchronizes again. Results are compared only after the guarded callback completes. V failure quarantines without E retry; eligible command failures use E; journal encoding/write failure hard-exits.
- CPU truth is constructed from the canonical registry, independent CPU BVH/executor and host pinhole rays, not another invocation of migrated Warp mathematics. Checks include physical rays, geometry, IDs/material indices, RGB/intensity, exact integers and fixed float tolerances. Three profiles and all four shadow/diagnostic combinations are collected. The 31x13 off-center, unequal-focal, rotated camera is asserted to cover all 21 hit IDs/materials; that actual GPU coverage is not yet verified here.
- Empty/resident-ray cases exercise 2/4-command dispatch and reset paths; camera cases use five commands. Counter checks include dtype/shape, nonnegativity, stack bounds, shadow decomposition and upper bounds. The new hand-derived microfixture invokes production kernels directly, not the recorded bundle, so its ordinary pytest execution does not violate the specifically recorded-bundle containment rule.
- Source searches found no new sacrificial imports under `optics`, `optical_worker` or `benchmarks`. The product A/K protocol and GPU/result activation files are unchanged. No performance or backend-selection evidence is produced.

These observations are static review plus the CPU checks below, not proof of real native resource behavior. In particular I did not certify dual-device ambient restoration, actual launch-owner finalization, input readback failure safety or hung real-GPU drain behavior from mocks or Windows skips.

## Fresh commands and observed results

Working directory was the candidate unless stated otherwise. Local Python: **3.14.0**, Windows PowerShell. No local Warp installation was found (`importlib.util.find_spec('warp')` returned `None`).

1. `git status --short`, `git status --porcelain`, `git rev-parse HEAD`, `git rev-parse 'HEAD^{tree}'`, `git branch --show-current`, and `git diff --stat 7de02e89c3e868e7554e5b1c385cfbbcf06570e1 HEAD`: clean tree and identities/scope above.
2. `git diff --check 7de02e89c3e868e7554e5b1c385cfbbcf06570e1 HEAD`: passed, no diagnostics.
3. `python -m ruff check` and `python -m ruff format --check`, each on these exact nine changed Python files:

   ```text
   tests/support/sacrificial_sources.py
   tests/support/sacrificial_supervisor.py
   tests/support/sacrificial_warp_case.py
   tests/support/sacrificial_warp_child.py
   tests/unit/optics/test_sacrificial_sources.py
   tests/gpu/test_warp_sacrificial_render.py
   tests/gpu/test_warp_shade_microfixtures.py
   tools/optical_sacrificial_warp_supervisor.py
   tools/optical_sacrificial_test_supervisor.py
   ```

   Results: **All checks passed**; **9 files already formatted**. No formatting writes.
4. `python -m pytest -q -p no:cacheprovider tests/unit/optics/test_sacrificial_sources.py tests/unit/optics/test_sacrificial_lifecycle.py tests/unit/optics/test_sacrificial_watchdog.py`: **41 passed, 21 skipped in 1.89s**. The skips are POSIX supervision tests, not successful runtime supervision evidence.
5. `python -m pytest -q -p no:cacheprovider tests/unit/optics/test_warp_render_command_contract.py tests/unit/optics/test_warp_fixed_shade_abi.py tests/unit/optics/test_warp_native_kernel_source.py`: **74 passed in 2.85s**.
6. `python -m pytest --collect-only -q -p no:cacheprovider tests/gpu/test_warp_sacrificial_render.py tests/gpu/test_warp_shade_microfixtures.py`: **74 tests collected in 0.32s**. Collection is not execution.
7. F1 fixture: created the external directory with `New-Item`; authored only the fixed source and check script using `apply_patch`. `python -I -S -c "import os,py_compile; p=r'C:\Users\AG\AppData\Local\Temp\c4-review-v2-cache-fixture-20260928\cache_probe.py'; os.utime(p,(1700000000,1700000000)); py_compile.compile(p,doraise=True)"` compiled the old constant. After the equal-length `apply_patch` change, `python -I -S -c "import os; os.utime(r'C:\Users\AG\AppData\Local\Temp\c4-review-v2-cache-fixture-20260928\cache_probe.py',(1700000000,1700000000))"` restored its timestamp. `python -I -S C:/Users/AG/AppData/Local/Temp/c4-review-v2-cache-fixture-20260928/check.py` exited 0 with the source-new/executed-old/ledger-accepted observations above.
8. `ssh -o BatchMode=yes -o ConnectTimeout=5 tileops-dev true`: timed out connecting to **183.166.183.33:40031**. Independently retried authorized fallback: `ssh -o BatchMode=yes -o ConnectTimeout=5 -i C:/Users/AG/.ssh/tileops_ed25519 ang.gao@192.168.80.101 true`: timed out connecting to **192.168.80.101:22**. No remote command ran and no remote repository was substituted.
9. Read-only dependency verification through the official NVIDIA Warp **v1.12.0** source links in F2. No package was installed and no GPU execution was attempted locally.

Reviewer tooling limitations, not candidate failures: the first workspace-root `AGENTS.md` lookup found no such file, then the candidate's instructions were read. Some combined reads exceeded output limits; relevant omitted sections were reread. A broad read-only filename search for local Warp sources produced truncated/noisy output and was narrowed; it is not validation evidence. All substantive findings above use specific source/command results.

## Required runtime evidence still missing

Both permitted SSH routes were independently unavailable. The final candidate has not been run here in the required Linux `robot_sim` environment (packet target: Python 3.12, Warp 1.12, Torch 2.9.1+cu128, NVCC 12.8). The dirty 101 development draft and earlier author counts were not used as substitute proof.

After repair, on an independently checked clean frozen revision, execute at minimum:

```sh
python -m pytest tests/gpu/test_warp_sacrificial_render.py tests/gpu/test_warp_shade_microfixtures.py tests/unit/optics/test_sacrificial_sources.py -q
python -m pytest tests/unit/optics/test_sacrificial_lifecycle.py tests/unit/optics/test_sacrificial_watchdog.py tests/unit/benchmarks/test_closed_worker_protocol.py tests/unit/benchmarks/test_optical_closed_worker.py -q
```

Use `/data3/ang.gao/anaconda3/bin/conda run --no-capture-output -n robot_sim` as specified in the packet, and freshly run affected recording/ABI/readiness regressions, including the exact source-cache and ownership-repair regressions. Required dual-device evidence cannot pass via a skip. The 31x13 correctness matrix, 56 microfixture executions within its collected test, actual S/V/E hangs/wrong ACKs, parent-loss cases, journal failures and clean C4c0 regressions all remain runtime obligations. No result of those executions is claimed by this report.

## Disposition and closed scope

Fix F1/F2 within their stated contract, freeze a new candidate and obtain fresh independent re-review plus the required real-environment evidence. Do not relax source identity, ownership or GPU acceptance requirements to obtain approval.

This report approves no new checkpoint. Earlier approvals remain bound to their own revisions and scopes. Product GPU preparation, full C4c1 acceptance, C4c2+, C4d/B3, host payload, a3b, result/baseline access, shipping and backend selection remain closed. No push, merge, baseline/result artifact, dependency installation or candidate modification was performed. The only created artifacts are the harmless external cache fixture and this report; none were deleted.
