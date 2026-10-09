# Independent Warp module preparation code review v1

Initiative: Q54 / P12.4 / C4 remaining render integration
Stage: independent code review before integrated native acceptance
Author: Codex independent reviewer /root/warp_module_code_v1
Version: v1
Date: 2026-10-10 (Asia/Shanghai; review began 2026-10-09)
Status: in_review — Request changes
Related Files: warp-module-preparation design-request v2 and design-review v1/v2; runtime-integration design-request v3; optical_worker/native_bootstrap.py; optical_worker/warp_setup.py; source policy/catalog; selected Warp runtime-environment v4; changed unit/native support tests
Owner Summary: The captured-source, private-cache, guarded serial-load and actual-executable custody implementation has a coherent bounded path, and both author success runs have independently readable terminal artifacts. One code defect remains: the claimed frozen kernel ABI/entry facts retain mutable annotation objects and omit the kernel entry hash, so later admission accepts changes that the accepted design requires it to reject. Native fault/hang acceptance and whole C4 remain separate pending gates.

## Identity and verdict

Fresh independent session `/root/warp_module_code_v1` received the review packet without the implementation conversation. I did not author, modify or check out the candidate, and did not spawn additional agents. I read AGENTS.md, collab/README.md and collab/REVIEW_PRINCIPLES.md fully; the Chinese principles were reread with explicit UTF-8 after the initial PowerShell default decoding was unsuitable. I read the required parent sections 4, 5, 5.1, 5.2, 6 and 8, the complete supplement v2 and both design reviews.

- Repository: `C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design`.
- Branch: `implementation/c4-supervised-waits`.
- Exact base: `0356108a958f34c7d17064a92d2cc7a339b07d0b`.
- Exact candidate HEAD: `4400cfd3adc1eee09c5d1f71f2dc711f0d68a189`.
- Initial and final `git status --porcelain=v1` were empty; HEAD remained unchanged.
- Base-to-HEAD delta: 16 files, 1109 insertions and 30 deletions. `git diff --check 0356108..HEAD` passed.
- Supplement SHA256 independently matched `194b3c25130df77954f5b2de2b556da81009b98f94efb861848ae06566a41a5b`.
- Read-only SSH independently observed the same clean HEAD at `tileops-dev:/data2/shared/ang.gao/robot_sim_warp_modules_wip`.

**Verdict: Request changes.** F1 is a substantiated code finding in this bounded module-preparation implementation. This verdict does not convert the acknowledged, unexecuted native fault/hang matrix into another code defect or imply that compilation acceptance completes C4.

## F1 — P2, blocking: freeze ABI values and the actual kernel entry identity

Location: `optical_worker/warp_setup.py:214–224` (`_definition_facts`), `:268–289` (snapshot and record construction), and `:341–346` (`validate_module_set`).

The function/kernel snapshot stores `arg.type` itself in `(arg.label, arg.type)`. Installed Warp 1.12 array annotations are mutable objects: `_ArrayAnnotationBase` has writable `dtype` and `ndim` slots (`warp/_src/types.py:4637–4659`). Thus changing an existing annotation's dtype or dimension also changes the object reachable through the retained snapshot. Comparing the newly created tuple to the old tuple cannot detect this in-place change. The analogous `var.type` struct field snapshot has the same shallow-reference pattern.

Separately, the kernel snapshot omits `Kernel.hash` and the resulting mangled entry name. Installed `Kernel.get_mangled_name` (`context.py:882–893`) derives the entry from the kernel key and the first eight hex characters of this hash. Changing `kernel.hash` changes the entry that later lookup requests, while every currently recorded fact can remain equal.

The module-hash check does not repair either gap. Pinned `Module.get_module_hash` (`context.py:2611–2635`) returns the retained ModuleHasher for the block dimension, whose `get_module_hash` (`:2084–2085`) returns its already computed `module_hash`. Direct mutation of an annotation or kernel hash does not invalidate that cached value. The retained executable and artifact hashes therefore still compare equal.

Trigger and consequence: after successful compilation/issuance, mutate an existing array argument annotation's `dtype` or `ndim`, or a kernel's entry hash, then run normal module-set validation in an armed setup segment. The actual validator returns the registered records instead of rejecting ABI/entry drift. Later recording can consequently reach an incompatible declaration or different symbol lookup before discovering the mismatch. This review does not claim an executed incorrect GPU render; it establishes that the newly promised admission boundary accepts a prohibited change.

Independent reproduction is saved outside the candidate at:

`C:/Users/AG/新建文件夹/c4_assets/test_review_warp_abi_alias.py`

Command from the candidate working directory:

```text
C:/Python314/python.exe -B -m pytest -q -s -p no:cacheprovider C:/Users/AG/新建文件夹/c4_assets/test_review_warp_abi_alias.py
```

Final result: **4 passed in 2.56s**, exit 0. These are three counterexample observations and one negative control:

1. In-place array dtype drift is admitted.
2. In-place array ndim drift is admitted.
3. Kernel hash drift is admitted even though the actual pinned `get_mangled_name` returns a different entry.
4. Replacing the annotation with a different descriptor is rejected with `Warp executable or kernel ABI changed`.

The repro invokes production `definition_owners`, `_definition_facts`, `issue_module_set` and `validate_module_set`, with the actual guarded lifecycle and exact owner adoption/registry. It also calls the existing runtime-binding validation before the array counterexamples. It executes only AST-extracted CPU metadata definitions from pinned installed `_ArrayAnnotationBase`, `Module.get_module_hash`, `ModuleHasher.get_module_hash`, and `Kernel.get_mangled_name`. Dependency objects and artifact/cache checks are CPU stand-ins; no Warp import, native handle or GPU execution is claimed. The exact installed `types.py` copy is `c4_assets/warp_setup_source/review_types.py`, SHA256 `9224fb234ff61f4a1c3338b7171d545171ab051fb625679568ea80a6af8321bd`, independently matched to the remote file. This isolates the defective production comparison while preserving the actual dependency behaviors that make the counterexample possible.

Existing checks miss this because module/definition object identities, function code, labels, options, executable identity and artifacts do not change; the old snapshot aliases the changed annotation. The current unit suite does not exercise successful module-set validation followed by these mutations. Existing runtime-binding validation checks target/stream/input custody and cannot detect a kernel declaration change. Source-loader audit checks captured source and module identity, not the already constructed mutable annotation's value.

Repair direction: retain exact owners but additionally snapshot immutable ABI values for supported scalar/array/struct types, including array dtype identity/structure and ndim and the relevant nested struct layout. Freeze the actual kernel key/hash/mangled entry after Warp has initialized the hash, and recheck it before later admission. Simply retaining another reference or comparing a cached module hash is insufficient. Account for the fact that the current `before` snapshot precedes `get_module_hash(256)`, which initializes kernel hashes; moving/freezing the relevant entry facts must preserve that legitimate transition. Add production-boundary negative tests for the original in-place changes and entry mutation, alongside an unchanged successful set and same-session executable reuse. Re-run the bounded tests and have the revised exact candidate independently reviewed; real native acceptance remains required afterward.

## High-risk areas inspected

- Held-source linecache publication occurs before `super().exec_module`, so decorators/inspect can consume held UTF-8 source. Existing stale entries are replaced once; loader/audit checks entry identity, mtime=None, original origin and exact reconstructed byte content. The new tests reach production loading, exercise disk changes, and reject clearing/replacing/mutating lines or mtime.
- Cache selection is inside the existing runtime setup guard before `wp.init` and optical decorators. It rejects preinitialized Warp and unsupported original config, creates an exclusive `/tmp/robot-sim-warp-*` directory, retains the path before subsequent fallible configuration, checks resolved nonsymlink identity/emptiness/mode0700, and pins the complete listed global options. PCH is disabled, CUBIN selected, and the checked defaults retain serial NVRTC loading and existing numerical settings.
- Preparation performs actual-IDLE entry checks, then one captured optical import guard followed by three nonnested load guards. Inside each load segment it holds the Module, definitions and original execs mapping before `wp.load_module`; it retains the published actual executable before further adoption/checks. The preheld mapping covers a public-wrapper exception after publication. Unpublished native temporaries remain within the stated trusted dependency exception-safety boundary.
- Initial artifact facts use the pinned module/hash/sm90 recipe, reject preexisting module cache paths and symlink/path escape, compare generated source/binary/metadata bytes, and compare executable metadata. Later validation checks exact registered set/record/module/mapping/executable identities, mapping variant set, handle, cached module hash, options and artifact drift. F1 identifies the narrower failure in ABI/entry facts; these other checks are not dismissed.
- Aggregate issuance follows successful DISARMED guards, checks record identity and successful guard sequences, and adopts the aggregate. It does not issue readiness or production execution authority.
- `_retain_failure` separates failure retention from the default source-terminal path. `_fail_warp_preparation` preserves the active/failed/pending guard path; for an actually IDLE CPU failure it validates retained runtime custody, applies unknown/incomplete scope precedence, retains the original failure and delegates disposition to the existing lifecycle. Unit tests reach entry, aggregate, repeat, unknown/missing owner/binding drift, guarded import/compile and publication/adoption failures. The real native counterparts remain pending.
- The current native support driver explicitly calls `_validate_runtime_binding_record` before `validate_module_set`; I did not report absence of that duplicate check inside the latter as a code defect. Future recording integration must preserve this composition as well as source/external auditing.
- No product PREPARE, readiness, warmup, graph/arena, renderer implementation, benchmark/results, baseline or backend-selection path was opened by this delta.

## Compiler inputs and flags

I compared `_DEFAULT_CONFIG` and `_MODULE_OPTIONS` to the copied pinned Warp config/context source, and traced `load_module`/Module.load/Module._compile through `build_cuda` to native `wp_cuda_compile_program`. The selected path is the existing in-process NVRTC compiler, with no new subprocess/service/raw repository runtime.core call. Public loading is explicit-module, target-bound, recursive=False, block_dim256 and max_workers0.

The global/module contract selects release, default optimization3, fast_math=False, fuse_fp=True, lineinfo=False, backward enabled, llvm_cuda=False, use_precompiled_headers=False and native sm90 CUBIN. Native `warp.cu:3753–3993` builds the include-path and sm90 architecture flags, C++17, NDEBUG, WP_VERIFY_FP undefinition, fmad=true, device-as-default-execution-space, extra-device-vectorization, restrict and its fixed diagnostic suppressions. The compiled toolchain's conditional Ofast-compile default is 0 where supported. No PCH, fast-math, time trace or extra external include directory is requested by this route.

The runtime profile has 1057 file rows versus v3's 1010: the 47 added rows are explicit native headers. All 47 SHA256 values independently matched the installed-source copies. Generated CUDA includes `builtin.h` with WP_NO_CRT. I followed its local include closure, including CRT/cuda_crt, array/fabric, math/intersection, mesh/BVH/tile, and volume's PNanoVDB path. The NVRTC conditionals avoid the host-only CUDA/standard-library includes: for example tile.h avoids cuda_runtime.h under __CUDACC_RTC__, and PNanoVDB suppresses stdint/string/math includes under that macro. The additional NanoVDB C++ headers in the literal inventory are not evidence that their unused host-oriented transitive closures execute in this path.

The profile already pins `warp/bin/warp.so` and `warp-clang.so`. Read-only remote SHA256 matched `4afdc3ddd8d4c7e4f68837e9b1f4268e767ef8527769321b152dde8cc3b11acd` and `f8e0f74720067ee5606f189463d55142928087bc45b59d0e3e5c2b2385425a3a`, respectively. `context.py` selects those exact binaries. `readelf -d warp.so` showed only libpthread, libdl, librt, libm, libc and the ELF loader as NEEDED, with ORIGIN RUNPATH, consistent with bundled compiler support. A dynamic-symbol search for NVRTC/nvJitLink names returned no matches (exit 1), so that search is not claimed as direct exported-symbol proof. Native cuda_util.cpp explicitly selects libcuda.so with libcuda.so.1 fallback. These pinned-source/binary observations are bounded static support evidence, not an exhaustive dynamic-library mapping observation or a new driver/toolchain attestation claim.

## Executed verification and native evidence limits

Executed the requested four suites with local CPython 3.14, no pytest cache provider and scratch basetemp outside the candidate:

```text
C:/Python314/python.exe -B -m pytest -q -p no:cacheprovider tests/unit/optics/test_warp_module_preparation.py tests/unit/optics/test_guarded_runtime_binding.py tests/unit/optics/test_native_bootstrap.py tests/unit/optics/test_native_source_custody.py --basetemp=C:/Users/AG/新建文件夹/c4_assets/review_warp_modules_v1_pytest
```

Observed **152 passed, 9 skipped in 275.10s**, exit 0. Unified exec session 31478 reached terminal completion. These Windows skips cover unavailable platform/symlink behavior and do not satisfy mandatory POSIX/native cases. Bootstrap tests create their own copied releases under the outside-candidate temp root; intentional source corruption and bytecode tests act on those copies. This suite is independent execution, unlike the author-reported broader test totals, which I did not rerun or treat as my result.

The scratch counterexample suite above also terminated, exit 0. Its earlier development runs used CPU stand-ins for the pinned metadata methods; the final result reported for F1 uses the actual extracted methods and mutable annotation class. No candidate code was fixed in this session.

Other executed read-only commands included Git status/revision/branch/stat/diff/diff-check; UTF-8 Get-Content and rg over production/tests/designs and pinned Warp sources; Get-FileHash for design/header/source pins; structured PowerShell JSON profile inspection; SSH Git/sha256sum/readelf/selected source reads; and scp of installed types.py into reviewer scratch. Some broad JSON displays were truncated, and one Windows rg glob was rejected; focused reads replaced them. One exploratory native evidence print included an oversized entire supervisor object and was truncated; the saved bounded script below produced the usable concise observations. These were inspection-output issues, not candidate failures.

During review, the original author supervisor finished both success cases at the frozen HEAD. I did not start, signal or take over either campaign. I independently read the evidence and ran this saved **read-only stdlib** artifact/terminal verifier through SSH:

```text
Get-Content -Raw -Encoding UTF8 C:/Users/AG/新建文件夹/c4_assets/review_warp_native_evidence.py | ssh -o BatchMode=yes -o ConnectTimeout=8 tileops-dev /home/ang.gao/miniforge3/envs/robot_sim/bin/python -I -S -B -
```

Exit 0. It verified all nine recorded artifact lengths/digests per case, ELF64 little-endian CUDA machine190 and flags0x5a055a for all three CUBINs, cache mode0700, three unchanged executable IDs across the six first-load/reuse observations, distinct cache paths, absent worker/supervisor PIDs and terminal supervisor summaries:

| Raw evidence directory | Case | Cache | Worker / supervisor | Terminal observation |
| --- | --- | --- | --- | --- |
| `/data2/shared/ang.gao/c4-warp-module-author-v2` | Go2 960x640 | `/tmp/robot-sim-warp-mha7wy_q` | 2685901 / 2685873 | worker0, EXITED, 11 normal completions, 12 arms |
| `/data2/shared/ang.gao/c4-warp-module-author-v3` | Go2 1920x1080 | `/tmp/robot-sim-warp-9bd9xq3e` | 2738162 / 2738154 | worker0, EXITED, 11 normal completions, 12 arms |

The v2 actual load observations show target/default/current-context device1 and the held stream before/after each load, with other primary contexts inactive. These records and the production source support successful serial compilation/reuse and terminal disposal in the author runs. They are not an independent reviewer-run native acceptance campaign, optical-output proof or performance result.

The new native observer intentionally bypasses the prior input-phase ScopedDevice receipt in the module-preparation phases and samples inside the load wrapper. Consequently these new load observations alone do not establish restoration to ambient0 *after* leaving each outer ScopedDevice/ScopedStream context, especially on failure. Preserve that distinction in the remaining native acceptance work; do not relabel input-phase evidence as module-compilation restoration evidence.

Mandatory remaining native acceptance includes the source-consumption/mutation and hostile cache/artifact/backend/target/stream/hash/ABI/foreign-record matrix at the production boundaries; guarded compiler/load error, publication-before-return, adoption failure and hang; true ARMED-before-import observations; IDLE entry and post-final-DISARMED issuance faults; exact-target emergency versus unknown/incomplete custody precedence; retained partial roots, zero next leaf/extra sync/guard, actual process exit and unrelated-process preservation; actual ambient restoration; and input/fixed-CUDA/source-admission regressions. CPU mocks and the two successful author cases do not close these gates.

## Process hygiene and scope

All reviewer tool calls and test sessions reached terminal results. The reviewer launched local CPU pytest/subprocess tests and short-lived shell/Git/hash/SSH/scp utilities, plus the read-only remote stdlib verifier. No reviewer GPU job, native compiler campaign, supervisor, installation, cache deletion, signal or process takeover occurred. The completed bootstrap test helpers explicitly reap their own subprocesses; the guarded-peer fixtures close on teardown. I independently observed the two successful author workers/supervisors absent through `/proc`; the coordinator's broader nine-PID cleanup statement and archived evidence are not substituted for an independent global process census. Scratch tests/source copies and this report remain outside the clean frozen candidate.

No approval is issued for this code until F1 is repaired and the revised exact candidate is independently reviewed. This report concerns the module-preparation/source-retention delta only. Even a later bounded code approval cannot approve integrated native acceptance, one-slot graph/timeline/arena, R_setup/R_final, ten warmups, final recording/seal/barrier, CPU-oracle render correctness, full lifecycle matrices or whole C4. C4c3/c4, C4d/B3, the planned systematic refactor and fresh whole review remain required. Product PREPARE, benchmark/results, baseline publication and backend selection remain closed.
