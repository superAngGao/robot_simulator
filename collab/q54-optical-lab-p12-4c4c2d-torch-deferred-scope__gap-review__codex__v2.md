Initiative: q54-optical-lab-p12-4c4c2d-torch-deferred-scope
Stage: independent bounded scope / diagnostic / deployment re-review
Author: codex, fresh independent session /root/torch_deferred_scope_gap_v2
Version: v2
Date: 2026-10-08
Status: accepted for the bounded first-Stream diagnostic scope only
Related Files: scope-request v2; unchanged gap-review v1; runtime-adoption design/review v2; cuda-ambient design/review v1; captured-import gap-review v1; frozen scratch drivers and deployment profile listed below
Owner Summary: Approve the frozen first captured Stream diagnostic and finite deployment delta. F1/F2 are closed for this next step: the post-binding observers and deliberate UNKNOWN remain inside the original runtime segment, and the exact third callback/present Triton probe are covered. No captured Stream execution or success was obtained by this review. Upload, readback, ordinary release and full C2d3 acceptance remain closed.

## Verdict, identity and isolation

**Approve**, only the exact candidate, profile and three diagnostic scripts below for the smallest next captured first-Stream run followed by deliberate UNKNOWN quarantine before runtime COMPLETE. No substantive blocking finding or additional repair comment was established. This is not whole-implementation approval, proof that the impending run will pass, or approval to proceed to upload if it does.

The reviewer received a fresh task packet, not the implementation conversation; did not author or modify the candidate; did not delegate. Read AGENTS.md, collab/README.md and collab/REVIEW_PRINCIPLES.md fully. Initial PowerShell decoding of Chinese text was corrected by UTF-8 reads. Read the required predecessor designs/approvals, complete prior deferred report and bounded cold-profile report. Oversized document output was split and operative truncated portions reread.

- Repository: `C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design`.
- Branch: `implementation/c4-supervised-waits`.
- Review base: `4b3585f7fbbf83d7956c63f2a04930f56218bb1a`; base tree `dde2fa206a45db856696a58d7fabcb8540204d44`.
- Actual HEAD: `e0a3b7a4858ce051442bdc6e785a92fde56686d5`; HEAD tree `8dc63d40ce1c9a96c8a6971c467bd39dcdc27cf3`.
- Initial/final local status empty; HEAD and frozen driver/request hashes unchanged. Base-to-HEAD contains exactly the 140-line request and one-line JSON profile, 141 added lines. `git diff --check` passed.
- Production: `a3d750bbaf79d760a6727e56146d277e1cb3b770`, tree `9982c6d9ec44e7232537643254d3c00e8f76d445`. Independently observed primary `tileops-dev:/data2/shared/ang.gao/robot_sim_c2d3_binding_wip` clean and detached at that revision before and after checks.
- `git diff --exit-code a3d750bbaf79d760a6727e56146d277e1cb3b770 HEAD -- optical_worker tests/support AGENTS.md collab/README.md collab/REVIEW_PRINCIPLES.md` passed. No production code, supervisor, source policy or operative workflow change is being approved here.

Independently verified frozen SHA256 identities:

| Artifact | SHA256 |
| --- | --- |
| `collab/q54-optical-lab-p12-4c4c2d-torch-deferred-scope__scope-request__codex__v2.md` | `2410bd2e2a41c5db27dd8a7d083f84fb3a215b26df4c8d0f6c39a3a771ecf81c` |
| `tests/data/runtime_environments/183_cuda_stream_author_v1.json`, raw bytes | `44f0833cdf507127695052b97b56864eb7b39783122d4b511240bcb6d6ef87d7` |
| Same profile, actual POSIX contract canonical digest | `b4e47505a751377d48336beba5508cdbe2a2adcccdd10ae0e463e0767a537a47` |
| `c4_assets/c2d3_captured_torch_stream_child_v3.py` | `aae8ae0f30832ef0ae05a887546fbd390598dfd55e5a1cfe7f1fdf9e83111ad3` |
| `c4_assets/c2d3_captured_torch_stream_supervisor_v3.py` | `617d3526e511e940b13ce6a215b78bd14fdffe2b1a0065b84a73ca3d17e7f20c` |
| `c4_assets/run_c2d3_captured_torch_stream_v3.py` | `4a9f5e3c583937cc21c34aabba93fb816ffe4428c4b2e6c341c1c9eea4381422` |
| Unchanged deferred gap-review v1 | `77631d10678e4dc20b3dcbb344891aaad7b63861bf93ecd77e9f9e521a030b77` |

Here and below `c4_assets` is the sibling directory outside the candidate repository. The profile is byte-identical to the external draft v9. During the review the author staged the already frozen three scripts at their literal `/data2/shared/ang.gao/` paths. They were absent on the first read; a final independent remote read and syntax compilation matched all three exact hashes. This staging did not change the candidate, profile, production checkout or review verdict. No one represented staging as execution.

## F1 closure: actual placement and failure flow

Inspected the complete child, supervisor wrapper and invoker, plus actual `native_bootstrap.py` runtime construction/adoption/failure and `guarded_lifecycle.py` segment/ticket/termination flow.

The pre-Stream hook runs while `initialize_native_runtime` is already inside its original `controller.segment("setup")`. It observes the pristine state, then delegates to the original `begin_device_leaf`; it does not replace the Stream constructor or permit/enter operation. The ordinary indexed runtime import and its declared scope already precede the hook. The raw getter uses the hash-pinned, already loaded cudart with RTLD_NOLOAD and typed arguments. It holds the library/function/closure in the already retained session graph. It checks raw0, no primary contexts and raw `_initialized=False` after observation, without invoking the warming Torch current-device getter first.

The second hook calls the captured original `_hold_runtime_owner` before any role test or observer. Only `runtime_setup_binding` enters the diagnostic. In the actual production method this call follows returned Stream retention, stream adoption, `_stream_identity`, external/source audits, binding population and registry insertion, and is still inside the same `with` block. The hook then calls `_require_setup()` and `require_adopted(owner, owner.stream)` before native observations. These check the actual live original ticket and exact held owners, rather than accepting the diagnostic's descriptive boolean field as proof.

Successful diagnostic observations lead to `mark_unknown_device_scope` followed by an unconditional RuntimeError. Post-adoption observation/assertion/I/O failure marks UNKNOWN unless the controller is already terminating or already unknown, then propagates. The surrounding segment's exception path records failure, retains the graph and terminates without normal COMPLETE. `runtime_ready`, the binding return, upload and readback are unreachable for this exact fixed production path and hook. The retained original hold can itself raise before the hook's local try; that exception still reaches the existing armed segment failure/quarantine path, with the original method's append-before-registration retention preserved. It cannot cause observer execution after a failed adoption.

The production `_segment_context` puts normal_body_returned/COMPLETE/DISARMED only after the yielded body returns and running/ticket checks pass. The revised hook never returns normally for the binding role. Unlike v1's proposed post-return placement, there is no native post-binding observation after disarming, new ARM, nested segment or renewed runtime budget.

The wrapper substitutes one literal child path in the actual repository supervisor's subprocess command. The child remains `-I -S -B`; assertion checks are active. Invoker target1, ACK1000 and operation120000 are unchanged. Its 650-second outer process ceiling does not establish per-operation supervision and is not counted as a guard result. The actual supervisor's clock, transport, kill/exit observation and wait/reap remain authoritative. This review did not execute their device path.

Independent CPU control-flow reproduction compiled the exact AST of the frozen `diagnostic_hold`, without rewriting its body, against inert fixtures. All nine cases passed: expected deliberate UNKNOWN/RuntimeError; unrelated role delegates unchanged; original hold failure; no live ticket; native-observer sentinel failure; evidence-write failure; external audit already terminated; unexpected Triton exec event; wrapper marked invoked. Original adoption precedes all observations; missing ticket/failed hold performs no native observer; failures never return a usable binding. These fixture tests substantiate local hook ordering and exception branches only, not native behavior, custody authenticity or actual deadline enforcement.

## F2 closure: queue, spec resolution and finite profile delta

Fresh file-only inspection of installed `torch/cuda/__init__.py` reconstructs exactly three top-level `_lazy_call` registrations in order: `_check_capability`, `_check_cubins`, `_register_triton_kernels`. The actual child requires those three exact retained function identities and length3. It requires the exact held `_LazySeedTracker` class and its entire raw state `{manual_seed_all_cb: None, manual_seed_cb: None, call_order: []}`. No queue clearing, callback replacement, seed invocation or lazy getter is used to manufacture the precondition. Installed `_lazy_init` appends only nonempty tracker calls and executes the actual queue.

The third callback constructs two `_WrappedTritonKernel` objects, resolves `find_spec('triton')`, and on the present branch defines/installs the two SparseCsrCUDA operators. The sparse kernel imports occur only inside the held wrappers' callable bodies. Fresh installed `torch/library.py` and retained exact-commit Python-dispatch/Library/Dispatcher/JIT listener source show schema/kernel registration and holder construction; the holder constructor retains the callable, and the listener installs a boxed-call closure. Registration does not invoke that closure or wrapper. This is the narrowly inspected registration path under the established trusted native dependency model, not permission for later Triton execution.

The new profile differs from the approved cold profile by exactly:

- Native file `os_native/libnvidia-ml.so.595.71.05`, 2621336 bytes, SHA256 `9eb4358b7fea76556657670a6ae6b0017eaa4256b56c421a36626bf8c2b5f3f5`.
- Python file `site/triton/__init__.py`, 1661 bytes, SHA256 `4ec1df745c6bfd03257370a2aab1534e7a31eaa62110cad2ff6c3b56a67cfc50`.
- Metadata file `site/triton-3.5.1.dist-info/METADATA`, 1663 bytes, SHA256 `bbc16183b5b32e3cbcd2d59f725614443ec8e3fed7d244b9c8403b908bb10e07`; one exact triton3.5.1 distribution row and one runtime package-file module row.

All old rows remain byte-for-value equal; no rows are removed. Roots, interpreter, settings, 42 publication rows, seven replacement rows, eight absent rows and all other categories are unchanged. There is no descendant or Triton native-library grant, fake absence, disabled-package bypass or importer exemption. Inspected both finite offline builders; neither changes executable permission from a running worker trace.

Read the production finder's rule, standard resolver, exact origin/package/path/file-digest validation, held delegate/spec/binding, file_resolved event and source-loader execution checks. Resolving a spec does not call create_module/exec_module. The child explicitly requires exactly one Triton file_resolved event, no record or sys.modules publication, exact held wrapper/delegate binding, name, origin, package location, SourceFileLoader type/name/path, exact two ops_table keys and both kernel_invoked flags false. An unexpected import still fails the diagnostic even if the finite root rule allows resolution.

A fresh CPU-only PathFinder call on the actual POSIX deployment returned the expected standard loader, origin and package location without importing Triton. Fresh production `RuntimeEnvironmentContract.from_payload` and `verify_files` passed for all 860 files, 759 named file modules, 19 distributions and the pinned interpreter. Windows path rules were not patched to produce that result. These checks do not substitute for actual captured finder events, which must still be observed in the proposed run.

The flag `kernel_invoked=False` is descriptive evidence, not by itself proof that no failing call was attempted: the installed wrapper sets it after the inner call returns. The inspected registration control flow and absence of kernel invocation are the reason the normal path is in scope. A failure/extra import is never accepted merely because this flag remains false.

## Source/deployment trust and limits

Verified every file in the four retained source manifests: 16/16 main native-chain files, 10/10 additional native files, 6/6 deferred-source files and 6/6 dispatcher/registration files. Manifest SHA256 values, respectively:

- `32a9b6deb348201e9893f8b7c0cb90d450dd1f9017cdf4a356de6e5e45ca9f91`
- `dac8a89314241f7f81ab0e5241a13d2bb240895e98a7f49b37969f1437d43985`
- `794a007a71935fe0ab1fc74809ca2100683eeeeb9403cde2e1af728a1128f79c`
- `f68a4a993b3807542d2a92a39166cf1186c89948426db1dc36f141fd4de1894e`

Checked the actual C10 GetDevice/ExchangeDevice/MaybeExchangeDevice and Torch lazy initialization/default-generator construction entrypoints relevant to unchanged native scope. C10's initial targetDeviceIndex=-1 plus pristine raw0 yields derived logical0; target exchange selects1; CUDA12 exit can restore logical0 without creating absent context0, leaving raw1. No implicit raw0 assumption is introduced by the hook. This round concentrates on F1/F2 and the new dependency delta; hash-checking entire manifests is not a claim to have repeated every prior allocator/property/stream native callee audit or every exception path.

Inspected the retained installed nvrtc disassembly and its matching library hash: it selects lazyNVRTC, addressing the direct-versus-lazy build question. The retained file-only NVML readelf/ldd audit shows the expected pthread/m/dl/c/rt dependencies and loader; builder v8 checks their existing profile hashes. Independently reverified the complete current profile, including those dependencies. Fresh `/proc/driver/nvidia/version` reports kernel module595.71.05. Fresh symlink resolution and hashing checked libnvidia-ml.so.1 to the exact admitted595.71.05 file and libcuda.so.1 to libcuda.so.595.71.05, SHA256 `76e0d9678d41cf6b6ae71d18549d88a963d3d434f08108baa12457eb53ca88d6`. No NVML/CUDA library was loaded by these checks.

Verified audit artifact digests: Triton probe `929b01634f061f6dbda1f05db2ccc83ee36b8aceed4b7999d9026bf3d4dae47a`; NVML `327ef49aa51d2ad75b4f7cc3ef1c0fbb7c47c3cf9f660a23215bc91cbb22e8da`; nvrtc binary evidence `2a19c660b78f7585b74cdd6545f33ae6a4e12f15d0d83e758a54b2a5b377e199`. All five stored Triton audit source/metadata strings match their own hashes. Actual deployment file verification, not those claimed hashes alone, checked current filesystem identity.

Native internals remain trusted pinned deployment dependencies. Snapshots cannot exclude transient context activity; process quarantine cannot undo unsafe cleanup performed inside native code before returning. The established source/native exception-safety obligation remains. No new sandbox, native shim, private context or broader target rule is introduced. Any contrary real side effect or unsafe native cleanup stops admission.

## Executed verification and artifacts

CPU-only commands used PowerShell UTF-8 reads, rg, Get-FileHash, Git revision/tree/status/diff checks; local `C:/Python314/python.exe -I -S -B`; primary SSH with BatchMode and ConnectTimeout8; remote `/home/ang.gao/miniforge3/envs/robot_sim/bin/python -I -S -B -`. Remote interpreter reports CPython3.12.13. No fallback host was used.

Retained exact independent scripts and logs under sibling c4_assets:

- `torch_deferred_scope_review_checks_v2.py` and `torch_deferred_scope_local_checks_v2.log`: request/driver/audit hashes, syntax, exact profile delta, four complete manifest hash checks, installed audit source checks.
- `torch_deferred_scope_remote_checks_v2.py` and `torch_deferred_scope_remote_checks_v2.log`: actual POSIX contract parsing/full filesystem verification, standard spec-only resolution, installed callback/lazy-init source, separate driver identities, pristine remote checkout and absence of backend imports.
- `torch_deferred_scope_hook_checks_v2.py` and matching `.log`: nine CPU-only exact-hook AST control-flow cases; all passed.
- `torch_deferred_scope_environment_tests_v2.log`: fresh `ssh tileops-dev 'cd /data2/shared/ang.gao/robot_sim_c2d3_binding_wip && /home/ang.gao/miniforge3/envs/robot_sim/bin/python -B -m pytest tests/unit/optics/test_runtime_environment.py -q -p no:cacheprovider'`; **52 passed in1.72s**, no skips.
- `torch_deferred_scope_final_remote_v2.log`: final staged script hashes/syntax, installed wrapper and Library source, final clean detached production revision and scoped process absence.

The remote verification script was supplied over stdin with `Get-Content -Raw -Encoding UTF8 ... | ssh ...`; it was not installed into the remote repository. Its PID346182/PPID346181 and start ticks230282028 were recorded; later ps showed neither present. Final read-only checker350443/350442 also exited; final scoped ps returned only its header (ps exit1 means no matching processes). No new device worker or supervisor was launched by this reviewer, so no fresh GPU residual or actual guardian reap claim is made. Runtime environment unit tests were CPU-only. Reviewer persistent writes are only the named scratch scripts/logs and this report.

Review-tool failures were corrected and not counted as passes: a nonexistent exploratory test filename, a Windows wildcard rg path, and a manifest parser initially expecting `sources` where one manifest uses `files`. Subsequent bounded reads/parser correction and checks passed. No product/runtime failure was observed because the device diagnostic was not run.

The author separately mentioned unrelated ordinary C1 nested-scope diagnostic results during this review. They were not used as evidence for this candidate, not re-reviewed here and cannot establish a captured Stream result.

## Exact next-step approval and evidence still required

This approval permits only the frozen v3 invoker/child/supervisor with the profile above, pinned robot_sim interpreter, clean productiona3d750b, target1, actual asymmetric_synthetic_7x3_v1 canonical materialization and unchanged ACK1000/op120000. Its invocation must pass the exact production revision, for example:

```text
/home/ang.gao/miniforge3/envs/robot_sim/bin/python -I -B /data2/shared/ang.gao/run_c2d3_captured_torch_stream_v3.py a3d750bbaf79d760a6727e56146d277e1cb3b770
```

The destination is exclusive and scripts are frozen. Preserve all artifacts and failed outcomes. Drift or a required new permission returns for review; do not normalize errors, learn permissions, change callbacks, expand device scope or renew a deadline to make this run pass.

Expected proof remains prospective: pristine raw0/uninitialized/all eight contexts absent and exact callback/empty seed state; actual captured Stream plus exact binding adoption; logical0/raw1/only context1 and streamdevice1; resolution-only Triton and expected uninvoked registrations; actual mapped native paths compared against the fixed inventory; intentional UNKNOWN inside the original runtime segment. Require four total ARM and only the three earlier normal COMPLETE records, zero runtime COMPLETE, terminal ARM/sync/release zero, hard exit74 and actual supervisor exit/reap under the unchanged budget. Record actual ordinal/UUID and PID/start/parent identities and scoped OS/GPU residual observations. The invoker's exit and partial summary assertions alone do not check all these obligations; inspect the actual observations and transition records. Missing/contradictory evidence remains failed or incomplete.

Still not obtained or approved: a real captured Stream success by this reviewer; successful complete runtime initialization; allocation/copy/events/input-ready/readback; ordinary target sync/release; native exception/hang matrix; full outer-device/inner-stream regression acceptance; both-lane canonical/replay/both real Go2 inputs; dual-GPU foreign/UNKNOWN and ACK/EOF/terminal matrix; final versioned custody aggregate and replay negatives; full independent C2d3 review. No R_setup/R_final, final graph seal, prepared sample authority or measured optical sample is created. C2e/f, C4c3/c4, C4d/B3, product GPU preparation, a3b, result/baseline, host payload, R2 and shipping remain closed. C4 remains active.

The unchanged complete report may be archived under the evidence-only exception, without modifying the verdict, required real-run evidence, approved bytes or gate boundaries.
