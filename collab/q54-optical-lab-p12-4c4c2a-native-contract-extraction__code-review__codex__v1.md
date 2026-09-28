# C4c2a native-contract extraction — independent code review

Initiative: q54-optical-lab-p12-4c4c2a-native-contract-extraction
Stage: review
Author: codex (fresh independent reviewer)
Version: v1
Date: 2026-09-29
Status: accepted
Related Files: C4c2a design-request/review v1; C4c arena design v7; C4c1 ABI v3; C4c1c review v8
Owner Summary: Approve only the bounded C4c2a extraction at 67b401878bb92371590885af7835be36430d3009. Independent checks pass 412 tests with no skips, reproduce all five frozen content contracts, and confirm exact moves and retained authority. The separately executed author full 135-case supervised suite also completed successfully. No substantive finding or mandatory evidence gap remains; no later-stage or product gate is opened.

## Identity and boundary

- Reviewer session: `/root/c4_c2a_contract_code_review_v1`, started independently without the implementation conversation. No candidate authorship or delegated agents.
- Base: `a1f6c8c87752559c7b25994d450a36c56c952b32`.
- Frozen HEAD: `67b401878bb92371590885af7835be36430d3009`.
- HEAD tree: `1aa693c73c5092cc25243c6cd0d804ae41c67670`.
- Local candidate: `C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design`, branch `implementation/c4-native-contract-extraction`.
- Runtime candidate: `/data3/ang.gao/robot_sim_c4_c2a_review_67b4018`, matching detached HEAD/tree on owner-authorized fallback `ang.gao@192.168.80.101:22`.
- Both candidates had empty tracked/untracked status at entry and final verification, with identical frozen HEAD/tree. No candidate edit was made. Scratch code and this report are outside the candidate.
- The HEAD delta from executable revision `0dcbf2a790adf3d695b9501e0ff01be564c8c06a` contains only the implementation note; executable/test content is identical. The full base delta is 24 files, 5,558 insertions / 3,880 deletions.
- Accepted design source-byte SHA-256 independently matches `b816775d3050471a9c33658f27f261e32a9819bf7bd74890b129302c4a019c48`.

Read `AGENTS.md`, `collab/README.md`, `collab/REVIEW_PRINCIPLES.md`, the full C4c2a design/approval and implementation note as UTF-8. Read C1 ABI v3 and C1c review v8, and retained C4c v7 provenance, implementation-slice, cleanup and closed-gate requirements. A truncated predecessor read was completed with its ending section. Historical review verdicts and author pass totals were treated as claims, not independent validation.

## Verdict: Approve

No substantive blocking or non-blocking finding was identified, and no mandatory evidence gap remains for this exact bounded extraction. Approval binds the recorded HEAD/tree and C4c2a scope only. It is not approval of full worker closure, runtime integration or parent-stage completion.

## High-risk implementation findings

### Exact move and unique ownership

Independent scratch AST comparison, anchored to the full immutable base revision, verifies **all non-import, non-module-docstring top-level statements** for native→native_input, receipts→provenance, timing→timing, cases→cases and workload_contract→workload_contract. This includes assignments, dataclasses, signatures, fields, serialization logic, registry cleanup closures and ContextVar construction, not merely function-name inventories or agreement between new aliases.

All seven lane-contract assignments exactly match the original eligibility assignments. Retained function/class ASTs in eligibility, device_bvh and device_scene are unchanged; the latter two lose exactly the moved host algorithm/types/helpers and canonical-hash helper. The extraction tests independently compare the moved mathematical AST and alias identities. CUDA arena has exactly two changed function definitions: `_canonical_c3_binding` and `_validate_c3_adapter_authority`. No other retained CUDA arena class/function AST changed.

The explicit compatibility imports point to the sole production objects, including private registries, timing ContextVars, readiness/prepared types and repository-consumed helper names. There is no sys.modules substitution, fabricated class module identity, wrapper duplicate body or second registry. Native-contract storage/readiness authority is not relocated. New production owners have no runtime instrumentation, benchmark eligibility or coordinator import. GPU preparation imports remain local to the explicit legacy preparation calls; those calls are not claimed to be CPU-only execution.

### Host construction, package exports and content anchors

The host median algorithm retains its sort/partition, stable primitive ordering, centroid/leaf behavior, normalized layout, bounds, quality computation, dtypes and topology serialization. Executed extraction checks cover zero/single/multiple primitives and degenerate centroids, both split strategies, repeated hashes and aliases. Frozen-content construction provides external anchors beyond self-consistency.

The initializer contains only its docstring, literal export list, `__dir__` and literal `__getattr__` dispatcher. Static inspection verifies all 172 explicit original import names are covered once with the same leaf mapping, except the two intended CPU aliases (`DeviceBvhBuildStats`, `build_host_median_bvh`) now directly use their real host owner. Runtime tests verify the unchanged public inventory, both leaf/package import orders, exact repeated object identity, `dir()` visibility and unknown-name rejection without new imports. There is no arbitrary loader, reflection fallback or losing-lane eager import.

Fresh CPU subprocess checks exercise every new contract entry and real packed-host fixture/replay construction. Their recorder registers forbidden attempts before raising, and the swallowed-Torch and benchmark negative controls pass. No runtime/benchmark/example/tool import attempt or optical GPU implementation side effect was observed. Real-runtime Warp-only leaf tests likewise observe no Torch or optical CUDA import. This is the specified contract-entry boundary, not full future-worker closure.

Independently reran the inspected `/data3/ang.gao/c4-c2a-content-probe.py` against this review tree and `/data3/ang.gao/c4_asset_frozen`. Its explicit candidate-root import, asset preflight, sole production materializers and exact expected-hash assertions all ran. All five cases and applicable scene/fixture/replay hashes matched, including both Go2 scenes and fixtures and the Go2 replay. Missing assets were not skipped or replaced. The frozen case definitions/content assignments themselves match the base AST. The separate legacy Go2 materializer does import the existing example importer; it is not evidence for the clean CPU entry boundary.

### Live C3/a2 authority and storage

Inspected the unchanged benchmark adapter path: exact a2 validation and source content checks, target allocation identity, actual stage/storage arrays, ordered scene/transform/acceleration/render signatures, canonical one-shot binding creation and registration of the exact binding/target/prepared/readiness tuple. The arena checks its configuration/target and actual live a2 pair before direct registry validation and canonical content checks; repeated binding preserves the existing exact tuple and digest conditions.

The new canonical consumer requires exact production class identity and invokes that known class's serializers directly. Registry validation calls the production function, never a callback obtained from the supplied receipt. Existing ordered ABI, content, device, storage/view, plan, slot/generation and C4 lease checks remain unchanged. Real CUDA tests reject canonical unregistered/copy receipts, injected claimed-authority methods, wrong target/readiness/replay allocations, same-content reuploads, same-object Torch storage rebinding and Warp pointer drift. An actually issued binding with an injected claimed-authority method still validates without calling it; no terminal event is recorded by binding.

A separate CPU bookkeeping probe confirms legacy and production imports see the exact same registry, wrong target/prepared/readiness identities fail individually, and binding garbage collection removes the registry entry. That bookkeeping probe uses an uninitialized receipt only to test registry mechanics and is **not** counted as live CUDA authority evidence. Real issuance/storage evidence comes from the GPU tests.

The supported-path process-local authority boundary remains unchanged; this is not hostile same-process Python protection. Serialization still does not mint live authority or result capability.

### Shared timing, hooks, admission and gates

Both timing import paths reference the same ContextVars, active trace state and runtime-scope identity registry. Executed tests prove both cross-import boundary directions, nested submission rejection, exact prior-state restoration after exceptions and real low-level allocation rejection. Real hook tests invoke the relocated host owner and the supported device-module alias separately through runtime entry resolution, observe the acceleration entry and reject `numpy.empty` before its body. Definitions were not replaced by disabled proxies or stale tokens.

Test-source closure changes add exactly the six production owner modules plus host_bvh; source-loader and policy implementations are unchanged. Executed source tests retain byte-digest validation, unknown/incomplete closure rejection, source-changed-before-compilation refusal, stale timestamp/unchecked-hash bytecode rejection and guardian-before-import/cache isolation checks. No sacrificial helper enters the product source policy.

Independent Git blob-byte comparison verifies unchanged native_contract, native_setup, worker, manifest, protocol, gate JSON and closed-worker runtime-contract JSON. The extraction tests also check the product supervisor. The full changed-path inventory contains no product request/launcher/AST/default/runtime activation edits. Existing host-seal rejection and a3b/result preflight denial tests pass. Cleanup ledger edits follow the real new symbol locations and explicitly retain only selected-lane consumers while requiring removal of losing-lane helpers/tables and unconsumed aliases at P12.4e; no permanent backend selector is introduced.

## Independent commands and observed results

Local commands ran in the candidate, with outside-tree scratch code:

```text
git status --porcelain=v1
git rev-parse HEAD
git rev-parse 'HEAD^{tree}'
git diff --stat/--name-only a1f6c8c HEAD
git diff --name-only 0dcbf2a HEAD
git diff --check a1f6c8c HEAD
Get-FileHash -Algorithm SHA256 <design>
python -B C:/Users/AG/新建文件夹/c4-c2a-review-static.py <candidate>
python -m ruff check $(git diff --name-only a1f6c8c HEAD -- '*.py')
python -m ruff format --check $(git diff --name-only a1f6c8c HEAD -- '*.py')
```

Identity, diff check, full AST/registry probe and lint all pass; all 22 changed Python files are already formatted. The static probe initially assumed every lazy branch retained its old module path; diagnostics identified exactly the two intended host-owner moves, after which its exact expected map was corrected. This was a scratch expectation error, not a candidate change. One exploratory Windows `rg` glob was corrected to `rg --files`; neither affected source.

Runtime transport:

```text
ssh -o BatchMode=yes -o ConnectTimeout=8 -i C:/Users/AG/.ssh/tileops_ed25519 ang.gao@192.168.80.101
```

Runtime prefix, from the matching detached review candidate:

```text
/data3/ang.gao/anaconda3/bin/conda run --no-capture-output -n robot_sim env CUDA_HOME=/data3/ang.gao/anaconda3/envs/robot_sim python -B
```

Independently observed Python **3.12.13**, Torch **2.9.1+cu128**, Warp **1.12.0**, three visible CUDA devices, and direct `nvcc --version` **12.8.93**.

1. `-m pytest -q tests/unit/optics/test_native_contract_extraction.py tests/unit/benchmarks tests/gpu/test_native_contract_runtime_imports.py tests/gpu/test_cuda_native_arena.py tests/gpu/test_optical_benchmark_native.py tests/gpu/test_optical_warp_native_commands.py`: **211 passed in 35.12s**, no skips, exit 0. Real target-1/ambient-0 arena restoration ran; no mocked ordinal substituted for it.
2. `-m pytest -q -p no:cacheprovider tests/unit/optics/test_sacrificial_sources.py tests/unit/optics/test_sacrificial_watchdog.py tests/unit/optics/test_sacrificial_lifecycle.py tests/unit/optics/test_sacrificial_leaf_descriptors.py tests/gpu/test_optical_cuda_lbvh_native.py tests/gpu/test_optical_warp_cuda_lbvh_comparison.py tests/gpu/test_warp_render_recording.py tests/gpu/test_warp_render_buffers.py`: **180 passed in 11.24s**, no skips, exit 0.
3. `python -B /data3/ang.gao/c4-c2a-content-probe.py /data3/ang.gao/robot_sim_c4_c2a_review_67b4018 /data3/ang.gao/c4_asset_frozen`: exit 0, five exact case/content rows with all applicable frozen fixture/replay digests; mandatory asset preflight ran.
4. `-m pytest -q -p no:cacheprovider tests/gpu/test_warp_sacrificial_render.py -k 'matches_independent_cpu_oracle or targets_second_real_gpu or exact_recorded_bundle_reused'`: **21 passed, 114 deselected in 142.24s**, no skips, exit 0. This freshly executes all twelve profile/shade CPU-oracle comparisons, real target-1/ambient-0 execution and all eight exact-bundle two-sentinel reuse cases through the supervised child and current admitted source closure.

Independent pytest total: **412 passed**, no skips. The content probe and AST/registry checks are separate, not added to pytest totals.

The independent checks are functional, not benchmark runs. Some overlap the author's functional supervised suite, so elapsed durations provide no performance evidence. Existing permitted Python/JIT caches may be used; the source-admission tests explicitly cover stale bytecode refusal and fresh bootstrap namespaces. No install, push, merge, baseline/result access or cleanup of other worktrees occurred.

## Author supplemental evidence and limitations

The author supplied source-identical CPU 413-pass, GPU 198-pass and C0 setup/timeline 19-pass summaries and outside-tree log locations. The complete supervised author suite finished before this verdict: **135 passed in 847.81s**, no skips/failures. I independently read all four final log summaries and computed their SHA-256 values. Those executions and totals remain **author supplemental evidence**, not my independent tests:

```text
/data3/ang.gao/c4-c2a-0dcbf2a-cpu.log
07112fa514c09098de6bb06e383a95327b1c4881780074346bfdb9c517d6aac5
/data3/ang.gao/c4-c2a-0dcbf2a-gpu.log
5f0aa299d5544bd660136cc2480eeedc36126c3b1a96c49914ad6cb82fe61b96
/data3/ang.gao/c4-c2a-0dcbf2a-setup-timeline.log
f7d9e668774315ca1f74610cdc20848c0973f5938f6d6f44cde8a4b345f16d39
/data3/ang.gao/c4-c2a-0dcbf2a-supervised.log
2e92c0667b52cf05858592fd3376bb76f9b18952e0fb56b04c51ebcd273371c4
```

This review independently exercises the affected source-admission mechanics and a real supervised subset, rather than claiming to rerun every historical native-debugger/terminal-fault control. Combined with the exact unchanged guard/supervisor implementation and the freshly completed full author regression, this is proportional verification for this extraction-only delta; no underlying C1c requirements are waived.

The primary host was reported repeatedly unreachable in the packet; this reviewer used the expressly authorized fallback after checking its candidate identity. No claim is made to primary-host ancestry/history reconciliation. Local Python 3.14 lacks the Torch/Warp runtime; runtime properties were tested in robot_sim, not waived or inferred from local failure. This is not a full-repository regression, hostile-Python/native-memory sandbox proof, cold-cache performance measurement or future worker source-closure certification.

## Approved and still-closed scope

Approved: the exact C4c2a extraction at the recorded HEAD/tree—sole production native-input/case/content/lane/provenance/timing owners, exact compatibility objects, host BVH/hash move, static deferred exports, exact CUDA C3/direct-registry consumers, bounded test-source closure and cleanup ownership updates.

C4c2 runtime integration, Warp timeline/R0–R1 arena, C4c3 static interop/R2 integration, C4c4 closure, C4d/B3, product GPU preparation, a3b/result/baseline access, host payload/shipping, backend selection and default changes remain closed. Earlier C1/C1c approvals retain only their own scope; this prerequisite cannot approve C1, C4c or C4 in aggregate. The later integration design remains required for materialization, source policy, readiness/warmup/guardian ordering, supervised waits, terminal events, lease authority, failure custody and idle close.
