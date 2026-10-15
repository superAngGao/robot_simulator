# C4c2a native contract extraction — independent design review

Initiative: q54-optical-lab-p12-4c4c2a-native-contract-extraction
Stage: review
Author: codex (fresh independent reviewer)
Version: v1
Date: 2026-09-29
Status: accepted
Related Files: C4c2a design-request v1; C4c arena design v7; C4c1 ABI design v3; C4c1c independent review v8
Owner Summary: Approve only the bounded C4c2a prerequisite design at 6084245689ae50976f772daa269cc92d19605568. The proposed single-owner extraction is implementable against the actual source graph without a worker source-policy waiver or an implicit runtime activation. No substantive finding. Implementation and its independent code review remain required; no parent-stage or product gate is opened.

## Identity and scope

- Fresh reviewer session: `/root/c4_c2a_contract_design_review_v1`. No implementation conversation inherited; no candidate authorship; no delegated agents.
- Local candidate: `C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design`, branch `design/c4-warp-timeline-r0-r1`.
- Base: `fd18b87dfa8e89e73b929576b52ff906e3c51bca`.
- HEAD: `6084245689ae50976f772daa269cc92d19605568`.
- HEAD tree: `e1a83972825cb90061c6f7f8e819347e93d33e81`.
- Design: `collab/q54-optical-lab-p12-4c4c2a-native-contract-extraction__design-request__codex__v1.md`.
- Design source-byte SHA-256: `b816775d3050471a9c33658f27f261e32a9819bf7bd74890b129302c4a019c48`.
- Base-to-HEAD delta is exactly that new 295-line document; no implementation, tests, gates, or operative workflow rules changed. `git diff --check fd18b87 HEAD` passes.
- Local tracked/untracked status was empty at entry and at final verification. No candidate files were edited and Python probes used `-B`.
- Remote candidate: `/data3/ang.gao/robot_sim_c4_c2a_design_6084245`. Independently observed empty status, detached HEAD, identical HEAD/tree and design digest. Remote operations were read-only identity checks; no remote Python, CUDA, benchmark or test suite was run.

Read `AGENTS.md`, `collab/README.md` and `collab/REVIEW_PRINCIPLES.md` completely. The Chinese principles were reread explicitly as UTF-8 after the initial default-encoding rendering. Read the full candidate, C1 ABI v3, and C1c v8 report (including its ending scope/limitations), and inspected the relevant arena-v7 provenance, worker/source-policy, implementation-slice and activation requirements. The large initial combined predecessor read was truncated; targeted reads supplied the relevant omitted boundary/provenance sections. Historical pass totals are not new verification evidence.

The evidence archive base follows approved C1c code `6d57fa789cb50e4191239a4d00f628f25824bb46`. Its v8 approval closes only the fixed supervised validation checkpoint, not C1 or C4 in aggregate. This review makes no new claim about the execution of that historical validation.

## Verdict: Approve

No substantive blocking or non-blocking finding was identified for this design revision. No mandatory design-stage evidence gap remains. Approval authorizes implementation of the enumerated C4c2a moves and consumer checks only, followed by fresh independent code review. It is not approval of an implementation that does not yet exist.

The design freezes the important choices instead of delegating them to implementation: sole owners and exact legacy aliases; unchanged a2 authority owner; one C3 identity registry; exact consumer type/canonical/direct-registry validation; one timing state; unchanged mathematical BVH implementation and frozen content; literal deferred exports; and distinct test-source versus product-source policies. It explicitly stops at the contract entry surface, with worker materialization, warmup/wait ordering, timeline, arena and lifetime composition reserved for the next design. This is a valid prerequisite boundary, not a claim to have completed those integrations.

## Reviewed source facts and requirements

### 1. Dependency cuts and CPU construction

`benchmarks/optical/native.py:14-37` already imports prepared/readiness definitions from `optical_worker.native_contract`, while its remaining dependencies include cases, eligibility, receipts, timing, workload content, `optics.device_bvh`, and `optics.scene_contract`. `eligibility.py:11-20` imports runtime instrumentation and timing. Merely aliasing its classes into production would therefore not provide the required clean entry surface.

The proposed split is feasible: cases and workload-contract definitions use CPU/stdlib data; the required eligibility definitions are literal input/lane/acceleration tables; receipts use those tables plus CPU serialization/weak references; timing is CPU context and trace management. The native helper module's Warp/Torch preparation and backend factory imports are function-local. Moving those existing helper bodies does not imply they are safe to invoke in a closed GPU-disabled worker, and the candidate explicitly forbids adopting that unguarded path into the worker.

`optics.scene_contract` uses the real `PackedOpticalSceneHost.from_registry` packer and `optics.registry`; the registry imports `physics.spatial`. Physics initialization is broad but its inspected reachable surface is CPU-safe. Fresh local cold-import probes separately imported `physics.spatial`, `sensing.optical`, `robot.model`, and `optical_worker.native_contract`, recording and rejecting forbidden imports before delegation. All four completed with no attempted Torch, Warp, benchmark, example, or tool import. The actual loaded-module lists included the normal physics/sensing/robot package initializers, not substitute packages.

These observations support leaving those initializers unchanged. They do not prove the future modules' closure: implementation must still execute the complete fresh-subprocess acceptance tests. I did not stub `optics`, move source in memory, or describe a hypothetical relocated package as tested production code.

`benchmarks/optical/workloads.py:35-79` contains the current frozen scene materialization routes and the local Go2 example importer. It is not moved by this proposal. The distinction between clean contract/host-construction tests and separate legacy Go2 materialization/hash tests is necessary and is stated explicitly. Full worker workload/camera materialization remains a later integration obligation.

### 2. Host BVH and package exports

`optics/device_bvh.py` currently imports `device_scene` and attempts Warp import at module load. Its CPU algorithm is separable: `DeviceBvhBuildStats`, `_HostNode`, `build_host_median_bvh`, `_surface_area`, CPU constants and topology hashing do not require device execution. The topology hash presently reaches the CPU `_canonical_sha256` through `device_scene`; the design correctly requires moving/reusing that helper without importing the GPU-bearing module.

Inspected the actual algorithm at `device_bvh.py:765-970`, including empty arrays, stable source-key ordering, longest-axis choice, sort/partition branches, breadth-first normalization, level ranges, quality calculation, dtype encoding and topology byte serialization. The contract preserves these bodies and their existing semantics, including edge behavior; it does not authorize mathematical cleanup or a second host implementation. AST-body comparison plus existing frozen digests are suitable independent anchors.

`optics/__init__.py` eagerly imports both lanes, including CUDA arena/render/LBVH, device scene/BVH, GPU runtime and Warp execution. A source-enumerated `__getattr__` with static relative imports can remove this package side effect without an arbitrary loader. Preserving `__all__`, explicit imports, repeated identity, both import orders, `dir()` and unknown-name behavior gives concrete compatibility acceptance criteria. The exception for explicitly requested star imports is coherent and does not turn star import into a worker entry.

### 3. C3 live authority and consumers

`benchmarks/optical/receipts.py:24-81` owns the existing weak-reference registry. Its value holds the exact target/prepared/readiness tuple; cleanup checks the same weak-reference identity before deletion. The benchmark CUDA adapter builds stage/storage evidence from the actual target and registers the resulting binding at `benchmarks/optical/cuda_arena.py:173`. Serialization alone does not execute that registration.

The existing real consumer has exactly the migration risk described by the proposal: `optics/cuda_arena.py:1965` recognizes module/name strings and obtains serialization methods from the supplied object/type; `:1982` calls a supplied object's authority method. The arena's `bind_c3_provenance` first validates its own target and live a2 pair, then validates adapter authority and canonical content. Replacing the latter checks with the real production class and direct shared validator is implementable without changing the target/storage/plan graph.

The design explicitly preserves ordered storage/dataflow/acceleration checks and the existing a2 physical-storage authority. Keeping `native_contract.py` as sole owner avoids redefining exact types or moving its live registries. Its attack matrix includes same-name fakes, canonical-but-unregistered values, copies, wrong tuples, ordered storage changes and actual pointer/storage drift; it also requires proving that an injected claimed-authority callback is never invoked. This is stronger than comparing two aliases or trusting a digest.

The registry remains supported-path process-local authority, not a hostile same-process Python sandbox. The accepted predecessor makes that trust boundary explicit. No new Warp adapter or result-capable receipt is authorized here.

### 4. Timing, instrumentation and legacy compatibility

`benchmarks/optical/timing.py` owns all three ContextVars, active trace state, hook-capability identity mapping and assertion functions. `native_timed_submission` rejects nested submission and resets context in `finally`; runtime hook scopes also reject nesting and restore their token. A complete single move plus explicit re-exports preserves those function globals and state objects. The design requires both cross-import directions, forbidden-entry observation and exceptional restoration; nested-scope coverage does not authorize newly allowing currently forbidden nesting.

Inspected native helper assertions, benchmark instrumentation's entry resolution/wrapping, and repository consumers in benchmark runner/schema/adapters and CPU/GPU tests. Private helpers such as `_validate_native_input_host_observations` and `_register_cuda_arena_c3_binding_authority` are real consumers, not optional implementation details. The explicit pre-edit alias inventory requirement covers them. Updating defining-module monkeypatches is necessary: re-export identity alone does not make assignments to a compatibility module patch the defining module's globals. The candidate explicitly requires real hook observations and owner-target updates, not weakening assertions when an old target stops working.

### 5. Source admission and closed gates

`tests/support/sacrificial_sources.py` maintains its own enumerated repository source closure, with byte-hash source admission. It currently lists the old benchmark modules and optics initializer. Adding the new sole-owner modules there is required for affected C1c tests and is materially different from changing the product source set.

Inspected `optical_worker/manifest.py` AST policy and runtime-contract validation, product worker PREPARE rejection, native arena host-seal rejection, and schema activation facts. Product `gpu_preparation_enabled` remains false; the worker rejects `PREPARE_FROZEN_ROW`; host payload sealing and result capability remain closed. Static imports in a literal deferred initializer do not themselves require a general `importlib`, `ctypes` or source-policy exemption. Conversely, future packed-ABI inspection and full worker closure are not approved simply because the contract modules have moved.

The candidate requires byte-identical frozen gate/runtime-contract JSON and unchanged product launcher/worker/AST/request surface. Because this HEAD is document-only, those surfaces are in fact unchanged. Follow-on source-policy and final setup ordering decisions are explicitly deferred for independent design, not silently waived.

## Independent verification and limitations

Executed read-only local checks in the stated candidate using PowerShell:

- `git status --porcelain=v1`, `git branch --show-current`, `git rev-parse HEAD`, quoted `git rev-parse 'HEAD^{tree}'`, base resolution, `git diff --name-status/--stat fd18b87 HEAD`, `git diff --check fd18b87 HEAD`, and `Get-FileHash -Algorithm SHA256` for the design.
- `git show --format=fuller --stat fd18b87 -1` to check the archive boundary.
- Targeted `rg` import/definition/call-site searches and UTF-8 source reads covering the modules and requirements discussed above.
- Four separate `python -B -c <import-attempt probe> <entry>` processes for `physics.spatial`, `sensing.optical`, `robot.model`, and `optical_worker.native_contract`. Local Python was 3.14.0, NumPy 2.5.2. The probe wrapped `builtins.__import__`, recorded forbidden roots before raising `ImportError`, then imported the real entry and checked the recorded attempts. Each recorded zero forbidden attempts.
- A separate calibration process attempted `import torch` inside `try/except ImportError` and `import benchmarks.optical`. Both rejected attempts were recorded exactly, with neither target present in `sys.modules`. Thus the limited diagnostic does not confuse swallowed imports with absence of attempts.
- Primary transport retry: `ssh -o BatchMode=yes -o ConnectTimeout=8 -p 40031 -i C:/Users/AG/.ssh/tileops_ed25519 ang.gao@183.166.183.33 "pwd"` timed out. This is a transport limitation, not a code/design failure.
- Owner-authorized fallback `ssh -o BatchMode=yes -o ConnectTimeout=8 -i C:/Users/AG/.ssh/tileops_ed25519 ang.gao@192.168.80.101` ran only remote git identity/status and `sha256sum` checks. It confirmed the matching clean detached candidate. No primary-host ancestry or history reconciliation claim is made.

One unquoted PowerShell `HEAD^{tree}` argument was misparsed; quoting it returned the tree recorded above. A few exploratory `rg` paths did not exist and were corrected to actual source paths. Neither issue changed files or affected a substantive conclusion.

No GPU suite is necessary to approve a pure design delta, and none was run. No device allocation, extension load, package installation, benchmark run, baseline access, result artifact, push/merge or candidate mutation occurred. The dependency probes are narrow observed CPU facts, not proof of future single-lane runtime closure or all possible import mechanisms. Implementation must provide the full calibrated import detector and actual migrated-entry tests specified by the candidate, frozen case/replay/fixture/Go2 identity checks, AST move comparisons, real C3/a2 adversarial evidence, affected dual-device regression and supervised-source tests. Historical C1c results are not substitutes.

## Exact approved and still-closed scope

Approved: design v1 at the recorded HEAD and digest for the sole-owner native input/manifest/case/content/lane identities, shared C3 receipts/registry, shared CPU timing context, host BVH extraction, literal deferred optics exports, exact CUDA C3 consumer checks, legacy aliases and the bounded test-source migration/acceptance work.

Still closed: C4c2 runtime integration, Warp timeline and R0/R1 arena, C4c3 static interop/R2, C4c4 closure, C4d/B3, product GPU preparation, a3b/result/baseline access, host payload/shipping, backend selection and default changes. Neither C1c's approval nor this prerequisite design approval approves C1, C4c or C4 in aggregate. The subsequent integration design must settle its own materialization, source-policy, warmup/readiness/guardian ordering, ACK-before-wait supervision, terminal event, lease authority, failure custody and idle-close obligations.

No additional follow-up beyond the candidate's stated implementation acceptance and later-design obligations is imposed by this review. Compatibility aliases remain one implementation, not a second data plane; the documented selected-lane cleanup and removal of unconsumed aliases remain required before broader completion can be claimed.
