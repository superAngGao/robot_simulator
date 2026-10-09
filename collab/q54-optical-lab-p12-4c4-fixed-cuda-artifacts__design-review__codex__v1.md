# Independent design review: fixed CUDA artifacts v1

Initiative: Q54 / P12.4 / C4 scope convergence
Stage: independent bounded design review
Author: codex, independent reviewer /root/fixed_artifacts_design_v1
Version: v1
Date: 2026-10-09 (Asia/Shanghai)
Status: in_review — review complete, changes requested
Related Files: fixed-cuda-artifacts design-request v1; runtime-integration design/review v3; remaining-setup gap-scan v1; compilation-admission design/review v3; setuptools-admission design/review v2; systematic-refactor implementation-note v1; scope-convergence proposal v1
Owner Summary: The replacement is architecturally viable and substantially simplifies the worker. Three bounded corrections are required: establish normal builder exit independently of systemd's success classification; freeze a builder environment that actually avoids CUDA initialization; and bind the new extension-set/entry tokens to named lifecycle adoptions. These corrections do not require retaining worker JIT or creating another general framework.

## Verdict and identity

**Verdict: Request changes.** This is a design verdict for the exact candidate below, not rejection of the deployment-artifact direction and not implementation or execution approval.

Fresh independent session `/root/fixed_artifacts_design_v1` received the task packet and repository access. I did not author or modify the candidate, spawn agents, execute tests, or run builds/services/GPU work. AGENTS.md, collab/README.md and collab/REVIEW_PRINCIPLES.md were read in full; the initially misdecoded Chinese principles were reread explicitly as UTF-8. Governing documents, relevant prior reviews and actual source paths were inspected. The coordinator supplied two additional source leads while the candidate remained frozen; both were independently checked against primary sources rather than accepted as findings on assertion.

- Repository: `C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design`.
- Branch: `implementation/c4-supervised-waits`.
- Base: `1a9f319708cdf1856fe73f1a0108acc78b9190e3`.
- Reviewed HEAD: `d9e764931d53b3e6b62934ef0cfe27932adaf058`.
- Sole candidate: `collab/q54-optical-lab-p12-4c4-fixed-cuda-artifacts__design-request__codex__v1.md`, 266 added lines.
- Independently recomputed Git LF blob SHA256: `48ccebb96f79289e08c9bf1a3cf93cd27475d1e08cb39f019fd282680e4e3406`.
- Working-byte SHA256 is identical.
- Initial and repeated local porcelain status: empty. HEAD unchanged. Base-to-HEAD `git diff --check`: passed.
- Source-only SSH observed the primary isolated tree HEAD as `92a8ac646ac182f2dac4dc4eef8a3dd0b331f1af`. The remainder of that read-only call stalled and was finally cancelled through its owned exec handle; actual terminal exit code was 1. No claim of independently refreshed remote cleanliness or installed-source hashes is made from that incomplete call.

## Findings

### R1 — P1: zero systemd-run status can publish an interrupted build

Location: candidate section 2, lines 74–94, especially 83–89; section 3 `builds` rows.

Trigger: the builder has produced the expected extension file, then its main process terminates by SIGTERM (or another default clean daemon signal) before the fixed build command completes normally. The owned cgroup subsequently empties. This can be an ordinary interruption; it does not require a hostile dependency or root attacker.

The specified `Type=exec` uses systemd's daemon clean-exit classification. Upstream v255 `service_sigchld_event` selects `EXIT_CLEAN_DAEMON` for this main process. `is_clean_exit` treats SIGTERM, SIGHUP, SIGINT and SIGPIPE termination as clean even with no SuccessExitStatus override. `systemd-run --wait` returns zero when Result is success, before examining other exit classifications. Thus waiter zero + Result success + completed output + empty cgroup does not establish normal builder exit. [systemd service classification](https://raw.githubusercontent.com/systemd/systemd/v255/src/core/service.c), [clean-exit implementation](https://raw.githubusercontent.com/systemd/systemd/v255/src/shared/exit-status.c), [waiter result handling](https://raw.githubusercontent.com/systemd/systemd/v255/src/run/run.c).

Consequence: the launcher may publish artifacts and a successful provenance manifest from an interrupted command. An output digest proves the selected bytes, not that the required fixed build transaction completed. The present prohibition on SuccessExitStatus overrides does not remove systemd's defaults. Cgroup emptiness proves process closure, not normal completion.

Fix direction: additionally require manager-observed main termination `CLD_EXITED` with status 0. Freeze how that observation is obtained and durably associated with the exact unit invocation before `--collect` makes properties unavailable. Use the existing manager/waiter facilities (for example a narrowly defined manager termination record or independently captured waiter termination summary); do not add PID tracking or restore the old subreaper framework. Keep command logs distinct from authoritative termination facts. Define the manifest representation of waiter status, main exit code/status and terminal unit/cgroup observation. Missing/ambiguous facts fail publication.

Required proof after repair: the normal builder fixture succeeds; a fixture with its expected output present followed by SIGTERM termination cannot publish despite a manager success result; nonzero exit and timeout cannot publish; all retain truthful terminal facts and the existing empty-cgroup/sentinel checks. These belong to the already planned installed-manager acceptance packet, not execution during this design review.

### R2 — P2: the proposed clean build environment does not establish the no-GPU-initialization promise

Location: candidate section 2, lines 48–51 and 64–71; section 6 item 3.

Trigger: a fresh builder imports `torch.utils.cpp_extension` for CUDAExtension/BuildExtension with the stated CUDA_HOME and explicit architecture, but without a separately frozen availability-check policy.

The independently read preserved installed `cpp_extension.py` (SHA256 `52ba4de5deedd894e2bdafc9b7f40526dbc2cc1e4c560bbae3b38d4a5dcab0c6`) calls `_find_cuda_home()` at module initialization; that function calls `torch.cuda.is_available()` even when CUDA_HOME is explicitly provided. Current repository `cuda_compilation.py:248` already acknowledges this native availability check and puts its old import inside device scope. PyTorch v2.9.1's default `is_available()` reaches `_cuda_getDeviceCount`; its source explicitly identifies CUDA runtime/driver initialization. Explicit TORCH_CUDA_ARCH_LIST avoids a different architecture-discovery path, not this import-time call. [Pinned PyTorch CUDA implementation](https://raw.githubusercontent.com/pytorch/pytorch/v2.9.1/torch/cuda/__init__.py).

Consequence: a literal implementation can perform CUDA driver initialization during a deployment build that the design describes as having no GPU initialization. This is not evidence that a kernel ran or that Torch's full lazy CUDA state initialized; those are different claims. The categorical requirement currently has no frozen mechanism preventing the actual native query.

Fix direction: define the exact builder-only environment and initialization invariant before importing Torch. A small plausible source-backed solution is `CUDA_VISIBLE_DEVICES=''` together with `PYTORCH_NVML_BASED_CUDA_CHECK=1` in each fresh builder: the pinned NVML-count path returns zero for an empty visible-device list before native NVML discovery, preventing the CUDA fallback at that point. Freeze and verify the installed path and distinguish these build variables from worker visibility. Merely setting the NVML flag is insufficient when discovery can fall back. No import monkeypatch or build sandbox is needed.

Required proof after repair: source preflight establishes the complete selected availability path and explicit architecture flags; the bounded build acceptance verifies that the builder neither imports its output nor invokes optical entries or GPU initialization. Missing/changed required environment values reject before dependency import. Do not substitute `torch.cuda.is_initialized() == False` alone for absence of the native availability call. The suggested environment is a design direction, not a claim of executed target proof.

### R3 — P2: new entry and set tokens lack independent named adoption joins

Location: candidate section 4, lines 181–200, especially the `cuda_extensions_bound` replay rules at 191–197.

Trigger/counterexample: keep a successful independent lifecycle/object table, all three completed ordinary file records, the six native-scope pairs, entry names/order and all distinct tokens fixed. Exchange only the two scene entry tokens under the names `update_triangles_world_aabb_out` and `update_planes_world_out` in the new binding event. Both tokens still refer to previously adopted objects, both entries still have required names in the descriptor, and uniqueness and all file/module joins still hold. The specified replay rules do not distinguish the two named entry roles. Similarly, an unrelated adopted HOST_OWNER can occupy owner_token unless the new set has an independently fixed adoption discriminator.

Consequence: a conforming but weak replay implementation can accept a semantically false named-entry/set binding. The live check of each callable's `__module__`/`__name__` is useful but not present in independent serialized adoption facts. A hash of the edited record cannot supply the missing relation. This is the same information loss the replaced compilation design v3 repaired, now restricted to the few genuinely new objects; ordinary file machinery already supplies module/spec/delegate joins and need not be duplicated.

Source evidence: `runtime_custody.py:329` supports role, scope, operation and temporal joins, but all are optional arguments. The ordinary file grammar at approximately 420–484 binds named file modules/specs/delegates; it has no named extension-entry fact. `compilation_contract.py:144` and `cuda_compilation.py:372,505` show the previous actual boundary assigning recipe-and-entry-specific adoption operations. The new design says only prior actual adoption and does not retain or replace those discriminators for entry/set objects.

Fix direction: freeze one literal adoption role/operation for CudaExtensionSet and one recipe-plus-entry operation per callable, assigned by the actual verified receiving boundary. Require those exact adoptions under guard3 before successful binding, with distinct direct guardian sequences and no root-only success. Continue joining module/spec/delegate tokens to existing completed ordinary records without adopting the same objects again. State exactly how each native create/exec site maps to its recipe, leaf descriptor and corresponding file call, using the existing scope grammar. This can remain one new event plus existing adoption fields; no continuation framework or extra ownership registry is required.

Required proof after repair: production collector/replay tests reject same-recipe and cross-recipe entry exchanges, unrelated adopted-owner substitution, root-only success, late/wrong-guard adoption, and mismatched recipe/operation scope pairs with the independent lifecycle and ordinary file records held fixed. The real positive and partial-adoption cases still preserve roots before fallible work. This counterexample is a static schema analysis, not a claimed execution of a future parser.

## Architectural assessment and non-blocking implementation notes

The proposed direction provides meaningful reduction of complexity. Current code contains a direct loader ledger, load_inline hook installation/restoration, a compilation continuation and duplicated compilation-oriented evidence, a setuptools transaction for import topology/alias/resource/metadata adaptation, and a custom compiler supervisor with libc ABI/pidfd/subreaper responsibilities. Ordinary deployed .so files eliminate the reason those mechanisms are live in the renderer. Existing fixed recipes and source modules make the extraction practical without duplicating kernels. The deployment command still needs build provenance and bounded process closure, which is an appropriate responsibility for a standard systemd service.

The use of BuildExtension is plausible: inspected installed-source material defines TORCH_EXTENSION_NAME and its standard compile handling; load_inline's PCH deletion occurs on its separate load_inline path. Explicit source headers and direct-light pybind bindings must reproduce the old wrapped function ABI, while actual emitted build inputs and numerical acceptance handle compiler-flag differences. The candidate correctly does not claim numerical equivalence merely from unchanged source names. I did not compile any extension or verify the frozen 120-second budget is sufficient.

The immutable artifact manifest has the right direction of trust: trusted launcher selection -> actual source/recipe/platform/output validation -> ordinary environment root/file/module rows -> actual worker module/entry custody. Build logs are provenance, not observed runtime readiness. Reusing immutable sets across fresh workers is a deliberate replacement of the old same-job-only policy, not untracked cache reuse. A compiler installation need not be rehashed inside each rendering worker. No hostile-root attestation is required under the accepted deployment boundary. R1 identifies the necessary terminal-fact addition; otherwise no new manifest framework is warranted.

Ordinary runtime environment v5 permits multiple fixed site roots, metadata files and extension rows. The new cuda_artifacts root can fit this schema; the implementation must enforce the proposed manifest/root/module cross-links and selected bytes before import. Source catalog/policy closure must reflect the new literal imports and remove the actual transitive JIT/setuptools dependencies. Current factories still call load_inline, and LBVH/direct-light still import compilation dependencies or Warp eagerly, so changing only one default loader is insufficient. The candidate explicitly covers those surfaces.

The ordinary external loader is reusable but needs the stated precise retention change. `_ExternalSourceLoader.create_module` currently retains a native return after the delegate call; enclosing that whole call in a target/stream helper without moving retention inside the helper would lose a return when context exit fails. Candidate lines 173–175 expressly require retention inside those contexts and are correct. The external session/registry already holds available roots; no new emergency sync or ordinary unload is authorized after failure. Extension loading after runtime binding and before the existing guard3 DISARMED is compatible with actual `initialize_native_runtime`; it must not move outside that guard or silently extend its deadline.

Factory resolution also has an achievable single route. Current LBVH workspace query and prepare each resolve a module independently through the cached loader; resolving once and passing the same registered object internally preserves the CUB query and ABI. Captured and compatibility paths can share the manifest/name/entry rules and existing kernel/factory logic while only captured preparation requires the exact session capability. One factual wording correction is useful: `CudaDirectLightNativeRenderer.__init__` at `optics/cuda_direct_light.py:295` currently accepts a bare module and does not possess `_PREPARED_EXTENSION_TOKEN`; the replacement must add or explicitly enforce its private preparation guard, rather than assume all three constructors already have that protection. The normative prohibition on caller modules already requires the behavior, so this is a non-blocking implementation note, not a fourth design blocker.

Deletion/migration is properly part of the same review packet. Generic phase constants should move once; historical schemas must retain their meanings under an explicit CPU-only reader boundary and not keep importing obsolete live worker machinery. Current tests/support/native_runtime_process.py contains compilation-specific observer/driver branches; those must be replaced by the fixed-module loading coverage, including actual scope observations and partial returns. Obsolete tests do not justify keeping a JIT fallback. The existing acceleration interop is separately approved behavior and should not be swept away as a generic backend adapter.

The candidate preserves the owner's actual goal and fair boundaries: input validation remains a prefix; no product PREPARE activation or speed comparison follows. C2e Warp timeline/arena, provisional graph, ten actual warmups, distinct final readiness/graph, one seal followed by guarded barrier/DISARMED, C2f correctness/failure acceptance, later C4 stages, joint reclaim/drain and whole-C4 refactor/review remain required. Moving already excluded preparation work cannot itself establish a performance gain. Warp NVRTC/module cache/ModuleExec custody remains an explicit unresolved concrete admission task rather than being mistaken for Python .so loading.

## Verification and limitations

Read-only commands included UTF-8 Get-Content with bounded ranges; rg/rg --files; git status, branch, rev-parse, diff --stat and diff --check; Get-FileHash; and local isolated `C:/Python314/python.exe -I -S -B` with stdlib subprocess/hashlib/pathlib solely to read/hash the Git blob and number candidate lines. The first inline Python hash command had a PowerShell quoting error and did not execute its body; the corrected stdin script succeeded. Some oversized combined displays were truncated and relevant sections were reread in bounded ranges. No candidate changes resulted.

Inspected high-risk source paths: optical_worker/native_bootstrap.py (`_ExternalSourceLoader`, `_ExternalAdmission`, `_RuntimeSetupBinding`, `_RuntimeObservation`, `initialize_native_runtime`, binding consumption and compiler wiring); runtime_environment.py; runtime_custody.py (actual file/adoption/scope grammar); cuda_compilation.py; compilation_contract.py; compiler_supervision.py; optics/cuda_extension_recipes.py; the scene/LBVH/direct-light preparation and loader boundaries; and tests/support/native_runtime_process.py. The supplied accepted parent and predecessor documents were inspected for governing scope, failure/readiness order and the replaced schema/ownership rules; author summaries were not treated as implementation evidence.

The preserved installed cpp_extension source was read independently and its digest matched the earlier installed-source identity above. Primary web sources inspected were pinned systemd v255 service.c/run.c/exit-status.c and PyTorch v2.9.1 torch/cuda/__init__.py, linked beside findings. These establish source-level counterexamples, not observed behavior of a newly launched installed-host service. The coordinator's installed-host observations and suggested fixes were leads only; an incomplete independent SSH call is not promoted to a completed inspection.

The one SSH command was:

```text
ssh -o BatchMode=yes -o ConnectTimeout=8 tileops-dev
  [read-only git rev-parse/status and sed/sha256sum of installed Torch sources]
```

It returned the expected remote HEAD, then produced no further output while polled through the same unified-exec session 81726. No restart, fallback host, build/service command or target interpreter was used. Once source-level review was complete, the coordinator explicitly authorized cancelling this owned read-only inspection. I sent Ctrl-C through that same handle with write_stdin; the tool returned actual terminal exit code 1, no session_id and no further output. This is observed cancellation of the owned exec/SSH call, not successful completion of the requested remote inspection. Its incomplete output is an inspection/transport limitation, not a candidate code failure. No remote worker/service was started; remote utility PIDs were not individually sampled, so I do not claim exhaustive remote process attestation.

No tests, compiler/Ninja invocation, service unit, fault/process experiment, target backend import, native observer/getter, GPU query/reset, package installation, remote write or product entry occurred. Only this outside-repository report was written. The sole cancellation was Ctrl-C to the owned read-only exec/SSH handle after explicit coordinator direction; no unrelated process was targeted. No worker/supervisor/compiler/GPU job was launched by this reviewer. All local tool commands reached terminal results; remote utility PIDs were not individually attested.

Required installed-manager and real-device proof remains the candidate's later bounded preflight/acceptance work, augmented by the concrete R1–R3 cases. No prior real output is reused as evidence of a new run. No finding demands unrelated hostile-root defenses, a generalized build platform, a new PID framework or a separate review project for each helper.

## Scope retained

This report requests a revised bounded replacement design and fresh independent review. It grants no implementation/native execution approval for v1. Existing accepted C2d3 scope is not revoked. C4 completion, CUDA artifact implementation acceptance, Warp compilation admission, C2e/f and activation, final readiness and warmups, C4c3/c4, C4d/B3, product PREPARE/submission, measurement/results/baselines, backend selection and losing-lane cleanup remain under their existing gates. The mandatory whole-C4 systematic refactor and independent acceptance remain unfinished.

The full report may be preserved verbatim as evidence with its exact reviewed identities, findings, limitations and verdict.
