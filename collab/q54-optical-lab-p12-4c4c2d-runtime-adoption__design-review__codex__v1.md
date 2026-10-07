# C2d3 runtime adoption — independent design review v1

Initiative: q54-optical-lab-p12-4c4c2d-runtime-adoption
Stage: design-review
Author: codex (fresh independent reviewer)
Version: v1
Date: 2026-10-07
Status: in_review
Related Files: runtime-adoption design-request codex v1; runtime-integration design-request codex v3; source-custody design-request v1/code-review v2; guarded-custody design-request v2/repair-design v2/reviews; optical_worker/native_bootstrap.py; native_setup.py; native_contract.py; native_input.py; guarded_lifecycle.py
Owner Summary: Request changes for two P2 external-admission contract gaps exposed by the installed dependencies: expected absent optional imports in the required Go2 path, and Torch-created modules that have no file-loader create/exec history. No implementation or GPU/product gate is approved. Review was read-only except this outside-repository report.

## Identity and verdict

Reviewer session: `/root/c4_c2d3_runtime_adoption_design_review_v1`, a fresh independent session given the review packet, not the author conversation. No delegated review or candidate edits.

- Repository: `C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design`.
- Frozen base: `ff2209e9fdcb1815a0e17a7dd97c399fc1e536e3`.
- Actual HEAD: `b54e82b34766eafb405b1b34ccb2491797b56617`.
- Sole candidate delta: `collab/q54-optical-lab-p12-4c4c2d-runtime-adoption__design-request__codex__v1.md`, 300 added lines.
- Independently verified candidate SHA-256: `3d61e39bf5552f96ba090ab9a7e4036032e11f8118acff625e719e09d6cbd1d5`.
- Initial and final local working tree: clean; final HEAD/hash unchanged. `git diff --check` exited 0.
- Primary remote source checkout: `/data2/shared/ang.gao/robot_sim_c2d2_code_v2`, actual clean HEAD `0dd684867a8d60a69978e878b26d98169d2b9b1c`, initially and finally. It supplies the unchanged implementation, not the local design delta.
- Verdict: **Request changes**. F1 and F2 below are blocking design clarifications, not claims that a new implementation has failed a runtime test.

Read AGENTS.md, collab/README.md and collab/REVIEW_PRINCIPLES.md completely, with a UTF-8 reread after the initial default-encoding display garbled Chinese. Inspected the candidate, predecessor source-custody contract and approval scope, guarded-custody design/repair and review requirements, parent runtime-integration setup/failure/source/validation gates, and actual relevant implementation call/data/state flows. Prior approval and author inspection claims were not substituted for evidence.

## F1 — P2: define expected absent optional lookups separately from poisoned imports

Location: candidate lines 119–130 and 138–145, line 223, and required Go2 positives at lines 263–265.

The design rejects unlisted external modules/loaders and makes an importer error swallowed by dependency code poison the session. It does not define a legal outcome for an expected optional module lookup that finds no file and never enters create_module/exec_module. The actual canonical Go2 path requires that distinction in the supplied environment.

Concrete source chain:

1. `optical_worker/workloads.py:72,81` imports/calls the canonical `optics.mjcf_visual.import_mjcf_visual_scene`.
2. `optics/mjcf_visual.py:41` imports trimesh, rather than a replaceable diagnostic materializer.
3. Installed trimesh 5.0.0 `trimesh/__init__.py:12–16` eagerly imports `boolean` and `collision`.
4. `trimesh/boolean.py:15–19` tries `from manifold3d import Manifold, Mesh`, catches BaseException and installs exception wrappers. `trimesh/collision.py:5–9` tries `import fcl`, catches BaseException and sets `fcl = None`.
5. A fresh read-only `python -I -S -B` stdlib PathFinder inspection of the approved robot_sim site-packages root found no spec for `manifold3d` or `fcl` (also none for `shapely`). This did not import any of those packages or NumPy/Torch/Warp/trimesh.

Trigger/reproduction: run the proposed cold guarded canonical Go2 materializer with that environment. Under the literal blanket rule, an ordinary absent optional probe must either be rejected as unlisted or become the swallowed import failure that permanently poisons the session. Thus the mandatory normal Go2 positive cannot complete. Alternatively, an implementer could silently ignore all caught ImportErrors to make it pass; that would also exempt dangerous failures after a native create/body has started and violate the stated custody rule. These are materially different policies, not a detail that can be left to the implementation.

Existing checks do not settle this: full distribution hashes do not distinguish an expected absent lookup from a started importer failure; mandatory Go2 tests reveal the contradiction only after implementation. Earlier CPU Go2 success did not run through this new external-admission policy. Installing optional dependencies, substituting a materializer or counting a skip is expressly not the repair authorized here.

Proposed direction: freeze a narrow optional-absence policy in the design. If permitted, it should identify the fixed expected requester/name/context, prove resolution ended before any delegated create/exec or dependency body, record an explicit absent outcome, and forbid fallback to other roots/hooks. Unlisted probes and every failure after admitted execution/creation begins must remain sticky, including when swallowed. Another valid direction is a separately authorized deployment/dependency change, with the required positive path kept blocked meanwhile. Do not broadly whitelist ImportError by exception class.

Required repair proof: both real Go2 materializations succeed through the actual cold captured path with the documented optional-absence outcomes; an unexpected missing name rejects; a module found outside the approved roots rejects; and injected create/exec/body ImportError—even for a permitted optional name and even when caught by the dependency—still quarantines with original exception/owners held and no COMPLETE/new ARM/sync. Verify absence means zero module-body/native-create execution, not merely a successful caught exception.

## F2 — P2: define custody and audit for dependency-created modules without a file loader

Location: candidate external-module resource class at lines 84–86; standard-file-loader/origin policy at lines 119–139; runtime-custody record at lines 238–247.

All external modules are described as having an exact spec/loader and create_module/exec_module sequence. The installed Torch creates actual module objects and puts them directly into sys.modules; those objects do not enter the proposed finder/delegated-file-loader sequence. The design needs an explicit successful owner/audit category for them. Otherwise a strict origin/ledger audit rejects required Torch behavior, or the implementation makes an unreviewed exception and pretends missing create/exec observations are complete.

Concrete source evidence in installed Torch 2.9.1+cu128:

- `torch/__init__.py:1056–1076` defines and immediately calls `_import_extension_to_sys_modules(_C)`. It recursively discovers native-created module members and inserts them with `sys.modules.setdefault(member_name, member)` at line 1072; its own comment names `_C._dynamo.eval_frame` as an example. There is no independent file-loader create/exec call for each inserted member.
- `torch/__init__.py:2268–2272` imports objects `ops` and `classes` from their owning file modules, then directly registers them as `torch.ops` and `torch.classes`.
- `torch/_classes.py:19–23,56` creates `classes = _Classes()`, a types.ModuleType subclass initialized as `torch.classes`. Its class-level `__file__ = "_classes.py"` is not a resolved deployment origin, and the object has not been created by a spec's loader. `_Ops` likewise subclasses types.ModuleType (`torch/_ops.py:1426`).

Trigger/reproduction: follow the unconditional normal Torch import body above. These sys.modules entries already exist before any later `import torch.classes` can reach the finder. Hashing only the executed parent file does not provide a missing module-level create/exec observation, nor decide how to validate this object's identity/origin. This is source-confirmed normal behavior, not a hostile in-process injection claim and not a runtime probe result.

The phrase “actual dependency owners” does not close the contract: it does not specify when a directly published object joins custody, which trusted creator establishes its origin, how a no-spec object is represented, or which ledger event replaces nonexistent create/exec outcomes. Trusting third-party native internals can justify a creator-root ownership boundary; it does not make an invented loader identity truthful. The current C2d2 loader only knows its captured file-module records and cannot supply this missing category unchanged.

Proposed direction: explicitly distinguish file-loaded external modules from a fixed, dependency-created module/alias category. Bind the latter to an exact retained trusted creator/root and the observed object identity, with a defined adoption/audit point before subsequent repository GPU leaves. Define legitimate aliases and null/not-applicable loader/spec/origin fields, partial-creator-failure custody, and replacement/extra-entry rejection. Do not permit arbitrary ModuleType objects or sys.modules additions merely by a torch name prefix, and do not fabricate a create/exec sequence for them. If direct module publication is instead forbidden, revise the dependency/implementation plan rather than approving an infeasible Torch path.

Required repair proof: actual cold Torch admission records these installed native-created and Python-created module objects with their correct creator ownership and no fictitious loader events; exact replacement, unlisted insertion and wrong-creator aliases reject; failure during creator execution retains the available root/objects and original exception through terminal disposition. Include the new category and legal outcomes in the design's evidence contract so code review checks an agreed policy rather than inventing it.

## High-risk areas checked without a further blocking finding

- The existing `native_setup` functions really start from NO_SETUP_GPU_WORK, import their own runtimes and catch failures internally through `_abort`. The proposed shared upload core plus separate C0 and guarded wrappers addresses a real incompatibility; wrapping the existing full functions after separate runtime initialization would not suffice. Destination adoption precedes copy in both existing loops. No copied upload implementation is authorized.
- Existing guarded source `fail()` is sticky and reaches terminal custody; guarded segments arm before yielding and complete/disarm only after success. A failed normal segment selects quarantine with no emergency retry. New adapter failure intake must preserve this ordering, including recording conservative UNKNOWN facts before irrevocable terminal selection where required; the proposal does not authorize bypassing it.
- The prepared storage registry freezes actual Torch storage/pointers/strides and Warp pointers/capacity/device facts, not just manifest labels. Its current full authority assertion also requires readiness. C2d3's no-R_setup/no-R_final rule means implementation must reuse/extract the storage-only validation without manufacturing readiness. This is an implementation obligation, not evidence that current public helpers already implement the new entry.
- `native_input.validate_prepared_native_input` performs actual readback through `.numpy()` or `.detach().cpu().numpy()`. Its shared `_validate_native_input_host_observations` explicitly supports already-adopted synchronized buffers. Future validation must keep real readback leaves/owners under supervision, not mistake a CPU comparison API for a no-GPU path. No readback was executed in this review.
- Installed Torch's backend-autoload switch and Warp's quiet-mode requirement are real source facts. Warp Runtime initializes device metadata and its ordinary greeting queries total memory; native `wp_cuda_device_get_memory_info` may retain a primary context. The proposal correctly distinguishes enumeration from context work. Quiet mode alone was not credited as a proof that the complete chain is target-only.
- Torch StreamContext observes the ambient current stream; both backends defer initialization. The proposed mandatory target1/ambient0 and unknown/foreign histories remain necessary. No implicit target-only scope, normal restoration success or absence of native constructor cleanup was inferred from API names or module retention.
- The process-owned pre-return native state and explicit trusted-native exception-safety obligation are honestly bounded. A Python guardian cannot undo native cleanup that already happened. Failure evidence of unsafe native cleanup must block admission, as the candidate says.
- Normal upload readiness is not final graph readiness. Terminal sync remains under the one emergency deadline through release and actual exit, with no emergency COMPLETE/DISARMED. The proposal leaves product dispatcher, samples, graph seal/quiescence and later gates closed and preserves the one-way evidence chain and cleanup obligations.

## Commands, observations and limitations

Local inspection used PowerShell UTF-8 Get-Content/Select-Object, rg, Get-FileHash and these Git checks:

```text
git status --short
git status --porcelain=v1
git rev-parse HEAD
git diff --stat ff2209e9fdcb1815a0e17a7dd97c399fc1e536e3 HEAD
git diff --name-status ff2209e9fdcb1815a0e17a7dd97c399fc1e536e3 HEAD
git diff --check ff2209e9fdcb1815a0e17a7dd97c399fc1e536e3 HEAD
```

Identity/scope checks match the packet; the separately executed diff-check returned 0 with no output. No candidate code/test/schema/workflow file changed.

All remote access used `ssh -o BatchMode=yes -o ConnectTimeout=8 tileops-dev` on primary 183. No fallback. Reads used git, sed, grep, find, ps and nvidia-smi; no remote files were written. The installed source root is `/home/ang.gao/miniforge3/envs/robot_sim/lib/python3.12/site-packages`.

Two short stdlib-only inspections were piped on stdin to:

```text
/home/ang.gao/miniforge3/envs/robot_sim/bin/python -I -S -B -
```

Core reproducible no-import check:

```python
from importlib.machinery import PathFinder
root = '/home/ang.gao/miniforge3/envs/robot_sim/lib/python3.12/site-packages'
for name in ('manifold3d', 'fcl', 'shapely'):
    print(name, PathFinder.find_spec(name, [root]))
```

Observed each result `None` in both runs. The first also read dist-info METADATA: Torch 2.9.1+cu128, warp-lang 1.12.0, NumPy 2.5.1 and trimesh 5.0.0. It found trimesh's expected SourceFileLoader spec. The second printed numbered source excerpts supporting F1/F2. Both explicitly printed `runtime_roots_loaded []` for NumPy/Torch/Warp/trimesh/fcl/manifold3d and exited 0. PathFinder inspected top-level file availability only; it did not load a backend or assert runtime correctness. These are current deployment observations, not the complete frozen environment manifest required for code approval.

Initial remote rg was unavailable, so grep/sed were used. That first command also lost grouping quotes through Windows SSH, causing shell command-not-found errors; a later grep pattern had invalid escaping. Those read-command failures were corrected with fixed-string or separate patterns and bounded reads. Large initial combined outputs were split where relevant text was truncated. These are inspection-tool issues, not product failures or acceptance runs.

Not executed: pytest, GPU imports/initialization, CUDA/Warp allocation/upload/readback, fault injection, dual-device tests, full dependency closure enumeration, native binary/driver audit, complete new evidence replay, or performance tests. Pure-design review did not mandate code suites, and unsupervised GPU probes were explicitly excluded. The counterexamples are static dependency/call-chain facts plus fresh no-import file-resolution observations. Neither parent nor historical test outputs were claimed as this round's runtime proof.

## Process work and round-end scoped hygiene

No worker/supervisor/test processes, background services or GPU work were launched. No installs, process signals, broad cleanup, GPU reset, push or candidate mutation.

- Initial read-only remote shell PID 3053811, PPID 3053783, start 2026-10-07 20:05:47 remote, completed. The first short stdlib inspector's PID was not sampled; its SSH command completed with exit 0.
- Second stdlib inspector PID 3080646, PPID 3080645, `/proc` start ticks 223931152, command exited 0. Final exact `ps -p 3080646,3080645,3053811 -o pid,ppid,lstart,args` returned only the header. Command completion plus this scoped absence was observed; no supervisor reap semantics are claimed for these source readers.
- Local inspection PowerShell PID 33932 started 2026-10-07 20:09:57. Same-call Win32_Process enumeration found no ssh.exe/python.exe/python3.exe residual. Short-lived read/git/SSH utility PIDs were not all sampled; this is not a complete OS process ledger or a claim of separate OS isolation for the reviewer agent.
- Initial remote Python inventory included unrelated language server 2587008 and another user's transient pre-commit processes. Final named Python inventory retained only the unrelated language server. None were touched.
- Initial GPU inventory contained MPS server PID 2238410. Final inventory also contained other users' active/idle pytest-xdist GPU processes: 3069162, 3069170, 3069175, 3069179, 3069185, 3069198, 3069203, and a sampled 3058628. The first seven were confirmed under UID 232071, parent 3068989, starts 20:08:06–07, not descendants of this review's source inspector; 3058628 was absent at follow-up. All were left untouched. No global-idle claim is made, and their existence is not an owned review residual.
- Final local and remote candidate checks remained clean at the revisions above. No unresolved owned hang was observed. This report is the only deliberate persistent output, outside the repository.

## Approval scope and remaining gates

No C2d3 implementation is approved by this Request changes verdict. Repair the external-admission contract and obtain a fresh independent design review for the new frozen document before implementation. Exact environment inventory, repository closure/policy pins, shared-core refactor, runtime binding and mandatory supervised real-device matrix remain future implementation/code-review work, not passed evidence.

Still closed: full C2d; final setup graph/coverage seal/barrier/warmup/R_setup/R_final; C2e/f; C4c3/c4; C4d/B3; product GPU preparation and native sample dispatch; a3b; result/baseline; all R2, host payload and S; shipping/activation. C4 is not complete. Existing C0 A/K and prior CPU approvals retain their exact original scope.
