# C2d3 runtime adoption — independent design re-review v2

Initiative: q54-optical-lab-p12-4c4c2d-runtime-adoption
Stage: design-review
Author: codex (fresh independent reviewer)
Version: v2
Date: 2026-10-07
Status: accepted
Related Files: runtime-adoption design-request codex v2/design-review v1; runtime-integration design-request codex v3; source-custody design-request v1/code-review v2; optical_worker/native_bootstrap.py; native_setup.py; native_contract.py; native_input.py; guarded_lifecycle.py
Owner Summary: Approve the exact v2 design for bounded C2d3 implementation in fixed supervised sacrificial drivers only. Previous F1/F2 are closed at the design-contract level. No new substantive finding was established. This is not code approval or evidence that real runtime admission, uploads, multi-device safety or product gates have passed.

## Identity, revision and verdict

Reviewer session: `/root/c4_c2d3_runtime_adoption_design_review_v2`, a fresh independent review session supplied the task packet and prior findings, not the implementation conversation. No subdelegation or candidate mutation.

- Local repository: `C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design`.
- Branch: `implementation/c4-supervised-waits`.
- Base: `ff2209e9fdcb1815a0e17a7dd97c399fc1e536e3`.
- Frozen and actual HEAD: `f668b5b3daa0124b02b474b423bce173e0dc1af4`.
- Previous candidate: `b54e82b34766eafb405b1b34ccb2491797b56617`.
- Active design: `collab/q54-optical-lab-p12-4c4c2d-runtime-adoption__design-request__codex__v2.md`.
- Independently checked design SHA-256: `1a0d3eed4dd182ddd5ea4ad0c19344cad735aabbdd317d77a14aad1932cc0224`.
- Archived previous report SHA-256: `32ae49cf06be44a2ca5d408269d96d28ad908798176d4c8999de22f64ffce729`, matching the packet.
- Initial/final local working tree clean; HEAD and design digest unchanged. `git diff --check` passed.
- Overall delta: only the v1 design, v2 design and unchanged v1 review, 847 added lines. Repair delta from b54e82b: v2 design and v1 report, 547 added lines. No implementation, test, gate, schema or operative collaboration-rule changes.
- Primary remote unchanged implementation: `/data2/shared/ang.gao/robot_sim_c2d2_code_v2`, detached clean HEAD `0dd684867a8d60a69978e878b26d98169d2b9b1c`, checked initially and finally.
- During review the coordinator supplied an optional exact remote design checkout. Independently checked `/data2/shared/ang.gao/robot_sim_c2d3_design_v2`: clean HEAD f668b5b3daa0124b02b474b423bce173e0dc1af4 and matching v2 SHA-256. No tests or writes were performed there by this reviewer.

Verdict: **Approve**, solely for the design implementation scope below. No substantive new finding or unresolved blocking design gap was established. F1/F2 closure is not a claim that their mandatory future runtime tests have already passed.

Read AGENTS.md, collab/README.md and collab/REVIEW_PRINCIPLES.md completely. Inspected the full candidate, prior complete report, v1-to-v2 text diff, predecessor source-custody contract/approval scope, parent runtime-integration setup/failure/source/validation rules, and relevant actual loader/controller/upload/storage/validation call and state flows. Prior approvals and author summaries were treated as requirements/history, not fresh execution evidence.

## Previous F1 — P2: closed in the revised design

Locations: v2 lines 163–196, 330–336 and 369–379.

Fresh source and no-import resolution checks reconfirmed the original trigger. Canonical Go2 `optical_worker/workloads.py:72,81` calls the production MJCF importer; `optics/mjcf_visual.py:41` imports trimesh. Installed trimesh 5.0.0 eagerly imports boolean/collision. Its `boolean.py:15–19` catches the absent manifold3d import, and `collision.py:5–9` catches the absent fcl import. A new isolated stdlib-only PathFinder check found no spec for manifold3d or fcl in the configured site-package root (also none for shapely). No backend or trimesh was imported by that check.

Section 4.1 now defines the previously missing legal outcome, with the policy decisions needed to implement it: literal full requested name, exact requester file/source identity, phase, actual active import context, fixed approved resolver/roots, no pre-existing module, no spec and zero create/exec/body execution. It records an absent_probe before raising the exact ordinary missing-module exception. It cannot fall through to another finder/root or authorize an invented parent. Repeated identical absence is allowed; a changed result is not.

The exception is narrowly pre-execution. Found optional files are deployment drift and reject before execution. Unexpected names, wrong requesters, alternate origins and resolution/audit exceptions poison. A create/exec/body ImportError remains sticky even when dependency code catches it, including an optional-name fault. Section 7 does not turn caught started-import failure into a healthy normal completion. This closes both alternatives identified previously: rejecting an ordinary required Go2 import path, or ignoring all caught ImportErrors to make it run.

The required proof is retained explicitly: both real canonical Go2 positives, complete frozen absence table, wrong requester/unexpected absence/found extra-file negatives, and swallowed started-import failure with actual create/body counters and retained custody. The literal full table belongs to the frozen implementation environment, as already stated; it must not be auto-learned at successful runtime admission. No installation or substitute materializer was authorized or used.

## Previous F2 — P2: closed in the revised design

Locations: v2 lines 94–112, 198–246, 351–360 and 369–376.

Fresh installed-source reads reconfirmed the original distinction:

- `torch/__init__.py:1056–1076` recursively publishes native `_C` module members using `sys.modules.setdefault`; those child modules do not have independent file-loader create/exec histories.
- `torch/__init__.py:2268–2272` publishes the exact `torch._ops.ops` and `torch._classes.classes` objects as torch.ops/classes.
- `torch/_classes.py:19–28,56` and `torch/_ops.py:1426–1439` show ModuleType subclasses and synthetic `__file__` values, which are not real deployment origins.

The v2 contract no longer requires fictitious file provenance for these objects. It distinguishes file modules from created modules, aliases, absent probes and explicitly required namespace packages. A pre-frozen literal publication entry binds exact sys.modules key, trusted creator root, fixed object path, publication phase and kind. Separate creation owner/publisher relationships must both be validated. Name prefixes, hashes, ModuleType checks and runtime enumeration cannot add permission.

The adoption point is explicit: retain the creator root before its body/call, cover unreturned native internals under section 3's process/trusted-dependency boundary, reconcile publication and adopt exact returned objects at creator completion before another repository GPU leaf. Aliases bind already-retained source identity; later checks preserve the same object/creator relationship. Partial failures retain available roots, module delta and original exception without pretending success or ordinary release. Unknown/replaced/extra/wrong-creator entries poison. Available-object scans after failure are retention, never authorization.

This provides the missing attainable custody category and truthful evidence representation. It does not claim that Python can intercept every native child creation or reverse internal native cleanup. Required actual cold Torch positives and replacement/insertion/wrong-creator/partial-creator negatives remain code-approval obligations. Exact table contents and actual publication ordering must be audited against the pinned deployment, not copied from an unrelated Torch build.

## Other high-risk areas checked

1. **One loader and one authority.** Current native_bootstrap admits the twelve CPU files, rejects NumPy/Torch/Warp, and makes failure sticky through the bound controller. It cannot already provide the proposed external admission. The design correctly extends that finder/session, requires new exact policy/catalog pins and frozen transitive repository closure, and forbids a second repository loader or external hook/plugin ingress. Existing CPU prelude versus guardian-first runtime boundary remains explicit.
2. **One upload core with distinct wrappers.** Actual native_setup public functions require NO_SETUP_GPU_WORK, import backends internally and use C0 `_abort`. Both loops already adopt destination before copy and directly block on input-ready event waits. The proposed shared core plus guarded runtime binding addresses those incompatibilities without copying loops, mixing abort policies or arming nested watchdogs. C0 signatures/fresh-epoch/A/K behavior stay mandatory regressions. Exact binding identity, one use, lane/target/stream/session checks and rejection before the next leaf are specified.
3. **Storage authority versus readiness.** Actual native_contract registration freezes array identity and physical allocation facts, including Torch storage/pointers/strides and Warp pointer/capacity/device facts. The current full assertion also requires NativeReadinessEvidence. Lines 285–288 now explicitly require reusing/extracting storage-only validation; they do not authorize minting R_setup/R_final merely to call the old combined assertion.
4. **Validation readback is real work.** Current native_input validation reaches `.numpy()` or `.detach().cpu().numpy()`. Its shared `_validate_native_input_host_observations` accepts completed host observations but does not certify their ownership. Lines 290–296 require actual adopted host readback storage, lane copy, supervised wait and shared CPU comparison, with fault coverage. This is neither a substitute source array nor product host delivery/B3.
5. **Scope facts precede GPU calls.** The design preserves CPU-only descriptors, monotonically growing dirty history, retention before the next leaf and conservative UNKNOWN for unaudited helper scope. Actual Warp Runtime calls wp_init and enumerates devices; its non-quiet greeting reads total_memory. Native `wp_cuda_device_get_memory_info` can temporarily retain an otherwise-unacquired primary context. Torch backend autoload defaults on, and StreamContext reads ambient current_stream. Thus disabling plugins, quiet Warp initialization, explicit target selection and the ambient/foreign matrix are necessary; none is incorrectly credited as complete target-only proof by itself.
6. **Failure ordering and terminal custody.** Actual source-session fail immediately marks the controller and can terminate; actual segment failure forbids normal COMPLETE and selects quarantine. New adapter failure intake must retain available roots/objects and commit conservative scope facts before irreversible terminal selection. This is required by the design, not permission to restore a failed controller. Normal success ends COMPLETE/DISARMED; terminal emergency stays armed through sole target sync, whole-graph release and actual process exit. No retry, second ARM or target-sync repair of unknown/foreign history is authorized.
7. **Trusted native boundary and evidence DAG.** Pre-return native state belongs to the process; native exception safety is explicitly trusted, not a property supposedly enforced by retaining Python modules. Evidence of unsafe native cleanup blocks admission. Runtime-custody records describe actual observations; parsing cannot mint authority or authorize release. They reference earlier source/manifest facts without requiring reverse references or rewriting those facts.
8. **Scope and exit plan.** Only canonical materialization and one lane's a2 input setup/validation/terminal disposal are proposed. No active lease, warmup, optical sample, final graph seal/quiescence or readiness is created. Source custody, runtime binding and selected ownership remain long-term; temporary fixed validation entry and losing backend removal stay governed by P12.4e after equivalent selected-worker tests. No gate or workflow mutation appears in this candidate.

## Verification commands, observations and limitations

Local reads used PowerShell UTF-8 Get-Content/Select-Object, rg, Get-FileHash and:

```text
git status --short
git status --porcelain=v1
git branch --show-current
git rev-parse HEAD
git diff --stat ff2209e9fdcb1815a0e17a7dd97c399fc1e536e3 HEAD
git diff --name-status ff2209e9fdcb1815a0e17a7dd97c399fc1e536e3 HEAD
git diff --stat b54e82b34766eafb405b1b34ccb2491797b56617 HEAD
git diff --no-index <design-v1-path> <design-v2-path>
git diff --check ff2209e9fdcb1815a0e17a7dd97c399fc1e536e3 HEAD
git diff --exit-code ff2209e9fdcb1815a0e17a7dd97c399fc1e536e3 HEAD -- optical_worker AGENTS.md collab/README.md collab/REVIEW_PRINCIPLES.md
```

Identity/scope and protected-path checks matched the packet; diff-check and protected-path diff returned 0 with no findings. No-index displayed the expected design changes and normal Windows LF/CRLF advisory; it was inspection, not a rewrite. Initial root REVIEW_PRINCIPLES.md lookup failed because the authoritative file is under collab; it was then fully read. Initial default-encoding README display was reread with explicit UTF-8. Large combined outputs were split and relevant source/contract ranges reread where truncated. These were read-tool issues, not product failures.

All remote access used `ssh -o BatchMode=yes -o ConnectTimeout=8 tileops-dev` on primary 183; no fallback. Remote git, grep, sed, ps, sha256sum and nvidia-smi were read-only. Dependency root:

```text
/home/ang.gao/miniforge3/envs/robot_sim/lib/python3.12/site-packages
```

A fresh short script supplied over stdin to `/home/ang.gao/miniforge3/envs/robot_sim/bin/python -I -S -B -` inspected installed METADATA, selected source text, process identity and file resolution. Core absence check:

```python
from importlib.machinery import PathFinder
root = '/home/ang.gao/miniforge3/envs/robot_sim/lib/python3.12/site-packages'
for name in ('manifold3d', 'fcl', 'shapely'):
    print(name, PathFinder.find_spec(name, [root]))
```

Each result was None. METADATA reported Torch 2.9.1+cu128, warp-lang 1.12.0, NumPy 2.5.1 and trimesh 5.0.0. The script printed `runtime_roots_loaded []` for NumPy/Torch/Warp/trimesh/fcl/manifold3d and exited 0. This proves current top-level file absence and source facts, not successful runtime admission or a complete immutable deployment inventory.

Separate remote source reads covered Torch native-module publication, ops/classes creation, backend-autoload switch and StreamContext; Warp runtime wp_init/enumeration/greeting and native memory-info context acquisition. The exact remote design checkout digest independently matched local bytes. All completed remote inspection commands returned 0.

Not executed: pytest or any implementation suite; backend imports, CUDA/Warp initialization, allocation/upload/readback, fault injection, native-child runtime enumeration, complete external distribution/native-binary hashing, driver-scope instrumentation, target1/ambient0 restoration tests, full new evidence replay or performance work. This is a pure design review; unrelated code suites are unnecessary, and unsupervised GPU probes were outside authority. Old source-custody tests, including old CPU Go2 positives, were not reused as this round's real-runtime proof.

## Scoped process hygiene

No worker, supervisor, test process, background service or GPU workload was launched by this reviewer. No package installation, process signal, GPU reset, deletion, push or candidate mutation occurred. The only intentional persistent reviewer output is this report outside the repository.

- Initial read-only remote shell: PID 3110373, PPID 3110368, start 2026-10-07 20:16:31 remote; command completed.
- Stdlib inspector: PID 3110744, PPID 3110743, `/proc` start ticks 223973808; SSH command completed with exit 0.
- Final exact `ps -p 3110373,3110368,3110744,3110743 -o pid,ppid,lstart,args` returned only the header. This is scoped absence after successful command completion, not a claim of guarded worker reap semantics.
- Final local inspection PowerShell PID 31124 started 2026-10-07 20:18:24. Same-call Win32_Process query found no ssh.exe/python.exe/python3.exe residual. Short-lived git/read/SSH utility PIDs were not all sampled; no complete OS process ledger or separate security isolation is claimed.
- Initial GPU inventory showed MPS server 2238410 plus unrelated pytest-xdist processes 3096260, 3096278, 3096287, 3096318, 3096293, 3096330 and 3096341. Final inventory showed MPS server 2238410. They were not launched or touched by this reviewer; their disappearance is not attributed to review actions. No global-idle claim is made.
- Final local and both remote checkout checks stayed clean at the exact revisions above. No owned unresolved hang or residual was observed.

## Approval scope and mandatory implementation follow-through

Approval binds v2 SHA-256 `1a0d3eed4dd182ddd5ea4ad0c19344cad735aabbdd317d77a14aad1932cc0224` at HEAD f668b5b3daa0124b02b474b423bce173e0dc1af4. It permits implementing only the specified C2d3 production-owned external runtime admission/custody extension, frozen absence/creator/namespace policy, exact runtime binding, shared guarded a2 upload/storage validation and supervised validation readback/terminal disposal, exercised only by fixed supervisor-owned sacrificial drivers. C0 behavior and existing source/guard ownership semantics must be preserved.

The following are non-blocking for this design verdict because the design already mandates them, but are mandatory before code approval: exact environment/transitive source/native-library and publication/absence tables; reviewed bootstrap policy/catalog byte pins; actual cold guardian-first module/creator observations; real canonical asymmetric/dynamic/both Go2 preparations; both backend upload/readback/storage fault matrices; dual-GPU cold/ambient/foreign/UNKNOWN proof; actual bounded exit/reap and retained-custody evidence; C0/C2d0–2 regressions and unchanged activation gates. No fixture-only positive, skip, installed substitute dependency, same-name object or parsed summary may stand in for these facts. Any required dependency behavior outside the frozen categories returns for scoped review, not wildcard admission.

Still CLOSED: complete C2d; final optical graph, coverage seal/final barrier/quiescence, compilation/warmup/R_setup/R_final and prepared sample authority; C2e/f; C4c3/c4; C4d/B3; product GPU preparation/native sample dispatch; a3b; result/baseline; R2, host payload and S; shipping/activation. **C4 is not complete.** This approval is permission to implement the reviewed design, not acceptance of an implementation or authority to open any of those gates.

The unchanged report may be archived separately under the evidence-only exception, referencing this exact approved parent revision. Such recording cannot alter the design, verdict, requirements, scope or gates.
