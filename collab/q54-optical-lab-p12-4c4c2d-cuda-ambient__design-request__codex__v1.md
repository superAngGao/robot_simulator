# C2d3 CUDA ambient selection: scoped runtime state and raw CUDA state

Initiative: q54-optical-lab-p12-4c4c2d-cuda-ambient
Stage: design-request
Author: codex
Version: v1
Date: 2026-10-08
Status: in_review
Related Files: runtime-adoption design-request v2 section6; torch-native-scope gap-review v1; torch-publication design-request v1
Owner Summary: Resolve the native-scope review's blocking ambiguity before full CUDA initialization/upload validation. Define restoration at the selected runtime's scoped API boundary, record the raw CUDA difference explicitly, and retain the absolute ban on unaccounted foreign context work. This draft grants no execution or later-stage approval.

## 1. Problem and bounded decision

The approved runtime-adoption design requires explicitly indexed operations and
ambient restoration through runtime scoped operations. Its term "ambient" does
not distinguish PyTorch's logical device from the CUDA runtime's current device.
The independent native-scope gap review v1 identifies this as F2, blocking the
full target1/ambient0 acceptance claim. Its separate cold-import diagnostic
approval does not decide this issue.

At installed Torch2.9.1+cu128 / CUDA12 on primary183, a pristine ordinary
`torch.cuda.Stream(device=torch.device('cuda', 1))` produces:

| Boundary | Torch initialized | Raw cudaGetDevice | Torch logical device | Primary contexts |
| --- | --- | --- | --- | --- |
| Before Stream | false | 0 | 0 implied by inspected cold C10 state, not queried | none of eight |
| After Stream | true | 1 | 0, observed | only device1 |

The source explains this behavior. CUDA12 `MaybeExchangeDevice(0)` uses C10's
thread-local targetDeviceIndex when device0 has no primary context; it avoids
`cudaSetDevice(0)`, which would initialize that foreign primary context. C10's
GetDevice reports logical0 afterward. No raw0 restoration occurred.

Proposed decision: for the CUDA lane, "restore ambient through runtime scoped
operations" means restoring **the selected Torch runtime's logical device and
scoped stream selection** on normal return. It does not require an absent
foreign primary context to be created to restore raw cudaGetDevice to0.
Raw current device/context remains a separately recorded fact. This is an
explicit interpretation of that requirement, never a claim that logical0 and
raw0 are the same thing or that a raw restoration test passed.

This applies to the sealed, single-selected-runtime setup worker and the
explicitly indexed CUDA preparation operations already authorized by C2d3.
It does not authorize unindexed raw CUDA calls, mixed-runtime implicit current
state, a backend bridge, or a change to physics publication/lease requirements.
The user-facing product goal and all performance/fairness gates are unchanged.

## 2. Retained hard requirements

1. HELLO's configured target remains the only device on which normal setup may
   acquire a primary context, allocate, create streams/events or queue work.
   Existing scope declarations happen before the corresponding calls. CPU
   bookkeeping for other ordinal slots is not mislabeled as device work, and
   actual foreign context creation is never mislabeled as bookkeeping.
2. Every device-bearing operation in the approved CUDA setup chain must use an
   explicit target/device/stream or an audited runtime scoped operation that
   selects the same target. Subsequent native APIs cannot assume raw ambient0.
   A newly introduced call that uses raw implicit current state needs its own
   source/scope review; it does not inherit permission from logical restoration.
3. No `cudaSetDevice(0)`, device0 stream/allocation fixture, private context,
   CUDA_VISIBLE_DEVICES remapping, device reset or extra foreign synchronization
   may be added to manufacture the positive target1/ambient0 case.
4. UNKNOWN and foreign-device history remain irreversible. The existing
   controller prohibits further normal work, normal COMPLETE, target sync and
   ordinary release after compromised scope. A later correct logical value or
   target-only snapshot cannot erase earlier foreign/unknown work.
5. A raised, partial or hung initialization is a failure, not an alternate
   restored state. Preserve the original error and all available owners; use
   existing quarantine/ACK/watchdog/exit/reap policy. No retry or recovery follows
   from an observer returning the expected logical device.
6. Warp's required target1/ambient0 case remains separate. This CUDA decision
   neither approves Warp's runtime behavior nor changes its device-scope rule.

## 3. Evidence and observation contract

The real supervised captured case must identify the actual pinned interpreter,
Torch/native files, allocator configuration, ordinal/UUID mapping, source
session/lane/HELLO target, and worker/supervisor PID/start/parent identities.
It must use the production captured source/import/runtime binding/upload path.
Fixed fault/observation selectors remain outside the production source closure.

Before the first actual Stream call, establish a cold process with no explicit
device0 fixture, record raw cudaGetDevice0 and all primary contexts absent, and
record Torch's raw `_initialized` false. Do not call Torch `_cuda_getDevice` or
`torch.cuda.current_device()` first: both can trigger lazy initialization and
would destroy the promised pristine ordering. The expected logical0 before
Stream is a source-derived conditional fact from cold C10 state plus raw0;
label it as derived, never as a fabricated observed native return value.

The approved ordinary v4 observation method is a candidate for the fixed real
test: use the already loaded hash-pinned libcudart with RTLD_NOLOAD and a typed
cudaGetDevice call, and query primary-context state through the audited pinned
entrypoint. Observe contexts after the raw getter and assert `_initialized`
remains false. Each observer's possible side effects need the same native scope
audit as other calls; a function name containing "get" grants no exemption.
No observer may load a different library or perform an unaccounted setter.

After the real Stream has been held before metadata inspection, record its
actual device1 and identity, all primary-context states, Torch logical0 and raw
cudaGetDevice1 separately. Record the exact source rationale for virtual
restoration. Do not report a common scalar "ambient restored=true" that hides
which API returned which value. The normal-chain checks must also establish
explicit target use through allocation, copies, event/ready wait and readback,
and logical/scoped selection restoration on each normal return. The sole setup
terminal synchronization remains against the exact target under the existing
protocol. No renderer or measured sample runs in this stage.

Before any pre-existing target stream exists, a positive pristine test must not
call a current-stream getter that creates one just to construct its fixture.
Once runtime binding has legitimately established the target stream, any
scoped-stream restoration comparison uses that actual held runtime state and
audited runtime getters; it does not infer stream identity from an integer
ordinal or claim physical completion from restored selection.

Snapshots alone cannot exclude transient native context creation/destruction.
The proof remains the inspected pinned call chain plus the approved trusted
native dependency obligation and the actual observations. This supplement
does not invent an impossible Python sandbox or waive unsafe native cleanup.
Any contradictory side effect stops admission and requires a new disposition.

## 4. Acceptance and negative cases

The captured pristine target1/ambient0 case passes this bounded restoration
requirement only when the ordering above is actual, logical device returns0,
the explicitly recorded raw state matches the audited scoped semantics, every
normal device leaf stays on target1, and foreign/UNKNOWN history is absent.
The target stream and all prepared arrays must retain the existing identity,
layout/content/storage/readiness checks. Parseable observations grant no live
module/runtime/storage/release authority.

Reject or quarantine, as dictated by the existing guard, if a pre-observation
Torch getter already initialized CUDA, device0 gains a context after guardian
creation, logical selection fails to restore, an unexpected raw-state change
contradicts the audited call chain, an implicit native call escapes target1,
the observer/library identity drifts, or initialization fails before return.
Deliberate foreign-context and UNKNOWN cases remain separate mandatory real
negative tests; they cannot be averaged into a successful normal case.

Include descriptive observations in the final versioned runtime-custody
aggregate cross-bound to actual guard and import/runtime order. No cyclic
prerequisite on earlier source or manifest hashes, and no new runtime token or
second lifecycle state authority. The precise aggregate schema remains the
implementation obligation of approved runtime-adoption section7; this draft
does not substitute an isolated restoration report for it.

## 5. Evidence, review and exclusions

Implementation code base: a3d750bbaf79d760a6727e56146d277e1cb3b770. Current
evidence-only predecessor:3932387b00b30644b9c2f29588ebcfb835125a1b.
This draft changes no production code. A new independent design reviewer must
check the chosen boundary against the overall native/fairness contract, actual
source and raw evidence before implementation or full Stream validation claims.

Independent native-scope report SHA256
5c20a33fbafddcf9dcb8c916f772a85a50ab4fc896367fd8c5ccefbaae11d8f3
is archived adjacent. Pristine ordinary evidence directory:
`/data2/shared/ang.gao/c2d3-a3d750b-torch-pristine-init-diagnostic-v4`;
scope observation SHA256
1b7def31a6c142ef239647a8e9d56d6315f2cd43dc37c98aa295159d3a365cfa.
Worker216938 returned74 and was reaped; parent216922/supervisor216928 exited,
scoped OS checks found all absent; only pre-existing MPS2238410 remained.
UNKNOWN throughout,1ARM,0normalCOMPLETE,0sync,no ordinary release. This is
ordinary diagnostic evidence, not a captured or a2 success.

Source manifests and installed binary/header evidence are identified in that
complete report. [Exact C10 device functions](https://raw.githubusercontent.com/pytorch/pytorch/5811a8d7da873dd699ff6687092c225caffcf1bb/c10/cuda/CUDAFunctions.cpp)
explain the logical/raw distinction; [CUDA12 documentation](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-runtime-api/group__CUDART__DRIVER.html)
describes context initialization by cudaSetDevice. These sources do not prove
the full captured lifecycle or grant execution authority by themselves.

No complete CUDA initialization/upload/readback, either full lane, both real
Go2 workloads, failure/hang/dual-GPU matrix, final aggregate or C2d3 acceptance
is asserted. C2e/f,C4c3/c4,C4d/B3,product GPU preparation,a3b,result/baseline,
host,R2/shipping remain closed. C4 remains active.
