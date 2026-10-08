# Native scope observations and aggregate custody: bounded gap request

Initiative: Q54 / P12.4 / C4c2 / C2d3
Stage: gap-request (not implementation permission)
Author: Codex
Version: v1
Date: 2026-10-08
Status: in_review
Related Files: runtime-adoption design-request v2 sections6-8;
cuda-ambient design-request/review v1; captured-cuda-ambient gap-review v1;
runtime-custody implementation-plan v3 and gap-review v3;
runtime-custody test-evidence v1 and implementation-note v2
Owner Summary: Check the remaining relationship between required real native
scope observations and the approved descriptive aggregate. Current v1 records
join imports, owners, storage and lifecycle but do not carry the raw/logical/
context/stream observations required by the earlier ambient decision. Determine
the smallest honest completion path before changing the production schema.

## Actual state and scope

Production aggregate implementation is unchanged from
`9b5702b4016824ea7fc5b5e7c108c5adf02a1bd1`.
Its plan v3 was independently design-approved at
`ab24a7fe6619362875596fd8c3694c0756b5ef4c`
(the authoritative plan digest is
`6dc1fa68fef830b5c5f631128cedb78a35f79ee31767879424945550fa483beb`).
There has been no independent code approval or whole-C2d3 acceptance.

The record's exact fields and event kinds are in
`optical_worker/runtime_custody.py`, with actual producers in
`optical_worker/native_bootstrap.py`. The live source/controller/guardian and
registered-storage authorities remain separate. Parsing never mints authority.
The emitter already supports immutable validated cuts, held failure tails and
one final attempt; older raw diagnostics are not accepted records.

The CUDA ambient design section4 explicitly requires descriptive scope
observations in the final versioned runtime-custody aggregate, cross-bound to
actual guard/import/runtime order. Its logical/raw distinction and ban on
unaccounted device0 context work remain binding. The current v3 aggregate plan
mentions historical ambient evidence but has no event/field for its actual
native values. This request asks the reviewer to verify whether that is an
unclosed requirement, not to accept a claimed failure from this summary alone.

## Existing actual evidence

* Historical positive CUDA ambient run at production `a3d750b`, independently
  preflight-reviewed:267 observations,132 actual helper leaves, logical0/raw1,
  only primary context1,42-array upload/readback, one terminal sync and reap.
  `c4_assets/c2d3-a3d750b-captured-cuda-ambient-author-v1` and the captured-
  ambient-and-terminal evidence v1 identify its full raw files and scripts.
  It used v0 diagnostics, not the new v1 aggregate.
* Actual new v1 records at9b5702b cover both lanes' four canonical inputs and
  six Warp normal/terminal cases. Local and remote archive
  `c2d3-9b5702b-runtime-custody-evidence-v1.tar.gz`, SHA256
  `eb08d4ed62f4271596a7b8a1e982f6cd7c2b7f3ab8ec7488319c6bb9aeab5510`.
  They prove their observed upload/readback/custody outcomes, not new raw ambient
  observations. Runtime-custody test-evidence v1 records limits and provenance.
* At4435593 four actual Warp/CUDA backend exec fault cases pass after repairing
  a missed Warp injection. The prior10pass/1failure attempt is preserved.
* Atbcc1bac the CPU/source/supervisor regression passes65 in172.65s after a
  stale12-module fixture was updated for the approved16-module closure.
  A70-case native/ACK/EOF matrix is currently executing on the clean frozen
  remote bcc1bac candidate. It is not complete and is outside this gap verdict.

Current accompanying test-only WIP adds11 collected, **unexecuted** cases:
real foreign stream return, exception after foreign stream creation (both
lanes), Go2 native-global pending/admitted faults (both lanes), CUDA module
replacement/stdlib alias observation faults and captured C1 stream-only copy.
These deliberately negative calls are not normal-scope admission or evidence.
The original initialization handler, not a test-written scope flag, must set
UNKNOWN after the pre-return exception. C1 observes actual context0 appearance,
marks the observed scope unknown and then enters the original failure path.
No production schema or native observer capability is added by those tests.

## Proposed minimal direction to assess, not yet implemented

Prefer extending the existing observation journal with one explicitly typed,
CPU-only native-scope event if required, rather than inventing another runtime
authority or duplicating lifecycle/storage facts. The actual fixed test driver
would append primitive observations it obtained under the original live ticket.
The collector itself would perform no GPU calls, dynamic import, getters,
callback execution or extra storage audit.

Any accepted design must freeze exact fields/enums and joins before code:

* CUDA values distinguish Torch logical ordinal, raw cudart ordinal, actual
  primary-context flags and current target raw stream. No common boolean
  "ambient restored". Before the first stream, the logical value is explicitly
  source-derived; no warming logical/current-stream getter is introduced.
* Warp values distinguish its CPU-selected default device, actual current
  driver context, actual primary-context flags and the already-created target
  stream. A cached Python `_context is None` alone is not native absence proof.
  Do not call `wp.set_device(0)` to establish a positive fixture: installed
  set_device calls Device.make_current and would touch the foreign context.
* Bind each occurrence to original lifecycle event/guard, actual runtime module
  and existing runtime binding where available. Before/after helper pairing
  must reference the real indexed leaf, retain actual returned owners before
  fallible observation/adoption, and introduce no new ARM or deadline renewal.
* Preserve contradictory snapshots as observations, not rewritten success.
  Existing source/controller poison and validated-cut selection govern errors.
  Parser validity must not be confused with an ambient acceptance verdict.
* Old v1 records and diagnostics must retain their historical meaning. Decide
  explicitly whether an additive event in the still-unaccepted v1 plan suffices
  or a schema revision is required. Do not silently migrate old execution files.
* If an outer descriptive attachment to the existing record is sufficient under
  the governing requirement, prefer it only with precise upstream/guard/prefix
  joins and failure-cut behavior. A free-floating JSON dump or matching count
  cannot stand in for the required binding. No cyclic hash into source/manifest.

This is a gap scan, not a request to approve unspecified payload fields. If the
requirement needs an amendment, the author will write the bounded exact design
and obtain independent design review before production changes.

## Warp observation feasibility and restrictions

The installed183 Warp1.12.0 sources are under
`/home/ang.gao/miniforge3/envs/robot_sim/lib/python3.12/site-packages/warp`.
`_src/context.py` Runtime initialization (around5280-5310) registers device
metadata and sets default cuda:0 when no driver context is current. Quiet mode
avoids greeting memory queries. get_stream(explicit1) obtains the actual target
stream. Device metadata registration and the native call chain must still be
reviewed; these statements are not an executed Warp ambient proof.

The existing profile already pins
`/usr/lib/x86_64-linux-gnu/libcuda.so.595.71.05`, SHA256
`76e0d9678d41cf6b6ae71d18549d88a963d3d434f08108baa12457eb53ca88d6`.
A possible fixed observation after real Warp init, before target stream, uses
RTLD_NOLOAD of those already loaded exact bytes and typed driver state queries;
no setter, primary-context retain, private context, reset or synchronization.
The ctypes library/functions would be strongly retained before later work.
Getter failures would poison the original segment. No GPU probe of this
proposal has run. Its native side effects require explicit review, not an
assumption based on a function's name.

Official references inspected by the author (not execution proof):
[primary context state](https://docs.nvidia.com/cuda/archive/12.3.1/cuda-driver-api/group__CUDA__PRIMARY__CTX.html)
describes flags and active/inactive state;
[current context](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-driver-api/group__CUDA__CTX.html)
describes a null result when none is bound. Inspect the pinned Warp wrapper/native
source and exact needed driver calls when judging this proposed observation.
The native exception-safety/trusted-dependency boundary remains unchanged.

## Requested review and closed gates

Fresh independent read-only gap review: identify substantive unmet requirements,
check whether the existing aggregate/evidence already closes them, evaluate the
minimum honest direction above, and state what exact design/native scope proof
is required before implementation or a new positive Warp observation attempt.
Do not author candidate fixes or run new GPU work. CPU repros and read-only
remote inspection may be used, with exact commands/results/limits preserved.

This scan does not approve whole C2d3, the running matrix, new native probes,
rendering, C2e/f, C4c3/c4, C4d/B3, product GPU preparation, a3b, result/baseline,
host, R2 or shipping. C4 remains the continuous execution objective.
