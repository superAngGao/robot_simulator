Initiative: q54-optical-lab-p12-4c4c2d-native-global-publication
Stage: implementation-note
Author: codex
Version: v2
Date: 2026-10-08
Status: in_review
Related Files: native-global-publication implementation-note v1 and implementation-gap-review v1/v2; approved native-global design v1; runtime-adoption design v2
Owner Summary: Repair remaining poisoned-evidence and same-frame replay findings. A separate supervised stack diagnostic located repeated configuration decoding and completed one instrumented actual Go2 Warp path; it is not acceptance. Cache only immutable validated configuration to remove that repeated work while preserving all live checks. Clean uninstrumented both-lane validation and whole C2d3 acceptance remain outstanding.

## Review and repair scope

Independent fresh reviewer `/root/native_global_implementation_gap_v2` reviewed
clean `9d1d7d025a4798cef8e7aff6318c530320ef7736`, closed the original four exact
counterexamples, and issued Request changes for two remaining evidence defects.
Full unchanged report is archived at `ed6c9bf`, raw SHA-256
`b4ac2229dbe3d9b43e9452ba9e128e9c02559ceccd3d5c97e8f4a71550832514`.
This is still bounded implementation gap review, not C2d3 code approval.

V2-F1 repair: serialization checks the issued frame registry, private global
bindings and immutable event facts even after poison. It never rebuilds a
missing event. Healthy module presence/metadata checks remain separate from
held-history checks so legitimate failed prefixes survive importlib unwind.
Tests enter failure through issuance, audit or another import, then mutate
tokens/erase/reorder history or alter frame facts and repeatedly request output.
Legitimate pre-return and post-return failure prefixes retain the exact issued
pending token without new admission; terminal stdlib remains available.

V2-F2 repair: request-entry observations cannot follow a known return of the
same frame. Return observations require returned/completed/failed terminal
snapshots, never started. Multiple globals at one return, same-frame requests
before return, ordinary nesting and unrelated sibling orders remain legal.

The source policy and active catalog are repinned. No deployment row, permission,
watchdog budget, optical preparation gate or final selection rule changes.

## Separate actual stack diagnostic on the previous candidate

The uninstrumented Go2 v10 attempt on9d1d7d0 timed out at materialization120s;
its exact artifacts/PIDs/hashes are preserved in independent gap-review v2.
An additional diagnostic used the SAME frozen remote checkout and actual
repository supervisor/driver via these outside-checkout launch wrappers:

- `/data2/shared/ang.gao/c2d3_stack_diagnostic_supervisor_v1.py`, SHA-256
  `979758fd23cfc59fbd2dbe38fc1b42fc7005be82de2b543d44c8ec1a35d7e857`.
- `/data2/shared/ang.gao/run_c2d3_go2_stack_diagnostic_v1.py`, SHA-256
  `2fc6514300853fabdb23cfd76d332fd44d50b67d65c1998c722e532d867461d7`.

Command:

```text
timeout 420 /home/ang.gao/miniforge3/envs/robot_sim/bin/python -I -S -B /data2/shared/ang.gao/run_c2d3_go2_stack_diagnostic_v1.py
```

The supervisor wrapper loads the existing supervise implementation unchanged and
adds only a stdlib faulthandler/runpy prelude to its actual fixed child command.
Ten-second stack dumps go to a separate file, not the IPC or stderr pipe. This
perturbs startup/module baseline and is explicitly DIAGNOSTIC ONLY: neither
normal acceptance nor performance evidence, even though actual code runs.
No alternate importer, preparation or cleanup implementation, extra permission,
deadline extension, package change or process signal is introduced.

Artifacts:
`/data2/shared/ang.gao/c2d3-9d1d7d0-go2-960-stack-diagnostic-v1`.
Local byte copy is under `../c4_assets/`. Parent/supervisor/worker PIDs:
4161835/4161850/4161851. Original ACK1000ms/operation120000ms, Warp target1,
real frozen Go2 960x640. Worker exited0; supervisor EXITED, six normal
completions, seven ARMs, no deadline expiry/kill. Materialization segment took
113.526s from authoritative supervisor trace; this is an operational diagnostic
duration, not a render/performance-gate result.

Twelve periodic stack samples: seven in repeated environment.to_payload()
decoding/canonical-value validation from admission.check_live(), one in lifecycle
replay during host adoption, three in canonical scene/input/BVH construction,
one in source compilation. This directly identifies repeated immutable-config
decoding as a substantial setup cost without claiming exact profiler percentages.

Actual captured diagnostic facts: the three declared native-global rows were
observed and admitted under the exact configured enclosing exec intervals. First
requests were numpy.random.bit_generator (cython_runtime), numpy.random._common
(_cython_3_2_8), and scipy._cyutility (_cython_3_2_5). The first differs from the
ordinary probe's zlib request; request_name is observed data, not a frozen creator
claim. Each object's issuer token/frame remains consistent in the retained
events. Real native upload/readback validated42 arrays. Manifest/content/topology
digests match the frozen canonical Go2 values. Worker records42 allocations,
42 H2D and42 D2H copies, two event create/record/wait operations each, five
runtime-import/init leaves, one terminal target sync, guardian RELEASED and
release_status=completed. No final execution readiness or sample was issued.
These diagnostic observations require an uninstrumented fresh rerun before
acceptance claims. Both-lane and complete parent matrices remain required.

Scoped ps for the three owned PIDs after exit returned only its header; NVML
listed only pre-existing MPS2238410 at60MiB. No other process was touched.

| Raw file | SHA-256 |
| --- | --- |
| request.json | 3a2324fff5acae37f354e9ed8ecc495beed8cd2574550c359a1fa6ec67350077 |
| identity.json | a745b885e967283d2def65a50ea8a6d4981a5b13ad34733ad16e339f557f53c7 |
| materialization_stacks.txt | c7caabb1947f0a8e482b48c4c801e33afbbfcf307e643e37327411b8668036a2 |
| canonical_materialization.json | c9ee9d14d1c6caf7d4823f9c440c7fd76874a819376498ae7dbf4684479e338e |
| input_validation_report.json | f76ba232ba326d720de31334fdab6608a265449b379a15adca1e5303e9eb9e20 |
| worker.json | 86ae50d868f76f85fadfd20419ca21cc697aba9e7bb949d4be504de2662046ac |
| supervisor_stdout.json | cdf852afb02e558742e78eec411c559c7ad394a9337338c0c64483950bdfa939 |
| custody_artifact.json | b60525596427e104fda4bce8855a4be67f5c406798b220a558b0e0fe89bbe72f |

## Equivalent immutable configuration work

The Go2 configuration contains925 modules and1010 files. Each check_live formerly
decoded/revalidated/re-encoded the full same immutable bytes before comparing
the mutable payload/indexes. `_immutable_environment_payload` now uses the
existing bounded metadata-cache pattern, keyed by COMPLETE canonical bytes,
returning only recursively immutable tuples and scalar values. Every public
to_payload call creates fresh mutable containers. Mutating returned dictionaries
or rebinding encoded cannot poison/reuse another configuration's validation;
new bytes are decoded and structurally validated. No cached mutable object is
returned. No file, source, module, loader, spec, event, session, scope, guardian,
execution or release observation is cached. Existing per-call payload/index
comparisons, bound digest checks and full verify_files audits remain unchanged.
This is an implementation cost reduction within the existing contract, not a
new permission policy or exemption from an audit. Its exact semantics and
negative tests require independent review together with the evidence repairs.

Local repaired CPU mechanics:260passed/3skipped across native-global publication,
runtime environment and import-admission files. Skips are not proof. Further
source/cold checks and the exact next frozen candidate are recorded separately.

Still missing: uninstrumented successful captured Go2, full both-lane canonical/
upload/readback/failure/hang/dual-GPU/terminal matrix, aggregate versioned runtime
custody schema and whole-stage independent review. C2d3/C4 remain incomplete;
C2e/f, C4c3/c4, C4d/B3, product GPU preparation, a3b, result/baseline, host payload,
R2 and shipping remain closed.
