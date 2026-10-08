Initiative: q54-optical-lab-p12-4c4c2d-torch-captured-import
Stage: test-evidence / frozen deployment candidate
Author: codex
Version: v1
Date: 2026-10-08
Status: in_review
Related Files: runtime-adoption design v2; torch-publication design v1 and implementation-gap-review v3; torch-native-scope gap-review v1; cuda-ambient design/review v1
Owner Summary: Actual captured Torch cold import reaches the fixed pre-Stream checkpoint. All 758 declared file modules complete, 42 publications reconcile, seven replacements admit and the exact multiprocessing alias is observed. The diagnostic then deliberately quarantines. This is no Stream/binding/upload/readback success and no C2d3 acceptance.

## Candidate and bounds

Production revision: a3d750bbaf79d760a6727e56146d277e1cb3b770, clean detached
primary183 isolated tree /data2/shared/ang.gao/robot_sim_c2d3_binding_wip.
Local documentation base: 1dc27d79485a80560c9a057b3a330868c584dc4f. All intervening
commits only archive reports and the independently approved ambient clarification.
The new tracked fixture tests/data/runtime_environments/183_cuda_cold_import_author_v1.json
is byte-identical to external author draft v7. Canonical environment SHA256:
2ffb9a81595b60855db8768f2892409c49d32d8ec97d41a3673b18e456492e07.
It is a bounded cold-import candidate, not a full CUDA/Go2 deployment freeze.
No production code, guard budget, source policy or activation changed here.

Environment: /home/ang.gao/miniforge3/envs/robot_sim/bin/python, CPython3.12.13,
Torch2.9.1+cu128, upstream5811a8d7da873dd699ff6687092c225caffcf1bb, primary183/H200.
Fixed captured child uses -I -S -B; plugins disabled before import; target1;
ACK1000ms and operation120000ms. The supervisor and captured session are the
production implementations; the scratch child only observes and injects a fixed
failure before indexed_runtime_stream. No native Stream call or a2 call executes.

The prior publication repair approval binds a3d750b and F1-F4 only (full report
SHA7d0cafe4e4c0a1985203605e195ff90b36b54ad30a51bcd5c827f04643f11b21).
Native cold-scope approval is likewise bounded to this import diagnostic, not
deferred initialization. CUDA ambient design approval distinguishes Torch logical
state from raw CUDA state; it does not certify the unexecuted full native chain.

## Frozen authoring sources and literal changes

All original authoring scripts, drafts and raw artifacts remain in local sibling
c4_assets and /data2/shared/ang.gao on183. Scripts author_cuda_profile_draft_v1.py
through v7.py check exact input hashes. Discovery never modifies worker permission.
The initial draft was assembled from a finite installed-file/source audit, fixed
canonical NumPy closure and mapped native-file inventory. It removed Warp and
added no broad namespace or prefix exemption. The fixture has857 files,758 named
modules,42 publications,7 replacements and8 exact absent probes. Its native-file
list does not yet contain the NVML file observed during ordinary deferred init.

Raw source evidence (SHA256):
- c2d3-offline-wheel-inventory-v1.json: 5e7e954d4c79ce323c9712d97c883d9311dc7182f9fb72891912ab007ab9099a.
- c2d3-torch-frozen-dependency-audit-v1.json: fba561913dceecbb5986debdff9de42d768b64fee0ffdb17c2ea65bdbf100951.
- c2d3-torch-mapped-files-audit-v1.json: 93f037133ea0b425da763f8e348a056098cf14b26e3dba1441d951b09fa05c2f.
- c2d3-torch-dill-sites-audit-v1.json: 80a77febaabb2408a0da25b12789342d80124719a96f1d7f9450ab41ef06c8c7.
- c2d3-torch-publication-sites-audit-v1.json: 4b28c1de7e2d323557e25ade7ec7deda3cb4399b063ea8c24418a3117b4efd3c.
- c2d3-torch-eager-optional-sites-audit-v1.json: 2548c01c7f0c234c20c37182c8743b801258a2b36923c70cc44d315514358bd5.

Original draft v1 incorrectly attributed every native publication to _C and the
ops/classes aliases to their creator files. Source inspection corrected this:
24 immediate publications have publisher_C;12 native attributes are published by
torch's recursive helper at root __init__.py1061/1076; four deferred native init
calls are made by torch.autograd, torch.distributed, torch.distributed.autograd,
torch.distributed.rpc. Creator remains the exact _C root and fixed attribute path.
ops/classes are created in their respective files and published by root torch
at2271/2272. The _dynamo.autograd_compiler key resolves attributes
_dynamo.compiled_autograd; key spelling is not its creator proof.

The eight absent probes preserve exact requester and phase. NumPy's existing
_distributor_init_local is materialization. Runtime rows are dill from
torch.utils.data.datapipes.utils.common; pynvml from torch.cuda; opt_einsum from
torch.backends.opt_einsum; torch._rocm_init from torch; monkeytype from
torch.jit._monkeytype_config; tqdm from torch.hub; pydot from
torch.fx.passes.graph_drawer. Files/roots were checked absent and pinned source
branches inspected before each finite change. TYPE_CHECKING sympy/_typeshed,
ROCm amdsmi, builtin-thread fallback and monkeytype descendants were not added.

## Preserved attempts and final actual prefix

Each artifact below has prefix c2d3-a3d750b-captured-torch-import-diagnostic- in
the two scratch locations. Earlier attempts are failures, not successive passes:

| Run | Draft | Observed result | Parent/supervisor/worker |
|---|---|---|---|
| v2 | v2 | unlisted torch._rocm_init |233142/233148/233177|
| v3 | v3 | missing creator publication at _C return |238104/238110/238118|
| v4 | v5 | unlisted monkeytype |253294/253299/253307|
| v5 | v6 | unlisted tqdm |256403/256409/256414|
| v6 | v7 | fixed intentional pre-Stream quarantine reached |268307/268312/268314|

No draft v4 run occurred: the ops/classes source defect was found and corrected
before launch. Final command on183:
`/home/ang.gao/miniforge3/envs/robot_sim/bin/python /data2/shared/ang.gao/run_c2d3_captured_torch_import_v6.py a3d750bbaf79d760a6727e56146d277e1cb3b770`
Output: /data2/shared/ang.gao/c2d3-captured-torch-import-v6.log, copied locally.
Child c2d3_captured_torch_import_child_v2.py SHA:
327fae1e3f0ec722371b993345f9ef9cdf18afea87e21e52735dde3bb2abb638.
That exact child was inspected in the native-scope independent review.

Final v6 facts:3840 external events;758 each resolve/create-start/create/exec-start/
complete;42 creator_publication;8 absent_probe. Replacement component has14 events
covering7 pending-to-admitted relationships, including parent-before-child
observation and parent-last admission. Exact __mp_main__ alias observed once.
Live external.audit() succeeds at the pre-Stream hook. Checkpoint records
torch_cuda_initialized=false and sample_submission_count=0. The hook then marks
UNKNOWN and raises its intentional RuntimeError; no successful runtime segment,
binding or upload is claimed. The three earlier COMPLETEs are CPU prelude/
configuration/materialization, not runtime completion.

Final actual supervisor:4ARM,3normalCOMPLETE, worker74, observed EXITED/reaped,
no deadline expiry, no kill. Worker:UNKNOWN,true; dirtyhistory[1] upper bound;
2runtime_import_or_init leaves;0emergency sync,0terminal ARM, release not_authorized,
full quarantine. Raw files identity.json and invoker log contain proc startticks
and parent identities (wrapper268306). Scoped ps for268306/268307/268312/268314
returned none after completion. NVML listed only pre-existing MPS2238410,60MiB;
it was untouched. No ordinary_atexit file exists.

Final v6 raw SHA256:
- runtime_initialization.json: dd3dcc2000353a1b5656793729168a2f05d0e1af55dd3835227debea21079f97.
- captured_import_checkpoint.json: 189dc04341b5ca92442ebd84936db6567987d4b65ea8afa7d1da5c00c1277038.
- custody_artifact.json: 4ea09730c054b5b3a7baf53acab470e1916eb7614895c43b04520b1c1de08b22.
- supervisor_stdout.json: 8b0208e416814860f788e5bc25be177302ccb590a1fded358a36cfeaaa94fa15.
- worker.json: e3598b9267f808a58fae3f5a265f6188bdb99711bbbdce898616b21f2721860e.

check_captured_torch_prefix_v1.py runs the existing native-global, replacement,
interpreter-alias and guarded custody validators on these actual components,
with complete import/replacement prefix required, plus exact file/publication
set comparison. Executed on pinned183 Python:passed. This remains component
replay, not the unfinished aggregate runtime-custody schema. Attempting to parse
this Linux deployment on local Windows correctly rejected POSIX absolute paths;
that is an environment mismatch, not a pass or a production defect.

## Remaining work and operational notes

The ordinary pristine v4 diagnostic separately observed before:_initializedfalse,
rawCUDA0,all8primarycontextsabsent; afterStream:logicalTorch0,rawCUDA1,onlycontext1.
It was UNKNOWN/quarantine from the outset and cannot supply captured success.
Native-scope and ambient reports retain its full provenance and limitations.
Source reasoning is still required for transient side effects and native cleanup;
snapshots and ELF initializer counts alone cannot prove their absence.

The next obligations are deferred native call-chain scope, nested device/stream
regression, actual captured Stream/upload/readback and full frozen deployment,
both-lane canonical/replay/both Go2, failure/hang/dual-GPU/terminal matrix, final
versioned runtime-custody aggregate and independent whole-stage review. None is
waived by this prefix. All later gates remain closed; C4 stays active.

Read-only source collection earlier had one SSH quoting error that launched an
unintended recursive grep in the author's remote home. Its exact owned grep244827
(parentbash244826) was identified and only that grep was terminated; scoped checks
confirmed both gone. No other process was killed. Fixed Python scripts were used
thereafter. All diagnostic workers/supervisors and CPU source-audit jobs from this
round have exited; historical failure artifacts and drafts remain intact.
