# C2d3 Torch dependency observations and replacement gap

Initiative: q54-optical-lab-p12-4c4c2d-torch-publication
Stage: test-evidence
Author: codex
Version: v1
Date: 2026-10-08
Status: draft
Related Files: runtime-adoption design v2; native-global-publication design v1;
native_bootstrap.py; runtime_environment.py
Owner Summary: Ordinary supervised Torch import and pinned source inspection
identify required dependency facts and a seven-module replacement incompatibility.
This is discovery evidence, not captured runtime admission, an environment
allowlist, target-only scope proof, a2 upload or C2d3 acceptance.

## Revisions and bounds

Local clean HEAD b88dbf1d5bf0a94951f3d0ecaa2bb4bdefd3ddc9; remote isolated
/data2/shared/ang.gao/robot_sim_c2d3_binding_wip clean at code revision
8bc53bfe25acff9d384a3536d249c0e3268538d0. The local delta is only the preceding
bounded native-global review archive. Primary tileops-dev was reachable.
No original-worktree change, package installation or fallback host was used.

The scratch child uses ordinary import machinery, not the captured finder.
It reuses NativeGuardedSetupLifecycle and the fixed existing supervisor via a
scratch wrapper that replaces only the fixed child script path. The original
1000 ms ACK and 120000 ms operation deadline are unchanged. A retained host list
precedes import; UNKNOWN is marked before Torch. TORCH_DEVICE_BACKEND_AUTOLOAD=0,
worker flags -I -S -B, explicit installed site path without site.main/.pth.
The available post-import module roots are appended to the retained list.
Intentional failure quarantines this list and hard-exits74, without normal
COMPLETE, emergency ARM/sync, ordinary release or atexit.

The diagnostic does not wrap each dependency constructor, observe each import
permission, or establish complete native ownership. Its lifecycle contains one
HOST_OWNER list and zero tracked device leaves. Neither zero tracked leaves nor
Torch's false _initialized flag proves no native/device side effects. UNKNOWN
is the conservative scope, and process supervision is the bounded terminal owner.

## Execution and immutable artifacts

Command (existing output directory must not be reused):

```sh
timeout 190 /home/ang.gao/miniforge3/envs/robot_sim/bin/python -I -S -B \
  /data2/shared/ang.gao/run_c2d3_torch_dependency_probe_v1.py
```

Parent42024, supervisor42027, child42034. Supervisor observed child exit74 and
wait/reaped it; supervisor and invoker exited0. One normal ARM, zero COMPLETE,
no deadline expiry or kill. Terminal reason input_eof, worker full_quarantine,
UNKNOWN true, release not_authorized, emergency sync not_attempted. No ordinary
atexit marker. A subsequent scoped ps found none of these three PIDs. No PID
start-time sample was captured for this short diagnostic; do not infer one.

NVML before and after listed only pre-existing MPS2238410 at60 MiB. This is a
residual observation, not a proof of absence of short-lived native contexts.
GPU0 UUID GPU-a4b256c9-c56a-0ef5-19fa-482196cc3274; GPU1 UUID
GPU-3a3112ff-837c-3b87-38d1-cb203d656589. Eight H200 were enumerated read-only.
No unrelated process was touched.

All script basenames below reside in /data2/shared/ang.gao and matching local
C:/Users/AG/新建文件夹/c4_assets. Diagnostic output directory is
/data2/shared/ang.gao/c2d3-8bc53bf-torch-dependency-diagnostic-v1, with a byte-copy
in local c4_assets.

| Script | SHA-256 |
|---|---|
| c2d3_torch_dependency_probe_child_v1.py | 5f953ae46216dc69b5fe3012253c13cbe02a1014e7dd43d9f266080ceb43c55e |
| c2d3_torch_dependency_probe_supervisor_v1.py | d455e7f3f21fc86bd19df62392cb2dc49a1c1b916368fcb4c789cd21f23aba81 |
| run_c2d3_torch_dependency_probe_v1.py | ea62e04f39344fdf154ac95257d02331627159c13da911174d27caa04895b268 |
| inspect_torch_replacements_c2d3_v1.py | f254835c9fd98f427739b6a6676bccb6a45bc4bd0b5a1702bc57808b99555e33 |
| audit_torch_frozen_dependencies_v1.py | 4c1873627d200ca06ec186335302373ddabe0dbead3eb60bfc76120cbbb54bbc |

| Diagnostic artifact | SHA-256 |
|---|---|
| request.json | e493bde020b217fb4b5eae87df3e5cf31ccb5e2e194151253c40198541d6694b |
| identity.json | e11ee3f88cf1f827779d3743fd89a909e36bd5893ae1abc1a749b8855719aac9 |
| ordinary_torch_dependency_observations.json | 4f76956c9baf2ad319b145f46b9efb5e39cca1541b25f84cd1bf62f44325a7a6 |
| diagnostic_failure.json | ac66be36248969ddd508352bded883e8537c5b2ffb7522215404ca1c5bb59b6f |
| worker.json | e783ba536bb7defb65d0a18dcab46a8ab503fa61fb708d2c8b9af9668d6daa71 |
| custody_artifact.json | b7b59fb88b005838b34cbb3b1f2df81bf3c3f363895c6de5f25ea036c03fa4a9 |
| supervisor_stdout.json | 4446bb26cb7aa0154162db3d5a491ccdcb208c49919cb5064eed318f0bf4e6e9 |
| supervisor_stderr.txt | e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 |

## Source and linker observations

Ordinary census observed956 new sys.modules entries, including709 Torch names,
84 NumPy names,4 torchgen names and typing_extensions.40 exact native attribute
paths were found by raw vars traversal from torch._C, including the non-name-derived
path [_dynamo, compiled_autograd] for torch._C._dynamo.autograd_compiler.
torch.ops/classes had exact identity with raw torch._ops.ops/_classes.classes.
These are observations for a future literal table; runtime enumeration grants
no permission, and no native creator timing was inferred from this census.

Seven additional objects have no ordinary final spec/loader/file metadata:
torch._VF and torch.backends with cudnn, miopen, mkldnn, opt_einsum, quantized.
Pinned source explicitly assigns a new ModuleType subclass to each same-name
sys.modules key. Six wrappers keep their original file module in raw m;
VFModule keeps torch._C._VariableFunctions in raw vf instead. torch.backends
replaces itself before importing its child packages. Existing file-module
identity checks reject this behavior. A separate bounded contract and independent
review are required; fake file provenance or a name/prefix exemption is invalid.
The census also contains multiprocessing's __mp_main__ alias, requiring separate
checking against the trusted stdlib boundary.

Read-only audit command:

```sh
timeout 100 /home/ang.gao/miniforge3/envs/robot_sim/bin/python -I -S -B \
  /data2/shared/ang.gao/audit_torch_frozen_dependencies_v1.py
```

Python48564/wrapper48563 exited0; scoped post-check found neither. It verified
actual ordinary-module/replacement source bytes against wheel inventory SHA
5e7e954d4c79ce323c9712d97c883d9311dc7182f9fb72891912ab007ab9099a, parsed possible
AST imports/dynamic calls/sys.modules assignments, and inspected4 ELF roots
plus35 ldd-resolved files. All standalone ldd diagnostics resolved. It did not
import Torch or run site startup. Its output is
/data2/shared/ang.gao/c2d3-torch-frozen-dependency-audit-v1.json, SHA
fba561913dceecbb5986debdff9de42d768b64fee0ffdb17c2ea65bdbf100951, matching local
copy.749 ordinary file modules,7 replacement source files,42 nonfiles.

Possible AST branches are not executed edges; ldd is not actual process-map
proof or complete dynamic native-loading closure. No environment profile was
created from these observations. Full C2d3 still requires frozen/source-audited
CUDA closure, captured actual upload/readback on both lanes and canonical/Go2
cases, faults/hangs/dual-GPU evidence, the aggregate runtime-custody schema and
whole independent review. All later product and comparison gates remain closed.

## Supplementary ordinary observation v2

A second exclusive diagnostic used the same protocol/UNKNOWN-before-import and
unchanged candidate, adding raw original/class/native-value relations and one
/proc/self/maps snapshot. Invoker run_c2d3_torch_dependency_probe_v2.py,
parent58788/supervisor58792/worker58794. The worker exited74 and was reaped;
supervisor/invoker exited0. One ARM, zero COMPLETE/sync/release, no deadline
expiry/kill or atexit marker. Scoped ps found none of those PIDs. Post-run NVML
listed existing MPS2238410 plus non-owned Python54178 at798 MiB; neither was
touched. No global-empty-GPU claim. PID start times were not captured.

All six raw m values are distinct original modules, and type(exposed) equals
the respective original's raw class attribute. Their raw package/file/path
metadata is null or absent. VF's raw vf equals raw _C._VariableFunctions, whose
actual type is torch._C._VariableFunctionsClass, not ModuleType. This ordinary
probe still does not retain/observe the original VF file module. The81 mapped
.so-like paths are one process snapshot, not a complete captured lifetime or
deferred initialization audit. Directory and local copy use diagnostic-v2.

| V2 script/artifact | SHA-256 |
|---|---|
| c2d3_torch_dependency_probe_child_v2.py | 292aa2e373a56773b47ca13c2d8f10a7859a1885d82595385ddccb8a19d6e3c2 |
| c2d3_torch_dependency_probe_supervisor_v2.py | 38c58bedf9e66576c10540d6ccd44ca4fb9705aac944aba7f20261f536a77c3a |
| run_c2d3_torch_dependency_probe_v2.py | 73f60cd4a828c49e74c6c7cbaa5e2ef697d299426303c50351b8ee81be060f70 |
| ordinary_torch_dependency_observations.json | 0d17823a3b78c5fe87c4d307db9afc1775443c17b08661a383abc77e4ce8b6a1 |
| identity.json | 7b0b1f7466c7c393ced938d3661fe40060297f95f6f5f87a3a9c753f7c597a30 |
| custody_artifact.json | 8898bd6f89c342a75a23596fbd44c5b7177ba3221daf7a66e192f7a692fbea73 |
| supervisor_stdout.json | 33c656c2c785f061472f50aa6d7c4ce6060c164ef5281142c53024893f5796da |

V2 request, intentional failure, worker lifecycle and empty stderr have the same
bytes/hashes as v1; they are newly written files from the v2 execution. The full
independent gap report was archived verbatim from final SHA
e8b0d0b8b9472231f23ca3a5290747ec25ce0b80e19abc5c7f311faad3725a12.
It identifies G1/G2 (replacement contract) and G3 (stdlib alias) and issues no
design/implementation/whole-stage approval.
