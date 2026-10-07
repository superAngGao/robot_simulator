# C2d3 Go2 dependency inspection and proposed-table evidence

Initiative: q54-optical-lab-p12-4c4c2d-runtime-adoption
Stage: test-evidence
Author: codex
Version: v5
Date: 2026-10-08
Status: in_review
Related Files: runtime-adoption design v2; metadata-probe gap-scan v1; absent-resolution design v1; runtime-cold-warp test-evidence v4
Owner Summary: Offline source/native-linker and ordinary CPU observations prepared a proposed Go2 table. A newly reproduced missing-spec protocol mismatch requires the separately reviewed supplement. No captured Go2 runtime, new GPU test, deployment-table approval or C2d3 completion is claimed.

## Scope and revisions

Inspected code remained `306842c994228c80cae31dc79ce01f0748d4ee1b` throughout
these observations. Documentation-only candidate `91b11d524f6fc4aad8a2b779327e5c4277cee73c`
then archived the independent gap scan and proposed absence-resolution design.
Both local worktree and remote isolated `/data2/shared/ang.gao/robot_sim_c2d3_binding_wip`
matched and remained clean. The original `/data2/shared/ang.gao/robot_simulator`
stayed clean at `f771e612beaeea126e26926738097512cd2d809c`.

Primary host was tileops-dev / 183.166.183.33:40031, hd03-gpu2-0031, H200.
No101 fallback, package install, dependency edit, GPU reset or process signal.
All observations below are author-side and outside the candidate. They are not
permission added automatically by a production importer.

Local artifact directory: `C:/Users/AG/新建文件夹/c4_assets`.
Remote directory: `/data2/shared/ang.gao`. Exact basenames below identify both
copies unless stated otherwise. Remote interpreter was robot_sim Python3.12.13,
under `/home/ang.gao/miniforge3/envs/robot_sim`.

## Offline source and linker audit

`c2d3-go2-frozen-dependency-audit-v4.json`:
`85b0d6e30df3f45e341e40401e9a1ada3d6734308ea607006cc1341f8f12928c`.

Source `audit_go2_frozen_dependencies_v4.py` (remote prefix `c2d3-`):
`1666479d13e98155833984d8e0502d9ef24d03839d022985c205a6f912fc95de`.
Run started01:33:01+08 under timeout90s, PID3483860 / parent3483858;
completed normally. It used only stdlib/read-only installed-file, ldd and
readelf diagnostics; it did not initialize Warp/Torch or CUDA.

The audit validates900 actual file-module identities against the previously
pinned wheel inventory, plus four non-file entries and four creator-self
aliases. These categories must not be collapsed into904 file modules. It records
static import statements,28 candidate absent-probe branches,139 process-mapped
native binaries, separate standalone linker resolutions and ELF dynamic tags.
Static function-body imports are possible edges, not automatically granted
permissions; explicit final tables still need review.

Installed optional branches were additionally read at their actual import/try
and metadata-probe locations. In particular util.has_module uses uncaught
find_spec, unlike the import/except branches. The two disabled adapters' body
probes are excluded from the28 absent rows: their bodies may not execute.

The single standalone unresolved linker diagnostic was SciPy's bundled
`libgfortran-83c28eba.so.5.0.0` needing `libquadmath-2284e583.so.0.0.0`.
The actual CPU process maps contain that quadmath binary; `_fblas` has RPATH
`$ORIGIN/../../scipy.libs`. The report keeps standalone results separate from
actual process mappings. It does not claim ldd alone proves runtime safety or
change LD_LIBRARY_PATH/install packages to silence it.

Earlier audit v1 failed before writing output because it treated a null-origin
namespace/created module as a file. Audit v2 stopped before output on the
standalone ldd diagnostic. These are author audit assumptions corrected in v3/v4,
not production importer failures or successful runtime admission evidence.
v3 output SHA: `fe6df21d76d80d3c786b462d18a31e833dc6743f9fd96d27ffe0776db13a7be2`.

## Creator and resource observations

Ordinary canonical CPU observer v3:
`c2d3-go2-960-cpu-dependencies-v3.json`, SHA
`ba3a9182dff260c08698f8c92183252d6ad3a0fa659e56402d136f91ef9bac3e`.
PID3478695 / timeout parent3478692, start01:30:43+08, normal exit.

The exact native creator `scipy.optimize._highspy._core` publishes `cb` and
`simplex_constants` during its own exec, not create. Both are distinct modules
whose sys.modules values equal the creator's actual raw attributes. This fits
the existing nonempty-attribute created-module relation. It is ordinary loader
observation, not proof under captured admission.

Ordinary canonical CPU observer v4 source `audit_go2_cpu_dependencies_v4.py`:
`1bb7a4790064266a6cde46ee1cd153c456ab810cddeaed097fdbda01dd2d8fe6`.
Run started01:45:04+08 with timeout90s, PID3497520 / parent3497517,
normal exit0. It adds a read-only Python audit hook for open attempts and records
actual trimesh resource-cache keys; no dependency or canonical implementation
is replaced. It imports neither Warp nor Torch and does not execute a GPU path.

Output `c2d3-go2-960-cpu-dependencies-v4.json`:
`ee48dc09dfb8ddd06f1b171148dd923aa19ee7a22f8186243a36bea7b786ce5a`.
Observed908 module entries,4548 observation events; canonical manifest remained
`ac3d950309917168d0bb5758e9916adcffcbb17bd68f6611e4fe0f3bdfed3a12`.
Actual resource cache contains only `creation.json` and `units_to_inches.json`.
NetworkX's discovered backend table is empty. Open attempts for distribution
entry_points files in this ordinary site-enabled observer are not automatically
required resources, successful reads, plugin execution, or worker admission.
The sealed worker must independently demonstrate its no-plugin property.

## Proposed table, CPU validation only

Outside-candidate file `183_warp_go2_author_v1.proposed.json`:
raw SHA `747e680530d2a02b8de372f7f0c660840b22b4cd3a0e375656a84e82a0898d60`;
canonical SHA `d2b1edc48c5a28a4ef71237fdf16d92d089f02f3fe7926c858d3e0c651146d3b`.

It combines the source-audited CPU closure with the existing25 Warp runtime
modules:925 file-module rules,1010 inventoried files, six distributions,
28 absent probes, two disabled adapters, two namespaces and six publications
(four creator-self aliases and two native-created children). It also pins the
two observed JSON resources. Extra native file records are inventory facts, not
Python import permissions. Existing synthetic fixture is untouched.

The offline author assembler `propose_go2_runtime_table_v1.py` checks exact
input artifact hashes and emits an apply_patch payload, not a runtime updater.
Its first Windows stdout used the console encoding for a Unicode artifact path;
the resulting newly created file was moved to the verified intended directory
and the two empty accidental directories removed non-recursively. Source now
selects UTF-8 output. No candidate or user file was overwritten/removed.

On183, actual RuntimeEnvironmentContract.from_payload followed by
verify_files(expected_sha256=contract.sha256) passed all1010 files in0.727s,
PID3503068 / timeout parent3503067, normal exit0 under30s bound. The process
asserted no NumPy/Warp/Torch/benchmark import. This validates parser/file facts
only, not runtime permissions, complete native execution or cold Go2 success.
The table remains a proposal outside the repository pending independent review.

## Independent gap and unchanged gates

Fresh session `/root/c4_c2d3_metadata_probe_gap_review_v1` independently
reproduced the four current metadata-probe failures and normal importlib tail
fallback behavior. Its unchanged report is archived in91b11d5; it is a gap scan,
not implementation approval. The absence-resolution design is a separate review.

No C2d3 code approval, real Go2 upload/readback, CUDA lane, runtime-custody schema
completion, new baseline or performance measurement is asserted here. Required
whole-checkpoint fault/multi-device/hang matrix remains pending. C2e/f,
C4c3/c4, C4d/B3, product GPU preparation, a3b/results/baselines, host payload and
shipping remain closed.

At01:52:01+08, a scoped `ps -u ang.gao` check showed no Python/test/worker process;
the observer and CPU-verifier PIDs had exited. NVML showed only preexisting MPS
PID2238410 using60MiB. This is no-residual evidence for these runs, not proof
that the historical101 hang is resolved.
