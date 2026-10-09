# Exact CPU dependency execution preflight

Initiative: Q54 / P12.4 / C4c2 / CUDA G1
Stage: independent code/execution preflight, dependency-only
Author: codex
Version: v1
Date: 2026-10-09
Status: in_review
Related Files: setuptools-admission design-request/design-review v2; compilation-admission design-request/design-review v3; setuptools-admission implementation-note v3/v4/v5; test-evidence v2/v3
Owner Summary: Review the exact production source/profile and fixed CPU runner before any real setuptools execution. Determine whether one bounded dependency-only run is ready. This is not a whole implementation acceptance, GPU/compiler preflight or higher-gate activation.

## Candidate and governing requirements

Repository `C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design`, branch
`implementation/c4-supervised-waits`. Implementation base:
`52fb36a8960596ddb2ac3decd8d129188e4b1eb4`. The exact candidate is the commit
containing this request and `tests/support/setuptools_dependency_process.py`;
the reviewer packet supplies its full HEAD. Freeze that commit and working tree.
Source/profile delta preceding the runner:
`127f59a51158e03b09e974f5bbf68247719e8345`.

Read AGENTS.md, collab/README.md and REVIEW_PRINCIPLES.md. The unchanged accepted
setuptools amendment v2 SHA256 is
`e8343ed7c9217a47ad45da5f055801cf43537a67f8059240453ed14078e27db8`;
compilation design v3 SHA256 is
`028b52c466ef46433235d4ec2495cd47d00200822cc64dfaea95614fd5bf26a2`.
Section8 item1 of the amendment explicitly requires this dedicated preflight
before actual CPU dependency execution. Prior approvals authorize design only.

Inspect the actual initialization helper, bootstrap file loader/finder/session,
shared compilation contract, environmentv6, source catalog/ASTpolicy and fixed
runner. Inspect parser/collector coupling where it affects these boundaries;
their complete native proof and whole implementation review remain later work.
Check the fixed profile against the actual pinned installed-source archives and
source call graph, including the two lazy roots and excluded platform branches.
No source rewriting or package installation is proposed.

## Exact execution under review

Target tileops-dev, isolated tree
`/data2/shared/ang.gao/robot_sim_cuda_compilation_wip`, target Python3.12.13:

```
/home/ang.gao/miniforge3/envs/robot_sim/bin/python -I -S -B /data2/shared/ang.gao/robot_sim_cuda_compilation_wip/tests/support/setuptools_dependency_process.py parent /data2/shared/ang.gao/robot_sim_cuda_compilation_wip /data2/shared/ang.gao/setuptools-cpu-<approved-head>-v1
```

The final argument must be a fresh absolute directory; mkdir0700 refuses reuse.
The parent verifies the clean revision and uses the existing CPU Peer / real
guard supervisor transport. A single isolated child uses a real source session,
source verification, custody binding, external admission, file/module loaders,
initialization owner, guardian, retention and observations. Its fixed entry is
`session.begin_setuptools_initialization`, then actual `import setuptools` and
`import setuptools.command.build_ext`. No alternate file importer is provided.
No native scope observer, subreaper, signal/pidfd compiler owner or GPU runs.

Explicit outer stand-ins (not production evidence): guards0/1 import the CPU
definitions and configure the real environment/admission; guards2..5 are empty
CPU guards. Phase moves to compiling_cuda_scene at guard6. The compile root's
input record contains descriptive existing token rows, with no real input upload
or validated runtime record. Its collector is an explicit SimpleNamespace with
`toolchain.inventory_checked=True`; it does not represent actual compiler
inventory. No `_Compilation` is created, and no cpp_extension module is forged.
These stand-ins are confined to the test runner and labelled in its output.

Before real dependency imports, a test-only audit hook rejects backend imports,
child subprocesses/fork/spawn, os.system and ctypes.dlopen. The repository source
capture's legitimate git subprocesses finish before that hook is installed.
No compiled third-party dependency is expected on this fixed setuptools path.
The parent has an exact-child timeout/kill/wait fallback (150s; reap10s), while
the CPU guard uses ACK1s and CUDA's existing120s operation budget. This is not a
compiler-tree supervision test; no descendant launch is permitted in the child.

## Observation stop and result criteria

Success requires actual RESTORED (never COMPLETE), setuptools/version83.0.0,
same-object distutils/setuptools._distutils alias, completed ordinary child
sources, real resource digest/read handles and metadata reads, original hooks
restored, singleton finder and exact P+vendor, source/admission/retention audits,
no backend modules and no completed initialization action. Serialize the actual
journal and source/file events under a distinct descriptive CPU schema, explicitly
marking no compilation/native scope/readiness. HOST_OWNER-only termination uses
the real controller with no synchronization callback, then process hard-exit.
The parent waits and writes PID/exit evidence. Failure uses the existing source
failure policy, preserves diagnostics and exits74; it is not counted as success.

This first run does not substitute for the complete negative dependency matrix,
v2 whole-record replay, actual three builds, failures/process/native scope cases,
or later whole-code approval. Counterexamples already exercised by CPU helper/
parser tests remain evidence of those bounded mechanics only. Higher gates,
timing outputs and product result/baseline stay closed.

## Permitted review work and evidence locations

Do not execute the new driver or import installed setuptools/backend packages
during preflight. Source/AST reads and existing stdlib/helper/parser/cold-source
CPU tests are permitted. Python3.14 local; target Python3.12 CPU test launcher
must use -I -S -B, explicit repo/site paths solely for pytest and disabled plugin
autoload. Relevant tests: test_setuptools_admission.py, test_setuptools_replay.py,
test_runtime_environment.py, test_native_source_policy.py,
test_compilation_environment_profile.py, source-only test_native_bootstrap.py.
Do not run the real materialization tests from the full test_native_sources.py.

Raw source archives and one-time authoring/file-resolution proof are in
`C:/Users/AG/新建文件夹/c4_assets`; exact hashes and observed commands are in
test-evidence v2/v3. No actual dependency success is claimed yet. The runner has
only been parsed/compiled without execution. Independently assess deficiencies;
do not infer readiness from the author notes or test counts.

Return a complete report with actual base/HEAD/tree status, commands, findings
and reproduction, exact execution scope approved or denied, limitations and one
of the standard verdicts. Write the report outside tracked candidate paths at
`C:/Users/AG/新建文件夹/c4_assets/setuptools_cpu_preflight_review_v1.md`.
The author will preserve the returned bytes unchanged. Do not modify candidate
implementation or fix it on the reviewer's behalf.
