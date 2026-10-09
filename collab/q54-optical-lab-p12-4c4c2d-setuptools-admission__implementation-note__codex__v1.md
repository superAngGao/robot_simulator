# Setuptools design approval and shared vocabulary implementation WIP

Initiative: Q54 / P12.4 / C4c2 / CUDA G1
Stage: implementation WIP and review/evidence handoff
Author: codex
Version: v1
Date: 2026-10-09
Status: draft
Related Files: setuptools-admission design-request/design-review v1-v2; compilation-admission design v3; systematic-refactor implementation-note v1; optical_worker/compilation_contract.py; tests/unit/optics/test_setuptools_contract.py
Owner Summary: The bounded initialization design passed fresh independent re-review after fixing ambiguous helper-role evidence. Implementation has begun with one shared immutable vocabulary and exact helper operation anchors. The actual initialization owner, environment v6, loader/resource/metadata integration and complete collector/parser are still pending. This is neither dependency success nor compilation/C4 acceptance.

## Exact approval and new code

Design v1 at b05f707b24eadae688299c1685c3ee47ed720601 received Request changes:
R1 showed that generic helper adoptions did not distinguish originals, wrappers
and delegates. V2 adds a closed operation table, source-event joins without
duplicate adoption, distinctness, permitted sharing and per-read occurrence rules.
It also clarifies namespace children and the CPU dependency proof's stop boundary.

Fresh reviewer `/root/setuptools_admission_design_v2` approved design v2 at
`d62da69a7d71ac1dafb92e1054630a39b934bd0a`, SHA256
`e8343ed7c9217a47ad45da5f055801cf43537a67f8059240453ed14078e27db8`.
The full unchanged report is archived in evidence-only commit
`2a745c69ebb363541efbc7b60757666b36cf635d`, report SHA256
`890497588c7f86c13952d6d4d7474630246cbab7b8610f4c34745a57d5fb03ae`.
It approves design only. The original Request-changes report is also unchanged
and tracked; it is not retroactively an approval.

Current code after that archive adds the fixed action field sets and helper
operation table to existing inert `compilation_contract.py`. Alias action repeats
resolve to the same anchors; metadata/resource bytes have different anchors;
resource handles use canonical integer occurrences. Ordinary file tokens and
failed partial tokens explicitly cannot receive a new helper role. No second
module loader, guardian or token registry was added. No live transaction uses
these definitions yet, and existing compilation/runtime schema versions are
deliberately unchanged until their actual implementation is ready.

## Local CPU verification

Command, with PYTEST_DISABLE_PLUGIN_AUTOLOAD=1:

`C:\Python314\python.exe -B -m pytest -q -p no:cacheprovider tests/unit/optics/test_setuptools_contract.py tests/unit/optics/test_cuda_compilation_custody.py tests/unit/optics/test_cuda_compilation_codec.py`

Result:18passed,11skipped in0.93s, exit0. All11skips are the existing codec suite's
explicit POSIX-only gate on this Windows host. They are not completed codec tests.

New tests use the actual NativeWorkerResourceGuardian to adopt15 distinct helper
objects with production anchors and pass its real records to the existing
_RuntimeReplay.adopted join. All105 pairwise helper exchanges reject with those
adoption/owner facts fixed. Duplicate adoption rejects without replacing the held
owner. Alias repetition/read reuse and ordinary/failed-owner misuse are tested,
including boolean/negative/noninteger occurrence rejection. This is actual local
guardian/join behavior using descriptive event fixtures, not a guarded setup,
new parser, transaction, source-session integration or target-native result.
Existing two actual compile-root retention tests also pass. git diff --check passes.

## Supplemental source/interface facts

Scratch `../c4_assets/setuptools_reader_design_probe_v1.py`, SHA256
`ab1c504fd4dc2e0743adeaf23ad6e17cbca9f8771e524fe0e61dc6b3ed8fa742`, ran with
`-I -S -B` on local3.14.0 and primary-host robot_sim3.12.13. Its proposed reader
interfaces returned the exact resource bytes and version83.0.0. Metadata Context
was exactly {'name':'setuptools'}, with read_text calls METADATA then PKG-INFO.
These are toy interface feasibility results; they do not prove production custody,
read-only handle enforcement or real dependency import. The initial scratch failed
because importlib.abc.ResourceReader is removed in3.14 and its toy finder blocked
lazy stdlib email.parser; corrected to importlib.resources.abc.ResourceReader and
standard-only delegates. Neither was a product failure or dependency execution.

Source-only `cuda-compilation-packaging-sources-v1.zip`, local ../c4_assets and
remote /data2/shared/ang.gao, has22files/90837bytes, SHA256
`52db25f3a79ef2df681dc4c49149d788610237ce9527c10899f3f496ff72bbde`.
All22size/digests match the previously frozen offline wheel inventory. It supplies
the actual site packaging26.1 sources for later closure authoring; vendor packaging
is not silently substituted. No packaging, setuptools, Torch or Warp was imported.
The source collector's isolated target sys.path was, in order:

- /home/ang.gao/miniforge3/envs/robot_sim/lib/python312.zip
- /home/ang.gao/miniforge3/envs/robot_sim/lib/python3.12
- /home/ang.gao/miniforge3/envs/robot_sim/lib/python3.12/lib-dynload

All SSH/CPU commands returned synchronously. No compiler, backend, subreaper,
native observer, worker/supervisor or GPU process was launched. The approved design
candidate stayed frozen throughout both independent reviews. Implementation began
only after the second review returned approval.

## Required next work

Implement environment v6 with exact pinned descriptor and alias/namespace parent
validation; the actual source-session initialization owner and topology/shim/alias
state machine; finite resource/named metadata integration; compilation-v2 journal
and source/lifecycle/retention joins using the shared vocabulary. Author the full
explicit frozen source closure before the original input cut. Obtain the dedicated
dependency-only exact preflight, then real dependency positive/failure evidence,
respecting its CPU stop before full COMPLETE. No fake cpp_extension completion.

Then complete the original compilation implementation/collector/parser/process
cases and renewed exact source/catalog/runner/observer/ABI preflight before real
native builds and faults. Source pins remain intentionally stale WIP. Whole C4,
its later stages, real regressions and systematic refactor/fresh acceptance review
remain the active goal; this vocabulary is not a substitute delivery objective.
