# C2d3 generated sysconfig module: bounded independent gap scan v1

Initiative: q54-optical-lab-p12-4c4c2d-runtime-adoption
Stage: independent sysconfig delegation gap-scan
Author: codex, independent reviewer
Version: v1
Date: 2026-10-08 (Asia/Shanghai)
Status: in_review
Related Files: runtime-adoption design-request v2; absent-resolution design-request/review v1; Go2 deployment-table review v1; optical_worker/native_bootstrap.py; optical_worker/runtime_environment.py
Owner Summary: Request changes for the bounded generated-interpreter-module gap. The failed Go2 attempt encountered an installed, interpreter-generated stdlib module outside CPython's static name lists, not an absent third-party dependency. A narrowly reviewed design supplement is required before extending delegation. No whole C2d3 code approval or new gate is granted.

## Verdict, identity and scope

Verdict: **Request changes**, limited to the generated sysconfig delegation gap and the design decision needed to repair it. This is not a whole-code review and does not revoke the earlier literal Go2 table's permission to attempt guarded validation. That review explicitly left captured closure unproved.

Reviewer session: `/root/c4_c2d3_sysconfig_gap_review_v1`. Fresh independent reviewer, not author of the candidate. Read AGENTS.md, collab/README.md and collab/REVIEW_PRINCIPLES.md. Prior reports and author observations were treated as claims to verify.

Frozen candidate HEAD on both machines: `89e89c15e1bcbca2ec1171d6814a9d4e9f478b83`.
This is a snapshot gap scan, not approval of a base-to-HEAD implementation diff.

- Local: `C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design`.
- Remote: `/data2/shared/ang.gao/robot_sim_c2d3_binding_wip`, primary `tileops-dev`, observed hostname `hd03-gpu2-0031`.
- Both working trees were clean at opening and closing checks; HEAD remained unchanged.
- Bootstrap SHA-256 on both: `517696c9ad170b612da90f4e47a5fd794a6c66f699d0f78b4682a8d54258f2ee`.
- Parent runtime-adoption v2 SHA-256: `1a0d3eed4dd182ddd5ea4ad0c19344cad735aabbdd317d77a14aad1932cc0224`.
- Absent-resolution request v1 SHA-256: `91c0b70ade9f0f3ec5c5b8f3ad222facf300436d2c8d09368f61947eec022cfb`.

## Independently established failure chain

The four requested artifacts were parsed directly under
`/data2/shared/ang.gao/c2d3-89e89c1-go2-960-guarded-author-v1`.

| Artifact | SHA-256 |
| --- | --- |
| external_failure.json | `2401dde57398cd803f049b688769e203131b10f69d9835d0c3a2488ac20abf99` |
| failure.json | `2e85fc9cecc90e479051e0647ab417a952068fc8d74c67ad8fd74cf722fc5696` |
| custody_artifact.json | `d7db839cfd17bb651ac176d1f022fbeff925bc75689f6538db24a87b84772b7a` |
| request.json | `d75019d6b8888c7424ca671ed4a83b63d852a3dcea5082ba2a7b47c77e5191b1` |

The request identifies the exact HEAD above, Warp, target ordinal 1,
`go2_static_960x640_v1`, catalog digest
`9200a3374399369cdab1948e7cb0688a3eed0d61b1018b4e35429fd2d7ab180c`, and environment
digest `d2b1edc48c5a28a4ef71237fdf16d92d089f02f3fe7926c858d3e0c651146d3b`.

The external ledger contains 615 events. Events 609-612 resolve/create/start
`scipy._lib._testutils`; 613 is permitted absence of `cython`; 614 is the first
failure, naming `_sysconfigdata__linux_x86_64-linux-gnu`. The artifact is failed
in `canonical_materialization`. This is not a cython optional-probe failure.

Installed SciPy `_lib/_testutils.py:12` imports sysconfig, then line 44 executes
`sysconfig.get_config_var('HOST_GNU_TYPE')`. Installed sysconfig.py:537-543
initializes POSIX variables by selecting `_get_sysconfigdata_name()` and invoking
`__import__(name, ..., ['build_time_vars'], 0)`. This is a required generated
interpreter configuration import, not an optional external package probe.

The candidate finder at native_bootstrap.py:314-331 delegates only when the
top-level name belongs to its frozen stdlib/builtin union. Otherwise this name
enters external admission. At lines 623-641 the name has no rule, namespace or
permitted absent row and fails `unlisted external import` before PathFinder or
any target loader is invoked. Source and the first-failure ledger corroborate
this exact causal route; the ledger itself does not serialize the original
exception text.

The top-level `SystemExit('mujoco_menagerie_robot_preview.py requires trimesh')`
is a misleading wrapper: optics/mjcf_visual.py:40-43 catches **any Exception**
from `import trimesh` and raises that message with the original exception as
cause. It does not establish that trimesh is missing. The ledger already shows
trimesh and its dependencies executing. Changing this diagnostic is not required
to fix delegation, and swallowing the underlying failure would be incorrect.

Custody evidence records failed setup sequence 2, full quarantine, no emergency
sync, no terminal ARM and release not authorized. The supervisor records actual
`process_exited`, return code 74 and EXITED; two earlier normal segments completed,
but the failed segment did not. identity.json reports worker 3552132 and parent
3552131; a scoped `ps -p 3552131,3552132` closing check returned no rows. These are
inspected artifacts of the author's captured supervised execution, not a newly
rerun captured execution by this reviewer, nor proof of the complete failure matrix.

## Interpreter facts and classification

Fresh CPU-only diagnostics used exactly
`/home/ang.gao/miniforge3/envs/robot_sim/bin/python -I -S -B -` on primary 183.
Observed CPython 3.12.13, Anaconda build, with isolated/no_site/dont_write_bytecode
all enabled and only BuiltinImporter, FrozenImporter, PathFinder at startup.
The exact startup path tuple was:

```text
/home/ang.gao/miniforge3/envs/robot_sim/lib/python312.zip
/home/ang.gao/miniforge3/envs/robot_sim/lib/python3.12
/home/ang.gao/miniforge3/envs/robot_sim/lib/python3.12/lib-dynload
```

`sys.stdlib_module_names` contains 301 names and builtin_module_names 31.
`_sysconfigdata__linux_x86_64-linux-gnu` is in **neither**, and was not loaded at
the diagnostic's selection check. PathFinder resolves it without execution to
the exact SourceFileLoader origin:

`/home/ang.gao/miniforge3/envs/robot_sim/lib/python3.12/_sysconfigdata__linux_x86_64-linux-gnu.py`.

Its SHA-256 is `07c6cf395105faefff17ea4a888e00858ca3da7e2f8f9266cb7953c772085683`.
The file identifies itself as generated system configuration. AST inspection
found one assignment to build_time_vars, whose literal HOST_GNU_TYPE is
`x86_64-conda_cos6-linux-gnu`. No generated module execution was needed to
establish its existence, content or origin.

Installed sysconfig.py SHA-256 is
`e9a916680e27278b130d8c4add824d55b952bef284bee354083ca9b436d5103e`;
SciPy `_testutils.py` SHA-256 is
`c0653e4b236caaf4f090becb912a9f39a350eeafdd53292163a82655782625ee`.

Classification: **generated stdlib delegation/admission-domain gap**. The file
belongs to the pinned interpreter installation. There is no demonstrated missing
third-party package or damaged generated file. Installing trimesh, SciPy, cython
or another dependency is not the indicated fix. Neither file existence nor an
ordinary successful import would prove the captured canonical path succeeds.

## Blocking finding F1 — P2: approved name domain omits required generated stdlib

Locations: absent-resolution design-request v1 section 2; native_bootstrap.py
lines 317, 624, 1146, 1782, 1916 and 1933; runtime_environment.py lines 66-69,
140 and 188.

Trigger: cold canonical Go2 materialization reaches installed SciPy's sysconfig
query before this generated module has been loaded. Consequence: safe but
unavoidable failed materialization under the current rules; no upload/readback
or complete Go2 proof.

The implementation's core classification follows the approved supplement's
literal rule: only the static stdlib/builtin union may reach internal delegates.
The design says those delegates are never invoked for nonmembers. Extending this
domain is therefore a **bounded design supplement**, not an undocumented table
maintenance fix. Parent runtime-adoption's trusted-stdlib statement does not
override that later explicit domain definition. A fresh independent design review
should freeze the generated-module exception before implementation.

The existing external schema cannot legally express this as an ordinary new row:

1. `_module` requires every dotted component to be an identifier. The actual
   name contains hyphens. Fresh invocation of the unchanged production validator
   raised `ValueError: invalid module name`.
2. Roots have only site/native kinds; imported file-module rows require site.
   The existing interpreter_native root could physically contain this file, but
   native roots cannot supply import rows. The sole site root is site-packages,
   which does not contain the generated file. Relative-path containment must not
   be bypassed with `..`, copying or symlinking a new package into place.
3. Relabeling/adding the actual stdlib directory as a site root conflicts with
   bootstrap's explicit site-root-not-on-sys.path check at lines 1429-1432 and
   changes the approved external/stdlib boundary. It is not an authorized way
   to reuse schema v3.

Furthermore, fixing only the finder would be incomplete. External audit at line
1146 skips baseline/repository/static-stdlib names, so a lazily loaded generated
module would subsequently fail as an unlisted publication. Cold preload checks
and topology name invariants also use the old domain. Preloading this module to
hide the gap is not a fix: cold checks currently reject it and it would bypass
the desired cold-path coverage. Maintain one frozen classification authority
across these uses instead of adding unrelated exceptions at each check.

Why existing checks missed it: representative stdlib delegation tests use names
such as quopri/colorsys/html/urllib, all in the static list; ordinary environment
observations and table spec checks did not exercise the new closed finder on
this generated dependency. The table review specifically withheld captured
closure approval. The fresh guarded failure is the counterexample.

## Required bounded fix direction and proof

This section proposes direction, not an approved implementation or complete new
design. Preserve the sole finder and exact interpreter roots/delegate bindings.

- Define a tiny interpreter-owned generated-module authority for this exact
  required full name, bound to the pinned interpreter identity and its expected
  generated file under the exact stdlib root. Freeze the expected origin and
  identity/digest in reviewed trusted configuration/policy. Do not derive new
  permission from observed names, a directory scan, any underscore rule,
  `_sysconfigdata_*` prefix, arbitrary `find_spec`, site roots or request input.
  Do not loosen the general external identifier grammar.
- Determine the default expected name from trusted pinned interpreter facts
  (sys.abiflags, sys.platform, sys.implementation._multiarch) or freeze its literal
  deployment value, independently of environment override input. In this pinned
  sysconfig source, lines 466-471 implement exactly that default formula but
  allow `_PYTHON_SYSCONFIGDATA_NAME` to override it.
- Freeze an explicit override policy before any relevant lazy import. The
  smallest policy is rejecting the presence of `_PYTHON_SYSCONFIGDATA_NAME`,
  including an empty value, rather than silently expanding authority. Any
  alternative normalization or exact-equal override policy must itself be
  documented, fixed-launcher-owned and reviewed. Recheck relevant configuration
  drift at the supported lookup/audit boundaries; a later arbitrary override
  must not select an otherwise accepted stdlib name to escape the intended
  configuration-module selection.
- Keep generated-module lookup top-level/exact-name and pinned-origin, using
  the captured standard delegates only. Define origin/loader failure, missing
  file, preload/cache state, post-load identity and audit behavior. It belongs
  to interpreter trust, not an external publication or absent-probe record;
  do not fabricate external create/exec custody. Other generated names remain
  denied until separately justified and frozen.
- Apply one consistent domain to finder dispatch, topology integrity, cold
  preload validation and external sys.modules reconciliation. Preserve intact
  topology checks before/after delegation, sticky failure, retained custody,
  no extra meta-path entry, no metadata/plugin discovery, and terminal-only
  stdlib availability after poison without reopening repository/external work.
- Repin affected source-policy/catalog bytes and add production-entrypoint tests:
  cold sysconfig.get_config_var under the sole finder with no generated preload;
  generated import followed by external audit; arbitrary/empty/late environment
  overrides; other underscore/prefix-sibling names; wrong origin/loader/root,
  missing/replaced/preloaded module and topology drift; after-poison stdlib use
  without clearing failure. Negative cases must check zero forbidden target
  execution and unchanged authority, not just exception text.
- Then rerun both required captured Go2 cases under the actual supervisor and
  inspect their ledgers/custody/canonical arrays before claiming this closure
  repaired. Full C2d3 still requires both lanes, real upload/readback and the
  parent failure/hang/multi-device/reap matrix plus independent code review.

Environment override concern was independently demonstrated in fresh isolated
CPU diagnostics: with sys.flags.isolated == 1 and ignore_environment == 1,
process-local assignment of `_PYTHON_SYSCONFIGDATA_NAME` to
`reviewer_unapproved_generated_name`, then to the empty string, made the installed
`_get_sysconfigdata_name()` return those exact values. No selected target was
imported; the environment change was confined to that disposable process and
restored. The initial host environment had no sysconfig override. Thus this is
a prospective repair requirement, **not** evidence an override caused the failed
author attempt. This sysconfig version reads no `_PYTHON_SYSCONFIGDATA_PATH`;
other versions must not inherit assumptions without inspection.

## Verification, limitations and closed gates

Commands used read-only Git status/revision/hash checks, UTF-8 source reads and
rg. Remote scripts were piped over SSH with BatchMode, ConnectTimeout=10 and
`timeout 15s` or `timeout 20s` around the exact robot_sim `-I -S -B -` interpreter.
They inspected interpreter metadata, PathFinder specs, source/AST and JSON
artifacts. No NumPy/SciPy/trimesh/Pillow/NetworkX/Warp/Torch module was imported
by these diagnostics. Explicit sys.modules assertions confirmed that property
in the configuration and contract diagnostics.

For the production `_module` validation only, the CPU contract was imported
normally with the repository temporarily added to this disposable interpreter's
sys.path. This is expressly **not captured-import proof**. An initial runpy
attempt failed on its relative import because it lacked a package context; the
subsequent normal CPU-contract import corrected that harness issue. One initial
malformed SSH stdin invocation also exited unsuccessfully and produced no usable
diagnostic; neither is counted as a candidate failure or a pass.

No full regression suite or GPU experiment was rerun by this reviewer. The parent
reported separate CPU/C0/synthetic regression results; they are not independently
reproduced evidence in this report and do not close the real Go2 counterexample.
No dependency installation, candidate edit, table broadening, source repinning,
backend import, CUDA call, process signal, GPU process query or fallback-host use
occurred. In particular, no other users' GPU2 process was touched. The only write
is this requested report outside the frozen candidate, using apply_patch.

The actionable outcome is a scoped design supplement followed by implementation
and independent re-review; not approval of a repair that has not yet been written.
Whole C2d3, C2e/f, C4c3/c4, C4d/B3, product GPU preparation, a3b,
results/baselines, R2/host payload, shipping and full C4 completion remain closed.
No ordinary import, source read, file hash or successful fail-closed termination
is promoted to canonical materialization or product readiness evidence.
