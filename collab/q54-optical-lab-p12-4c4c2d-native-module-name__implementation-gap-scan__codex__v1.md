# C2d3 native-name implementation: independent bounded gap scan v1

Initiative: q54-optical-lab-p12-4c4c2d-runtime-adoption
Stage: bounded implementation gap-scan, not code approval
Author: codex, independent reviewer
Version: v1
Date: 2026-10-08 (Asia/Shanghai)
Status: in_review
Related Files: native-module-name design-request/design-review/gap-scan v1; optical_worker/runtime_environment.py; optical_worker/native_bootstrap.py; native_source_policy.py; native_sources.py; tests/support/native_runtime_process.py
Owner Summary: 本次限定实现扫描未建立需先修的实质实现问题。此结论不是 C2d3 code approval，不批准 CPU 子集代替父级验收。最终 HEAD 的定向 CPU 测试和额外反例通过；真实 Go2 v7 仍以 unlisted module publication 失败，完整证据和后续门禁继续关闭。

## Identity, independence and exact scope

Reviewer session: `/root/c4_c2d3_native_name_implementation_gap_v1`.
The reviewer did not author the candidate; this was a fresh bounded assignment,
not an implementation-conversation continuation. AGENTS.md, collab/README.md
and collab/REVIEW_PRINCIPLES.md were read completely. The selected design,
pre-implementation gap scan and independent design-review report were inspected.
Truncated combined outputs were followed by bounded reads; UTF-8 was explicit.

Local: `C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design`.
Remote: `tileops-dev:/data2/shared/ang.gao/robot_sim_c2d3_binding_wip`.

Implementation base: `a4cc1a6b85144997e3823d61f440c889c2921c2b`.
Frozen final HEAD, independently observed on both ends:
`a50b227d38ac33e7e56109fb5b6807e485c98be4`.
Final local tree: `ae4f6eded3a398ed7015964c9413117058849bad`.
The delta is b11ff9e + 787da2e + 672a247 + a50b227: nine files,
339 insertions and 10 deletions. b05c1e1 is the design; a4cc1a6 only archives
its review and is not an earlier implementation acceptance.
Both candidates were clean at opening and closing checks. No candidate edit,
install, commit, private-instruction change or shared README change was made.

Independently matched remote hashes:

- Design request v1: `b0eb29a1d03e54606be7149c20e5a037f0bbafe5f59e3f522f9526afda62dadd`.
- Design-review v1: `ac9e28ff7ea8925c388534de02defe68e22afd370dbcdbbaf981b5916e835739`.
- Catalog v9: `c5a10ad522472d3e1733515833569c7e570e7d62a284f7d097edc68072ef8779`.

## Bounded conclusion — no substantive implementation finding established

No new severity/location/repair finding was substantiated within the assigned
native-name policy, boundary ordering, exact identity, sticky failure and two
optimization changes. This is deliberately **not an Approve or code-approval
verdict**. The gap scan is complete; whole-C2d3 acceptance remains blocked by
the unfinished obligations below. No CPU-subset approval is issued.

### 1. One metadata authority; no new resolver/alias capability

`runtime_environment.py:29-31` contains one literal tuple relation, exactly
`scipy._lib._uarray._uarray -> uarray._uarray`. The helper at lines 112-120
requires an extension inventory row and `is_package is False`; ordinary file
modules return their full import name. `_validate` calls this helper at line
225 during construction, before an executable loader is invoked.
`expected_file_module_name` at lines 436-443 requires the exact contract type,
a valid module-name input, and membership in its declared file-module rows.
Namespaces/publications/undeclared names receive no expectation or permission.
There is no new serialized caller field, copied binary inventory, suffix rule,
dual-name acceptance, normalization or observed-name learning.

`native_bootstrap.py:1166-1171` uses only this bound contract query and reads
the raw held module dictionary. Missing, None, integer, empty and unexpected
names reject; a string subclass cannot satisfy the exact-str check. The new
name is not used by `find_spec`, lookup keys, phase selection or publication
admission. Audit lines 1236-1246 still reject a new `uarray._uarray` key,
including the same held module and None. The existing optional `uarray` probe
and creator-self alias rules are not enlarged.

### 2. Retain create's return before validation; preserve execution identity

`create_module` lines 483-509 retains the actual returned object in the
already-held admission's records at line 498, then validates its raw name at
501, then proceeds to self-alias checks/adoption/event recording. It does not
demand importlib-initialized spec/package/loader at raw create return. This
keeps the returned owner reachable even when validation fails and importlib
removes or never inserts the qualified sys.modules entry.

At exec entry (`:511-529`), exact module, created state, qualified sys.modules
entry, held spec and wrapper are required before delegated execution. The
full `check_module` is now called before recording exec_started. On return,
check_live remains before publication completion; check_module runs again
before a successful completed event. Later audits recheck each held record.

`check_module` lines 1173-1224 preserves qualified-key object identity, exact
spec/wrapper, spec name/parent, package, file/origin and package paths. The new
nonpackage branch rejects both a package-shaped spec and an added __path__.
Correct intrinsic text does not mask another object, copied spec or loader,
wrong file/package/origin, or an independently substituted delegate.

### 3. Delegate copying is rejected by identity, not equality

`_check_binding` at lines 439-458 compares the held wrapper and delegate with
`is`, and captured source with `is`, retaining exact origin equality. This
removes the former tuple-equality weakness for equal-valued standard loaders.
It also rechecks exact SourceFileLoader/ExtensionFileLoader type plus qualified
delegate.name and delegate.path. A copy of the original delegate cannot pass
because its name/path compare equal. Identity checks run at create/exec and
again at post-exec/full audit module checks.

### 4. The optimizations cache metadata, not observations or liveness

672a247's `_file_module_names` (`runtime_environment.py:123-137`) caches one
tuple of immutable name pairs, keyed by the complete exact canonical bytes.
On a miss it decodes and structurally validates those bytes. It does not cache
a module, loader, file verification, runtime observation, session/guardian
state, or environment digest verdict. Changed encoded bytes select/revalidate
a different key; ordinary verify_files still rereads physical files. The
added test explicitly changes bytes on the same contract object and changes
the physical native fixture after warming metadata; both reject appropriately.

a50b227 factors `_check_binding` from `_check`. It does not remove the latter's
check_live call. Exec still invokes `_check` before execution and check_live
after execution. Audit still invokes check_live and verify_files once before
the module loop, then performs each module's current exact binding and metadata
checks. Checked auxiliary callers include disabled resolution/consumption,
fallback confirmation and creator-self reconciliation: their enclosing
entrypoints retain live validation. The optimization avoids repeating the
whole environment traversal per module; it does not memoize a valid module
or skip the current object relationships. The successful repeated-audit test
counts exactly one check_live per audit and no additional native create/exec.

This is a supported-boundary assessment, not a claim of instruction-level
sandboxing or a new hostile-concurrent-thread guarantee. The trusted native
pre-return/process-owner limitations are not solved or expanded here.

### 5. Failure remains sticky; diagnostics are not upgraded evidence

Create/exec/audit exceptions enter `fail` (`native_bootstrap.py:648-684`).
The first exception and actual returned record remain reachable, repeat intake
retains further available owners, and source-session failure is one-way.
Restoring the name or catching the first exception does not permit a later
exec/audit, create, successful completion or new permission. The independent
CPU probes below exercise this directly, but use unit custody prerequisites;
they are not proof of the full real guardian/fault matrix.

The tests/support driver addition reads native metadata from actual held
records and reports qualified identity separately from module_name. It does
not change a production import capability or manufacture an alias event.
Its `q54.c2d3.driver-observation.v0` output remains diagnostic, not the still
missing finalized/versioned C2d3 custody schema. Adding descriptive fields
cannot retroactively upgrade old dumps into acceptance evidence.

Policy is v10 and both cold-bootstrap and production catalog selectors are
v9. The 787da2e consistency test guards the earlier b11ff9e selector omission;
final targeted policy tests pass. Historical 83-failure and early source-exit
claims were supplied by the author, not independently rerun at that old HEAD.

## Original counterexample and current real-run evidence

The original gap-scan F1 was the genuine pinned extension returning intrinsic
`uarray._uarray` under qualified SciPy spec/import identity. The old global
name equality rejected that module. The bounded repair direction was an
explicit reviewed metadata expectation, never deleting name validation or
granting an alias. The current code implements that direction, and synthetic
production-wrapper positives/negatives exercise the replacement rule.

I independently inspected the author's final-HEAD Go2 v7 artifacts, without
starting or changing that run:
`/data2/shared/ang.gao/c2d3-a50b227-go2-960-guarded-author-v7`.

| Artifact | Independently calculated SHA-256 |
| --- | --- |
| request.json | 2b7a788f168b554db2340996634e35893374a077ac026214786a24e4b88fba98 |
| identity.json | 6a76638ac79dc27bc1560cbd3900afe769af84af6a1dc17f3045012626a8bdf6 |
| external_failure.json | a05d85148a57ef557e6bc9e33e6a39d45b344a50d91029a38ba3bdbff157fe39 |
| custody_artifact.json | 23ae27cdbafde33320e4eff48ea79ab7ea469a3ef08a27a613d10c3df2bdb4a0 |
| supervisor_stdout.json | 856b8ce28ea1f5633d0ad1b8c750b841972c3d15fd1a8da921a05b7ad4e823d2 |

The request matches final HEAD/catalog above, Warp, target 1, fixed Go2
960x640 case and environment canonical digest
`d2b1edc48c5a28a4ef71237fdf16d92d089f02f3fe7926c858d3e0c651146d3b`.
The configured budget remains 120000 ms / 1000 ms acknowledgment.
Events 2322-2326 show the exact qualified uarray file resolved, create_started,
created, exec_started and completed once. This corroborates passage beyond
the original failure; it is not complete successful final audit evidence.

The run has 4554 external events. PIL.ImageFile completes at 4552; audit fails
at 4553 with `unlisted module publication`. The failure metadata names audit,
not the unexpected key; this scan did not establish the new publication's
identity or authorize its admission. Diagnose that separately without adding
an automatic exception or relaxing audit. There is no successful guarded Go2
materialization/upload/readback result here.

Worker PID 3651806 / parent 3651805 exited 74. Supervisor summary records
EXITED, no expired deadline, no kill request, three ARM entries, only two
normal completions, and input_eof terminal reason. Worker summary records
failed setup sequence 2, full quarantine, release not authorized, zero
emergency sync attempts and zero terminal ARM attempts. `complete=true` in
the supervisor is protocol/exit completion, not workload success. This is
independent inspection of author artifacts, not a newly executed reviewer
Go2/custody test or full custody validation.

The packet reports v5/v6 reached their original 120-second deadlines and were
killed/reaped by the supervisor. Those histories were not rerun here and do
not prove success or no-hang risk. Neither they nor v7's normal failed exit
replace the still-missing both-lane/fault/hang/dual-GPU matrix.

## Independent verification and actual commands

Local inspection used Get-Content -Encoding utf8, rg, git diff/show/log,
status and rev-parse. `git diff --check a4cc1a6 HEAD` passed on both machines.
The initial requests for workspace-root AGENTS.md and repository-root
REVIEW_PRINCIPLES.md returned missing-path errors; the actual repository
AGENTS.md and collab/REVIEW_PRINCIPLES.md were subsequently read completely.
An unquoted PowerShell `HEAD^{tree}` was misparsed; the quoted command gave
the exact tree above. Neither error was treated as candidate failure.

Fresh CPU test command, remote shell content passed in a PowerShell
single-quoted SSH argument:

```sh
cd /data2/shared/ang.gao/robot_sim_c2d3_binding_wip &&
export PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 &&
scan_tmp=$(mktemp -d /tmp/c2d3-name-gap-v1.XXXXXX) &&
echo REVIEW_TEMP=$scan_tmp &&
timeout --kill-after=5s 60s /home/ang.gao/miniforge3/envs/robot_sim/bin/python -B -m pytest \
  -q -p no:cacheprovider --basetemp=$scan_tmp \
  tests/unit/optics/test_runtime_environment.py \
  tests/unit/optics/test_runtime_import_admission.py \
  tests/unit/optics/test_native_source_policy.py
```

Observed: **339 passed in 10.87 s**, exit 0, no timeout.
Scratch root: `/tmp/c2d3-name-gap-v1.jQB3W7`, outside the candidate.
Installed interpreter: robot_sim CPython 3.12.

The first attempted invocation incorrectly used a double-quoted PowerShell
SSH command with backslash-dollar escaping. PowerShell attempted local
`mktemp`, which was unavailable, and malformed --basetemp consumed the next
test filename. Pytest exited with **164 passed, 124 setup errors in 3.00 s**;
the setup errors were FileNotFoundError for a malformed path ending in
`/ tests/unit/optics/test_runtime_environment.py`. This was a reviewer command
quoting error, not a candidate defect and not counted as successful evidence.
The subsequent correct invocation above reran all three intended files.
Remote git status remained clean; no candidate directory was created by the
failed mkdir. No unsafe cleanup or broad deletion was attempted.

Additional independent scratch probe, created with apply_patch outside the
candidate:
`C:/Users/AG/新建文件夹/c4_assets/c2d3_native_name_extra_probe_v1.py`.
Executed with:

```powershell
Get-Content -Encoding utf8 -Raw 'C:/Users/AG/新建文件夹/c4_assets/c2d3_native_name_extra_probe_v1.py' |
  ssh -o BatchMode=yes -o ConnectTimeout=15 tileops-dev 'cd /data2/shared/ang.gao/robot_sim_c2d3_binding_wip && timeout --kill-after=5s 30s /home/ang.gao/miniforge3/envs/robot_sim/bin/python -B -'
```

Observed: **16/16 passed**, exit 0. Eight independent mutations at each of
exec-return and later audit: copied delegate, changed delegate path, copied
spec, copied wrapper, package, file, name and copied sys.modules object.
Each checks failure, exact held returned module, original exception identity,
once-only create/exec, and continued rejection after swallowing the exception
and restoring the name. The name cases specifically restore the sole bad
field and still cannot recover. These use the repository's actual wrapper
and synthetic standard-loader callbacks plus unit custody prerequisites;
no actual native binary or backend is executed. No second import algorithm
or duplicate expected-name mapping was introduced.

Artifact readers ran over SSH stdin with robot_sim `python -I -S -B -` under
20-second timeout. They read JSON/source bytes and hashes, not GPU APIs.
The probe ran PID 3656967 / timeout parent 3656966. Artifact readers ran
3657218/3657217 and 3657560/3657559. All returned normally. Final command:

```sh
ps -p 3656967,3656966,3657218,3657217,3657560,3657559,3651806,3651805 -o pid,ppid,stat,args
```

returned only the header and status 1 (no matching processes). The pytest
execution session also returned exit 0 and is closed. No observed reviewer
hang/background session or v7 worker/parent remained. This bounded process
check is not a proof of every possible GPU/native failure mode.

No GPU work was started by this reviewer, no other-user process was signaled,
and no package/environment file was modified. -B/PYTHONDONTWRITEBYTECODE and
no-cacheprovider prevented reviewer bytecode/cache writes in the candidate.
Normal installed caches may be read. Pytest scratch stays outside the tree;
the extra probe's disposable fixture directories were normally removed by
its own temporary-directory context. Only this report and the named scratch
probe were deliberately written locally, both via apply_patch.

## Evidence still required; all broader gates remain closed

The author's local168/remote493-or494/GPU6/C0-30 and final five-file CPU382
summaries were not independently rerun in full and are not this review's
test totals. Historical outputs are not promoted to final-HEAD executions.

Still required before whole C2d3 acceptance: diagnose the new v7 publication
failure without weakening admission; a successful real pinned-extension
captured path with final audits and distinct observed/import name evidence;
both canonical Go2 materializations and actual guarded Warp/CUDA upload and
readback; the finalized versioned runtime-custody schema and negative replay
validation; full unexpected/swallowed failure, custody, watchdog/hang,
dual-device/dual-GPU and exit/reap matrices; and independent whole-stage code
review of the exact finished revision. No synthetic CPU test substitutes for
those real execution or process/device ownership properties.

This scan identifies no additional bounded repair direction beyond retaining
the implemented exact checks and completing those evidence obligations. It
does not approve unfinished C2d3, full C2d/C4, C2e/f, C4c3/c4, C4d/B3, product
GPU preparation, a3b, result/baseline, R2/host payload or shipping. No new
stage, live lease, renderer, final readiness, measured sample, performance
result, product activation or release authority is opened. Existing cleanup
and losing-lane obligations remain unchanged. The report digest is supplied
separately to avoid a self-referential hash.
