# C2d3 exact generated-stdlib admission: independent design review v1

Initiative: q54-optical-lab-p12-4c4c2d-runtime-adoption
Stage: independent generated-stdlib design review
Author: codex, independent reviewer
Version: v1
Date: 2026-10-08 (Asia/Shanghai)
Status: accepted
Related Files: generated-stdlib design-request v1; runtime-adoption design-request v2; absent-resolution design-request/review v1; Go2 deployment-table review v1; sysconfig gap-scan v1; go2-dependencies test-evidence v6; optical_worker/native_bootstrap.py; optical_worker/runtime_environment.py
Owner Summary: Approve the exact bounded DESIGN supplement. The intrinsic-name, fixed-origin generated CPython module is a feasible addition to interpreter trust, not external-table permission. No implementation, captured Go2 success, whole C2d3 acceptance, GPU matrix, or later gate is approved.

## Verdict and frozen identity

Verdict: **Approve**.

Reviewer session: `/root/c4_c2d3_generated_stdlib_design_review_v1`.
This is a fresh independent review session; the reviewer did not author the
candidate. AGENTS.md, collab/README.md and collab/REVIEW_PRINCIPLES.md were read
in full. The initial PowerShell encoding rendered Chinese incorrectly; the
principles were reread in full using explicit UTF-8. Author evidence and prior
verdicts were treated as claims to check, not approval of this proposal.

Comparison base: `89e89c15e1bcbca2ec1171d6814a9d4e9f478b83`.

Exact candidate HEAD: `791a39c65261e66345381e9d0d4054c54a73d556`.

Candidate: `collab/q54-optical-lab-p12-4c4c2d-generated-stdlib__design-request__codex__v1.md`.

Candidate SHA-256: `0c90f2d331e79898edfd5182d6036b2c1594d207bb12d8b259ba5b31106956a9`.

Local repository: `C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design`.

Remote repository: `/data2/shared/ang.gao/robot_sim_c2d3_binding_wip`, primary
SSH alias `tileops-dev`. No fallback host was used.

Both trees were clean at opening and closing checks, with this exact HEAD and
matching design digest. The base-to-HEAD delta is only three documentation
files: the 158-line candidate, 142-line test-evidence v6, and 255-line verbatim
gap scan. No code, table, shared workflow or private instruction changes are in
this delta. `git show --check HEAD` passed on both machines. The frozen candidate
was not modified.

## Findings

No substantive blocking or non-blocking design finding was established.
Accordingly there is no severity/location/trigger/consequence/repair item to
close in this design revision. Required implementation proof below remains
mandatory; its absence is not represented as a code pass.

The production code still rejects the generated module. That is the explicitly
identified pre-implementation condition, not a contradiction in the proposed
repair and not evidence that this repair is already implemented.

## Independent assessment of high-risk areas

### 1. Real gap and bounded classification

The historical artifact and installed sources agree on the failure chain.
External event 614 names `_sysconfigdata__linux_x86_64-linux-gnu` immediately
after SciPy `_lib._testutils` starts and its permitted cython absence is recorded.
The installed file imports sysconfig at line 12 and calls
`sysconfig.get_config_var('HOST_GNU_TYPE')` at line 44. CPython's `_init_posix`
imports the generated data module using its selected name.

Current `native_bootstrap.py:314-331` admits ordinary interpreter modules only
through the frozen static root-name set. The generated name is outside that
set and reaches external admission's unlisted-name rejection at lines 623-641.
The external audit at line 1146 would also reject a lazily loaded generated
module if only finder dispatch were patched. Cold preload checking at line
1916 and topology name checks at line 1782 are additional related sites.

The external schema is not a legitimate workaround: `_module` at
`runtime_environment.py:66-69` requires identifier components, so the hyphenated
name is invalid; imported file rows require a site root at line 188. The
stdlib directory is a cold interpreter path, and treating it as an external
site root conflicts with `native_bootstrap.py:1429-1432`. The proposal retains
all three boundaries and makes no table/schema permission change.

### 2. Authority derives from the trusted interpreter, not observation

Design sections 1-2 constrain authority to CPython 3.12 POSIX, exact startup
intrinsics, a canonical cold stdlib directory and exactly one derived source
path. There is no failed-import learning, directory enumeration, prefix grant,
site-root fallback, request-selected name or evidence-minted capability.
Unsupported interpreter configurations do not inherit this special permission.

The relevant intrinsic values actually exist on the pinned interpreter, and
the specified formula matches its installed sysconfig implementation. The
generated file is a regular non-symlink directly under the canonical stdlib
directory, which is already on the isolated cold path. Its parsed source has
one assignment to a literal build_time_vars dictionary. Capturing its bytes,
length and hash without executing it is feasible before singleton installation.

This is a trusted-deployment snapshot, not authentication against a privileged
actor changing the interpreter or a claim that a source hash alone proves
executed bytecode. Standard SourceFileLoader/cache behavior remains within the
expressly trusted immutable interpreter deployment, not the repository
captured-byte execution contract. No second repository loader is needed.

### 3. Override policy and cold startup are explicit

Design section 3 rejects presence, not merely a nondefault value, of
`_PYTHON_SYSCONFIGDATA_NAME`. Empty and correct-default overrides are therefore
unambiguously rejected. The check is at bootstrap, topology entry and existing
audits, not only the generated-name lookup branch. This matters because
CPython's sysconfig consults os.environ even in isolated mode and because cached
module/configuration queries need not enter the finder.

Freezing the intrinsic fields and rechecking drift avoids turning later
mutation of sys.platform/ABI/multiarch into new authority. The design also
forbids helper monkeypatches. It preserves cold rejection of a preloaded
generated object; extending the classification helper must not accidentally
turn that object into an accepted cold baseline. These are implementation
requirements already stated in sections 2-3, not optional testing conveniences.

### 4. Exact resolution, audit and identity are coherent

The one generated full name has its own match, not ordinary top-level stdlib
prefix semantics. PathFinder is held through the existing frozen binding and
receives only the frozen stdlib directory. Exact SourceFileLoader type,
name/origin/path, nonpackage status, source snapshot and no reload target are
required. Dotted children, sibling ABI names and fallback roots gain no rights.

The existing topology registry, singleton finder, session audits and external
ledger reconciliation provide the required integration points. One internal
classification authority can serve dispatch and reconciliation while keeping
the generated-module-specific integrity audit separate from an ordinary
stdlib skip. Existing baseline checks must not bypass that audit.

The first successful boundary establishes a held module identity only after
its exact metadata/source checks; later boundaries require that same object.
The design does not claim instruction-level interception of hostile in-process
sys.modules mutation. Import caching and the interval before first boundary
remain inside the trusted sealed-worker model, which is explicitly inherited
from the parent. This is not a new adversarial Python sandbox.

### 5. Failure and ownership do not acquire a recovery shortcut

Descriptor/override/source/topology drift must follow existing sticky failure.
In an armed segment it cannot produce COMPLETE, another ARM, retry, fallback
synchronization or normal release. Cold rejection remains before third-party
execution. Terminal stdlib reporting is conditional on intact topology and
cannot reopen repository/external work or clear failure.

No external create/exec/publication events are invented for the trusted stdlib
module, and provenance is not an authority token. There is no new GPU leaf,
lease, event role or timed-acquire work. The proposal retains the original
guardian/controller terminal policy and the parent native-dependency trust
obligations.

### 6. Scope and required proof remain bounded

Section 5 specifies production-entrypoint cold positive, override and mutation
negatives, source/spec/loader/origin faults, preload/replacement rejection,
external-audit integration, cached repeats and platform restrictions. It also
retains policy/catalog repinning, existing regressions and both real captured
Go2 workloads, not just ordinary CPU import success. Sections 5-6 explicitly
withhold whole C2d3 acceptance and preserve parent cleanup obligations.

## Verification performed

Read the accepted parent runtime-adoption v2, absent-resolution design/review
v1, Go2 literal-table review v1 and this commit's gap-scan v1 and test-evidence
v6. Inspected the actual finder, external resolution and audits, topology
checks, cold installer, environment binding, materialization entry and schema
validation. Author regression totals were not rerun or promoted to independent
proof.

Read-only repository commands included:

```text
rg --files / rg -n <relevant paths and symbols>
Get-Content -Encoding UTF8 <documents and source>
git status --short
git rev-parse HEAD
git diff --stat 89e89c15e1bcbca2ec1171d6814a9d4e9f478b83 HEAD
git show --check HEAD
Get-FileHash -Algorithm SHA256 <candidate>
sha256sum <candidate>                         # remote
```

Three fresh remote CPU-only diagnostics were passed over stdin, without creating
a script or artifact in the candidate:

```text
ssh -o BatchMode=yes -o ConnectTimeout=10 tileops-dev \
  'timeout 20s /home/ang.gao/miniforge3/envs/robot_sim/bin/python -I -S -B -'
```

Observed interpreter: CPython 3.12.13, Anaconda build, executable
`/home/ang.gao/miniforge3/envs/robot_sim/bin/python`.

The first diagnostic read intrinsic values, source/AST, a nonexecuted
PathFinder spec and historical JSON artifacts. Observed:

- ABI flags `''`, platform `linux`, multiarch `x86_64-linux-gnu`.
- Exact derived module is absent from both static name lists and not preloaded.
- Root `/home/ang.gao/miniforge3/envs/robot_sim/lib/python3.12` resolves to itself
  and is a cold sys.path entry.
- Source is a regular non-symlink, 71,715 bytes, SHA-256
  `07c6cf395105faefff17ea4a888e00858ca3da7e2f8f9266cb7953c772085683`.
- Exact SourceFileLoader, exact name/origin/path, no package locations.
- sysconfig.py SHA-256
  `e9a916680e27278b130d8c4add824d55b952bef284bee354083ca9b436d5103e`.
- SciPy _testutils.py SHA-256
  `c0653e4b236caaf4f090becb912a9f39a350eeafdd53292163a82655782625ee`.
- No initial sysconfig-name override. AST literal HOST_GNU_TYPE is
  `x86_64-conda_cos6-linux-gnu`.

The second diagnostic installed a scratch singleton finder only in its
disposable interpreter, delegating ordinary static stdlib names and the one
exact generated name through the held standard bindings. It checked fixed
paths/hooks, source bytes and override absence, then invoked actual installed
sysconfig.get_config_var. The generated module was unloaded before the query,
loaded through that exact branch once, returned the expected HOST_GNU_TYPE,
and retained identity on a repeated cached query. Topology stayed unchanged.

In that same disposable process, empty, correct-default, arbitrary and foreign
ABI override values were each observed by the real `_get_sysconfigdata_name`
helper and rejected by the scratch explicit boundary check. Each environment
change was removed before continuing. This demonstrates why the proposed
boundary checks are necessary and feasible; it is not testing production
override enforcement, which has not yet been implemented.

The first two diagnostics asserted that NumPy, SciPy, trimesh, PIL, NetworkX, Warp and
Torch had not been imported. They exited normally, PIDs 3572287 and 3573000.
A final scoped `ps -p 3572287,3573000 -o pid=,args=` returned no rows (status 1,
normal no-match result), after the clean remote Git/hash checks.

### Explicit bytecode-cache question

Before this report was handed off, the author asked whether existing pyc use
conflicts with the frozen source snapshot. The reviewer had already considered
the cache/trust distinction above and independently checked the concrete facts
in a third read-only diagnostic; no cache was created, changed or executed by
that diagnostic.

`-B` prevents bytecode writes, not reads. The actual spec.cached is
`/home/ang.gao/miniforge3/envs/robot_sim/lib/python3.12/__pycache__/_sysconfigdata__linux_x86_64-linux-gnu.cpython-312.pyc`.
It exists, is 73,163 bytes, and has SHA-256
`eaef36a5f037996ad3354637943b1377a1b9225f75511c9a252cb8bf56321e60`.
The timestamp-based header has flags 0, source mtime 1790144460 and source size
71715, matching the source stat. Decoding its code object with marshal without
execution and comparing against a fresh compile of the actual source at the
same filename yielded equality. The actual installed SourceLoader.get_code
source returns valid cached bytecode before the conditional bytecode-write
branch controlled by sys.dont_write_bytecode. The generated module remained
absent from sys.modules in this diagnostic, and the no-third-party-import
assertion also passed.

Decision: this is not a blocking contradiction under the exact proposal.
Section 1 explicitly assigns the generated module the same immutable deployment
trust as ordinary stdlib, while section 3 explicitly returns a standard
SourceFileLoader spec. The source snapshot detects source drift and describes
the trusted deployment; it is not a promise to compile/execute only held source
bytes or attest cache contents. A manipulated or inconsistent interpreter
bytecode cache is not made safe by the source hash, and this approval makes no
such claim. Requiring captured-source-only loading, cache hashing/attestation
or cache bypass would be a different, stronger contract than this standard
interpreter delegation supplement. The observed current cache match is a
deployment fact, not a proof of a general bytecode-integrity guarantee.

### Historical guarded evidence independently inspected

Directory: `/data2/shared/ang.gao/c2d3-89e89c1-go2-960-guarded-author-v1`.

| File | Independently matched SHA-256 |
| --- | --- |
| request.json | d75019d6b8888c7424ca671ed4a83b63d852a3dcea5082ba2a7b47c77e5191b1 |
| external_failure.json | 2401dde57398cd803f049b688769e203131b10f69d9835d0c3a2488ac20abf99 |
| failure.json | 2e85fc9cecc90e479051e0647ab417a952068fc8d74c67ad8fd74cf722fc5696 |
| custody_artifact.json | d7db839cfd17bb651ac176d1f022fbeff925bc75689f6538db24a87b84772b7a |
| supervisor_stdout.json | c47ddd3a2681edd659d6e086ff4dda70c0e15173bebe7428cb9cc757da4b20c8 |

The external ledger ends with the exact generated-name failure at event 614.
The top-level SystemExit message says the preview requires trimesh; this is
not evidence that trimesh is missing. Custody records full quarantine, release
not authorized, failed sequence 2, no emergency sync and no terminal ARM.
Supervisor summary records EXITED, observed return code 74, two completed
earlier normal segments, no expired deadline and no kill request.

These are checked historical author artifacts, not a new supervised execution.
An initial diagnostic printed too much supervisor JSON and its tool output was
truncated; the subsequent diagnostic printed and checked the small supervisor
and worker summaries directly. Truncation is not counted as runtime evidence.

## Limitations and implementation proof still required

The scratch finder is a protocol feasibility experiment, not a production
implementation, integration test or second implementation proposed for use.
No candidate edits, dependency installation, backend imports, CUDA calls, GPU
queries, process signals, actual Go2 materialization, fault/hang/reap matrix,
Windows/non-3.12 execution or broad regression suite were performed. Other
users' GPU2 work was untouched. No real-device property is claimed from these
CPU diagnostics or the historical failure artifact.

The implementation review must examine the unified classification at every
dispatch/audit use, ensure the generated audit cannot be hidden by a baseline
skip, preserve strict cold rejection, and exercise source/metadata/object and
override mutations through actual production entrypoints. Stable exact spec
and loader relationships, unchanged intrinsic/descriptor state, no fallback
execution and sticky failure must be asserted, not inferred from exception
text. Tests must cover cached queries followed by audits, not assume every
query invokes the finder. Source-policy/catalog bytes must be repinned in a
new generation, without rewriting historical evidence.

After bounded implementation, full parent acceptance still requires both
mandatory Go2 workloads, both complete lanes, actual guarded Warp/CUDA
upload/readback, complete runtime-custody evidence, native/partial/fault/hang/
multi-device/exit-reap coverage and independent code review. This design
approval is not approval of a CPU-only substitute or the current C2d3 code.

## Exact authorization boundary

Approved only: implementation of this generated-stdlib design v1 at the exact
digest above, adding the single intrinsic-derived CPython 3.12 POSIX source
to interpreter trust under the stated cold/path/source/override/audit rules.
No general generated-module prefix, new external permission, schema change,
environment-table expansion, second finder, plugin resolver or monkeypatch is
authorized.

Whole C2d3 remains unapproved. C2e/f, C4c3/c4, C4d/B3, product GPU preparation,
a3b, results/baselines, R2/host payload, shipping and full C4 completion remain
closed. No renderer, final readiness, measured sample, product activation or
performance result is established. Existing P12.4e losing-lane and temporary
validation-driver cleanup obligations remain unchanged.

The only write by this reviewer is this report, created with apply_patch
outside the frozen candidate for unchanged archival by the author.
