# C2d3 exact native module-name contract: independent design review v1

Initiative: q54-optical-lab-p12-4c4c2d-runtime-adoption
Stage: independent native-module-name design review
Author: codex, independent reviewer
Version: v1
Date: 2026-10-08 (Asia/Shanghai)
Status: accepted
Related Files: native-module-name design-request/gap-scan v1; runtime-adoption design-request v2; native-self-alias design-request v1; generated-stdlib design-request/design-review/implementation-note v1; Go2 deployment-table review v1; optical_worker/runtime_environment.py; optical_worker/native_bootstrap.py
Owner Summary: Approve only the exact bounded DESIGN supplement. One already-declared native file module may have the fixed intrinsic name while retaining its qualified import identity. This is not implementation acceptance, import/alias permission expansion, captured Go2 success, whole C2d3 approval, or any later gate.

## Verdict and frozen scope

Verdict: **Approve**.

Reviewer session: `/root/c4_c2d3_native_module_name_design_review_v1`.
This is a fresh independent session, without the author's implementation
conversation. The reviewer did not author this proposal. AGENTS.md,
collab/README.md and collab/REVIEW_PRINCIPLES.md were read in full; the initial
PowerShell encoding issue was corrected by rereading the principles as UTF-8.
Author notes, prior approval statements and historical tests were treated as
claims/evidence to inspect, not approval of this new design.

Base: `f97186cd9c3746ccf9ea0c98e9753076d2072011`.

Frozen HEAD: `b05c1e1f7678b21750ded24b59a73f08fc0e83fa`.

Design: `collab/q54-optical-lab-p12-4c4c2d-native-module-name__design-request__codex__v1.md`.

Exact LF SHA-256: `b0eb29a1d03e54606be7149c20e5a037f0bbafe5f59e3f522f9526afda62dadd`.

Local repository: `C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design`.
Remote repository: `/data2/shared/ang.gao/robot_sim_c2d3_binding_wip`, primary
SSH alias `tileops-dev`. No fallback host was used.

Both trees were observed clean at opening and final checks, with the exact HEAD
above. The local UTF-8 content normalized to LF and the remote raw file both
matched the design digest. The base-to-HEAD delta contains four documentation
files only: PROGRESS.md, generated-stdlib implementation-note v1, this design,
and the unchanged gap-scan report. No code, deployment table, shared workflow
or private instruction changes are in this delta. Local `git diff --check`
and remote `git show --check HEAD` passed. The candidate was not edited.

## Findings

No substantive blocking or non-blocking design finding was established.
There is therefore no new severity/location/repair finding to close in this
revision. The existing code's strict-name rejection is the acknowledged
pre-implementation condition, not a claim that the proposed repair already
works. The implementation and whole-stage evidence obligations below remain
mandatory and are not waived by this verdict.

The independent gap-scan report was matched at SHA-256
`3ecd6b80366b03c7e2ed784b82a4379d7e3cc9e6dc7e94b86fafcd23a2acf4ec`.
Its F1 is resolved at the design level by an explicit new policy decision here;
that earlier Request changes verdict did not itself authorize implementation.

## Independent assessment

### 1. The original counterexample is real and this repair is narrow

Current `native_bootstrap.py:1143-1187`, specifically line 1177, requires
`module.__name__ == name` for a file record. The environment file-module rows
at `runtime_environment.py:177-191` contain the import identity and inventory
reference, not a distinct runtime-name field. The acknowledged failure follows
directly from that code, not merely from the author's diagnosis.

The actual Go2 table identifies `scipy._lib._uarray._uarray` as a nonpackage
extension in materialization, referring to
`scipy/_lib/_uarray/_uarray.cpython-312-x86_64-linux-gnu.so`. The file independently
matched 115080 bytes and SHA-256
`ca83664f94c09db3b054260ec762112b16fc6543d259b3477f26fa6f6639c7c0`.
It is not one of the four creator-self alias roots, and the table grants no
publication for `uarray._uarray`.

Design sections 1-2 expressly supersede the generic name equality for this
single declared file module, not for other extensions or aliases. The proposed
relation does not resolve/import its value, manufacture a parent package, or
change sys.modules. Qualified import key/spec name and intrinsic module name
are distinct facts; neither alone proves object or file identity.

### 2. Single authority and version boundaries are implementable

Design lines 54-89 choose one literal immutable relation in the existing CPU
contract module. The expected-name query is scoped to a file module already
declared in the validated environment. Its exceptional row must reference an
extension and be nonpackage; incompatible declared rows reject during environment
validation, before dependency execution. An absent exceptional row does not
grant permission or make unrelated environments require SciPy.

This fits the existing `_validate`/`RuntimeEnvironmentContract` construction and
repeat verification path (`runtime_environment.py:106`, `:380-412`). The existing
fixed optional-adapter tuple demonstrates the source-policy pattern, but is not
a reason to duplicate binary inventories or introduce another launch input.
The bound environment continues to own paths, phases, versions and hashes.

Keeping runtime-environment schema v3 is coherent: there is no new serialized
field or silently defaulted caller override. Acceptance changes through a new
source-policy/catalog generation, not by reinterpreting an old receipt as proof.
Both the bootstrap classes and CPU contract are already in the captured source
closure (`native_bootstrap.py:51-53`, `native_sources.py:53`). Current policy v9
and catalog v8 are historical inputs, not approval of the future changed bytes.
The implementation must repin the changed policy/contract/bootstrap exactly.

The proposed runtime-custody record remains a separate, still-required versioned
schema. Design lines 129-135 require separate import identity and observed name,
rule/kind/boundary validation, and no implicit upgrade of old diagnostic dumps.
This is consistent with runtime-adoption v2's requirement to freeze its actual
field order/enums and negative replay checks during implementation. This design
does not approve that as-yet-uncompleted schema or the current diagnostics as it.

### 3. Create-return retention and importlib initialization order are feasible

The current wrapper retains the returned module in `admission.records` at
`native_bootstrap.py:481`, before fallible self-alias validation, explicit
controller adoption and event recording. That registry belongs to the already
held admission/source session. The proposed raw-name check has an available
insertion point immediately after this retention, before another native leaf.
It must not be moved ahead of retention or postponed until successful adoption.

Installed CPython 3.12.13 source confirms the ordering:
`importlib/_bootstrap.py:813` calls create; line 819 initializes module attributes;
lines 928/935 establish the qualified sys.modules entry and call exec. Its
`_init_module_attrs` at lines 733-793 does not overwrite an already non-None
__name__, and initializes loader/package/spec/file as applicable.

I independently executed the exact installed extension with its standard
ExtensionFileLoader, an observation-only wrapper and real module_from_spec in
a disposable CPU interpreter. The raw create-returned object had:

| Boundary/property | Observation |
| --- | --- |
| Raw __name__ | `uarray._uarray` |
| Raw __package__, __spec__, __loader__ | all `None` |
| Raw __file__ | exact inventoried extension path |
| Exec entry | held object, qualified sys.modules key, exact spec and wrapper, spec.loader, package/parent and file/origin all match |
| Exec return | same intrinsic name and exact identities |
| Native calls | create once, exec once |
| Keys referring to the module | only `scipy._lib._uarray._uarray` |
| Short parent/alias | neither `uarray` nor `uarray._uarray` present |

Thus design lines 116-125 avoid a real ordering error: requiring complete spec/
package/loader at raw create return would reject this legitimate object; waiting
until after importlib initialization permits those checks without fabrication.
Missing/None native names must still reject at the earlier supported raw boundary,
before importlib can populate a missing name. Ordinary SourceFileLoader's None
create return is not a returned module to inspect; the production wrapper's
Python branch itself constructs `ModuleType(self.name)` at line 478.

This observation wrapper was only a feasibility diagnostic. It was not a second
production importer, guardian/custody test, captured admission, canonical Go2
run, or proof that the future implementation performs these checks.

### 4. Identity, audit and failure requirements remain strict

The proposed query replaces only the file record's expected __name__ value.
Design lines 93-127 preserve exact held module, qualified key, spec, wrapper,
delegated-loader binding, package and origin relationships. Current resolution
and binding sites are `native_bootstrap.py:720-757`; exec begins at `:491`;
completion checks at `:517`; later audit at `:1190-1230`. The existing exec-entry
check covers fewer fields than the proposal requires: implementation must add
the specified full initialized checks there, not merely edit line 1177.

The same formal query must serve create-name, exec and audit comparisons, with
raw name validation where required. It cannot learn a name from the object or
use an alternate test/benchmark dispatch table. Comparing accepted name metadata
must never mask replacement of spec, wrapper, delegate or returned object.

The existing external audit rejects unlisted sys.modules entries at lines
1206-1210. The design retains that rule for an extra short key even if it refers
to the held object or contains None. It neither converts the intrinsic name into
a creator-publication row nor grants another create/exec. Creator-self alias
absence/reconciliation semantics remain separate and unchanged.

The current `fail` path at lines 625-661 retains exceptions and available deltas
on repeated intake, while terminal notification occurs once. Retaining the exact
returned owner in records is essential because importlib deletes its qualified
sys.modules entry when exec raises. Design lines 122-127 preserve sticky failure,
original error and custody, prohibit COMPLETE/new ARM/retry/normal release or
fallback sync, and forbid recovery when an enclosing importer catches the error.
The parent trusted-native/pre-return ownership limitation is not relabeled a
Python-wrapper safety guarantee.

### 5. Acceptance and cleanup are bounded and testable

Section 4 supplies meaningful positives and adversarial negatives at distinct
supported boundaries, not just matching string labels. In particular it requires
the exceptional row to reject the full import key as a runtime name, wrong/missing/
non-string/empty names, wrong kinds/packages, identity substitutions, changed-back
observed mutations, unlisted short aliases, wrong phase/preload/reload/file drift,
and repeated read-only audit behavior. These would distinguish a correct narrow
implementation from global relaxation, dual-name acceptance or late learning.

The exact real extension must subsequently pass through the captured driver
without rewriting metadata and with once-only real create/exec and retained
identities. Ordinary import observations and synthetic delegates cannot satisfy
that criterion. Both actual Go2 materializations and guarded Warp/CUDA upload/
readback remain required, as do the full parent both-lane/fault/hang/dual-device/
exit matrix, versioned runtime-custody evidence and independent code review.
A subsequent different failure remains a failure and needs separate review.

Section 5 introduces no new experimental runtime path and retains the selected
dependency policy only while needed. Reviewed dependency cleanup removes obsolete
relations; existing losing-lane/temporary-driver obligations remain in force.
No new long-lived alternate importer or automatic permission-discovery mechanism
is authorized.

## Verification, provenance and limitations

Read the parent and supplement documents listed in Related Files, the actual
source/loader/validator/audit paths above, source-policy/catalog declarations,
the production-entrypoint unit fixture, and the four-file documentation diff.
The generated-stdlib implementation note's passing and failed historical test
totals were not independently rerun or promoted to this review's proof.

Local read-only tools: UTF-8 Get-Content, rg, git status/rev-parse/diff/show,
and SHA-256 over normalized LF design bytes. Broad outputs that truncated were
followed by narrower reads of the relevant source/document sections.

Remote diagnostics were passed over stdin, without writing a script, cache or
artifact into the candidate:

```text
ssh -o BatchMode=yes -o ConnectTimeout=10 tileops-dev \
  'timeout 20s /home/ang.gao/miniforge3/envs/robot_sim/bin/python -I -S -B -'
```

The inventory verifier used timeout 30s. All four diagnostic processes exited
normally; none reached its timeout:

| PID / parent | Fresh execution |
| --- | --- |
| 3618849 / 3618848 | real native create/init/exec metadata feasibility, no backend imports |
| 3619068 / 3619067 | historical failure artifact hashes and bounded inspection |
| 3619890 / 3619888 | actual CPU environment parser/verify_files, precise terminal summaries, installed importlib source |
| 3620617 / 3620616 | historical CPU trace hash/boundary scan and exact importlib functions |

The real interpreter was CPython 3.12.13, Anaconda build, at the packet's exact
robot_sim path. The CPU environment verifier explicitly added only the repository
path to import its real contract; it asserted no NumPy/SciPy/Torch/Warp/trimesh/
PIL/NetworkX import. The native diagnostic directly selected only the inventoried
extension and asserted no other SciPy/NumPy/Torch/Warp import. Neither diagnostic
is represented as an isolated captured worker. No package install or site edit
occurred; -B prevented bytecode writes.

The actual table passed `RuntimeEnvironmentContract.from_payload` and
`verify_files(expected_sha256=...)` using independently fixed canonical digest
`d2b1edc48c5a28a4ef71237fdf16d92d089f02f3fe7926c858d3e0c651146d3b`.
This verifies current CPU deployment facts, not the missing metadata policy.

Historical guarded artifacts independently matched under
`/data2/shared/ang.gao/c2d3-f97186c-go2-960-guarded-author-v3`:

| Artifact | SHA-256 |
| --- | --- |
| external_failure.json | 85ef3338a51beadca0ca05307dcb24a4e28d25222cdb5c99ed0ca81247cf5b5c |
| failure.json | 55fab24bf3a637d363b37a0c8288f14c44307585d64fa8bae75841e793c9aed6 |
| custody_artifact.json | c28e0ee245812a335b7b95bf7680f7a92a45b3cd311201829e669f61127a6007 |
| supervisor_stdout.json | 586badbade5867619c23b38c03c11c200fdc5e1910a1cb358a4e2f66592fd68e |

Events 2322-2325 identify resolved/create-started/created/exec-started for the
qualified extension; event 2326 is its first failure. Retained diagnostics show
the intrinsic name with correct file/package and exact spec/wrapper relations.
The supervisor summary records EXITED, code 74, two earlier completions, no
expired deadline and no kill request. Worker summary records full quarantine,
unknown scope, failed sequence 2, release not authorized, zero emergency sync
and zero terminal ARM. The outer 'requires trimesh' SystemExit is not evidence
of a missing installed package. These are checked historical records, not new
supervised execution or a complete custody proof.

Historical `/data2/shared/ang.gao/c2d3-go2-960-cpu-dependencies-v4.json` matched
SHA-256 `ee48dc09dfb8ddd06f1b171148dd923aa19ee7a22f8186243a36bea7b786ce5a`.
An independent scan found one name mismatch at each boundary: among 74 nonnull
create_returned names, 903 nonnull exec_started names and 901 nonnull exec_returned
names. Each mismatch was this same extension. These counts are historical trace
observations, not newly executed imports or a guarantee about every workload.

Final remote check at `2026-10-08T03:21:28+08:00` still showed clean candidate HEAD
and matching design SHA. Scoped `ps -p` for all four reviewer PIDs and their
timeout parents, plus historical packet PIDs 3594945/3594988/3594989, returned no
rows (status 1 is the normal no-match result). All local tool invocations also
terminated. No reviewer background process or observed owned hang remained.

Not run: pytest regressions, a repaired implementation, real captured positive,
canonical materialization, GPU upload/readback, fault/hang/dual-GPU/exit matrix,
or full C2d3 code review. No backend/GPU API or GPU residual query was used;
no GPU work was started. Other users' processes were neither signaled nor
modified. No native exception-safety guarantee or real-device property is
inferred from this CPU review.

## Exact authorization and closed gates

Approved only: implementing the exact design digest above, with the one literal
native-name expectation in the existing production contract, unchanged qualified
import/file/object authority, specified retention/check/failure ordering, required
tests, and next source-policy/catalog pinning. No new import, alias/publication,
caller field, resolver, package modification or evidence-minted capability is
approved. The implementation still requires its independent review and mandatory
parent evidence; this verdict is not a CPU-subset approval of unfinished C2d3.

Whole C2d3, full C2d/C4, C2e/f, C4c3/c4, C4d/B3, product GPU preparation, a3b,
result/baseline, R2/host payload and shipping remain closed. No live lease,
renderer, final readiness, measured sample, performance result, product
activation or output-delivery authority is established.

The sole reviewer write is this requested report via apply_patch outside the
candidate, for unchanged archival. Its digest is reported separately to avoid
a self-referential hash.
