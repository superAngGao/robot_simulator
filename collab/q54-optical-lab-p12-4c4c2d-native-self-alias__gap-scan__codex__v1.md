# C2d3 native self-alias contract gap scan v1

Initiative: q54-optical-lab-p12-4c4c2d-runtime-adoption
Stage: gap-scan
Author: codex, independent reviewer
Version: v1
Date: 2026-10-08
Status: in_review
Related Files: runtime-adoption design-request/review v2; optional-adapter design-request/review v2; optical_worker/runtime_environment.py; optical_worker/native_bootstrap.py; tests/unit/optics/test_runtime_import_admission.py
Owner Summary: The current nonempty creator-attribute-path scheme cannot truthfully express the four observed SciPy native self-aliases. A narrow, explicitly reviewed creator-root identity selector is needed; parent-package attribution is not a legitimate substitute. A separate synthetic importer-mechanics check also found skipped late failure custody after the first failure. Neither conclusion approves incomplete C2d3 code or opens a product/GPU gate.

## Identity, scope and verdict

Exact reviewer session: `/root/c4_c2d3_dependency_alias_gap_v1`.

Verdict: **Request changes**, scoped only to contract adequacy for truthful native self-alias representation. This is an independent gap scan, not a C2d3 code-approval checkpoint and not a request to approve a CPU subset.

- Frozen and independently observed HEAD: `56acbd375180f5ea89cf8fca3e3e7f854dbc4954`.
- WIP implementation parent: `b5f7737fb3788b15d02bad420533ac1fe94db1f9`.
- Local repository: `C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design`, branch `implementation/c4-supervised-waits`.
- Remote isolated repository: `/data2/shared/ang.gao/robot_sim_c2d3_binding_wip`, same HEAD.
- Primary SSH host: `tileops-dev`, `183.166.183.33:40031`; no fallback host.
- Interpreter: `/home/ang.gao/miniforge3/envs/robot_sim/bin/python`, observed CPython 3.12.13.
- Initial and final local and remote working trees were clean. Parent-to-HEAD diff contains only the 152-line runtime-cold-Warp evidence v3 record; diff-check passed.
- Runtime-adoption design-request v2 SHA-256: `1a0d3eed4dd182ddd5ea4ad0c19344cad735aabbdd317d77a14aad1932cc0224`.
- Optional-adapter design-request v2 SHA-256: `877da7e5476ff85e9273b591481447a5c76ba4636bcdee10100af8098ae675eb`.

Read AGENTS.md, collab/README.md and collab/REVIEW_PRINCIPLES.md fully, and the accepted contract/supplement relevant to this scope. This session did not author the candidate or subdelegate. Author observations were treated as claims, not permission or acceptance evidence.

## F1 — P2: the creator itself is not representable by the current selector

Locations: runtime-adoption design-request v2 section 4.2; `runtime_environment.py:220–244`; `native_bootstrap.py:387–439,951–994`.

The schema requires every publication to use an admitted file-module creator, admitted file-module publisher and a **nonempty** list of identifier attributes. Reconciliation starts from the held creator module and follows `vars(value).get(attribute)`. The scheme can express dependency-owned attributes, such as a module object held on its actual creator root. It has no explicit way to select the held root object itself.

Fresh independent observation of the existing canonical `go2_static_960x640_v1` CPU materializer found:

| Published short key | Actual qualified native module/root | Returned module and alias identical | First observed publication |
| --- | --- | --- | --- |
| `_cyutility` | `scipy._cyutility` | yes | its own delegated native exec |
| `_csparsetools` | `scipy.sparse._csparsetools` | yes | its own delegated native exec |
| `_moduleTNC` | `scipy.optimize._moduleTNC` | yes | its own delegated native exec |
| `_ni_label` | `scipy.ndimage._ni_label` | yes | its own delegated native exec |

For every row, the short key was absent before and after `create_module`; the qualified key was present at `exec_module` entry; after delegated exec, both keys referred to the exact module returned by create. The native module's `__name__` and `__spec__.name` remained the qualified name. These are extra names for already file-loaded objects, not additional file loads or newly created child modules.

The observer wrapped the actual standard extension delegates, forwarding each create/exec exactly once. No extension bytes or behavior were replaced. Nested context was also observed: `_cyutility` loaded inside `scipy._lib._ccallback_c` exec, and `_csparsetools` inside `scipy.spatial._ckdtree` exec. The short-key changes occurred inside the respective inner module's exec, not merely somewhere in a broad parent import. Attribution here is to the observed native delegate boundary; this is not instruction-level C tracing or proof of native internal exception safety.

Canonical CPU materialization completed, and its native manifest SHA-256 was:

`ac3d950309917168d0bb5758e9916adcffcbb17bd68f6611e4fe0f3bdfed3a12`

This matched the separately hashed author observation. The existing asset preflight, render-source materializer, fixed fixture constructor and CanonicalNativeInput constructor were called unchanged. No copied mesh importer or substitute scene was used. Torch/Warp imports were explicitly denied by the observer and were absent after execution.

### Legitimate mapping alternatives were checked

The four roots have no direct self-reference attribute. A second fresh fixed CPU-extension probe traversed the entire finite graph reachable by the production selector's raw `vars()`/identifier-component semantics. It found no nonempty path back to any root, both at native exec-return and after the ordinary import completed. It exhausted 1,244 objects for `_cyutility`, 23,861 for `_csparsetools`, 23,805 for `_moduleTNC` and 23,811 for `_ni_label`; the 100,000-node safety bound was never reached. It did not invoke dynamic `getattr` or traverse dictionary/list items that the production selector cannot traverse. This is an observation of the installed dependency object graph with an observational loader wrapper, not permission to use incidental importer-internal references as creator provenance.

There are real parent/consumer references after import, which must not be ignored:

- `scipy._cyutility` is available as the `_cyutility` attribute on `scipy`.
- `_csparsetools` is held on `scipy.sparse` and `scipy.sparse._lil`.
- `_moduleTNC` is held on `scipy.optimize`, and as `moduleTNC` on `scipy.optimize._tnc`.
- `_ni_label` is held on `scipy.ndimage` and `scipy.ndimage._measurements`.

Those modules are holders/importers, not the observed native creation/publication root. In the second probe, all four package-parent attributes were still absent at delegated native exec-return, the current production reconciliation boundary. They were bound later by importlib. Thus using a parent as creator and publisher would both misstate provenance and postpone the check to an unrelated completion; using a parent creator with the actual native publisher would also fail at the actual boundary. The approved contract explicitly requires exact creation owner and publisher, not merely any convenient reference to an equal object.

Other purported workarounds are not valid:

- Treating a short key as a file module invents a load that was never requested or executed and contradicts its qualified metadata.
- Marking it `created_module` still needs the missing selector and misstates the same-object relationship.
- A namespace or absent/disabled-optional row is not an executed native self-alias.
- Adding a synthetic self attribute, deleting the short alias or modifying SciPy would change dependencies and evade the requested contract.
- Broad name permission, runtime-discovered table insertion, a name/hash-only check or a second importer is forbidden by the accepted design.

A fresh synthetic payload passed through the existing production schema produced the exact rejection for the natural root selector `attributes=[]`:

`publication requires a fixed nonempty attribute path`

This is not a production Go2 environment-table experiment. No deployment table was authored or granted new permission.

### Minimal explicit contract gap and proof needed

The missing policy is a narrowly defined **creator-object identity selector** for a self-alias, distinct from the existing nonempty creator-attribute selector. A reviewed supplement should choose its exact encoding (for example an explicit selector kind, or a narrowly restricted empty path); this report does not silently choose or implement a schema change.

For the observed case, the source is the exact held file-module record for the qualified native root, and the publisher is that native root's observed execution. The alias must equal that previously retained module object, not merely an object with matching metadata. The qualified file-module record keeps its real create/exec provenance; the short name receives alias/publication evidence only. Both names must remain associated with the same identity and source owner.

The supplement needs to freeze:

1. Selector semantics, allowed kind and exact source/publisher relationship; normal attribute publications must retain their existing rules. No general empty-path waiver for created modules, arbitrary roots or cycles.
2. Relevant create/exec timing. These four aliases arise during exec, after the native create return has already been put in the retained file-module registry. The existing before-exec retention is usable; these observations do not justify new pre-return publication permission. Truly pre-return native state remains under the existing process/native-trust obligation.
3. Exact phase and completion checks; unlisted, missing, early, replaced, wrong-source and wrong-publisher aliases fail. Repeated audits are read-only and preserve identity, not new admissions.
4. Versioned schema/evidence compatibility and a later, separately reviewed frozen table grounded in pinned dependency facts. The four observations above are evidence for that work, not a table-authoring authorization or proof of closure completeness.

Required proof after repair includes production-schema positives/negatives, existing attribute-publication regressions, cold real CPU observations of these exact four roots through the captured production importer, and alias/qualified-key replacement, omitted/extra alias, wrong creator/publisher, wrong phase, repeated audit and partial-exec/record/adoption-failure cases. Each failure must keep exact owners/errors and prohibit successful continuation. Then the whole mandatory C2d3 canonical, both-lane, both-Go2, upload/readback and real-device matrices and independent code review still remain. No trace or this report substitutes for them.

The existing schema is correctly rejecting an unrepresentable entry; the discovery is not a reason to loosen checks ad hoc. The accepted Torch-oriented attribute contract did not explicitly close this native self-root case. The optional-adapter supplement changes only two pre-execution denials and does not solve or authorize this alias case.

## F2 — P2: separate related implementation finding, late failure custody is skipped

Location: `native_bootstrap.py:542–561`, especially `if self.failed: return`; callers at `create_module`/`exec_module` failure intake. This is an existing sticky-custody implementation obligation, **not** part of the F1 contract-adequacy verdict and not a full review of the unfinished candidate.

The first failure records its exception and takes a `sys.modules` delta snapshot. Every later `fail()` call returns before retaining its exception, taking a new available-object snapshot or processing its native-create flag. A trusted enclosing module can catch the inner started-import failure, publish another module, then itself fail. The session remains failed, which is correct, but the late object's exact identity and the outer exception are omitted from this admission object's failure-custody containers.

Exact reviewer repro is preserved outside the candidate at:

`C:/Users/AG/新建文件夹/c4_assets/c2d3-alias-contract-probe-v1.py`

It loads the existing importer fixture with `runpy.run_path`, uses `importer.__wrapped__(tmp_path, pytest.MonkeyPatch())`, and calls the actual production schema/finder/loader/admission methods. Its source-session and guardian prerequisites are the existing unit doubles, not real guarded lifecycle objects. Source fixtures are:

```python
# c2ext/__init__.py
import sys, types
try:
    from . import child
except ImportError:
    pass
sys.modules['c2ext.late'] = types.ModuleType('late')
raise ValueError('outer_after_late_publication')

# c2ext/child.py
raise ImportError('inner_started_failure')
```

After `importlib.import_module('c2ext')` raises the outer ValueError, the probe obtains `late = sys.modules['c2ext.late']` and checks:

```python
admission.failed                              # True
any(x is late for x in admission.retained)     # False
'c2ext.late' in admission.records              # False
any(x is error for x in admission.failures)    # False (outer ValueError)
[str(x) for x in admission.failures]           # ['inner_started_failure']
```

The sole failed event names `c2ext.child`; no second intake records the parent failure. Importlib removed the failed parent key; its originally retained module still exists. The late module remains in the mutable sys.modules table at this observation point. This repro does **not** claim it was already destroyed, freed native storage or escaped quarantine; it proves missing explicit stable retention of the newly available identity and the original outer exception by this admission path. Incidental reachability through mutable global tables is not the required exact failure snapshot.

The issue is not manufactured by allowing normal admission after failure: the late insertion is done by the already-executing trusted parent body, which does not consult the finder. Inspection of actual `_SourceSession.fail` shows it marks the controller and calls `terminate`; `NativeGuardedSetupLifecycle.terminate` returns termination evidence. Thus one must not assume that Python dependency code cannot catch the original import exception merely because sticky failure was set. No real guarded late-publication run was executed here, so final worker-exit behavior is not claimed.

Proposed direction: separate idempotent terminal-state/notification handling from monotonic failure-custody intake. Subsequent available owners and original exceptions must still be strongly retained in the already-held graph, without retrying normal adoption, ordinary release, synchronization, ARM or successful admission. A native-create scope escalation must not be silently skipped merely because an earlier failure was Python-side; keep conservative terminal handling.

Existing tests establish initial partial retention and stickiness, but do not publish a new object after the initial snapshot or check the subsequent exception. Six unchanged relevant cases passed in this review, including the first-failure swallowed-error test; that does not close this counterexample.

Proof required: reproduce the exact late-delta case through production importer methods, check exact late object and each exception retained before unwind, test removal/rebinding of the mutable publication key without losing the held object, and cover repeated intake without duplicate terminal operations. Extend to actual supervised partial native construction/exec and recording/adoption failures under the existing mandatory matrix. The synthetic fixture is not native exception-safety or GPU evidence.

## Other high-risk checks and bounded conclusions

- `create_module` retains the returned native object in the admission registry before explicit `hold()` can fail. Its spec/delegate are held before delegation. No missing-root finding was established for this straight-line returned-object path.
- `exec_module` reconciles before its completed state/event and routes reconciliation failure into failure custody. The observed aliases are therefore compatible with the existing timing once root identity is expressible; no dependency rewrite or pre-exec fabrication is needed.
- Initial failure intake scans available module deltas for retention only, not permission. F2 concerns later intake, not a recommendation to auto-admit any scanned name.
- Native state before create return and native exception safety remain explicit trusted-process/dependency obligations. This review did not inspect generated SciPy C implementations or inject a real native constructor fault, and cannot claim to reverse native internal cleanup.
- Existing alias/created-module provenance and exact retained-source semantics must remain explicit in the supplement. No generic ModuleType/hash/name shortcut or laundering through a parent importer is acceptable.

## Verification commands and results

All SSH calls used:

```text
ssh -o BatchMode=yes -o ConnectTimeout=8 -o ServerAliveInterval=5 -o ServerAliveCountMax=2 tileops-dev
```

Local read-only inspection used `rg`, UTF-8 `Get-Content`, `Select-Object`, `ConvertFrom-Json`, `Get-FileHash`, and:

```text
git status --short
git status --porcelain=v1
git rev-parse HEAD
git branch --show-current
git diff --stat b5f7737fb3788b15d02bad420533ac1fe94db1f9 HEAD
git diff --check b5f7737fb3788b15d02bad420533ac1fe94db1f9 HEAD
```

Remote identity checks repeated git HEAD/status/log and interpreter version. Author v1 script/report SHA-256 values were independently matched locally and remotely:

- `/data2/shared/ang.gao/c2d3-audit-go2-cpu-dependencies-v1.py`: `13fc3a184f1fdf6b0b86d0635a258081bbcc60656319ce4cce30d4386ac26b84`.
- `/data2/shared/ang.gao/c2d3-go2-960-cpu-dependencies-v1.json`: `e34cc5d099ccac9988b9c83d629c4b1ff9205efac52fc02d4df9fd0b6de9c571`.

The author v1 script was read; its report's revision, manifest and aliases were inspected. Its original execution was not counted as this review's execution. A later author message mentioned a v2 observer; this reviewer did not run or rely on that artifact.

Three new reviewer-written scripts were sent over stdin; no remote scratch source file or output report was created:

```powershell
Get-Content -Raw -Encoding UTF8 C:/Users/AG/新建文件夹/c4_assets/c2d3-native-self-alias-observer-v1.py | ssh -o BatchMode=yes -o ConnectTimeout=8 -o ServerAliveInterval=5 -o ServerAliveCountMax=2 tileops-dev "timeout 50s /home/ang.gao/miniforge3/envs/robot_sim/bin/python -I -S -B -"
Get-Content -Raw -Encoding UTF8 C:/Users/AG/新建文件夹/c4_assets/c2d3-native-self-alias-path-check-v1.py | ssh -o BatchMode=yes -o ConnectTimeout=8 -o ServerAliveInterval=5 -o ServerAliveCountMax=2 tileops-dev "timeout 30s /home/ang.gao/miniforge3/envs/robot_sim/bin/python -I -S -B -"
Get-Content -Raw -Encoding UTF8 C:/Users/AG/新建文件夹/c4_assets/c2d3-alias-contract-probe-v1.py | ssh -o BatchMode=yes -o ConnectTimeout=8 -o ServerAliveInterval=5 -o ServerAliveCountMax=2 tileops-dev "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 timeout 30s /home/ang.gao/miniforge3/envs/robot_sim/bin/python -I -S -B -"
```

All returned exit 0, without timeout. The first completed the real ordinary CPU canonical materialization and the second completed all fixed-path inspections described above. The third reproduced the schema rejection and F2, then called `pytest.main` with `-q -p no:cacheprovider` for these unchanged nodes:

```text
tests/unit/optics/test_runtime_import_admission.py::test_creator_publication_has_no_invented_file_execution
tests/unit/optics/test_runtime_import_admission.py::test_publication_requires_exact_held_creator_and_complete_table
tests/unit/optics/test_runtime_import_admission.py::test_swallowed_started_import_error_is_sticky_and_retains_partial_module
```

Result: **6 passed in 0.52s** (the second node has four cases). Pytest plugin autoload was disabled. These tests use production import mechanics with unit custody prerequisites and do not execute native binaries.

Reviewer scratch SHA-256 values:

| Local artifact under c4_assets | SHA-256 |
| --- | --- |
| `c2d3-native-self-alias-observer-v1.py` | `e5602e6ea77c3b6469dbea844385cd87e60d4bbb2b4aab14fe22cd05b606fcd2` |
| `c2d3-native-self-alias-path-check-v1.py` | `1b892dfab5dd1a4c245b1f363ba55db350d8e4bc872d4ecd558c2501b3ff60e9` |
| `c2d3-alias-contract-probe-v1.py` | `66480576eadda83fe2f81954b0b2b68100c9a0615aae1a10d15b61291956b66d` |

## PID, cleanup and limitations

| Activity | Python PID | Parent timeout PID | /proc start ticks |
| --- | ---: | ---: | ---: |
| Canonical CPU observer | 3427268 | 3427267 | 225546473 |
| Fixed native root/path observer | 3430719 | 3430718 | 225560377 |
| Contract/late-delta probe and six regressions | 3432057 | 3432056 | 225568642 |

Each foreground SSH command completed normally with exit 0. The subsequent exact check:

```text
ps -p 3427268,3427267,3430719,3430718,3432057,3432056 -o pid,ppid,lstart,args
```

returned only the header. This is scoped OS absence after normal command completion, not a claim of the product supervisor's wait/reap protocol, global host idleness or GPU-residual validation. No process signals, broad kills or GPU queries/resets were issued. No GPU backend import or GPU operation was run.

There were no candidate, tracked-source, deployment-table, asset or site-package edits; no installs, native-library changes, commits or shared-policy changes. Persistent reviewer artifacts are only this report and the three scratch scripts outside the candidate. The synthetic probe used the existing fixture to create tiny modules in its own `TemporaryDirectory` and cleaned that scoped directory on exit; pytest used ordinary temporary fixture directories. Bytecode and pytest repository cache writes were disabled. No material user data was deleted.

Initial default PowerShell decoding garbled Chinese instructions; the complete authoritative principles were reread with UTF-8. Some large combined displays were truncated; relevant contract/source ranges were reread in smaller UTF-8 chunks. These were display limitations, not product failures.

Not executed: production captured Go2 admission with a frozen complete environment, the 1920x1080 case, actual guarded/native fault injection, schema repairs, real supervisor fault/reap tests, Torch/Warp initialization, either GPU lane, uploads/readbacks, multi-device/ambient-state checks, GPU residual checks, all required C2d3 regressions, or complete remaining dependency/native-source closure. The successful ordinary CPU manifest is useful corroboration, not an output-array equivalence acceptance test or worker-admission proof.

## Handoff and closed gates

First obtain an independently reviewed bounded contract supplement for F1; do not improvise a table mapping with a false parent creator/publisher. F2 can be addressed as a substantiated implementation counterexample under the existing sticky failure-custody requirements, with new independent review of the eventual frozen changes. This report neither requires nor grants approval of the current unfinished C2d3 implementation.

Full C2d, C2e/f, C4c3/c4, C4d/B3, product GPU preparation, a3b, result/baseline, R2, host-payload and shipping remain closed. No prepared sample authority, final optical readiness or benchmark result is established. C4 remains incomplete.
