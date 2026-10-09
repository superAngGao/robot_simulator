# Independent setuptools CPU execution preflight

Initiative: Q54 / P12.4 / C4c2 / CUDA G1
Stage: independent code/execution preflight, dependency-only
Author: codex, independent reviewer /root/setuptools_cpu_preflight_v1
Version: v1
Date: 2026-10-09 (Asia/Shanghai)
Status: in_review — Request changes
Related Files: setuptools-cpu-preflight review-request v1; setuptools-admission design-request/design-review v2; compilation-admission design-request/design-review v3; setuptools-admission implementation-note v3/v4/v5 and test-evidence v2/v3
Owner Summary: Request changes. Two reproducible startup blockers prevent the exact frozen runner from reaching dependency initialization: its profile digest identifies Windows CRLF bytes rather than the clean target Git release, and its blanket subprocess audit hook prevents mandatory subsequent source audits from running Git. Actual dependency execution remains closed. The bounded helper/source checks pass, but do not cure either runner integration failure.

## Verdict and exact scope

**Verdict: Request changes.** Do not execute the proposed real dependency runner at this revision as an approved CPU integration run. R1 and R2 below are blocking for this exact execution scope. They are code/runner integration failures, not missing setuptools installation, missing GPU availability, or compiler/toolchain failures.

The review concerns readiness for one fresh, isolated dependency-only run through the real source session, external admission, initialization owner, guardian and ordinary dependency loaders, stopping at RESTORED. It is not whole-code approval. No installed setuptools/backend import, new dependency runner execution, real materialization test, compilation, native device execution, or full v2 compilation acceptance was performed. COMPLETE, compiler/native preflight, actual three builds, process/native failure matrices, C4 acceptance, readiness, timing and product/baseline results remain closed.

I did not author or modify the candidate and did not delegate this review. The initial packet was independent of the implementation conversation. During review the author supplied a hypothesis about the subprocess hook; I independently checked its production call chain and reproduced it without executing the new runner. Prior reports and author test counts were treated as claims, not approval evidence.

## Candidate identity and filesystem state

- Local repository: `C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design`.
- Local branch: `implementation/c4-supervised-waits`.
- Implementation base, independently confirmed as merge-base: `52fb36a8960596ddb2ac3decd8d129188e4b1eb4`.
- Frozen HEAD: `32fba6096f6425e70a4b671d2e14800dd10c2f8f`.
- HEAD tree: `b899a44e5b7bcd82b7e07e6550da71d30302c822`.
- Target: primary SSH alias `tileops-dev`; `/data2/shared/ang.gao/robot_sim_cuda_compilation_wip`, detached at that same HEAD/tree. No fallback host was used.
- Target interpreter: `/home/ang.gao/miniforge3/envs/robot_sim/bin/python`, CPython 3.12.13, Anaconda, GCC 14.3.0. Executed target Python used `-I -S -B`.
- Initial and final local and target `git status --porcelain` were empty. `git diff --check BASE HEAD` passed. No implementation edit, checkout, reset, dependency installation or environment mutation was made by the reviewer.
- A clean Git status does not imply byte equality between Windows and Linux working files. R1 is exactly such a difference.

Independently checked digests:

| Artifact | SHA256 |
| --- | --- |
| Accepted setuptools amendment v2, local bytes | `e8343ed7c9217a47ad45da5f055801cf43537a67f8059240453ed14078e27db8` |
| Accepted compilation design v3, local bytes | `028b52c466ef46433235d4ec2495cd47d00200822cc64dfaea95614fd5bf26a2` |
| Target runner release bytes | `29fa6575d5226d4b4f0c825e76ec52d720b39e3a08589236d4d6c670032b0a8d` |
| Source catalog v13, local and target | `17387af6b8188c40707da7e1835f639e87d1464ffa664b7cdcdb301eedd9d399` |
| Local CRLF compilation profile / runner's expected digest | `1725bfd714059d60010644e9160ef9022d27ce4d32ee89e2bae229979cb8dea8` |
| Actual clean target LF compilation profile | `0cb6288345fc3a2644e61bb01dc6e5897ee58d52eb71050c31b2e76f8ea60753` |
| Helper source archive | `8f5a7f25d4e7654cea2e157d44af708cc14b912edcdedbcc68e7852dcffdaac3` |
| Packaging source archive | `52db25f3a79ef2df681dc4c49149d788610237ce9527c10899f3f496ff72bbde` |
| Target importlib source archive | `3a41be7920961a96cb36e0aa442ecbc0ec0ff2f805ceadae9fadba917fab8289` |
| Profile author evidence archive | `17ebe27d160499e8e971d51b246fd6d2e9cfb0cb99dc7198ef4bad746f2766de` |
| Setuptools Conda installed-file facts | `45203dcfcd97a7b925b8c72061b71b38187328e7205e08ced3f28c98a097f403` |
| Offline wheel inventory | `5e7e954d4c79ce323c9712d97c883d9311dc7182f9fb72891912ab007ab9099a` |
| Target stdlib facts | `739e34613c16da9afc15d323790c38de198820d9735c5278b86a9566a95891e7` |

## Findings

### R1 — Blocking / high: the pinned profile digest is not the released target file

Locations: `tests/support/setuptools_dependency_process.py:23,53`; `tests/data/runtime_environments/183_cuda_compilation_author_v1.json`; the request's exact target command and source/profile evidence claim.

Trigger: run the proposed command against the clean exact target checkout. Its profile has LF endings and SHA256 `0cb6288345fc3a2644e61bb01dc6e5897ee58d52eb71050c31b2e76f8ea60753`. The runner requires `1725bfd714059d60010644e9160ef9022d27ce4d32ee89e2bae229979cb8dea8`, which is the local Windows CRLF working-file digest. `git ls-files --eol` reports local `i/lf w/crlf` and target `i/lf w/lf` for this profile. Both trees are Git-clean.

Consequence: the child fails at line 53, before `identity.json`, before the source session, and before the `try` handler that would write `failure.json` and exit 74. The proposed real integration cannot begin; merely checking target HEAD and clean status does not catch this.

Independent reproduction used only read-only hashing and the exact hash assertion, not the runner. The target isolated Python check raised `AssertionError` at the comparison. Separate `sha256sum` and `git ls-files --eol` confirmed the cause. The target profile's parsed content subsequently passed the production schema and installed-file verifier; that does not satisfy this raw-byte pin.

Why existing checks miss it: author source/file validation used uploaded authoring bytes, while Git normalized the committed file; profile parser tests check content rather than joining the exact released raw file to the runner constant. A clean checkout is therefore insufficient proof of this particular deployment identity.

Repair direction: pin and review the exact Git-release bytes that the target will execute; keep existing CRLF artifacts and their historical evidence unchanged. Freeze any line-ending policy deliberately, and bind the new runner/evidence to the actual target bytes. Do not weaken the check by silently normalizing bytes at runtime or accepting either digest without an explicit reviewed contract.

Required proof: fresh target clean checkout at the revised frozen HEAD, independently compared Git blob/working-file/runner digest, profile/schema/installed-file verification using that exact profile, and a new execution preflight before real dependency execution.

### R2 — Blocking / high: the subprocess audit hook prevents mandatory source revalidation

Locations: `tests/support/setuptools_dependency_process.py:34,67-70,79,94,135`; `optical_worker/native_bootstrap.py:2626-2637,3634`; `optical_worker/native_sources.py:146-157,221-236`.

Trigger: after correcting R1, the runner calls `_verify`, installs `_cpu_only`, then calls `session.bind_guarded_custody`. `_cpu_only` rejects every `subprocess.Popen`. Binding performs `self.audit()`, which calls `snapshot.check_loaded_modules()`, `verify_unchanged()`, `capture_native_source_snapshot()`, and `_load_catalog()`. `_load_catalog` executes `git rev-parse HEAD` and `git show <revision>:<catalog>`. These are not confined to initial capture; later explicit session audits require them too.

Consequence: the first binding audit raises the audit-hook RuntimeError before constructing the controller and before dependency initialization. Source failure latches true. Moving the hook past binding alone is not a complete fix because the subsequent guarded setup and final source audits take the same path. The packet's claim that legitimate Git subprocesses have all finished when the hook is installed is false for the inspected implementation.

Independent cold-source reproduction: loaded the actual bootstrap by file, installed and verified the exact CUDA source session, then installed a minimal audit hook rejecting `subprocess.Popen`, and called the actual `session.bind_guarded_custody` with a socket pair. The empty configuration was intentionally never reached: production binding audits before parsing it. The observed blocked event was:

```text
["subprocess.Popen", "git", ["git", "rev-parse", "HEAD"],
 "/data2/shared/ang.gao/robot_sim_cuda_compilation_wip"]
FAILED_LATCH True
RESERVED_AFTER []
```

The traceback passed through `native_bootstrap.py:2637,3634` and `native_sources.py:236,223,316,150`, then target stdlib `subprocess.py:1823`. The test caught the expected RuntimeError and exited zero; this is a reproduced failure, not a passing integration run. It imported no installed setuptools/backend, executed no new runner, and created no controller, compiler or device operation.

Why existing checks miss it: source tests run with the normal Git verification path available, while helper/replay tests use descriptive outer contexts and do not combine the new runner hook with actual source-session binding. Parsing the runner does not exercise this interaction.

Repair direction: reconcile the test-only execution restriction with required production source audits. Preserve source revalidation and the backend/compiler prohibition. Any allowance for fixed source-verification processes must be narrow and reviewed, with truthful process ownership/timeout behavior; do not retain the assertion that no descendant launch is possible if Git children are allowed. Do not broadly disable the restriction, bypass production audits, or substitute a test importer to make the integration pass.

Required proof: cold-source CPU counterexamples through actual binding and subsequent audits under the revised restriction, showing allowed verification remains functional and forbidden backend/process/library operations are still rejected. Account for any legitimate child lifetime under parent timeout/failure. Freeze the resulting source/runner/profile and obtain a fresh independent preflight before installed dependencies execute.

## Inspected high-risk areas

I read the shared review rules, complete preflight packet, accepted dependency amendment and compilation requirements, referenced implementation/evidence notes, and actual source changes. The following checks narrow the findings; they are not whole-implementation approval:

- The initialization owner is issued by the source session and tied to its compilation root, external admission, controller and thread. Tokens and retention use the existing observation registry/root; helper adoption uses the shared literal field/operation mapping. Construction is retained before fallible initialization/adoption. Bootstrap owns ordinary file loading and delegates the bounded dependency decisions.
- The actual shim call chain in the pinned source is `setuptools -> _distutils_hack.override -> do_override -> ensure_local_distutils -> shim -> importlib.import_module('distutils')`. The source's original spec producer creates the real custom alias loader. Inspected create/exec metadata checks correspond to same-object source/alias transitions. Restoration belongs to the owner and source-loader finally path; failure remains sticky.
- Topology code appends the declared vendor path to the existing path list and retains singleton custodian identity. Top-level stdlib PathFinder lookup uses original P rather than the appended vendor directory. Vendor and reserved dependency imports require the issued transaction; alias descendants use ordinary file/namespace rules.
- Resource/version operations are tied to the executing `jaraco.text` / `setuptools.version` module, current guard/thread, held bytes and actual reader/distribution objects. The pinned jaraco expression uses stdlib `files(...).joinpath(...).read_text(encoding='utf-8')`; source version lookup uses stdlib metadata. The resource's read-only handle differs from a literal BytesIO, but the real stdlib adapter helper tests exercise its intended interface. The exact installed expression remains unexecuted.
- Raw archive/source joins independently matched all 84 newly added module rows, sizes and digests to the helper/packaging archives and parsed them with AST. Every predecessor modules/namespaces/absence/publication/native-global/replacement/files/distributions row was preserved. This is a finite source/profile check, not a proof that runtime will reach no additional import.
- Inspected the lazy `monkey.patch_all -> _patch_distribution_metadata -> _core_metadata` route and Torch `_jit_compile`'s hipify/GeneratedFileCleaner imports, including their occurrence before its HIP-only work. The second route remains outside this CPU run. Python 3.12 selects stdlib tarfile in jaraco.context; jaraco.text's older resource fallback is excluded; Windows/JNA appdirs branches do not apply on the fixed Linux target. Cython, dl and `_distutils_system_mod` absence probes are explicit. `distutils.compat.numpy` is a compatibility source name, not a numpy import.
- Inspected shared action fields/role anchors and parser coupling for exact source/adoption/retention joins, action/read ordering, failure and v1/v2 separation. Helper/replay tests cannot prove a complete real v2 compilation record. The runner does not invoke that parser or fabricate COMPLETE.
- The runner explicitly creates only outer input/device/toolchain stand-ins, never `_Compilation` or fake cpp_extension. Its result schema is descriptive; assertions require RESTORED, completed ordinary dependency records, actual alias identity, restored originals, read handles/version, source/admission/root audits, and no completed initialization action. Its real controller terminal call supplies no synchronization callback, and the parent owns the exact child wait/kill fallback. R1/R2 prevent that path from being reached.

## Verification commands and observed results

Read-only inspection used `git status`, `git rev-parse`, `git merge-base`, `git diff`, `git ls-files --eol`, `rg`, UTF-8 `Get-Content`, `Get-FileHash`, target `sha256sum`, and stdlib `zipfile/hashlib/json/ast` archive reads. Initial PowerShell quoting of `HEAD^{tree}` was corrected using a quoted literal; initial stdin Unicode-path/output-encoding issues were corrected by using `Path.cwd().parent` and explicit UTF-8 output. These tooling errors changed no candidate files and were not treated as code failures.

Independent target pytest launcher was sent on stdin to:

```text
ssh tileops-dev /home/ang.gao/miniforge3/envs/robot_sim/bin/python -I -S -B -
```

It set `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`, changed to the exact isolated repository, and inserted only that repository and `/home/ang.gao/miniforge3/envs/robot_sim/lib/python3.12/site-packages` for pytest. Pytest used `-q -p no:cacheprovider`.

| Independent check | Observed result |
| --- | --- |
| `test_setuptools_admission.py`, `test_setuptools_contract.py`, `test_setuptools_replay.py`, `test_runtime_environment.py` | 130 passed in 3.61s, exit 0; reserved imports blocked by an additional reviewer audit hook and absent afterward |
| Source-only `test_native_bootstrap.py` | 77 passed, 1 skipped in 78.04s, exit 0; reserved modules absent afterward |
| Exact target raw profile pin | AssertionError; reproduced R1 |
| Exact target LF profile `RuntimeEnvironmentContract.from_payload` and `verify_files(expected_sha256=environment.sha256)` | 1,791 files verified, 4.372s, exit 0; reserved imports blocked and absent afterward |
| New runner AST parse and `compile(..., 'exec')`, without executing compiled code | Passed under target Python; runner digest recorded above |
| Cold-source production binding with subprocess-denying hook | Expected RuntimeError at first post-verification Git request; sticky source failure true; no reserved modules; reproduced R2 |
| Base-to-HEAD whitespace check and final identity/status checks | Passed; local and target remain clean at frozen HEAD/tree |

The bootstrap skip is the host-specific unsupported-platform behavior case: the test source skips it on the supported Linux interpreter and covers it on Windows. It is not a passing execution of that case. Test fixtures may create temporary scratch repositories/files and owned cold-source children; no tracked candidate file was changed and each launcher completed. No bytecode or pytest cache was requested.

The author separately reported 181 source-policy/profile/capture-only passes on this same HEAD. I do not count that author run as an independent reviewer execution. I independently checked archive/profile byte joins and performed the target production file verification above.

Minimal R1 reproduction, from the target exact tree, imports stdlib only:

```python
import hashlib
from pathlib import Path
p = Path('/data2/shared/ang.gao/robot_sim_cuda_compilation_wip')
b = (p / 'tests/data/runtime_environments/183_cuda_compilation_author_v1.json').read_bytes()
print(hashlib.sha256(b).hexdigest())
assert hashlib.sha256(b).hexdigest() == '1725bfd714059d60010644e9160ef9022d27ce4d32ee89e2bae229979cb8dea8'
```

Minimal executed R2 source-only reproduction, under the same isolated interpreter (not the dependency runner):

```python
import hashlib, importlib.util, socket, sys
from pathlib import Path
root = Path('/data2/shared/ang.gao/robot_sim_cuda_compilation_wip')
revision = '32fba6096f6425e70a4b671d2e14800dd10c2f8f'
spec = importlib.util.spec_from_file_location('source_test', root / 'optical_worker/native_bootstrap.py')
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)
entry.__name__ = '__main__'
sys.modules['__main__'] = entry
session = entry._install(root, revision, 'cuda', hashlib.sha256(
    (root / 'optical_worker/data/native_sources/v13.json').read_bytes()).hexdigest())
entry._verify(session)
seen = []
def forbid_subprocess(event, args):
    if event == 'subprocess.Popen':
        seen.append((event, args[0], args[1], str(args[2])))
        raise RuntimeError('independent cold-source probe blocks child launch')
sys.addaudithook(forbid_subprocess)
a, b = socket.socketpair()
try:
    session.bind_guarded_custody(a, {})  # audit fails before configuration parsing
except RuntimeError as error:
    print(str(error), seen, session.failed)
else:
    raise AssertionError('expected source audit to attempt git')
finally:
    a.close()
    b.close()
```

## Limits and next checkpoint

No substantive finding beyond R1/R2 is asserted for this bounded preflight. The absence of further findings is not approval of the whole initialization implementation, replay schema, compiler supervision or native path. Full negative dependency integration, real alias execution, actual jaraco/version reads under the custodian, complete v2 record replay and later native/build matrices remain unproved as required by the accepted designs.

The exact command under review was the packet's `python -I -S -B .../tests/support/setuptools_dependency_process.py parent <exact-tree> <fresh-absolute-run-directory>`. It was deliberately not executed. A revised implementation must remain frozen for a new independent review; this report does not authorize trying the old runner, modifying its pins on the target, or broadening its process restrictions during a live run.
