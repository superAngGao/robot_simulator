# C2c5 CPU source-policy primitive — independent code review

Initiative: q54-optical-lab-p12-4c4c2c-source-policy
Stage: review
Author: codex
Version: v1
Date: 2026-10-03
Status: in_review
Related Files: optical_worker/native_source_policy.py; tests/unit/optics/test_native_source_policy.py; benchmarks/results/q54_p12_4/cleanup_ledger.json; source-policy implementation-note codex v1; runtime-integration design codex v3 section 4
Owner Summary: Request changes. The pinned renderer fails on the actual Python 3.12 worker interpreter. Two ordinary, explicit source spellings also escape the declared dynamic-name/literal-reflection checks. No runtime admission or later-stage approval is given.

Reviewer session: `/root/c4_c2c5_source_policy_review_v1`, fresh independent reviewer, not the candidate author. Repository instructions and both collaboration documents were read. Author evidence was treated as a claim, not proof; the deployment failure was independently reproduced.

Verdict: **Request changes**.

## Exact revision and boundary

- Base: `7f32e6869e47fe31ea65a67ea317e3554144b41a`.
- Frozen HEAD: `3a365ea38c34b6ceb7a13ce52e1cdc6c0144d207`.
- Local repository: `C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design`, branch `implementation/c4-supervised-waits`.
- Remote repository: `/data3/ang.gao/robot_sim_c2c5_3a365ea`, detached HEAD at the same exact revision, reached at `ang.gao@192.168.80.101` with the supplied identity. Primary-host timeout/fallback authorization was supplied by the coordinator; the reviewer did not repeat the primary transport attempt.
- Local and remote `git status --short` were empty at inspection and after tests. Candidate remained unchanged. Exactly the four files listed in Related Files changed against the base, with 466 added lines.
- Scratch probe and this report live outside the candidate under `C:/Users/AG/新建文件夹/c4_assets/c2c5_review_v1`.

This review covers only the CPU byte/AST verifier and file helper, its tests, cleanup obligation, and accurate bounded documentation. It does not require or assert a complete per-lane source catalog, actual import ledger, loader/bootstrap, hostile-Python sandbox, runtime authority, or GPU execution. In particular, accepting a caller-rehashed ordinary source is intentionally only consistency checking, not admission. Findings below concern explicit forms of the very operations the primitive claims to reject, not prevention of arbitrary Python behavior inside a trusted process.

## Findings

### F1 — P1: interpreter-dependent AST pin rejects the exact reviewed renderer on the deployment interpreter

Location: `optical_worker/native_source_policy.py:26`, `:75-79`; affected positive test at `tests/unit/optics/test_native_source_policy.py:37`.

Trigger: validate the unchanged LF Git blob for `optics/warp_native_render.py` on `/data3/ang.gao/anaconda3/envs/robot_sim/bin/python` (Python 3.12.13).

Independent measurements:

- Full source SHA-256 on both interpreters: `636019f65594ac83dce1b386f4b3f684053e419a8cf81ba5feddce237cf318e4`, exactly the pinned release.
- Python 3.14.0 function `ast.dump(..., include_attributes=False)` SHA-256: `aece3ed687339523c475e1d145ef0edd4214af1cc2a70750c54a5741106bc410`, matching the candidate pin.
- Python 3.12.13 function AST SHA-256: `72772288a637b898e0116ce2b4b187aafe16dece32632e60dabb1465c45092f1`.
- Both byte validation and actual POSIX file validation reject with `read-only packed snapshot function has drifted` on Python 3.12.

Consequence: the one intended ctypes exception cannot accept its correct reviewed release in the required environment. This is a code portability defect, not missing Torch, GPU access, a changed source file, or a test-only mismatch. AST dumps are interpreter-produced representations; omitting location attributes does not make their serialized form stable between these interpreter versions.

Existing checks miss it locally because the constant and local tests use the same Python 3.14 representation. The deployed focused suite independently produced **89 passed, 2 failed**: the true renderer-positive failure and the newline/comment test encountering the earlier AST-pin failure instead of its expected full-source-pin rejection.

Repair direction: use a deliberately stable exact function-source representation/digest, or an explicitly specified canonical AST representation compatible with every supported interpreter. Retain the independent whole-release byte pin and the exact-function/ctypes restrictions; do not normalize input source to silently accept different releases.

Required proof: actual byte and file positive validation in Python 3.12.13, local supported-interpreter validation, and unchanged mutation/alias/write/comment/newline negatives. Re-run the focused tests and relevant CPU regressions at the revised frozen HEAD.

### F2 — P2: ordinary imported aliases of forbidden dynamic builtins pass

Location: `optical_worker/native_source_policy.py:95-101`.

Independent negative repro (with the SHA-256 of these exact bytes, path `optics/probe.py`):

```python
from builtins import eval as run
run("1 + 1")
```

The verifier accepts this source on both Python 3.14 and 3.12. It also accepts:

```python
from builtins import compile as build
build("x=1", "probe", "exec")
```

The equivalent `run = eval` is rejected. The first ordinary Python alias was independently executed in a separate harmless `python -I -S -c` command and returned `2`; it is an actual builtin alias, not an invented module or a speculative runtime interpretation.

Consequence: the advertised forbidden dynamic-name/alias source rule does not cover a straightforward language spelling of the same operation. No mutation of verifier internals, untrusted object, dynamic string construction, or hostile-process exploit is needed.

Why missed: `ImportFrom` checks only the module root (`builtins` is allowed), while imported names and aliases reside in `ast.alias`, not `ast.Name`. Current negative tests check assignment aliases, attribute access and direct calls, but not this import form.

Repair direction: explicitly reject forbidden dynamic builtin imports/aliases in the relevant import syntax, preserving any legitimately needed ordinary builtin imports. This does not require implementing arbitrary Python data-flow analysis.

Required proof: negative tests for ordinary `from builtins import eval/exec/compile/__import__` spellings with and without aliases, plus positives for permitted builtin imports and unchanged production-source checks.

### F3 — P2: qualified builtin literal reflection bypasses the explicitly implemented literal-reflection rule

Location: `optical_worker/native_source_policy.py:116-123`.

Independent negative repro (again exact self-consistent bytes under `optics/probe.py`):

```python
import builtins
builtins.getattr(runtime.core, "wp_cuda_launch_kernel")(1)
```

This is accepted on both interpreters. The same call spelled `getattr(runtime.core, "wp_cuda_launch_kernel")(1)` rejects with `native source reflects a forbidden runtime attribute`; direct `runtime.core.wp_cuda_launch_kernel(1)` also rejects.

Consequence: an explicit literal-reflected native ingress that the implementation note and section 4 say remains forbidden is classified as acceptable solely because the ordinary builtin has its module qualifier. The full-source renderer pin does not help: this is an ordinary non-exception source file being checked by the primitive. A later trusted literal manifest can prevent admission, but is not a substitute for the narrower AST rule this API already promises.

Why missed: the string check recognizes only `Call.func` nodes that are `ast.Name`. In this exact spelling the call function is `ast.Attribute`, and neither `builtins` nor `getattr` itself is forbidden by the attribute check. Current literal-reflection negatives exercise only the unqualified builtin or direct subscript.

Repair direction: cover the explicitly qualified builtin reflection spelling (and ordinary imported builtin reflector aliases if supported by the intended rule). Keep the requirement bounded to explicit source constructs; this finding does not demand blocking computed strings or arbitrary hostile reflection.

Required proof: qualified builtin `getattr`/`setattr`/`delattr`/`hasattr` with each forbidden literal rejects, with harmless literal reflection and policy-definition string data still passing. Include the exact repro above and the existing direct-form negatives.

## Other inspected behavior

- Full source and function pins are separate. The full renderer pin rejects any rehashed changed release; it is not merely trusting the caller-provided digest. Actual existing `_render_launch_snapshot` reads packed values, checks exact ctypes identities/address correspondence, and does not create CDLL handles or writable casts.
- Plain top-level sole ctypes import, exact snapshot-node membership, permitted attributes and load contexts are checked. Rehashed alias/write/cast changes to the pinned renderer fail closed; whole-source pinning also blocks changes outside the function.
- Canonical relative POSIX paths, known production roots/two named launchers, `.py` suffix, source/digest exact types and lowercase digest shape are checked. The roots are documented namespace limits, not a complete import allowlist.
- File traversal rejects symlinked relative components and verifies final containment/regular-file status before reading. Existing file and directory symlink tests passed without skips locally and remotely. The helper intentionally does not solve a later loader's TOCTOU or bytecode behavior. Missing/invalid root exception-type consistency and concurrent filesystem replacement were not promoted into unsupported sandbox requirements.
- The checker parses/reads rather than importing the inspected source. Independent imports showed no `numpy`, `torch`, `warp`, or `optics` in `sys.modules`; existing cold `-I -S` CPU-import test passed. A source containing a top-level `raise RuntimeError` was accepted without being executed.
- C0 manifest/protocol files, current allowlist and disabled preparation contract are unchanged. Search found no consumer of the new policy outside its own definitions/tests. There is no premature loader/dispatcher activation.
- The cleanup entry supplies selected-lane collapse/removal requirements at P12.4e and does not claim that backend selection has occurred. The implementation note accurately identifies the unimplemented source closure/import-ledger/bootstrap work.

## Independent verification and exact commands

Environment: local `C:/Python314/python.exe` 3.14.0; remote `robot_sim` Python 3.12.13. The standalone probe uses `-B`, runs pytest with `-p no:cacheprovider`, records test-child PID/parentage, applies a 90-second wait bound, and observes/reaps the child. No timeout occurred. The probe wrapper intentionally reports the child exit code in output rather than propagating it; the remote wrapper exiting zero is **not** evidence of a passing pytest run.

Read-only revision/diff checks:

```text
git status --short
git rev-parse HEAD
git branch --show-current
git diff --stat 7f32e6869e47fe31ea65a67ea317e3554144b41a 3a365ea38c34b6ceb7a13ce52e1cdc6c0144d207
git diff --name-only 7f32e6869e47fe31ea65a67ea317e3554144b41a 3a365ea38c34b6ceb7a13ce52e1cdc6c0144d207
git diff --check 7f32e6869e47fe31ea65a67ea317e3554144b41a 3a365ea38c34b6ceb7a13ce52e1cdc6c0144d207
python -m ruff check optical_worker/native_source_policy.py tests/unit/optics/test_native_source_policy.py
```

Results: clean exact trees; expected four-file delta; no whitespace errors; Ruff `All checks passed!`.

Local probe:

```powershell
python -B C:/Users/AG/新建文件夹/c4_assets/c2c5_review_v1/independent_probe.py C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design
```

Remote probe (stdin only; no remote candidate file created):

```powershell
Get-Content -Raw -Encoding UTF8 C:/Users/AG/新建文件夹/c4_assets/c2c5_review_v1/independent_probe.py | ssh -o BatchMode=yes -o ConnectTimeout=10 -i C:/Users/AG/.ssh/tileops_ed25519 ang.gao@192.168.80.101 '/data3/ang.gao/anaconda3/envs/robot_sim/bin/python -B - /data3/ang.gao/robot_sim_c2c5_3a365ea'
```

Each probe executes the equivalent focused command using its exact interpreter:

```text
python -B -m pytest -q -p no:cacheprovider tests/unit/optics/test_native_source_policy.py tests/unit/benchmarks/test_closed_worker_protocol.py
```

- Local: **91 passed in 2.19s**, child exit 0.
- Remote: **89 passed, 2 failed in 0.98s**, child exit 1, failures identified in F1.
- All F2/F3 accepted/rejected classifications were independently observed on both interpreters.
- Actual local renderer-file validation rejects against LF release digest, consistent with the explicitly documented CRLF checkout restriction, while local Git-blob validation passes. Actual remote renderer-file validation fails at F1 despite correct release bytes.
- An exploratory synthetic `from warp.context.runtime.core import ...` also passed syntactic checking, but was not used as a finding because that import path was not established as a real supported Python module. This distinguishes actual counterexamples from speculative syntax.
- Initial local and remote attempts mistakenly named `tests/unit/optics/test_closed_worker_protocol.py`; each exited immediately with “file or directory not found” and no tests run. File discovery corrected the path above. These are reviewer invocation errors, not product failures.
- Initial root-level `REVIEW_PRINCIPLES.md` lookup was wrong; the required complete document was then read from `collab/REVIEW_PRINCIPLES.md` before review conclusions.

## Round-end process hygiene

- Reviewer app-server identity observed: Windows PID 31768, parent 4408, creation 2026-10-03 03:01:32 local, Codex app-server executable under the VS Code extension. It is infrastructure, not a test residual.
- Local probe PID 21868, parent PowerShell PID 33592; owned pytest PID 20228, parent 21868. Probe observed pytest exit 0 and reaped it. A subsequent `Get-CimInstance Win32_Process` query for these PIDs (also earlier inspection shell 31256) returned no entries.
- Remote probe PID 2044245, parent 2044236; owned pytest PID 2044270, parent 2044245. Probe observed pytest exit 1 and reaped it. Subsequent `ps -o pid,ppid,lstart,args -p 2044245,2044236,2044270` returned only the header: none remained. Numeric identities are supported by launch-time parentage and the bounded launch interval; no PID was used for a kill operation.
- Remote `nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader` returned no compute-process rows. No GPU tests, worker launch, or supervisor process were requested in this CPU round.
- Final remote HEAD remained exact and clean; final observation time was 2026-10-03 06:13:58 +08:00. No hang, timeout, unresolved owned residual, broad kill, package installation, GPU reset, or unrelated worktree mutation occurred. This is scoped round evidence, not a diagnosis of any historical hang.

## Gaps, disposition and closed gates

The broader benchmark/core regression collections reported by the author were not independently re-run by this reviewer; their reported results are not substituted for independent evidence here. The focused CPU collection, real POSIX byte/file repro, negative snippets, static call-flow inspection and unchanged C0 regression are sufficient to substantiate this request for changes.

No concurrent symlink race, malicious in-process verifier monkeypatching, arbitrary computed reflection, complete native dependency closure, actual import-ledger enforcement, runtime source consumption, or GPU lifecycle property was tested or claimed. Those are either explicitly outside this trusted-process primitive or require later separately scoped integration work. Cold-import tests and module-state observations do not alone prove a future complete transitive source ledger.

F1, F2 and F3 require repair and a fresh independent review of the revised frozen candidate. Do not count this review as approving C2c5, complete C2c, C2d/e/f, C4, product GPU preparation, B3/host delivery, shipping, a3b, result publication or baseline generation. C0 protocol/allowlist and all existing gates must remain unchanged during the bounded fixes.
