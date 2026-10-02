# C2c5 CPU source-policy primitive — independent code re-review

Initiative: q54-optical-lab-p12-4c4c2c-source-policy
Stage: review
Author: codex
Version: v2
Date: 2026-10-03
Status: accepted
Related Files: optical_worker/native_source_policy.py; tests/unit/optics/test_native_source_policy.py; benchmarks/results/q54_p12_4/cleanup_ledger.json; source-policy implementation notes v1/v2 and review v1; runtime-integration design codex v3 section 4
Owner Summary: Approve with comments for the exact bounded CPU read-only primitive at 05d9ccf. Prior F1/F2/F3 are independently closed on Python 3.14 and actual POSIX Python 3.12. One non-blocking mutation-test quality follow-up remains. No source catalog, loader, runtime authority or later gate is approved.

Reviewer session: `/root/c4_c2c5_source_policy_review_v2`, fresh independent reviewer, not the candidate author. Read AGENTS.md, collab/README.md and collab/REVIEW_PRINCIPLES.md completely, then inspected the accepted section 4, prior report, repair claims, production code, tests, original-base diff and repair diff. An initial root-level REVIEW_PRINCIPLES.md lookup failed; the correct complete document was read before review work continued. Author assertions and test counts were not used as independent proof.

Verdict: **Approve with comments**. No unresolved blocking production-code finding or required evidence gap for this bounded primitive.

## Exact candidate and scope

- Original base: `7f32e6869e47fe31ea65a67ea317e3554144b41a`.
- Prior reviewed candidate: `3a365ea38c34b6ceb7a13ce52e1cdc6c0144d207`.
- Reviewed frozen HEAD: `05d9ccf370824f36cf757e5a251ad244de89b988`.
- Local: `C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design`, branch `implementation/c4-supervised-waits`.
- Remote: `/data3/ang.gao/robot_sim_c2c5_05d9ccf`, detached HEAD at that exact revision, accessed through authorized fallback `ang.gao@192.168.80.101:22`. The coordinator supplied the primary-host timeout and fallback authorization; this reviewer did not repeat the primary transport attempt.
- Both trees were clean before and after tests. Candidate files were not edited. Original-base delta contains only the new policy and tests, cleanup-ledger addition, implementation notes v1/v2 and archived review v1: six files, 764 inserted lines.
- Scratch probe and this report are outside the candidate under `C:/Users/AG/新建文件夹/c4_assets/c2c5_review_v2`.

Approval covers the CPU byte/AST checker and read-only file helper, the narrowly pinned existing ctypes snapshot exception, tests, cleanup obligation and bounded documentation. It does not establish complete C2c source admission. Self-consistent caller hashes remain consistency checks, not execution authority. The production-root list is a path namespace restriction, not an import allowlist.

## Prior finding closure

### F1 — closed: stable exact function bytes on both supported interpreters

Inspected `native_source_policy.py:74` onward. AST now locates the single top-level function but does not serialize it into the pin. Verification hashes its exact physical UTF-8 lines including the final newline. The whole renderer release pin is unchanged and independently enforced after the AST checks.

Independent results from Python 3.14.0 and deployed Python 3.12.13:

- Full LF Git release SHA-256: `636019f65594ac83dce1b386f4b3f684053e419a8cf81ba5feddce237cf318e4`.
- Function source SHA-256, independently extracted using `ast.get_source_segment` plus the physical trailing newline: `725c7f315fdba4ad6df78e31a149865fd97e26da3c95df24678dbadba0459199` on both interpreters.
- AST dump digests still differ: Python 3.14 `aece3ed687339523c475e1d145ef0edd4214af1cc2a70750c54a5741106bc410`; Python 3.12 `72772288a637b898e0116ce2b4b187aafe16dece32632e60dabb1465c45092f1`. They are no longer policy inputs.
- Byte validation passes on both. On Linux, the actual checkout file equals the Git blob byte-for-byte and `validate_native_source_file` passes against the frozen digest; no reconstructed fixture substitutes for that positive.
- Windows checkout bytes differ from LF release bytes, as documented. The focused test uses a declared LF fixture there, not a claim that CRLF is an admitted release.
- Rehashed CRLF rejects at the function pin; appended comments reject at the whole-release pin; valid write and ctypes-escape modifications inside the snapshot reject at the function pin; aliased ctypes import rejects at the plain-import check.

### F2 — closed: explicit forbidden builtin imports and aliases reject

Inspected the ImportFrom branch, including absolute `builtins` scope and wildcard handling. The original complete snippets independently reject on both interpreters with `native source imports a forbidden dynamic builtin`:

```python
from builtins import eval as run
run("1 + 1")
```

```python
from builtins import compile as build
build("x=1", "probe", "exec")
```

The focused tests cover eval/exec/compile/__import__, aliased and unaliased, plus wildcard imports. Permitted `from builtins import bytes as copy_bytes` still passes. This remains a syntax policy, not arbitrary Python data-flow analysis.

### F3 — closed: explicit literal-reflected ingress rejects regardless of call spelling

Inspected the forbidden string Constant check and its immediate Call/keyword/Subscript contexts. The original repro now independently rejects on both interpreters:

```python
import builtins
builtins.getattr(runtime.core, "wp_cuda_launch_kernel")(1)
```

Imported reflector alias, module alias, keyword literal, `__builtins__["eval"]` and `vars(...).get(...)` forms also reject. The focused tests exercise all four reflector names against all four forbidden attributes and four ordinary spelling styles. Harmless reflection and inert policy-definition containers pass.

The conservative rule also rejects a harmless call carrying exactly a forbidden string. This is clearly disclosed in implementation note v2 and encoded in the policy digest; it does not purport to perform runtime inference. It does not introduce an execution channel or broaden the approved exception.

## Non-blocking comment C1 — P3: ensure mutation tests produce valid Python

Location: `tests/unit/optics/test_native_source_policy.py:84`, `:85`, `:89`; source occurrences at `optics/warp_native_render.py:520` and `:2973`.

Trigger: the write/escape parameters use unrestricted `renderer_release.replace(b"snapshots = []", ...)`. This matches both the target four-space-indented snapshot function and an earlier eight-space-indented block. The inserted four-space line breaks the earlier block, so these two tests currently pass because parsing raises SyntaxError, not because a valid forbidden mutation reaches the pin/checker.

Consequence: these two regression cases alone do not prove the intended semantic mutation rejection. Existing `pytest.raises(ClosedWorkerProtocolError)` accepts both parse failures and policy failures. This is non-blocking because the exact reviewed function and module remain pinned, and the reviewer independently constructed both syntactically valid mutations scoped only to `_render_launch_snapshot`, asserted `ast.parse` success first, and observed function-pin rejection on both supported interpreters.

Bounded follow-up: scope replacement to the identified function and assert parseability before invoking the verifier, optionally matching the expected rejection. No policy or scope expansion is needed. Retain this comment and disposition in the evidence record; any later test edit is a separate reviewed delta, not part of this frozen approval. Proof needed for that follow-up is both valid mutants rejecting on Python 3.12 and the focused collection remaining green.

Disposition: the coordinator acknowledged C1 during this review and committed to fix mutation construction plus the parseability assertion in the next full source-admission candidate, with a fresh review of that delta. The current candidate remains frozen; its malformed mutation cases will not be represented as proof of valid write rejection.

## Other inspected invariants

- The snapshot implementation reads actual packed descriptor/type/address/byte values. The exception admits one plain top-level ctypes import and only exact permitted load-context references inside the pinned function. Existing module/function pins prevent rehashed modifications from becoming a broader exception.
- Byte verification checks canonical relative production `.py` paths, exact bytes/string types, lowercase SHA-256 shape, exact digest match, UTF-8 parseability and the policy. It returns no object/capability and imports no checked source.
- File verification resolves the trusted root, rejects symlinked relative components, checks containment and regular-file status, and reads bytes. File/directory symlink tests pass without skips. No later-loader TOCTOU or bytecode-cache guarantee is claimed.
- The checker imports only its CPU protocol/manifest dependencies. Cold `-I -S` test passes; independent module observations contain no numpy/torch/warp/optics. A source with top-level `raise RuntimeError` passes without execution.
- The complete diff leaves C0 manifest, protocol, dispatcher paths and runtime contract unchanged. Explicit `git diff --quiet` against C0 manifest/protocol/worker/runtime-contract paths exited 0. The focused C0 tests confirm old policy version and disabled preparation contract. Search found no production consumer of the new verifier outside its own definitions; no loader or activation was added.
- Cleanup ledger records P12.4e selected-lane collapse/removal, including removal if no selected consumer adopts the verifier. It does not imply backend selection or completed cleanup.

## Independent commands and results

Read-only inspection included:

```text
git status --short
git rev-parse HEAD
git branch --show-current
git diff --stat 7f32e6869e47fe31ea65a67ea317e3554144b41a HEAD
git diff --name-only 7f32e6869e47fe31ea65a67ea317e3554144b41a HEAD
git diff 3a365ea38c34b6ceb7a13ce52e1cdc6c0144d207 05d9ccf370824f36cf757e5a251ad244de89b988 -- optical_worker/native_source_policy.py tests/unit/optics/test_native_source_policy.py
git diff --check 7f32e6869e47fe31ea65a67ea317e3554144b41a HEAD
python -m ruff check optical_worker/native_source_policy.py tests/unit/optics/test_native_source_policy.py
```

No whitespace errors; Ruff `All checks passed!`. Repeated exact clean revision checks passed.

Probe commands:

```powershell
python -B C:/Users/AG/新建文件夹/c4_assets/c2c5_review_v2/independent_probe.py C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design
Get-Content -Raw -Encoding UTF8 C:/Users/AG/新建文件夹/c4_assets/c2c5_review_v2/independent_probe.py | ssh -o BatchMode=yes -o ConnectTimeout=10 -i C:/Users/AG/.ssh/tileops_ed25519 ang.gao@192.168.80.101 '/data3/ang.gao/anaconda3/envs/robot_sim/bin/python -B - /data3/ang.gao/robot_sim_c2c5_05d9ccf'
```

Each probe runs and checks the actual child exit of:

```text
<exact interpreter> -B -m pytest -q -p no:cacheprovider tests/unit/optics/test_native_source_policy.py tests/unit/benchmarks/test_closed_worker_protocol.py
```

- Local exact interpreter `C:/Python314/python.exe`, Python 3.14.0: first **175 passed in 2.49s**, final **175 passed in 2.22s**, both child exit 0.
- Remote exact interpreter `/data3/ang.gao/anaconda3/envs/robot_sim/bin/python`, Python 3.12.13: first **175 passed in 1.36s**, final **175 passed in 1.31s**, both child exit 0.
- First independent write mutation used the same broad replacement as the candidate test and revealed C1. The scratch probe was corrected, not the candidate. The final probe adds a valid ctypes-escape mutation and parses all renderer mutants before checking rejection. The final two runs establish the valid policy negatives described above.
- Tests use no pytest cache provider and inherit PYTHONDONTWRITEBYTECODE=1. Both wrappers assert the child exit code, reap it, and recheck clean status. No timeout occurred.

The coordinator separately reported 278 broader remote CPU passes. This reviewer did not rerun that broader collection and does not substitute its reported result for the independent evidence above. No GPU test, install, runtime launch, benchmark result or baseline was requested or performed.

## Process hygiene

- Reviewer infrastructure observed: Windows Codex app-server PID 31768, parent 4408, creation 2026-10-03 03:01:32 local. Inspection PowerShell PID 18240 had parent 31768. This infrastructure is not a test residual.
- First local probe PID 27064, parent PowerShell 21316; pytest PID 35488, parent 27064. Final local probe PID 35132, parent PowerShell 33632; pytest PID 13740, parent 35132. All pytest children were observed exit 0 and reaped. Scoped Win32_Process lookup for all six PIDs returned zero matches after both rounds.
- First remote probe PID 2057746, parent 2057664; pytest PID 2057776, parent 2057746. Final remote probe PID 2058996, parent 2058909; pytest PID 2059026, parent 2058996. Launch records include parentage and timestamps; all pytest children exited 0 and were reaped. Final `ps -o pid,ppid,lstart,args -p 2058996,2058909,2059026,2057746,2057664,2057776` returned only its header.
- Final `nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader` returned no rows. Final remote HEAD remained exact and clean at 2026-10-03T06:24:20+08:00; local clean check at 06:24:17+08:00.
- No hangs, timeouts, kills, GPU reset, unresolved owned residual or unrelated worktree modification occurred. No process-name-only ownership inference was used. These are scoped current-round observations, not a historical hang diagnosis.

## Remaining boundaries

No arbitrary hostile in-process monkeypatching/computed-reflection defense, concurrent filesystem race closure, full transitive repository source catalog, actual import-ledger enforcement, trusted bootstrap/source consumption, runtime authority, or GPU lifecycle acceptance is proved or claimed. These exclusions are accurate in the code and notes, not silently omitted acceptance work for this primitive.

Only the named CPU C2c5 primitive at the exact reviewed revision is approved with C1 retained as a non-blocking follow-up. Complete C2c, C2d/e/f, C4 completion, GPU preparation, B3/host delivery, shipping, a3b, result publication and baseline generation remain closed. Existing C0 protocol/allowlist and all product gates remain unchanged. Recording this unchanged report and commands/results in an evidence-only commit does not authorize other code/test/policy changes.
