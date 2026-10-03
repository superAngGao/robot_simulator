# Independent C2d0 captured-source CPU bootstrap code re-review v2

Initiative: q54-optical-lab-p12-4c4c2d-source-consumption
Stage: review
Author: codex
Version: v2
Date: 2026-10-03
Status: accepted
Related Files: source-consumption design-request codex v1; design-review codex v1; code-review codex v1; review-followup codex v1; optical_worker/native_bootstrap.py; optical_worker/native_source_policy.py; optical_worker/native_sources.py; tests/unit/optics/test_native_bootstrap.py
Owner Summary: Approve the exact revised CPU-only C2d0 implementation. Both original identity findings independently reject on the revised production verifier, and restoring mutated identities cannot revive failed sessions. Independent local and remote regressions pass; actual POSIX entry and mandatory real Go2 checks have no remote skips. No GPU, integrated lifecycle, activation or result authority is approved.

## Identity and verdict

Reviewer session: `/root/c4_c2d0_source_bootstrap_code_review_v2`, a fresh independent session receiving the review packet, not the implementation conversation. The reviewer did not author or modify candidate files. Scratch probes and this report are outside the candidate.

Verdict: **Approve**.

- Original base: `29b2055061d38211546774f510e70545701bc0e9`.
- Rejected prior candidate: `0fae0a4c6df4f5527196009ce994aee3f0b7385d`.
- Exact approved revised HEAD: `1f0576bfc607d24d2f5a04838051497f90a5592a`.
- Local checkout: `C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design`, branch `implementation/c4-supervised-waits`.
- Remote checkout: clean detached `/data3/ang.gao/robot_sim_c2d0_code_v2` on `ang.gao@192.168.80.101`.
- Accepted design SHA-256 independently checked: `bf63e0fa34a37668845c2672a46e052b8f919890904b24d7bf040852e5c81f7d`.
- Complete archived prior review SHA-256 independently checked: `9486ae73be138940fd90bc393879b6ea332a49f500494caa1933843a26486dbe`.
- Actual remote bootstrap SHA-256: `18e7bf22e88fa71dc150ddfe85151cc05498514784d85ac59571aab0fee0b10a`.
- Actual remote catalog v2 SHA-256: `a1795363dacfe24eb9b7b8f2c9a3fe4852108e150af8e0c11f2ad887073faf01`.

Initial and final local/remote status were clean at the exact revised HEAD; diff whitespace checks passed. The original dirty `/data3/ang.gao/robot_sim_c4c1b2` was not touched. The packet's twice-unreachable primary SSH transport was accepted as the routing condition; this reviewer used the explicitly authorized fallback and did not retry primary183. Author tests and author probe output were not counted as independent proof.

## Findings and closure of prior findings

No unresolved substantive blocking or non-blocking finding was established. There is no new severity/location/trigger/consequence/repair request. Verification limits below remain explicit and do not expand approval.

### F1 (prior P2), bootstrap owner replacement: closed

At `optical_worker/native_bootstrap.py:238`, `_SourceSession` strongly retains the actual initial `__main__` module, loader, spec and package metadata. It requires the canonical main name and configured bootstrap file. `check_live()` requires `sys.modules['__main__']` to remain that exact object and checks its frozen metadata. `_verify()` catches failures and sets the session's terminal failed flag; `audit()` has the same failure behavior.

The unchanged original repro replacing `__main__` with a fresh `ModuleType` containing only the copied file path now raises `SourceBootstrapError: bootstrap owner identity/metadata drift`. No success report is returned. Independently run submitted regressions also cover deletion, copied-dictionary impersonation, and name/file/package/loader/spec replacement, asserting failure and repeat rejection.

An additional independent actual-POSIX probe copied the entire bootstrap dictionary into a new module, observed rejection and `session.failed`, restored the original `sys.modules['__main__']` object, then verified again. Both Warp and CUDA sessions remained permanently rejected. This tests restoration rather than merely repeating the still-mutated input. The bootstrap is explicitly retained as the interpreter's trusted initial owner, not misrepresented as initially executed or retroactively authorized by the captured-source loader.

### F2 (prior P2), spec name/package drift: closed

At `optical_worker/native_bootstrap.py:278`, the audit derives the expected package from the fixed initializer/module layout, then checks module name/package, spec name/parent, loader identity, origin/file and package search locations together. Existing exact module/loader records and completed trace checks remain in force.

The unchanged original leaf-module spec-name and package mutations now each raise `SourceBootstrapError: module metadata drift`. Submitted tests independently run here cover both the initializer and leaf module and terminal rejection. Additional independent probes, on both lanes, changed each field separately, observed failed state, restored the field and retried: every session remained rejected. Real direct entry on Python 3.12 and explicit local LF release entry on Python 3.14 still pass, preserving actual eager/relative imports.

## Inspected correctness and scope

Read `AGENTS.md`, `collab/README.md` and `collab/REVIEW_PRINCIPLES.md` in full, as well as the complete accepted C2d0 design, design review, prior code review and repair note. Inspected the complete bootstrap and bootstrap tests, native source verifier/policy, affected policy tests, source fixture and Go2 tests, relevant C0 process tests, base-to-HEAD diff and rejected-to-revised diff. Prior conclusions were checked against actual code rather than treated as authority for this candidate.

The repair delta is confined to strong bootstrap-owner retention/checks, canonical spec/package audit, twelve negative regression cases and required source/policy/catalog repins, plus evidence records. Loader/finder class pins are unchanged. The base-to-HEAD production scope is the new CPU bootstrap, catalog v2, native AST v3 exception, source-ledger/catalog updates and cleanup obligation. C0 worker/protocol/manifest, both tools launchers, shared workflow rules and activation contracts have no diff.

High-risk properties checked:

- Cold start requires isolated CPython `-I -S -B`, rejects preloaded reserved repository modules/root search paths, and consumes the single installation slot even when initialization fails. The fixed main entry accepts only root/revision/lane/catalog digest/result FD, not a function, callback, arbitrary module or fault selector.
- Capture binds trusted configured HEAD and catalog digest to actual Git/disk bytes, checks canonical bounded JSON, path containment/non-symlink spelling and captured source digests. It captures before executing the exact six-module CPU core. The full existing source/schema/AST verifier then replays and compares the complete held-byte map; no second substantive policy algorithm was introduced.
- The first finder denies reserved names outside the CPU subset before downstream resolution. Loader identity/name/source/origin are bound to the session. `get_code()` compiles the held source through the inherited stdlib helper, not ordinary cached-code loading. Execution records retain module/loader owners before running the body and mark completion afterward; exceptions permanently fail the session. Normal nested package imports remain possible, while reload/re-execution/duplicate-loader paths fail.
- Audit combines actual source/loaded-module replay, exact registry/module/loader identity, canonical module/spec/package/path metadata and one started/completed trace per core module. Repairs close the prior omitted identities rather than replacing these observations with hashes or caller-filled facts.
- Post-capture source drift cannot execute its altered body through this loader and fails before a report. Timestamp-matched and unchecked-hash stale caches for both initializer and leaf are independently exercised by the submitted tests; ordinary imports demonstrate the poison is effective, while the real bootstrap succeeds without consuming or rewriting it.
- Native AST v3 pins complete bootstrap LF bytes and physical loader/finder definitions. Existing renderer/regex exceptions remain unchanged. Actual policy rejects parseable caller-rehashed mutants; a self-consistent hash is not treated as independent release authorization.
- CPU report framing and bounded inherited FD output remain descriptive. The fixed launcher tests kill/reap owned verification/response stalls while the parent remains alive. No product dispatcher, GPU preparation, benchmark sample or GPU-quiescence receipt is introduced.
- The cleanup ledger explicitly replaces this temporary CPU terminal path only through later reviewed integration and retains the selected loader/source closure at P12.4e. No unrestricted callback survives that stated migration.

The trusted interpreter, stdlib, configured reviewed release/root and launcher remain assumptions. This approval is not a hostile same-process sandbox, arbitrary filesystem attacker defense or proof of initial bootstrap execution provenance before the interpreter runs it.

## Independent executions

Local interpreter: `C:/Python314/python.exe`, CPython 3.14.0. Remote: `/data3/ang.gao/anaconda3/envs/robot_sim/bin/python`, CPython 3.12.13. No packages were installed. Local release fixtures explicitly package LF bytes; actual unmodified POSIX release entry was separately tested remotely.

### Local four-file suite

Executed through reviewer scratch `c4_assets/c2d0_review_v2_runner.py`, which starts the actual command below, prints its actual PID/parent, waits with a 300-second bound and propagates its exit:

```text
C:/Python314/python.exe -B -m pytest tests/unit/optics/test_native_bootstrap.py tests/unit/optics/test_native_source_policy.py tests/unit/optics/test_native_sources.py tests/unit/benchmarks/test_closed_worker_protocol.py -q -rs
```

Observed actual exit 0: **268 passed, 6 skipped in 134.32s**. Skips were precisely two actual POSIX bootstrap entries, two actual POSIX source-release checks and two real Go2 materialization cases without the local configured asset. All corresponding remote cases ran without skip. Local Torch-dependent full benchmark-contract tests were not attempted or substituted with mocks; the six-file remote run below covers them in their configured environment.

### Remote six-file suite

Waited for the author's revised full suite to finish. Before starting independent runtime testing, independently checked that its timeout/pytest/sampled child PIDs 2204201/2204222/2205979 were absent, plus exact remote HEAD/status/diff and source/catalog digests. The check's exit 1 came only from `ps` finding no such processes; preceding Git/digest/interpreter commands completed successfully.

The same scratch runner executed the following actual pytest command with `ROBOT_SIM_TEST_GO2_MODEL_DIR=/data3/ang.gao/c4_asset_frozen/out/external/mujoco_menagerie/unitree_go2` and a 300-second owned-child deadline:

```text
/data3/ang.gao/anaconda3/envs/robot_sim/bin/python -B -m pytest tests/unit/optics/test_native_bootstrap.py tests/unit/optics/test_native_source_policy.py tests/unit/optics/test_native_sources.py tests/unit/benchmarks/test_closed_worker_protocol.py tests/unit/benchmarks/test_optical_closed_worker.py tests/unit/benchmarks/test_optical_benchmark_contracts.py -q -rs
```

Observed pytest exit 0 and SSH exit 0: **323 passed in 135.56s, no skips**. This includes both actual POSIX source-entry lanes, both real Go2 CPU materialization lanes, exact policy/cache/identity checks and C0/contract regressions. Existing benchmark contract imports may use installed dependencies in the pytest process; this is not GPU bootstrap admission or device execution acceptance.

### Original unchanged repro

Independently verified and used the original `C:/Users/AG/新建文件夹/c4_assets/c2d0_review_probe_v1.py`, SHA-256 `401e0dc1813ce4d381a7218924952336b40b947da6a1be2c2abbb099ac0b701c`, unchanged with only the revised root argument. It calls actual production `_install()`/`_verify()` in the same isolated test-only harness as before. No production callback or alternate loader was created.

```text
PROBE_PARENT 2211066 2211043
CHILD spec_name 2211068
RESULT spec_name exit 0 REJECTED SourceBootstrapError module metadata drift
CHILD package 2211100
RESULT package exit 0 REJECTED SourceBootstrapError module metadata drift
CHILD main_replacement 2211137
RESULT main_replacement exit 0 REJECTED SourceBootstrapError bootstrap owner identity/metadata drift
ORIGINAL_REPRO_SSH_EXIT 0
```

The child exit 0 here means the scratch script successfully observed rejection; it is not a successful production verification. Each child had a 30-second communicate deadline and was waited/reaped.

### Additional restoration-after-failure proof

Reviewer scratch `C:/Users/AG/新建文件夹/c4_assets/c2d0_review_v2_terminal.py`, SHA-256 `9cb7b4c03ea8f9287fcf349b63e0c3bebaabd4083ef1ff4da75aae08a973bf4f`, calls the real `_install()`/`_verify()`. For each lane it tests copied-dictionary main replacement, leaf spec-name/package mutation and initializer spec-name/package mutation. Each starts with a successful verifier run, applies one mutation, requires rejection/failed state, restores that identity and requires rejection/failed state again. All ten children returned exit 0 with `REJECTED_AND_PERMANENT`; outer SSH exit 0. Each had a 30-second owned-child deadline. It adds no product entrypoint.

Scratch runner SHA-256: `b198ae35d7cb07e9985aef7e514548c53cd3b07dd076d27f5909dd1f6cbe80dc`.

### Transport and repository commands

Used `git rev-parse HEAD`, `git branch --show-current`, `git status --porcelain=v1`, base/revised `git diff`, `git diff --check`, `Get-FileHash`, `Get-Content -Encoding utf8`, `rg`, remote `sha256sum` and interpreter `-V`. The explicit no-diff check covered C0 worker/protocol/manifest, both tools launchers and shared workflow rules. No candidate source was normalized or rewritten.

All remote scratch sources were transported without writing the candidate, using this pattern, substituting the stated original/runner/terminal scratch filename:

```powershell
$reviewSource = Get-Content -Raw -Encoding utf8 C:/Users/AG/新建文件夹/c4_assets/c2d0_review_v2_runner.py
$reviewEncoded = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($reviewSource))
ssh -n -o BatchMode=yes -o ConnectTimeout=8 -i C:/Users/AG/.ssh/tileops_ed25519 ang.gao@192.168.80.101 "echo $reviewEncoded | base64 -d | /data3/ang.gao/anaconda3/envs/robot_sim/bin/python -I -S -B - /data3/ang.gao/robot_sim_c2d0_code_v2"
Write-Output "REVIEW_SSH_EXIT $LASTEXITCODE"
```

## Process hygiene and residual checks

- Local runner PID 16184, parent 20852; pytest PID 1788, parent 16184. Both creation times were 2026-10-03 08:18:11 local. Sampled isolated bootstrap child 22860, parent 1788, creation 08:19:37. Actual pytest/wrapper exit 0; all three subsequently absent in CIM checks.
- Remote runner PID 2209708, parent 2209684, `/proc` starttime 84248838; pytest PID 2209709, parent 2209708, starttime 84248840. The runner sampled the real descendant graph every 0.5 seconds. Example nested branch: 2212492 parent 2209709 start 84257863, children 2212568/2212612 parent 2212492 starts 84257953/84258021. Later C0 branches: 2213595 parent 2209709 start 84261661, child 2213597 start 84261665; 2213617 parent 2209709 start 84261725, child 2213622 start 84261728. Full sampled PID/parent/start output is in the review-session transcript.
- Original repro parent/children and actual exits are listed above. Their short-lived starttime fields were not retained; scoped postchecks found all absent.
- Terminal probe parent PID 2214451, parent 2214428. Its ten children, all parent 2214451, were 2214453/2214470/2214500/2214530/2214564/2214595/2214616/2214650/2214687/2214767, with `/proc` starts 84265437/84265548/84265755/84265965/84266173/84266380/84266502/84266736/84266966/84267199, respectively. All returned actual exit 0 and were explicitly reaped.
- A final exact `ps -p` query included every sampled suite PID and every original/terminal probe PID (70 identities total); it returned only the header. Scoped `pgrep -af '[p]ytest'`, `'[n]ative_bootstrap'` and `'[o]ptical_closed_worker'` returned no matches. Final local CIM scope scans for the reviewer scripts and bootstrap pytest command returned no process.
- Remote `nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader` returned no rows after the suite/original-repro round and again after terminal probes. Local query listed unrelated existing desktop applications only, no owned test PID. None were touched.
- No review timeout was triggered. Submitted live-parent stall tests performed their intentional owned-child kill/reap and passed. No broad process kill, GPU reset, cache deletion, package installation, asset modification, Git reset or unrelated cleanup occurred.
- Sampling does not guarantee capture of every very short-lived Git/child PID. Read-only SSH session PIDs were not individually ledgered. Explicit subprocess returns/reap, sampled ancestry and final exact/scoped residual scans are the available evidence, not exhaustive descendant supervision or GPU-quiescence proof.
- Tests generated ordinary temporary fixtures/caches outside tracked candidate files. Final local and remote HEAD/status/diff checks remained clean. One initial inspection tried the root-level `REVIEW_PRINCIPLES.md`; it was corrected to the complete `collab/REVIEW_PRINCIPLES.md` before review work. This was an inspection-path error, not a code failure.

## Approval scope and remaining gates

Approval binds only revised HEAD `1f0576bfc607d24d2f5a04838051497f90a5592a`: the CPU-only C2d0 direct source bootstrap, held-byte six-core loader/registry, exact owner and metadata audit, pinned native policy/catalog, descriptive verification report, corresponding regressions and cleanup obligation. Both prior P2 findings are closed; no further repair is required for this bounded checkpoint.

C0 remains byte-for-byte unchanged; this review neither withdraws nor broadens its historical approval. Local unmodified CRLF checkout execution is not claimed to match POSIX release bytes. No full GPU test was appropriate or performed for this CPU-only checkpoint, and no descendant-Git-hang fault injection was added.

C2d integrated guarded lifecycle/termination, guardian custody before GPU-capable import/JIT, worker/supervisor dispatcher and IPC, startup/HELLO/per-request composition, GPU preparation, C2e/f timeline/arena and same-driver real-device acceptance, separate R0/R1 activation, C4c3/c4, C4d/B3 joint obligations, R2/S/host shipping, a3b, baseline and benchmark-result authority all remain closed. C4 is not complete. Future adoption must use the reviewed loader under the separately reviewed runtime guard; this CPU success must not be reused as GPU cleanup, quiescence or result evidence.
