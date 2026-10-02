# Independent C2d0 source-consumption design review v1

Initiative: q54-optical-lab-p12-4c4c2d-source-consumption
Stage: review
Author: codex
Version: v1
Date: 2026-10-03
Status: accepted
Related Files: source-consumption design-request codex v1; runtime-integration design-request codex v3 and review codex v3; source-catalog review codex v2 and test-evidence codex v1; optical_worker/native_sources.py; optical_worker/native_source_policy.py; tools/optical_closed_worker.py
Owner Summary: Approve the exact CPU-only C2d0 design addendum. Its source-only bootstrap is feasible against the current six-module CPU core and full verifier. This is permission to implement the bounded design, not approval of a loader implementation, GPU/lifecycle integration, activation or results.

## Identity, revision and verdict

Reviewer session: `/root/c4_c2d0_source_consumption_design_review_v1`, a fresh independent design-review session. The reviewer received the review packet, not the implementation conversation, and did not author or edit the candidate.

Verdict: **Approve**.

Base: `129697a969e69d7b75f11094828694b5e80b25e5`.

Reviewed HEAD: `57fea1c1a6824e704a6e2b590c26fbc0616552de`.

Candidate: `collab/q54-optical-lab-p12-4c4c2d-source-consumption__design-request__codex__v1.md`.

Independently verified SHA-256: `bf63e0fa34a37668845c2672a46e052b8f919890904b24d7bf040852e5c81f7d`.

Local repository: `C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design`, branch `implementation/c4-supervised-waits`.

Remote repository: `/data3/ang.gao/robot_sim_c2d0_design_v1` on `ang.gao@192.168.80.101`, detached HEAD.

Initial and final local/remote status were clean at that exact HEAD and digest. The base-to-HEAD change is one added design document, 243 lines; no source, test, C0, gate or shared-workflow changes. The packet reported the primary host's transport timeout and owner-authorized fallback; this reviewer used fallback101 and did not independently retry primary183.

## Findings

No substantive blocking or non-blocking design finding was established. Accordingly there is no defect severity/location/trigger/consequence/repair item to report. The limitations and mandatory implementation evidence below are retained scope boundaries, not assertions that unimplemented behavior passed.

## Design analysis against actual code

Read `AGENTS.md`, `collab/README.md` and `collab/REVIEW_PRINCIPLES.md` completely. Read the complete candidate, accepted runtime-integration v3 design, source-catalog review v2 and evidence v1; checked the runtime v3 approval scope. Author claims and prior approval were not substituted for source inspection or independent diagnostic output.

Inspected actual `native_sources.py`, `native_source_policy.py`, the eager package initializer, CPU-core import dependencies, manifest AST policy, protocol canonical codec/limits and old direct worker bootstrap. High-risk conclusions:

- The gap is real and accurately bounded. C0 sets `dont_write_bytecode` before ordinary imports; existing C2c6 `check_loaded_modules()` calls `verify_unchanged()` and inspects actual module metadata, but does not establish which earlier cached code executed. Independent package-init cache probes below reproduce that distinction. This addendum does not misrepresent C2c6 as pre-execution proof or silently modify C0.
- The exact CPU core is implementable. `optical_worker/__init__.py` eagerly imports lifecycle and protocol; native_sources imports native_source_policy; policy imports manifest and protocol. Those dependencies close over exactly the six named repository modules. The full verifier needs only stdlib plus this core, including subprocess Git checks; it need not import NumPy, Torch or Warp.
- Initial authority is explicitly the trusted configured release and directly executed script, not a self-issued hash. The bootstrap-before-self-verification limitation is acknowledged. Holding source bytes, verifying release/catalog and digests, loading the pinned CPU core, then invoking the existing full validator is a legal order within this trust model. A copied source snapshot or descriptive CPU report is not runtime authority.
- The hash dependency is acyclic: exact bootstrap source pins reside in native policy; the policy digest and all source digests reside in catalog data; configured release/catalog identity is external to the bootstrap. Bootstrap source contains no self-digest. Changing bootstrap therefore requires reviewed policy/catalog/release updates, not solving a self-referential digest.
- Existing `capture_native_source_snapshot()` is suitable for full replay: it validates exact literal lane inventory, schema, policy identity and actual source bytes, and retains source bytes in the snapshot. Bootstrap's first-pass capture is not a second full admission validator. The candidate requires byte-for-byte replay and differential checks of its limited parser against the production codec/validator; implementation may not replace those with hash-only equality.
- The proposed finder intercepts the reserved repository roots before later finders, with no repository root on normal `sys.path`; the CPU subset blocks even catalog-listed GPU modules. Per-name loader binding, held-byte `source_to_code`, no default `SourceLoader.get_code`, and no fallback to pyc/native/zip/namespace sources directly address the cache-consumption defect. Normal package import cycles can retain Python's partially initialized module behavior without admitting explicit loader re-execution.
- Started/completed/failed records belong to actual loader execution and exact module/loader identities. Their one-way references to captured bytes and the permanent failed-session state prevent a source hash alone from masquerading as execution. Retaining owners before execution and auditing sys.modules afterward are explicit requirements; the existing metadata-only audit is reused but not mistaken for the new execution registry.
- Native AST v2 really forbids importlib. The candidate explicitly requests a v3 exception restricted to one production-package script, exact imports/API uses and whole-source/physical-definition pins, while retaining the existing ctypes/regex exceptions. This is an intentional reviewed policy amendment, not an assertion that the old policy already permits it. Whole-source pins independently prevent caller-rehashed valid-syntax mutants from broadening the exception. Exact final pins necessarily belong to the later implementation candidate.
- C2d0 process failure is CPU-only and terminal: bounded launcher deadline, actual exit/reap, no retry or partial success and no GPU-quiescence claim. Later CUDA-capable imports require guardian custody and armed supervision before execution and must retain runtime-v3 failure ordering. The new CPU path does not authorize adopting unguarded imports/JIT into C2d.
- Section 7 specifies positive and adversarial production acceptance for cache poisoning, source drift, unauthorized resolution, replay/reload/identity replacement, AST mutants and live-parent hangs. The temporary CPU self-check has an explicit replacement/removal route; it is not a permanent parallel shipping worker. All parent-stage and product gates remain closed.

## Independent verification

This is a documentation-only design checkpoint. There is no production C2d0 implementation or committed C2d0 test to execute. Tests were proportional CPU diagnostics of the premises, not GPU or worker-lifecycle acceptance.

Scratch diagnostic: `C:/Users/AG/新建文件夹/c4_assets/c2d0_design_review_v1/probe.py`, created by this reviewer outside the candidate. First remote stdout/stderr logs are alongside it. The second direct run's output is retained in the review-session transcript. The author-provided original cache-probe source was read, but neither its output nor the author's separate six-core probe was counted as independent evidence.

### Local and remote cache diagnostic

Ran the reviewer probe on local `C:/Python314/python.exe` (CPython 3.14) and remote `/data3/ang.gao/anaconda3/envs/robot_sim/bin/python` (CPython 3.12.13). It created only fresh disposable temporary package fixtures outside the candidate, compiled stale `__init__.py` bytecode, replaced source with equal-size frozen bytes, and restored the original nanosecond timestamps. It tested both TIMESTAMP and UNCHECKED_HASH caches in isolated `-I -S -B` child interpreters.

For both cache modes on both interpreters:

- Ordinary import executed `STALE!` despite the frozen source bytes.
- A diagnostic SourceLoader overriding get_code to compile held frozen bytes executed `FROZEN`; its get_data raised on any attempted loader filesystem read.
- Each child returned actual exit 0 and was waited/reaped.

This confirms the stated cache premise and the basic held-byte loading mechanism. The diagnostic is deliberately not a production implementation: it does not prove policy pins, production identity-registry enforcement, reload attacks, trusted-configuration admission, or launcher hang behavior.

### Actual POSIX CPU-core and verifier diagnostic

The remote part read the actual six source files from the frozen POSIX checkout, installed a diagnostic source-only finder for those exact names, and imported `optical_worker.native_sources`. The root was not added to sys.path. Other reserved repository-root imports raised before resolution. It then called the actual `capture_native_source_snapshot()` and `check_loaded_modules()` for each lane at the actual HEAD.

Observed on each of two independent remote executions:

- Warp full verifier: 65 source entries; CUDA: 70.
- Actual catalog SHA-256: `f92183879cbfcd0f2817b2dc2b2333e5b85f1e683936ce0e520d9cad0c534262` for both lanes.
- Loaded repository audit: exactly the six proposed core files.
- Twelve actual diagnostic records: one started and one completed for each of optical_worker, lifecycle, protocol, native_sources, native_source_policy and manifest, in the real nested import order.
- No numpy, torch or warp import.
- No GPU import/work, package install, candidate writes or cache deletion.

These checks validate existing-source dependency closure and replay feasibility, not the proposed new bootstrap source/catalog or production-loader security properties. The loader diagnostic is intentionally permissive compared with the proposed policy and is not approved product code.

### Commands executed

Repository inspection used `Get-Content -Encoding utf8`, `rg`, `git rev-parse HEAD`, `git branch --show-current`, `git status --porcelain=v1`, `git diff --stat`, `git diff --name-status`, and `Get-FileHash` on the paths identified above. Exact substantive checks included:

```text
git diff --check 129697a969e69d7b75f11094828694b5e80b25e5 HEAD
git diff --name-status 129697a969e69d7b75f11094828694b5e80b25e5 HEAD
C:/Python314/python.exe -I -S -B C:/Users/AG/新建文件夹/c4_assets/c2d0_design_review_v1/probe.py
```

The local command returned exit 0. Diff checks passed locally and remotely. Remote source transport and actual direct invocation were:

```powershell
$reviewSource = Get-Content -Raw -Encoding UTF8 C:/Users/AG/新建文件夹/c4_assets/c2d0_design_review_v1/probe.py
$reviewEncoded = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($reviewSource))
$reviewRemote = "echo $reviewEncoded | base64 -d | /data3/ang.gao/anaconda3/envs/robot_sim/bin/python -I -S -B - /data3/ang.gao/robot_sim_c2d0_design_v1"
& ssh -i C:/Users/AG/.ssh/tileops_ed25519 -o BatchMode=yes -o ConnectTimeout=10 ang.gao@192.168.80.101 $reviewRemote
Write-Output "DIRECT_SSH_EXIT $LASTEXITCODE"
```

Observed `DIRECT_SSH_EXIT 0`; the remote pipeline ends with the probe interpreter. The first invocation used `Start-Process -WindowStyle Hidden -PassThru`, redirected logs and `WaitForExit()`. Its PowerShell ExitCode field was blank, so the completed direct rerun above supplied independently observed transport/probe exit rather than inferring it from a report field.

Initial/final remote checks used the same SSH identity and ran, in the exact remote repository:

```sh
git rev-parse HEAD
git branch --show-current
git status --porcelain=v1
git diff --check 129697a969e69d7b75f11094828694b5e80b25e5 HEAD
sha256sum collab/q54-optical-lab-p12-4c4c2d-source-consumption__design-request__codex__v1.md
/data3/ang.gao/anaconda3/envs/robot_sim/bin/python -V
```

Two inspection-tool errors were corrected and are not test failures: an initial broad directory enumeration produced excessive output and its owned tool session 79269 was interrupted; an initial `Get-Content REVIEW_PRINCIPLES.md` missed the collab directory and was followed by reading the correct complete file. One attempted PowerShell command used unsupported brace path syntax and failed parsing before execution; it was rerun with explicit paths. No candidate mutation resulted.

## Process and round hygiene

No production worker, supervisor or GPU process was launched. Probe children were synchronous owned CPU interpreters with explicit 30-second communicate deadlines; no timeout or kill was needed. Linux identities below include `/proc/<pid>/stat` starttime:

- First remote probe: PID 2145371, parent 2145347, start 83950976; parent start 83950974. Children 2145372/2145373/2145374/2145375, parent 2145371, starts 83950981/83950982/83950987/83950988; each actual exit 0.
- Its local SSH: PID 32732, parent 31152, creation 2026-10-03 07:29:30 local time; waited and subsequently absent. Its blank ExitCode field is disclosed above.
- Direct remote rerun: PID 2146860, parent 2146837, start 83956451; parent start 83956449. Children 2146861/2146862/2146863/2146864, parent 2146860, starts 83956456/83956457/83956462/83956463; each actual exit 0. Direct SSH/probe exit 0 was observed. This rerun's short-lived local SSH PID was not individually retained.
- Local probe: PID 11392, parent 7360, tool-observed actual exit 0; child PIDs 13692/29348/16424/21676 each actual exit 0. Local child start timestamps were not retained. All were absent at the first postcheck. A later PID 11392 was explicitly identified as a reused unrelated `svchost.exe`, parent 1232, creation 07:30:52; it was not treated as a probe residual or touched.

After each remote round, scoped `ps -p ... -o pid,ppid,lstart,args` checks found the listed remote parents/probes/children absent. Final combined check of all twelve remote PIDs returned only the header. `nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader` returned no rows remotely after each round. The local GPU query listed unrelated existing desktop applications, no owned test PID; no cleanup was attempted. A final local CIM scan found no python/ssh process whose command line referenced the reviewer scratch path.

Read-only inspection SSH sessions and transient Git subprocesses were not individually ledgered; successful bounded subprocess returns and the scoped postchecks are the available evidence. No owned residual or stall was observed. No broad kill, GPU reset, installation, Git reset, repository GC, or unrelated resource cleanup occurred. This is scoped hygiene evidence, not proof against every possible future hang.

## Approval scope, limits and next proof

Approval binds only the exact design document/digest at the reviewed HEAD: a direct-entry stdlib CPU bootstrap, six-module captured-source core, single full-verifier reuse, exact source-only loader/session facts, narrow release-pinned native AST v3 exception, versioned catalog update, bounded CPU launcher/tests and stated cleanup path. No substantive design finding requires revision before implementing that scope.

Implementation review must still verify every section 7 requirement against the actual production bootstrap, not this probe: both release catalogs and supported interpreter behavior; timestamp/unchecked-hash core and initializer cache attacks; module/spec/package and loader-registry identity; denied bodies never executing; reload/re-execution/replacement failures; capture drift; malformed/escaped/foreign/native-extension/sourceless resolution; exact AST pins and valid-syntax rehashed mutants; unchanged C0/renderer/regex regressions; bounded live-parent hang termination and observed reap. No existing or scratch test run here substitutes for those obligations.

No full GPU test was appropriate to this doc-only CPU design checkpoint. GPU-capable source was only read by the existing verifier; CUDA/Warp execution, guardian custody during import/JIT, worker IPC, startup/HELLO/request integration and runtime failure supervision are not tested or approved. C2d lifecycle completion, C2e/f, separate R0/R1 activation, C4c3/c4, C4d/B3, host/shipping, a3b, baseline and result authority remain closed. C4 is not complete, and no C0 bytes or historical approval meanings change.
