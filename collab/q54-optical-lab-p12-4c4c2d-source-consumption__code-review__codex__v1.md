# Independent C2d0 captured-source CPU bootstrap code review v1

Initiative: q54-optical-lab-p12-4c4c2d-source-consumption
Stage: review
Author: codex
Version: v1
Date: 2026-10-03
Status: in_review
Related Files: source-consumption design-request codex v1 and review codex v1; optical_worker/native_bootstrap.py; optical_worker/native_source_policy.py; optical_worker/native_sources.py; tests/unit/optics/test_native_bootstrap.py
Owner Summary: Request changes for two independently reproduced identity-audit omissions. The held-byte loader and existing regression suites pass, but the verifier still reports success after replacing the bootstrap module or changing a core module's spec name/package. These are bounded promised-invariant failures, not a request for a hostile same-process sandbox.

## Identity, revisions and verdict

Reviewer session: `/root/c4_c2d0_source_bootstrap_code_review_v1`, a fresh independent code-review session. The reviewer received the task packet, not the implementation conversation, and did not author or edit the candidate.

Verdict: **Request changes**.

Base: `29b2055061d38211546774f510e70545701bc0e9`.

Frozen reviewed HEAD: `0fae0a4c6df4f5527196009ce994aee3f0b7385d`.

Local checkout: `C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design`, branch `implementation/c4-supervised-waits`.

Remote checkout: `/data3/ang.gao/robot_sim_c2d0_code_v1`, provided clean detached checkout on `ang.gao@192.168.80.101`; independently verified HEAD and clean status before and after testing. Local initial/final status was also clean. The dirty original checkout was not touched.

Accepted design: `collab/q54-optical-lab-p12-4c4c2d-source-consumption__design-request__codex__v1.md`, independently verified SHA-256 `bf63e0fa34a37668845c2672a46e052b8f919890904b24d7bf040852e5c81f7d`. Its implementation approval is not substituted for this code verdict.

Remote catalog v2 SHA-256: `400d28d408f08c7e2e0576e23bb901021f23b5f3e67070b0025a957f713c9763`.

Remote bootstrap source SHA-256: `f61f500de8d41807d4af5f02fd900c7af697057e598468dba79ba3e6801e7521`.

The supplied primary-host transport timeout was accepted as the routing condition; this reviewer used the owner-authorized fallback101 and did not independently retry primary183. No network/GPU deployment was performed.

## Findings

### F1 — P2: The sole bootstrap module owner is not retained or audited

Location: `optical_worker/native_bootstrap.py:234` (`_SourceSession`), `:248` (`audit`), and `:283` (`_install`); relevant downstream check at `optical_worker/native_sources.py:263`.

The session retains the six loaded core modules but no exact bootstrap `__main__` owner. The source snapshot audit recognizes `__main__` by its allowed file path; it does not establish its object identity. Consequently an entirely new module containing only the copied bootstrap `__file__` passes the combined verifier:

```python
# After a normal production _install() and successful _verify(session):
replacement = types.ModuleType('__main__')
replacement.__file__ = entry.__file__
sys.modules['__main__'] = replacement
entry._verify(session)  # Returns status cpu_sources_verified.
```

Independent actual-POSIX reproduction at the frozen HEAD returned child exit 0 and a complete successful CPU report. The replacement did not contain the executing bootstrap's functions, loader, spec, or globals. The six core records and source bytes were unchanged; no registry internals or validation function were forged.

Consequence: the reported seven-entry repository ledger can include a replacement bootstrap object, despite the design's exact sole bootstrap-owner and module-replacement invariants. This is an identity-attestation omission, not proof that altered repository source or arbitrary external code can execute through the loader. No hostile-Python sandbox is requested.

Why existing checks/tests miss it: `session.audit()` iterates only `_CORE`; `snapshot.check_loaded_modules()` permits the exact `__main__` path and intentionally skips its normal spec check. Existing replacement tests target `optical_worker.native_sources`, not `__main__`.

Repair direction: retain the actual directly executing bootstrap module at the trusted initialization boundary and bind it to the session. Audit that `sys.modules['__main__']` is still that exact owner, with the intended direct-entry metadata, before reporting success. This should not pretend that the bootstrap itself was initially executed by the captured-source loader or retroactively authorized.

Required repair proof: the replacement above, deletion, and path-copy impersonation must fail closed and permanently invalidate the session; ordinary direct source entry on both interpreters/lanes must continue to pass. Update release pins/catalog as required by changed source.

### F2 — P2: Core spec names and package identities can drift without rejection

Location: `optical_worker/native_bootstrap.py:258` in `_SourceSession.audit()`.

The audit checks `module.__name__`, loaders, file/origin, and package search locations, but omits `module.__spec__.name` and `module.__package__`. Neither omission is repaired by the full source-ledger verifier. Both independent mutations below still produce a successful report:

```python
module = sys.modules['optical_worker.native_sources']
module.__spec__.name = 'foreign.alias'
entry._verify(session)  # cpu_sources_verified
```

```python
module = sys.modules['optical_worker.native_sources']
module.__package__ = 'foreign.package'
entry._verify(session)  # cpu_sources_verified
```

Each mutation ran in its own isolated real production session and returned child exit 0. Module object, loader, source bytes and registry were not replaced.

Consequence: success is reported for a module that no longer has the promised canonical spec/name/package identity. `__package__` and spec parent/name participate in Python's relative-import and module semantics; checking only `__path__` is not a complete package-identity check. No GPU behavior or arbitrary-code exploit is inferred from this CPU-only counterexample.

Why existing checks/tests miss it: tests cover a foreign loader, file, module object, package path, reload and re-execution, but not these two identity fields. The source snapshot check verifies spec origin only.

Repair direction: compare `__spec__.name` to the canonical bound name and `__package__` to the expected package derived from the frozen package/module layout, alongside the existing loader/origin/path checks. Preserve normal package initialization and relative imports.

Required repair proof: independent spec-name and package mutations for both a leaf module and the package initializer must reject before a success report and leave the session permanently failed; both lanes and Python 3.12/3.14 direct entry must remain valid. Re-run the existing identity, cache, policy and C0 regressions after repinning.

## Inspected behavior and scope control

Read `AGENTS.md`, `collab/README.md`, and `collab/REVIEW_PRINCIPLES.md` completely. Read the complete accepted C2d0 design and design-review record, the entire new bootstrap and tests, source snapshot verifier, native AST policy and affected existing tests. Inspected the seven-file base-to-HEAD diff and runtime call/data/state flow.

The actual diff contains the new bootstrap, catalog v2, narrow native policy v3 updates, source catalog/ledger changes, new bootstrap tests, the policy-version test update, and cleanup-ledger additions. Explicit diff checks found no changes to C0 worker, protocol, manifest, either tools launcher, or closed-worker runtime contract. Existing native v1 catalog history remains present. No dispatcher or activation gate was added.

High-risk checks that were satisfactory within this revision's bounded scope:

- Cold installation requires CPython isolation/no-site/no-bytecode flags, rejects reserved repository modules already loaded, avoids putting the configured root on normal `sys.path`, and permanently consumes the one installation slot even on a cold-start failure.
- Capture compares configured HEAD/catalog digest with actual Git and disk catalog bytes, validates canonical limited JSON/path containment and source digests, and retains source bytes. The loaded core then reuses the existing full catalog/source/AST validator and compares the resulting byte map exactly; the bootstrap does not duplicate the full policy algorithm.
- The finder precedes fallback finders and denies reserved names outside the exact six CPU modules. SourceLoader `get_code()` compiles held bytes through stdlib `source_to_code`; it does not call ordinary cached-code `get_code()`. The loader is bound to a canonical name, source object, origin and session. Package initialization and nested relative imports run through real stdlib execution semantics.
- Execution is recorded before module body execution, successful completion is recorded afterward, and exceptions permanently fail the session while retaining the attempted owner/record. Reload, repeated execution, duplicate loader attempts and tested core identity replacements fail closed. F1/F2 identify the remaining omissions rather than asserting that the whole registry is ineffective.
- Full source audit repeats current-byte verification, so post-capture source drift cannot be silently reported as success. Its held loader does not execute the new body. Stale bytecode tests independently demonstrate that normal imports execute the cached sentinel while this actual bootstrap does not; they include initializer and native_sources, timestamp-matched and unchecked-hash caches, and unchanged cache bytes after the run.
- Native policy v3 narrows the importlib exception to the bootstrap, pins its complete source and physical loader/finder class bytes, and preserves prior regex/renderer pins. Parseable caller-rehashed mutants are rejected by the actual policy. It is intentionally not a general Python security sandbox.
- The fixed CPU report has no worker protocol dispatcher, GPU preparation, backend selection, arbitrary callable selector, results authority or GPU-quiescence claim. The temporary self-check's replacement/removal conditions are present in the cleanup ledger.

The existing tests exercise capture errors, missing/sourceless/symlinked source, catalog malformations and traversal, preloaded modules, held-source behavior, fallback-finder denial, compiler failure, post-capture drift, policy mutants and live-parent verification/response stalls. All executed tests passed; those facts do not invalidate the three additional counterexamples.

## Independent verification

Local interpreter: `C:/Python314/python.exe`, CPython 3.14.0. Remote interpreter: `/data3/ang.gao/anaconda3/envs/robot_sim/bin/python`, CPython 3.12.13. No packages were installed. Local LF packaging fixtures are explicit test fixtures; actual POSIX checkout entry was separately exercised remotely without packaging or newline normalization.

### Repository checks

Executed locally/remotely as appropriate:

```text
git rev-parse HEAD
git branch --show-current
git status --porcelain=v1
git diff --stat 29b2055061d38211546774f510e70545701bc0e9 HEAD
git diff --name-only 29b2055061d38211546774f510e70545701bc0e9 HEAD
git diff --check 29b2055061d38211546774f510e70545701bc0e9 HEAD
```

Diff whitespace checks passed. Local `Get-FileHash` verified the accepted design digest; remote `sha256sum` verified the catalog/bootstrap digests listed above. Clean status was reconfirmed after all tests. Read-only inspection included `rg`, `Get-Content -Encoding UTF8`, and targeted diff/read ranges.

### Local test execution

```text
C:/Python314/python.exe -m pytest tests/unit/optics/test_native_bootstrap.py tests/unit/optics/test_native_source_policy.py tests/unit/optics/test_native_sources.py tests/unit/benchmarks/test_closed_worker_protocol.py -q
```

Observed exit 0: **256 passed, 6 skipped in 98.29s**. The skipped cases were actual POSIX checkout checks (two bootstrap and two source-catalog lane cases) and the two Go2 source materialization cases without a configured local real asset. Corresponding actual POSIX/Go2 checks passed in the remote suite. Full local Torch-dependent benchmark contracts were not run by this reviewer; no missing local dependency was installed and no author test output was counted as independent evidence.

### Remote six-file execution

Waited until the author-reported prior suite's exact PIDs 2180034/2180059 had exited before starting the independent suite. Actual independent command in the clean frozen checkout:

```sh
ROBOT_SIM_TEST_GO2_MODEL_DIR=/data3/ang.gao/c4_asset_frozen/out/external/mujoco_menagerie/unitree_go2 timeout --signal=TERM --kill-after=15s 300s /data3/ang.gao/anaconda3/envs/robot_sim/bin/python -B -m pytest tests/unit/optics/test_native_bootstrap.py tests/unit/optics/test_native_source_policy.py tests/unit/optics/test_native_sources.py tests/unit/benchmarks/test_closed_worker_protocol.py tests/unit/benchmarks/test_optical_closed_worker.py tests/unit/benchmarks/test_optical_benchmark_contracts.py -q
```

Observed SSH/command exit 0: **311 passed in 118.18s**, no skips. This included both actual POSIX direct-entry lanes, real Go2 CPU asset materialization, renderer/regex policy and C0/contract regressions. No GPU product execution acceptance is inferred from these CPU tests.

A separately PID-ledgered remote bootstrap-only rerun completed with actual exit 0: **44 passed in 34.37s**. Its wrapper and child identities are below.

### Independent identity reproductions

Scratch source, outside the candidate:

`C:/Users/AG/新建文件夹/c4_assets/c2d0_review_probe_v1.py`

SHA-256 after adding the optional separately bounded bootstrap-suite mode: `401e0dc1813ce4d381a7218924952336b40b947da6a1be2c2abbb099ac0b701c`.

The script imports the actual production bootstrap in an isolated test harness, binds it as `__main__`, calls its actual `_install()` and `_verify()`, changes one promised identity field/object, then calls the actual verifier again. This follows the same test-only internal-entry pattern as the submitted tests; it is not a new production entry or a duplicate loader. Each case has a 30-second owned-child deadline and explicit communicate/reap.

Execution transport (no remote candidate source edits):

```powershell
$reviewSource = Get-Content -Raw -Encoding UTF8 C:/Users/AG/新建文件夹/c4_assets/c2d0_review_probe_v1.py
$reviewEncoded = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($reviewSource))
ssh -n -o BatchMode=yes -o ConnectTimeout=8 -i C:/Users/AG/.ssh/tileops_ed25519 ang.gao@192.168.80.101 "echo $reviewEncoded | base64 -d | /data3/ang.gao/anaconda3/envs/robot_sim/bin/python -I -S -B - /data3/ang.gao/robot_sim_c2d0_code_v1"
```

Observed probe/SSH exit 0. Exact distinguishing output:

```text
PROBE_PARENT 2187625 2187602
CHILD spec_name 2187627
RESULT spec_name exit 0 ACCEPTED <successful CPU report below>
CHILD package 2187774
RESULT package exit 0 ACCEPTED <same successful CPU report>
CHILD main_replacement 2187804
RESULT main_replacement exit 0 ACCEPTED <same successful CPU report>
```

All three complete returned JSON reports were byte-identical. Their exact common report was:

```json
{"catalog_sha256":"400d28d408f08c7e2e0576e23bb901021f23b5f3e67070b0025a957f713c9763","execution_trace":[{"module":"optical_worker","operation":"started"},{"module":"optical_worker.lifecycle","operation":"started"},{"module":"optical_worker.lifecycle","operation":"completed"},{"module":"optical_worker.protocol","operation":"started"},{"module":"optical_worker.protocol","operation":"completed"},{"module":"optical_worker","operation":"completed"},{"module":"optical_worker.native_sources","operation":"started"},{"module":"optical_worker.native_source_policy","operation":"started"},{"module":"optical_worker.manifest","operation":"started"},{"module":"optical_worker.manifest","operation":"completed"},{"module":"optical_worker.native_source_policy","operation":"completed"},{"module":"optical_worker.native_sources","operation":"completed"}],"lane":"warp","repository_modules":["optical_worker/__init__.py","optical_worker/lifecycle.py","optical_worker/manifest.py","optical_worker/native_bootstrap.py","optical_worker/native_source_policy.py","optical_worker/native_sources.py","optical_worker/protocol.py"],"repository_revision":"0fae0a4c6df4f5527196009ce994aee3f0b7385d","schema_version":"q54.p12.4.native-source-execution-check.v1","source_snapshot_sha256":"7b5b742015ade17eab0ff96fccc4f0107675d481386bef61671a249aa2ece6e4","status":"cpu_sources_verified"}
```

Appending `--bootstrap-suite` to that transport selected the scratch wrapper's fixed pytest invocation; it did not add a selector to the production bootstrap. Its output was:

```text
PROBE_PARENT 2191338 2191315
BOOTSTRAP_SUITE_CHILD 2191340 PARENT 2191338 START 84163861
44 passed in 34.37s
BOOTSTRAP_SUITE_EXIT 0
```

## Process hygiene and limitations

- Local independent pytest PID 7848, parent 20156, creation 2026-10-03 08:01:06 local time; actual exit 0 observed. Its sampled isolated child PID 34988, parent 7848, creation 08:01:40, ran a fixed fallback-denial test. Both were absent in final CIM checks.
- Local SSH PID 5668, parent 22200, creation 08:01:31, carried the independent remote six-file suite. Command exit 0 was observed; the PID was absent afterward. The remote full-suite pytest/timeout PIDs were not successfully captured during that run; this is a process-ledger limitation, not a claim that they were individually reaped by the reviewer. Scoped postchecks found no remaining pytest/bootstrap/closed-worker process.
- The identity probe parent PID 2187625 (parent 2187602) and children 2187627/2187774/2187804 were explicitly printed by the running processes. All three child return codes and the outer exit were observed as 0. Their short-lived starttime fields were not retained. Scoped `ps -p` postchecks found all four absent.
- The PID-ledgered bootstrap rerun parent was 2191338 (parent 2191315); child 2191340 had `/proc/<pid>/stat` starttime 84163861. Both had `ps` start time Sat Oct 3 08:05:01 2026. Child exit 0 and outer SSH exit 0 were observed. Final scoped `ps` found both absent.
- Final remote `ps -p 2180034,2180059,2187625,2187627,2187774,2187804,2191338,2191340 -o pid,ppid,lstart,args` returned only its header. `pgrep -af pytest`, `pgrep -af native_bootstrap`, and `pgrep -af optical_closed_worker` returned only the current inspection shell containing those literal command strings, not a test or worker. Remote compute-process queries returned no rows after the observed rounds and at the final check.
- Final local CIM checks found no retained test/SSH PID and no python/SSH command matching the reviewer pytest/probe/ledgered-rerun scope. Local `nvidia-smi` listed existing desktop applications only, no owned test PID; these were not touched.
- The submitted live-parent hang tests deliberately kill/reap their owned CPU child after the fixed deadline. They passed in the independent runs. The reviewer did not issue a broad kill, GPU reset or unrelated cleanup. No independent scratch-probe timeout occurred.
- The review does not claim exhaustive descendant-process supervision, CUDA quiescence, GPU import/JIT custody, or later C2d lifecycle correctness. CPU subprocess Git timeouts and the submitted launcher were inspected, but no separate descendant-Git-hang injection was performed.
- Tests created ordinary pytest caches/temporary fixtures and packaged LF release trees outside tracked candidate files. No candidate tracked file was edited; local/remote trees remained clean. No cache deletion, installation, Git reset, or asset modification was performed.
- Two inspection errors were corrected: initially reading `REVIEW_PRINCIPLES.md` from the root instead of `collab/`, and a remote `rg` process scan on a host without `rg` (with SSH quoting also splitting its alternation). The complete correct principles file was then read, and process checks were rerun with `ps`/`pgrep`. Neither error is represented as a code/test failure.

## Scope and next checkpoint

No implementation approval is granted at this frozen revision until F1 and F2 are repaired and independently re-reviewed with these original counterexamples. Existing passing results support the held-source CPU mechanism and unchanged regressions; they do not close the missing identity checks.

The intended next approval remains only the exact CPU C2d0 bootstrap/source-loader, pinned native policy/catalog, descriptive report, tests and cleanup scope. C0 is byte-for-byte unchanged and its prior approval is neither withdrawn nor broadened by this review.

C2d integrated guarded lifecycle/termination, GPU-capable imports and preparation, worker/supervisor dispatcher, C2e/f real-device acceptance, separate R0/R1 activation, C4c3/c4, C4d/B3, R2/S/host shipping, a3b, baseline and benchmark result authority remain closed. C4 is not complete. There are no additional substantive non-blocking findings; the verification limitations above remain disclosed.
