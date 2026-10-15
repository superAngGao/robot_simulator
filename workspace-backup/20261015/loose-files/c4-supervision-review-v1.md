# C4c1c steps 1–2 independent code review

Initiative: q54-optical-lab-p12-4c4c1c-supervision
Stage: review
Author: codex (independent reviewer)
Version: v1
Date: 2026-09-28
Status: in_review
Related Files: C4c1 ABI design v3 and v2; tests/support/sacrificial_lifecycle.py; tests/support/sacrificial_watchdog.py; tests/support/sacrificial_protocol_child.py; tools/optical_sacrificial_test_supervisor.py
Owner Summary: Request changes. The clean candidate passes the focused 72-test suite and real supervisor checks, but its owner-transfer error paths can lose a reachable owner before quarantine.

## Identity and exact scope

- Reviewer session/task: `/root/c4_supervision_review_v1`. This is a fresh reviewer session supplied with the review packet, not an author self-review. No subagents were used.
- Base: `c8c675accb18ecb5fe91c0761bb3aafb73a5d2b0`.
- Reviewed HEAD: `a597e8a3d5346fd0a7c18491267274b7428cbfc4`.
- HEAD tree: `b6634e4f5254f0eeddffc5509cd14fe01257dbea`.
- Local candidate: `C:/Users/AG/新建文件夹/p12_4_c4_sacrificial`, branch `implementation/c4-sacrificial`.
- Remote candidate: `/data3/ang.gao/robot_sim_c4_supervision_review1` on `ang.gao@192.168.80.101`, branch `review/c4-supervision-v1`.
- Both trees were independently checked clean before and after verification and resolve to the exact HEAD above. The remote pathname does not identify an older candidate.
- V3 design SHA-256 as read locally: `291d4248633a9398eb4466b5e3a0ec514e04b810c5f6575886a4fab2804e1626`.
- Read AGENTS.md, collab/README.md, collab/REVIEW_PRINCIPLES.md, and the complete v3 and v2 design requests. Inspected the implementation note as a claim, not as approval evidence.

The diff is eight added files, 1,205 added lines: one implementation note (78), tests/support/__init__.py (1), lifecycle support (282), fixed CPU child (87), watchdog support (187), lifecycle unit tests (287), watchdog unit tests (146), and the standalone test supervisor (137). No existing production C0 source, allowlist, manifest, protocol, gate, or product A/K code is modified. Searches found no imports/references to this new mechanism in optical_worker, optics, or benchmarks.

The review is only CPU guardian/state authority plus the separate test-only S/V/E supervisor. It does not review or authorize a GPU driver, actual setup-owner adoption, physical storage freeze, recorded GPU execution, or result production.

## Verdict: Request changes

One blocking medium-severity finding remains. The supervisor behavior exercised here is sound in the tested fixed CPU scenarios; that does not close the owner-transfer boundary defect below. No approval is granted for this exact checkpoint until repair and independent re-review.

### F1 — Medium — Failed owner adoption drops later reachable owners before quarantine

Location: `tests/support/sacrificial_lifecycle.py:141–167`, especially the per-owner loops at 149–154 and 164–165. The downstream null rejection is in `optical_worker/lifecycle.py:302–303` (unchanged production code).

Trigger: A valid active owner-producing leaf transfers an exact tuple containing two real reachable owners with an invalid owner between them, for example `(Owner('first'), None, Owner('last'))`. This can occur in an erroneous/partially constructed helper return or partial-owner capsule. Call either `finish_setup_leaf(..., owners=...)` or `fail_setup_leaf(..., partial_owners=...)`.

Actual flow: the first owner is adopted. Production guardian validation rejects None. Neither wrapper first retains the complete incoming tuple in a safe owning capsule. The later valid owner is only held by the call/exception frames, not the guardian. Once the handled exception traceback is discarded, that owner finalizes. The wrapper is POISONED and ownership-unknown, but quarantine subsequently retains only the first owner. A later quarantine cannot recover the lost owner.

Consequence: CPU lifetime authority does not fulfill strong retention for all reachable transferred owners on its own adoption-error path. No setup quiescence or release was authorized. The current fixed CPU child always supplies a single valid Owner, so this review does **not** claim that its existing scenarios destroy GPU resources, nor that any GPU execution was run. Its immediate hard exit inside the catch also retains the active traceback until exit; relying on that incidental traceback is not whole-graph guardian adoption. The defect is in the step-1 CPU transfer boundary and should be repaired before connecting real setup builders.

Why existing checks miss it: tests cover a copied permit, missing/empty owner return, and partial-owner failure with valid owners. They do not inject an adoption failure in the middle of a returned or partial-owner tuple and then check all previously reachable owner finalizers after exception-frame cleanup.

Independent proof: external scratch file `C:/Users/AG/新建文件夹/c4-supervision-review-v1-repro.py`, executed against the exact remote candidate under robot_sim Python with `-I -S`. It invokes only CPU lifecycle code. Observed output:

```text
finish exception WorkerSetupProtocolError guardian cannot adopt a null owner
finish before_quarantine POISONED unknown_owner True resource_count 1 finalizers ['last']
finish after_quarantine QUARANTINED resource_count 1 finalizers ['last']
fail exception WorkerSetupProtocolError guardian cannot adopt a null owner
fail before_quarantine POISONED unknown_owner True resource_count 1 finalizers ['last']
fail after_quarantine QUARANTINED resource_count 1 finalizers ['last']
```

Repair direction: secure strong ownership of the complete reachable input capsule before per-element validation/adoption can raise, then keep it quarantined if canonical owner registration fails. Do not solve this by merely validating the whole tuple and raising earlier: that still leaves the passed valid owners unretained. Preserve monotonic ownership-unknown/poison state and do not permit ordinary release. The author should choose the narrow capsule/retention design without modifying product cleanup authority unnecessarily.

Required post-repair proof: for both finish and failure paths, inject an invalid middle element or controlled adoption exception, remove caller/traceback references, collect garbage, and prove every real passed owner remains alive before and after quarantine. Verify ordinary release/emergency recovery cannot upgrade incomplete ownership. Retest the existing state/protocol suite and exact clean-worker regressions.

## Inspected high-risk behavior

- Production guardian composition uses exact owner identity and its process-quarantine registry, not the serialized owner type name as authority.
- Leaf permits are identity-consumed; copied permits and active-leaf overlap fail. Launch positions use exact integer checks, latch before the driver leaf, and distinguish attempted from normally returned counts.
- Snapshots are detached diagnostics, not authority-bearing inputs. Descriptors remain trusted fixed-driver declarations; this checkpoint cannot prove a real device/resource graph from those literal fields.
- S completes only after an exact disarm ACK. V/E cannot disarm through the protocol and remain armed through the fixed child's inspection/release/exit. Sequence booleans are rejected rather than accepted as integer equality.
- Supervisor deadlines use its own monotonic clock and precede ACK generation. Its parser bounds frame size and uses canonical JSON. Wrong role/sequence, EOF and missing ACK prevent callable entry in the tested paths.
- Raised waits quarantine rather than initiating a second synchronization. The real normal V hang occurs after two launch-latch journal entries, is killed by the supervisor, and produces no inspection, release, or owner-finalizer marker.
- The fixed subprocess entry uses `-I -S`, a repository-fixed child path, inherited descriptors, and enumerated CPU scenarios. There is no caller-selected executable/module/callback on the IPC surface.

## Executed verification

Remote transport used the explicitly authorized 101 fallback and `C:/Users/AG/.ssh/tileops_ed25519`. Runtime was `/data3/ang.gao/anaconda3/envs/robot_sim/bin/python`, Python 3.12.13; pytest/Ruff commands were run through `/data3/ang.gao/anaconda3/bin/conda run --no-capture-output -n robot_sim` from the clean remote candidate.

1. `git status --porcelain`, `git rev-parse HEAD`, quoted `git rev-parse 'HEAD^{tree}'`, branch checks, `git diff --stat/--numstat/--name-only BASE HEAD`: exact identity and eight-file scope confirmed. Final local and remote status remained clean.
2. `git diff --check c8c675accb18ecb5fe91c0761bb3aafb73a5d2b0 HEAD`: passed.
3. `python -m pytest -q tests/unit/optics/test_sacrificial_lifecycle.py tests/unit/optics/test_sacrificial_watchdog.py tests/unit/benchmarks/test_closed_worker_protocol.py tests/unit/benchmarks/test_optical_closed_worker.py`: **72 passed in 10.29s**. This is a new reviewer execution, not the author's prior run. The clean-manifest integration cases passed; no earlier scratch failure was reclassified as success.
4. `python -m ruff check tests/support/sacrificial_lifecycle.py tests/support/sacrificial_watchdog.py tests/support/sacrificial_protocol_child.py tests/unit/optics/test_sacrificial_lifecycle.py tests/unit/optics/test_sacrificial_watchdog.py tools/optical_sacrificial_test_supervisor.py`: **All checks passed**.
5. F1 repro command (external script streamed over SSH, no candidate edits):

   `Get-Content -Raw -Encoding UTF8 'C:/Users/AG/新建文件夹/c4-supervision-review-v1-repro.py' | ssh -i C:/Users/AG/.ssh/tileops_ed25519 ang.gao@192.168.80.101 '/data3/ang.gao/anaconda3/envs/robot_sim/bin/python -I -S - /data3/ang.gao/robot_sim_c4_supervision_review1'`

   Both independent transfer variants reproduced F1 as shown above; process exit code 0 means the diagnostic script completed, not that the safety assertion passed.
6. An additional external script, `C:/Users/AG/新建文件夹/c4-supervision-review-v1-parent-loss.py`, was streamed with the same command pattern. It launches the **actual unchanged supervisor**, waits for the child's ACKed V blocking marker, and closes the parent-liveness pipe. **PASS**: supervisor reports `reason=parent_lost`, `killed=true`, worker return code `-9`, terminal deadline still armed, no restart, and no inspection/release/finalizer markers. New artifacts reside at `/tmp/c4-review-parent-loss-almpiza8`, outside the candidate.

Non-mutating inspection hiccups: the initial unquoted PowerShell `HEAD^{tree}` was parsed incorrectly and was repeated with quoting; an initially guessed implementation-note filename and tests/conftest.py did not exist. The correct implementation note was subsequently read. These are command/path errors, not product failures.

## Limits, residual risk, and closed gates

All executed scenarios here were CPU-only; no Warp/GPU driver, native input setup, kernel, recorded bundle, or device probe was run. Real OS supervision was tested on Linux/POSIX, not Windows. This review does not claim malicious in-process Python capability isolation: the state API explicitly trusts a fixed single-threaded child driver and its fixed callables. Full import-ledger proof, actual allocation/command ownership, storage identity freeze, numerical CPU oracle, submit-then-raise device faults, and the complete GPU setup/launch/drain matrix remain future work.

Candidate tracked files were not modified. Standard test/tool caches may exist as ignored files; newly produced diagnostic artifacts and both repro scripts were kept outside the candidate. No old outputs were used as execution proof.

Even after F1 is fixed, approval may cover only C4c1c steps 1–2 at the newly reviewed revision. C4c1c GPU-driver execution and full C4c1 acceptance, C4c2–C4c4, C4d/B3, C0/C4 joint shutdown, host payload, product preparation, product GPU activation, submission/result/performance evidence, backend selection, a3b, baseline access, and shipping gates remain closed.
