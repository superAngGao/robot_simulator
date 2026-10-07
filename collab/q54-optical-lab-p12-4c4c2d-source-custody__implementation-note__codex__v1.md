# C2d2 CPU captured-source custody implementation

Initiative: q54-optical-lab-p12-4c4c2d-source-custody
Stage: implementation-note
Author: codex
Version: v1
Date: 2026-10-07
Status: in_review
Related Files: design-request/design-review v1; code-review v1; native_bootstrap.py
Owner Summary: Implement the approved CPU-only loader/guardian composition;
repair the independently reproduced no-ACK test oracle, without changing
production deadlines, admission or activation. This is not C4 completion.

Base af509cb5e3ed4103a1035a1298cd20bb652f45e4. Initial code candidate
f7692b4b7bfd50521956c5c83fcb2bc94049906a was Request changes (P2 F1).
Full independent report is archived verbatim in code-review v1, SHA256
9d58f848fd9e9ced7ec9462da418c31b8e523a69827c9a14ed82f8197952558f.

## Actual changes

- Extend the same captured-byte loader/finder/session with literal six-core,
  ten-prelude and twelve-definition phases. Existing direct CPU entry still
  reports only the six-module core; no new product role or module selector.
- Construct the exact captured guarded controller/configuration once. Retain
  distinct launch and actual client configuration identities; adopt session,
  snapshot, bootstrap and all prelude/core modules into the existing guardian.
- Fixed native_setup/native_contract definition import registers each partial
  module before guardian adoption and compilation/body. Perform full source
  audit inside the actual guarded setup segment before COMPLETE/DISARMED.
- Carry original errors, including pre-guard adoption failures and swallowed
  nested errors, into existing irreversible quarantine before unwinding.
  Equal-hash snapshot/client-config/source replacements are not authority.
- Pin changed bootstrap/loader/finder bytes in policy v4/catalog v3. Literal
  lane inventories, ctypes/regex exceptions and protected C0 stay unchanged.
- Add fixed CPU probes and real POSIX child/supervisor faults; extend the
  existing test supervisor rather than implement a second lifecycle reducer.

Spec precision: exact retained module, loader and source identities plus
ModuleSpec field checks are enforced. An equal-valued shallow ModuleSpec copy
is accepted; there is no claim that every spec-object replacement fails.
This descriptive object does not supply new code, loader or live authority.
The review's non-blocking observation is retained, not rewritten as a rejected
attack or an additional security boundary.

## F1 repair and proof obligation

Initial independent POSIX run:2failed28passed112.25s. Both no-ACK tests wrongly
required child exit74 and its final marker even when the unchanged supervisor
deadline won the race and killed/reaped the child. Full actual release repro
confirmed arm->deadline->kill->exit, no ACK/COMPLETE/body, and absent worker tail.

Repair branches on validated observed evidence only for no-ACK cases. The
self-exit case still requires full worker quarantine and no-runtime marker.
The killed case requires committed deadline/kill/actual exit ordering and
matching returncode; a missing worker tail is allowed, never inferred complete.
Both require exactly one ARM, zero COMPLETE, no ACK write, no body entry,
unchanged parent liveness and no destructor/atexit. Other fault assertions stay
strict. Two extra fixed test cases delay initial transport receive by400ms,
before supervisor ARM commitment, so the original client ACK deadline precedes
the original supervisor deadline and the self-exit ending is exercised. No
production timer, protocol, committed deadline or worker admission is changed.
Required rerun is the original30 cases plus these two self-exit cases, on both
lanes, then fresh independent review. An outer test timeout is failure only.

## Evidence and limits

Initial author local source suite26passed30POSIXskips123.51s; local CPU
controller/evidence/wait/policy476passed10.52s. Initial author remote process
run1failed9passed32.04s at F1. Independent CPU broad752passed2asset-skips163.12s,
then actual Go2 asset rerun2passed45.26s (full commands in review v1).
Author remote benchmark-contract suite41passed8.58s; local attempt25failed/
16passed because Torch was absent, reproduced as torch.empty hook ImportError,
not accepted as product evidence and no package installed. Author Go2 rerun
2passed46.03s; earlier45s outer-bound attempt ended without complete summary,
not credited. Post-repair results must be recorded separately, not inferred.

Primary183 is reachable again. Remote code lives in new isolated
/data2/shared/ang.gao/robot_sim_c2d2_code_v1, robot_sim Python3.12.13. Original
/data2/shared/ang.gao/robot_simulator is untouched clean mainf771e61. Local
Python3.14 has noTorch/Warp; POSIX supervision is validated remotely. Frozen
Go2 archive was relayed to new c2d2_frozen_assets on183, no install/gate change.
Scoped round-end process checks and GPU inventory found no owned residual;
pre-existing MPS2238410 was untouched. This does not diagnose the old101hang.

Still closed: third-party/native-create custody; actual Torch/Warp admission,
GPU import/init/upload and real multi-device faults; full graph/seal/barrier/
warmup/readiness; full C2d/e/f and C4c3/c4; C4d/B3; host/S/shipping, product GPU
preparation, a3b, result/baseline and activation. Exact-path cleanup ledger
records temporary probes for P12.4e; neither candidate backend is removed.
