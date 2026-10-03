# C2d1 guarded custody: initial code review candidate

Initiative: q54-optical-lab-p12-4c4c2d-guarded-custody
Stage: implementation-note
Author: codex
Version: v1
Date: 2026-10-03
Status: in_review
Related Files: design-request v2; design-review v1/v2; guarded_lifecycle.py;
guarded_evidence.py; wait_protocol.py; native_sources.py; cleanup_ledger.json
Owner Summary: Implement the approved CPU custody interface and supervised
fault tests. This is an unapproved implementation candidate, not GPU admission
or C4 completion. Exact code review HEAD is the commit containing this note.

## Scope and code facts

Base: `8cbf8645123a3d7e3f6cf4bbc1434fa3a6612472`.
Intermediate author validation snapshots: `4c64b69`, `cc2fae8`; neither is an
independent approval. This candidate also adds the final context single-use
checks, failure-time strong retention and its four negative tests.

Approved interface: guarded-custody design v2 at `f1f4394`, original SHA-256
`00aa16cfef0dfd83b7f44081f845c9a3397d79435b4b3cf319e27c467d840983`.
Independent design review v2 approved with one whitespace comment. This code
delta removes only that proposal's line260 trailing space; it does not change
the design semantics or retroactively rename the approved digest.

`NativeGuardedSetupLifecycle` subclasses the existing C0 lifecycle and reuses
its guardian, descriptor/permit, owner registration and scope expansion.
Neither upload loop nor C0 termination is copied. Actual `native_setup._abort`
dispatches to its override; a normal failed guard cannot fall back to A/K or a
new emergency synchronization. C0 lifecycle/native_setup/worker and both C0
launcher/supervisor source files are unchanged from the base.

The new controller constructs the exact client from its private socket/config.
Contexts require a real ACK before body/leaf entry and actual DISARMED before
normal completion. A single-use context wrapper rejects premature/repeated
enter/exit; direct protocol misuse cannot silently reuse the generator.
Wrong-thread/reentrant failures also strongly retain the controller immediately,
before later owner-thread termination. Termination's custody event observes
that retention and does not add a duplicate registry entry. The process-wide
registry is append-only and has no recovery/reset API.

An actual healthy IDLE can arm a non-completable emergency-role envelope for
cpu_cancel (zero sync) or target_sync (at most one indexed sync selected by the
trusted driver). Ordinary reference release and exit remain within that same
deadline. Failed normal guards, foreign/unknown scope and failed terminal ACK
retain/quarantine instead. Release-before/after-failure states preserve remaining
custody rather than illegally transferring a RELEASE_AUTHORIZED/RELEASED guardian.
No finalizer grants release. The fixed test driver always hard-exits; quarantine
or cleanup failure uses74, successful clean/synced terminal intent uses0.

The guard client adds only exact live/just-completed history inspection and
original ACK-field retention. Copied, stale, wrong-role/foreign and changed-field
tickets reject. Wire version/messages, guard budgets, supervisor states and C0
notifier behavior are unchanged.

`guarded_evidence.py` implements the v3 schema/reducer and inert parsers, plus
worker/supervisor artifact cross-validation. It derives scope conservatively
from full descriptors/owner facts, even if the subsequent notification is
missing. It separates release authorization, start, completion, failure,
terminal intent and actual supervisor-observed exit. Hashes are descriptive,
not authenticity or live authority. Hand-authored legal fixture summaries do
not come from the reducer; deletion/duplication/reorder/identity/extra-field
mutants and false summaries are challenged separately.

Literal native source inventory adds the two production modules and repins
catalog v2's source digests. Catalog format and policy version are unchanged;
the CPU-only bootstrap still admits exactly its existing six cold-core modules.
No runtime role, product request, callback or GPU preparation is enabled.
Cleanup ledger records selected-worker adoption and removal of unused modes/
transitional test drivers only after equivalent retained fault coverage.

## Author verification so far (not independent acceptance)

Primary183:40031 timed out; authorized fallback101 used in new detached
worktrees. Original dirty checkout preserved. Environment is robot_sim
Python3.12.13 remotely; local C:/Python314/python.exe. No dependency installed.
101 hardware query identifies H200 NVL at index2, A100 at0/1. These are CPU tests;
no mock ordinal result is offered as real-device verification.

- Initial new/guarded-core local suite:133passed after correcting a test's
  cross-thread ACK observation race. The client may receive bytes before the
  supervisor records ack_written; the precommitted exact identity/deadline,
  not one transient state label, is the safety fact.
- `4c64b69` remote controller/guard core/real process suites:165passed19.87s,
  actual exit0, no skips. The first fixture driver used exit0 for some terminal
  failure dispositions; revised driver uses74 without ordinary cleanup.
- `cc2fae8` remote15-file regression:624passed167.75s, actual exit0, no skips.
  Includes C0 A/K, C1c source/watchdog/lifecycle/leaf, C2b core/process, C2d0
  source bootstrap, policy/catalog, real Go2 asset and benchmark contract tests.
- Final context/negative fixture/core local run:177passed2.63s, actual exit0.
  Earlier local source-policy combination338passed16.68s. Ruff changed Python
  files passes. Catalog generation validates every exact LF release source
  through the real AST policy; no verifier-side newline normalization added.

Remote broad command (replace cwd with exact reviewed worktree):

```sh
ROBOT_SIM_TEST_GO2_MODEL_DIR=/data3/ang.gao/c4_asset_frozen/out/external/mujoco_menagerie/unitree_go2 timeout --signal=TERM --kill-after=15s 420s /data3/ang.gao/anaconda3/envs/robot_sim/bin/python -B -m pytest tests/unit/optics/test_guarded_lifecycle.py tests/unit/optics/test_guarded_evidence_fixtures.py tests/unit/optics/test_guarded_custody_process.py tests/unit/optics/test_guarded_wait_core.py tests/unit/optics/test_guarded_wait_process.py tests/unit/optics/test_native_bootstrap.py tests/unit/optics/test_native_source_policy.py tests/unit/optics/test_native_sources.py tests/unit/optics/test_sacrificial_watchdog.py tests/unit/optics/test_sacrificial_sources.py tests/unit/optics/test_sacrificial_lifecycle.py tests/unit/optics/test_sacrificial_leaf_descriptors.py tests/unit/benchmarks/test_closed_worker_protocol.py tests/unit/benchmarks/test_optical_closed_worker.py tests/unit/benchmarks/test_optical_benchmark_contracts.py -q -s -p no:cacheprovider
```

The22 fixed new process cases record supervisor/worker PID, parentage, actual
returncode and real supervisor EXITED trace. CPU cleanup/pre-exit/sync hangs
are killed/reaped by that supervisor with the parent alive, not scored from
the outer pytest timeout. Marker assertions distinguish destructor/ref release
from quarantine and reject atexit. CPU sync fixtures are not CUDA calls.

For the624-test run, timeout2302667(parent2302584), pytest2302692(parent2302667),
sampled bootstrap2303318(parent2302692) started09:34:03/09:34:30 remote time.
Their final exact PID query was empty. New fault workers2302864..2303031 had
observed exit0/74/-9 as recorded by the process harness. Final process-name
scan showed only unrelated long-lived Python2568/626988, left untouched;
GPU compute query empty. Initial165-run workers also had observed exits and
the subsequent scan found no owned Python/timeout. Local short-lived test PID
creation metadata was not captured; all command handles returned exit0.
No historical unidentified101 hang is claimed diagnosed.

## Remaining acceptance and phase gates

Run the final candidate's fresh regression and independent code review; prior
snapshots' passing runs do not approve the final context delta automatically.
Review must check actual state/trace/resource behavior, not this note or counts.
Full C2d1 approval requires the reviewed real CPU subprocess proofs. Broader
source-frozen guarded-role integration, pre-import module/session custody,
actual Warp/Torch upload/foreign-pre-return/unknown failure matrices and real
dual-GPU tests remain later C2d work. Final graph/warmup/barrier/readiness and
C2e/f integration remain incomplete. C4c3/c4, C4d/B3, product GPU preparation,
a3b, host/shipping and result/baseline gates remain closed.
