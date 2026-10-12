Initiative: cuda-render-experiment-archive
Stage: implementation-note
Author: codex
Version: v1
Date: 2026-10-12
Status: draft
Related Files: optical_worker/; optics/; tests/support/; tests/unit/optics/; physics-correctness-handoff__request__owner__v1.md
Owner Summary: Owner requested preserving the CUDA rendering replacement experiment on a new remote branch and moving the next session to physics correctness. This is an unfinished experiment snapshot, not an implementation approval or release.

## Archive identity and scope

- Repository: https://github.com/superAngGao/robot_simulator
- Archive branch: `archive/cuda-render-experiment-2026-10-12`.
- Source checkout: `C:/Users/AG/新建文件夹/c4_worker_completion`.
- Previous branch: `implementation/c4-worker-completion`.
- Previous committed HEAD: `157a3948d5f4f97dffe831a1f2bcc39ca57ef207`.
- This archive retains that complete history and the 11 modified/new experimental files present at handoff, together with this note, the physics handoff, and the existing regression stdout.
- The owner's explicit instruction is to push the unfinished experiment for preservation. No independent approval is claimed for the newly archived delta; ordinary implementation review and real-device gates remain incomplete. Main is not being merged or advanced.

## Most recent unfinished delta

Canonical render sources retain the existing optical registry so the fixed final-readback driver can reuse CPU rendering as a numerical reference. The new `tests/support/final_worker_numeric.py` stages selected frame references, validates frozen parameters and channel payloads, and compares results between RESOLVE and RELEASE. Selection covers initial samples, the last sample, and camera-wrap samples 119/120 when present. Native source catalog v18 was refreshed for the changed owners. Tests cover the numerical checker, driver integration, and failure propagation.

## Evidence and limits

- Existing local Windows CPU regression stdout reports **290 passed in 268.43s**. It is copied unchanged to `cuda-render-experiment-archive__test-evidence__codex__v1.txt` from `../c4_assets/final_numeric_integration_v1.txt`.
- This is a previous run, not a fresh archive-time run. The retained stdout does not contain the full invocation. It is not evidence of real-device acceptance, a complete C4 result, physical fidelity, or CUDA/Warp performance.
- Some driver tests use modeled native behavior. Their passing results cannot substitute for actual CUDA/Warp execution and lifecycle evidence.
- **Known unresolved startup issue:** the acceptance harness starts the supervisor using `-I -S`, while its newly imported numerical reference module imports NumPy. That isolated supervisor cannot rely on site packages. CPU reference preparation also needs placement outside existing child/supervisor timeout budgets. A possible next step is preparation in the ordinary acceptance parent and a fixed reference handoff; it has NOT been implemented or approved here.
- EXECUTE observations/raw-parser/sentinel coverage and remaining G3–G5 / complete D1 evidence still require work if this experiment is resumed.
- No new native GPU campaign was performed for this delta. There is no completed fair CUDA-versus-Warp speed conclusion to carry forward.
- Do not automatically resume the old optical C4 objective in the physics session.

## Verified physics baseline for the next session

Read-only checks on 2026-10-12 found:

- GitHub `main`: `a20d75ecae624de3be054916142386477f0a352c`.
- Primary host `tileops-dev`, `/data2/shared/ang.gao/robot_simulator`: clean `main` at `f771e612beaeea126e26926738097512cd2d809c`.
- The server main and this experiment's previous HEAD have identical Git trees for `physics`, `robot`, `rendering`, `sensing`, `rl_env`, `domain_rand`, and identical `simulator.py`.
- In particular, `physics` tree = `1bf3c590a9644a7375c97d61d6bd8479f63fe0c7` and `rl_env` tree = `642c5e69fc6d95ca856cfc565363fced9d7ba9a2`.

The new physics effort should branch from the verified server main in an isolated worktree. Recheck these facts at session start; do not reset an existing checkout or assume the older GitHub main is the latest physics baseline. Other local/remote working directories are not cleaned or implicitly included by this snapshot.
