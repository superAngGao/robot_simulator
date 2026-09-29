# C2c1 F1 repair: portable structural AST identity

Initiative: q54-optical-lab-p12-4c4c2c-cpu-owners
Stage: review-followup
Author: codex
Version: v1
Date: 2026-09-30
Status: in_review
Related Files: CPU owners review codex v1; implementation 234268a
Owner Summary: Repair the independently reproduced test portability defect. No production code, algorithm, source admission or activation gate changes in this follow-up.

## Finding and repair

Independent review of 234268a returned Request changes, F1 P2. Local Python
3.14 and remote Python 3.12 format empty AST fields differently in `ast.dump`.
The four failures are a test/fixture defect, not a missing-dependency waiver.
The complete independent report is archived verbatim with SHA-256
`f3082f40284707a1984109d9e81844e0a978b655c2a6e7cc08b7e09229bbca6b`.

The test now hashes explicitly versioned structural JSON: node type, ordered
field names and values, including empty lists and None. Only empty PEP 695
type_params on function/class nodes is normalized away because Python 3.10/3.11
do not have that semantic no-op field; nonempty fields are retained. Unsupported
constant forms fail instead of being silently erased. Location metadata remains
excluded. The exact importer relocation normalization is unchanged.

All 36 fixture digests were regenerated from `git show c969296:<original path>`,
not from new production bodies. Original source SHA-256s and line spans remain
unchanged. New negative tests change algorithm, argument order, default value
and imports; each changes the structural identity. Empty fields/None and the
specific absent/empty type_params equivalence are directly asserted.

Local ownership/camera run: 88 passed in 2.57s on Python 3.14; Ruff passed.
Remote Python 3.12 repair validation is still required before approval. Local
Torch is absent; a3a hooks run in the prescribed robot_sim environment instead.
No dependency was installed or check skipped to claim the missing coverage.

## Preserved first-round evidence

Author remote regression at 234268a: 488 passed, four AST failures in 112.95s;
all selected GPU tests completed. Monitor returncode 1, watchdog_reason null.
Logs under `/data3/ang.gao/c2c1_234268a_validation_logs`, mirrored under local
`c4_assets/c2c1_234268a_validation_logs`; pytest.log SHA-256
`76fb24b37c9d42b54e2b51bd75114681ab512f255711e9bc18fa350f0c90c921`.
The exact command, environment, revision and process/GPU snapshots are retained.
Both real Go2 scene/fixture identities and 1080p replay passed production-only
CPU materialization; go2.log SHA-256
`bea359cbea1342c52a3c26a5365e2882929686db9524aea03adf9082ce4c57e7`.
The independent reviewer separately reproduced these asset/camera checks.

Post-round ps found monitor/test 2359942/2359974, Go2 2366144/2366145 and the
last observed supervisor/worker 2370234/2370239 absent. nvidia-smi showed only
unrelated PID2000951, untouched. Reviewer processes also exited as recorded in
the report. No observed owned hang; no claim of a system-wide or future guarantee.

C2c1 remains pending re-review. C2c as a whole, C2d/e/f, C4d/B3, R2/S/host,
product preparation/submission, a3b, result and baseline gates remain closed.
