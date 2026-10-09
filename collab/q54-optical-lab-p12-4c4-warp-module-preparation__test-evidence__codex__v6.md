# Native Warp post-compilation rejection matrix

Initiative: Q54 / P12.4 / C4 guarded compilation
Stage: author native evidence, not independent acceptance
Author: Codex
Version: v6
Date: 2026-10-10
Status: in_review
Related Files: warp-module-preparation design-request v2, test-evidence v4/v5;
  warp-admission-support code-review v1; tests/support/native_runtime_process.py;
  tests/support/cuda_artifact_runtime_acceptance.py
Owner Summary: All 15 fixed real Go2 post-compilation rejection cases completed
  at the reviewed 4fe6b3a support revision. Original failed guards quarantined
  without another native leaf, completion or synchronization. This is not render
  execution, allocation acceptance, speed evidence or whole C4 completion.

Exact clean remote source revision: 4fe6b3ae71ac6aa1586cad5f4f7ac66c6717a2ec,
/data2/shared/ang.gao/robot_sim_warp_modules_wip on tileops-dev. Python 3.12.13,
Warp 1.12.0, H200 target 1 / ambient 0, the unchanged inventoried Go2 profile v4
SHA256 02429134207218e87ca1305e6c89cc7b34948cda41499f1ac7399ac961ffbeeb.
The runner verified source revision and cleanliness before every case and used
the original 1000 ms ACK / 120000 ms operation budgets. No reset, cache deletion,
dependency change or unrelated process signal was used.

Command: remote robot_sim Python -I -B
/data2/shared/ang.gao/run_warp_module_admission.py. Tool session 96843 observed
terminal exit 0. Parent 3095592. Raw directories have prefix
/data2/shared/ang.gao/c4-warp-module-author-v and versions 22 through 36:

| Versions | Actual mutation after three real module compiles |
| --- | --- |
| 22–24 | kernel ABI dtype, rank, entry hash |
| 25–29 | compilation config, module options, cache permissions, artifact bytes, redirect |
| 30–33 | executable mapping, copied module set, set session field, copied compilation record |
| 34–36 | runtime binding target, stream, captured source linecache |

Each case independently reached guard 10 after ten successful setup completions.
The reviewed runner required the selector's exact original exception type and
message, worker exit 74, eleven ARMs / ten normal completions, actual worker and
supervisor disappearance, one failed original guard, full quarantine, zero
emergency synchronization and zero further device leaf after the fault marker.
It replayed the existing source/runtime/custody artifacts and rejected a success
binding, ordinary atexit cleanup or unexpected additional ARM/COMPLETE. All 15
cases passed those checks. The session-field mutation is not a separately issued
foreign session; target/stream substitutions are CPU identity rejection before
a GPU operation, not foreign-device execution tests.

After parent exit, archive_warp_module_author_v5.py verified all 31 distinct
owned PIDs absent before archiving all 15 raw directories, exact author scripts,
batch result and profile. Archive and adjacent summary copied locally:
../c4_assets/c4-warp-module-author-evidence-v5.tar.gz and .json;
remote copies are in /data2/shared/ang.gao. Archive SHA256, verified locally:
6ac564235e6171b727fac86246052fa2847ce45489160ad7d65e78a54b68c354.
Also preserved c4-warp-admission-batch-v1.json. The final nvidia-smi snapshot
contained only unrelated MPS PID 2238410, untouched. Failure caches remain on
the server; this archive does not claim to include every compiled cache byte.

The subsequent original Warp input-scope / CUDA factory-input regressions run
separately at the same frozen revision; they are not claimed complete here.
Compilation still requires its fresh independent native acceptance. The newer
local output-allocation approval and fixed-render-storage implementation have
not been run by this matrix. Numerical rendering, graph construction, warmups,
final readiness/barrier, full C4/refactor and later product gates remain open work.
