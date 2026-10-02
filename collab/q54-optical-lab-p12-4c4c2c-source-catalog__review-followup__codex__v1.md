# C2c6 mandatory Go2 driver repair

Initiative: q54-optical-lab-p12-4c4c2c-source-catalog
Stage: review-followup
Author: codex
Version: v1
Date: 2026-10-03
Status: in_review
Related Files: source-catalog review codex v1; tests/unit/optics/test_native_sources.py
Owner Summary: Fix the mandatory real-asset test driver, not the production guard.

Independent review of `f8e9c4b015fcb78e464f9a62d32e97a97e39a31c`
found one P2: both Go2 tests omitted the required actual asset evidence and failed
before materialization. Author remote regression was 319 passed / 2 failed;
reviewer focused regression was 216 passed / 2 failed. These were test defects,
not environment failures or missing assets. The original full review is retained.

The cold script now calls production `preflight_case_asset` for each real Go2
case and passes its returned evidence to the existing materializer. It retains
both resolutions, both lanes, and actual-module-ledger/no-runtime-import checks.
There is no fabricated evidence, fallback scene, weakened guard, or new skip.
Only the test driver changes; all production source/catalog bytes stay fixed.

Separate author and reviewer read-only diagnostic scripts already exercised this
correct call chain at the original candidate: both lanes observed 51 repository
modules, no Torch/Warp/benchmark/test/example imports. Those probes narrowed the
defect but did not approve or repair the committed tests. The revised committed
tests must pass remotely with actual Go2 assets in a fresh independent review.

The author probe PID 2099830 and timeout parent 2099809 exited; scoped post-check
also found reviewer PIDs 2098385/2098384 absent. GPU compute list was empty.
Primary 183:40031 still timed out at this continuation; authorized 101 remains
the validation host. No claim is made about the cause of the historical hang.

Old C0, C2d/e/f activation, GPU preparation, C4 completion, B3/host delivery,
shipping, a3b, results and baseline remain closed and unchanged. No new policy.
