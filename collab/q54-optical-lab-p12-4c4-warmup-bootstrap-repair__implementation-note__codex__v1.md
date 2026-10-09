# Warmup bootstrap ownership repair

Initiative: Q54 / P12.4 / C4 runtime integration
Stage: bounded repair after native warmup bring-up failure
Author: codex
Version: v1
Date: 2026-10-10
Status: in_review
Related Files: runtime-integration design-request codex v3 sections 4/5.1/8/9; fixed-render-warmup code-review codex v2
Owner Summary: Remove warmup's reverse import of the process bootstrap. The existing source-session entry validates its original upload/runtime binding and passes those objects to preparation, matching the existing storage/graph pattern. No source admission is relaxed.

Base: 0dbc13b9613e6238cba58b63ba0b0b7b8e8f0845 (evidence-only successor of approved warmup code 3a3cd3a).
Candidate: commit containing this note; exact HEAD and completed test results in the review packet.

The first real Warp Go2 960x640 warmup attempt at clean native3a3cd3a failed in
setup guard13, before any warmup frame or optical launch. SourceBootstrapError:
repository import outside CPU core or reload. Worker74 and supervisor EXITED;
13 normal completions, full quarantine, zero emergency sync, no deadline/kill.
Batch99669 terminated exit1. Parent3444918/supervisor3445180/worker3445181 are
absent; live PID/start-tick observations were retained before their exits.

Raw remote directory: /data2/shared/ang.gao/c4-render-warmup-author-v1.
Failure archive: c4-render-warmup-failure-evidence-v1.tar.gz under remote
/data2/shared/ang.gao and local ../c4_assets; SHA256
7cc7eb78fa664aa1439fbc61730700996433189cceaeafc8653ca8d5dd64b67c.
This failed batch never started CUDA or the remaining seven cases.

The raw error does not name the module. Code inspection and a new reproduction
identify warmup's `from .native_bootstrap import _UPLOADED_INPUTS`: the actual
bootstrap executes as __main__ and is deliberately not an importable repository
module. Ordinary unit imports cached it under its package name and hid this
integration defect. With that bootstrap import rejected, the new test failed
on both lanes before the repair (2 failed,52 deselected in1.69s), at that exact
statement. This local test models the cold import restriction; it does not
claim a complete real loader/GPU execution.

The repair keeps the registry in the existing process entry. Under the original
guard, the entry checks exact uploaded input identity, validates the retained
runtime record and passes runtime/target/stream to the existing private helper.
No module alias, extra registry or source-policy exemption is introduced.
Existing warmup custody, execution, failure behavior and readback are unchanged.
The regression executes ten completed transactions with bootstrap reimport
forbidden on both lanes; additional tests reject missing/replaced uploaded
input before any optical work and retain full quarantine.

Author verification with C:/Python314/python.exe -B -m pytest -q -p no:cacheprovider:
native_render_warmup, native_warmup_support and the unchanged F1 repro wrapper
passed66 in104.43s (handle94831 exit0). Native source policy/sources/bootstrap
passed281 with13 platform/asset skips in167.92s (handle10401 exit0). These CPU
checks do not substitute for the real rerun. Observed test PIDs41140/3780 and
their parents7272/31376 are absent in the final scoped query; no matching test
invocation remains. The local downloaded failure archive matches its remote
SHA256. git diff --check passed.

Source inventory and phase admission are unchanged. The existing bootstrap
class/source pins and three affected source hashes are refreshed in catalogv16;
catalog SHA256 f6208cb33d29476878925e81f4f7e3c0665ca84f718aac4ff042f35b5a5fc70a.
Real warmup output must be rerun after fresh independent review. No image,
numerical correctness, timing, later-stage activation or whole-C4 completion
is established. The systematic whole-C4 refactor remains outstanding.
