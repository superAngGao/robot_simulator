# C4c1c independent review archive and runtime blocker

Initiative: q54-optical-lab-p12-4c4c1c-gpu-validation
Stage: test-evidence
Author: codex
Version: v3
Date: 2026-09-28
Status: in_review
Related Files: GPU-validation review v3; review-followup v3
Owner Summary: Code findings closed; required Linux/GPU proof remains blocked.
No checkpoint or product gate is approved by this evidence-only record.

Reviewed candidate: `9db8146f9c818dd580ce73d886f7afe4c5d22d73`.
Reviewed tree: `2b68fcb160fae4c0bd34062aa1792b591573142d`.
Independent session: `/root/c4_sacrificial_gpu_review_v3`, fresh context.
The complete report is preserved verbatim in the adjacent review v3 file.
Its SHA-256 matches the external original:
`ebcfb3cdf10b5e026fc56ca02b8dabbd7da7e74287238f12bf978ee3306ce3c5`.

The reviewer closed the prior source-cache and implicit-readback code defects,
raised no new substantive code finding, and returned **Blocked**, not approval.
Its fresh CPU checks total 152 passed and 29 platform skips; the 79 GPU cases
were collected only. See the original report for exact commands and limits.

Additional author-side Windows Python 3.14 check on the same frozen candidate:

```text
python -m pytest tests/unit/benchmarks/test_native_readiness_authority.py tests/unit/benchmarks/test_closed_worker_protocol.py tests/unit/physics/test_cuda_publication.py -q
```

Observed: **46 passed in 2.54s**. This is separate author evidence, overlaps some
reviewer checks, and is not added to the independent test total or GPU proof.

At approximately 2026-09-28 22:08-22:13 Asia/Shanghai, the author retried the
primary SSH alias `tileops-dev` with BatchMode and a five-second connect timeout,
then the authorized fallback `ang.gao@192.168.80.101` with the configured key.
Both connections timed out (183.166.183.33:40031 and 192.168.80.101:22).
Read-only adapter inspection showed WLAN up and local VPN adapters disconnected;
this is an observation, not proof of the remote cause. No remote command ran.

The final outcome of the earlier remote dirty-draft test process remains
unknown. After connectivity returns, inspect that process/log state first,
preserve its worktree, establish this exact candidate in a clean checkout and
run the required POSIX, real-device, two-device and C4c0 regression selections.
Old draft outputs cannot establish this revision's correctness. A new
independent review must assess the resulting runtime evidence before advancing.

This commit contains only the unchanged review and this evidence record under
the collaboration protocol's evidence-only exception. It does not modify
README, review principles, code, tests, design, gates or the verdict. C4c1c is
not accepted; C4 is incomplete. C4c2-c4, C4d/B3, product GPU preparation, a3b,
host/shipping paths, result and baseline outlets remain closed. No dependency
installation, remote push, merge or backend/default selection occurred.
