# C4c2 camera contract: approval record

Initiative: q54-optical-lab-p12-4c4c2-camera-resolution
Stage: test-evidence
Author: codex
Version: v1
Date: 2026-09-29
Status: accepted
Related Files: camera-resolution design request v1, design review v1,
code review v1 and implementation note v1 in this directory
Owner Summary: Archive independent approval of the CPU camera prerequisite,
not approval of runtime adoption or parent C4 completion.

Approved implementation: `1bcb622af82d820c7247dced82b58c459b89233c`.
Approved design: `848b5a2abf24f15021b1dcab4c859beefce031ed`.
Independent code-review session: `/root/c4_camera_resolution_code_review_v1`.
Report copied verbatim; SHA-256:
`f021339bd7a11b4ff27a700148e528180cacf603bab0c0d70f7c03e0e40a3a41`.
Verdict: Approve, no substantive findings for the bounded CPU-only scope.

Reviewer independently ran Windows 122 tests and Linux 183 tests, and rejected
370 adversarial payload mutations on each host. See the full report for exact
commands, environments, source inspection, limitations and frozen revisions.

Author's separate Linux run on the same clean detached revision produced
183 passed in 13.55s, with no skips. Log:
`/data3/ang.gao/c4-camera-1bcb622-cpu.log`, also copied to
`C:/Users/AG/新建文件夹/c4_assets/c4-camera-1bcb622-cpu.log`.
SHA-256: `ec26b939202fd155d11d5f3d6ac488a867210e0645ebfd0fb1ee90c6910c4339`.
Remote checkout: `/data3/ang.gao/robot_sim_c4_camera_code_1bcb622`.
Host: authorized fallback 192.168.80.101 after primary 183 SSH timeout.
The author's local matching four-file selection produced 122 passed in 5.62s.

This archive is evidence-only. It changes no code, tests, design requirements,
gate, runtime manifest or workflow rule. The existing case/content hashes are
unchanged; no GPU work, timing sample, result or baseline was produced.
The new digest remains descriptive until independently reviewed integration
binds actual native ABI scalars and exact runtime objects. C4c2 integration,
C4c3/c4/d, B3, product GPU preparation/submission, host/shipping, a3b and result
activation remain closed. The overall C4 objective remains unfinished.
