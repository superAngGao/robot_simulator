# C2d1: cold import probe isolation

Initiative: q54-optical-lab-p12-4c4c2d-guarded-custody
Stage: implementation-note
Author: codex
Version: v6
Date: 2026-10-03
Status: in_review
Related Files: implementation-note v5; code-review v4
Owner Summary: Test-only isolation correction after a real combined regression
failure. No production, policy, catalog or activation changes.

The 15-file remote author regression at 8b1918d completed with exit1:
8 failed, 761 passed in 172.28s. All eight failures were the new in-process
import probes asserting the runtime was absent from sys.modules; combined
collection had already imported it. The four cold supervised process cases
passed. This run is not reported as passing or as GPU safety evidence.

Run each import probe in a fresh -I -B CPU child instead of unloading runtime
modules. Caught/propagated abort, cause, retained graph and before-unwind
evidence assertions remain single-source in the fixed test harness. The socket
peer moves to a shared stdlib-only support module. Child PID/PPID and returncode
are checked; communicate is bounded and exceptional cleanup kills/reaps only
that owned child. This does not change the product worker protocol.

An intermediate local attempt imported the pytest unit module in the isolated
child and failed because local pytest is user-site only (8 failed,101 passed).
The final harness imports no pytest/unit module. No environment installation or
isolation relaxation was used. Final local lifecycle tests:109 passed in6.87s.
Ruff import ordering was corrected afterward; remote rerun/review follow on the
frozen candidate. No results are inferred from earlier passing revisions.

Post-round local/remote scoped process checks found no pytest/guarded worker
residue. Remote GPU PID2483521 is unrelated and was left untouched. The original
unknown hang is not claimed diagnosed. Original dirty checkout is preserved.

All exclusions in v5 remain: C2d1 CPU custody only, no runtime adoption, complete
setup/readiness, product GPU preparation, a3b/result/baseline, R2/host/S or later
C4/B3 activation. Fresh independent review remains required.
