# C4c1c CPU owner-transfer failure closure

Initiative: q54-optical-lab-p12-4c4c1c-supervision
Stage: implementation-note
Author: codex
Version: v2
Date: 2026-09-28
Status: in_review
Related Files: supervision independent review v1; sacrificial_lifecycle.py
Owner Summary: Retain the whole returned/partial owner capsule before per-owner registration can fail.

The complete independent v1 report is preserved verbatim. F1 is repaired at
both `finish_setup_leaf` and `fail_setup_leaf`: after exact active permit and
tuple checks, the guardian first adopts the entire passed tuple as a strong
host-side retention capsule. Individual resource registration follows. Thus
an invalid member cannot leave later reachable owners only in temporary call
or traceback frames. Any per-owner registration failure irreversibly poisons
ownership, clears the active permit and prevents emergency recovery/release.
The full capsule follows the production guardian into process quarantine.

Six tests cover invalid first/middle/last members on both transfer paths. They
drop caller and exception references, force GC, verify every real owner is
still alive, require sync-based release to fail, drop the outer lifetime
object after quarantine, and check the owners remain retained without any
finalizer marker. The failure-path tests also use a descriptor not claiming
an owner return, proving adoption failure itself sets unknown ownership.

Author scratch run (101 robot_sim) of the two new test files and existing
C0 protocol unit file: **70 passed in 5.33s**. Fresh clean-candidate integration
and independent re-review follow. No GPU driver or product path is added;
the v1 exact scope and all activation exclusions remain unchanged.
