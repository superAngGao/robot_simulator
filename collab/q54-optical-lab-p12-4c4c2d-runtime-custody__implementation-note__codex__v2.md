# Fixed native fault and protocol matrix continuation

Initiative: Q54 / P12.4 / C4c2 / C2d3
Stage: implementation-note
Author: Codex (implementation session)
Version: v2
Date: 2026-10-08
Status: implemented
Related Files: runtime-custody test-evidence v1; runtime-adoption design v2;
tests/support/native_runtime_process.py; tests/support/guarded_custody_process.py;
tests/gpu/test_native_runtime_fault_matrix.py
Owner Summary: The first actual fixed-fault run stopped on a missed Warp test
injection after10 passes. This revision repairs that test entrypoint and adds
the required stage-specific ACK/EOF cases. The74 collected lane cases are WIP,
not a passing matrix or independently approved implementation.

Base is `8c794db7e7f3003dd54031b070a983252227a078`; no production source closure,
catalog, admission rule, supervisor core or watchdog duration changes.

At that exact clean revision, the remote CPU/POSIX driver/parser/manifest
regression passed57 in23.70s. The GPU matrix
`/data2/shared/ang.gao/c2d3-8c794db-runtime-fault-matrix-v1` then produced
**1failed,10passed in300.48s**, with11 live external OS identity chains and
no observer errors. Eight real NumPy create/exec before/after exception cases
(both lanes) had worker74 and current aggregate cuts. Two real native-create
hangs were killed/reaped under their unchanged Warp30s / CUDA120s deadlines.

The failure was `runtime_backend_exec_before_raise-warp`: no fault marker was
written, and the real worker completed normally. The test correctly rejected
this as missing injection. The captured Python wrapper executes its frozen
bytes via `super().exec_module` on `importlib.abc.SourceLoader`; it never calls
the retained `SourceFileLoader.exec_module` delegate selected by the original
test. The repair wraps the actual `SourceLoader` execution for the fixed
`warp._src.context` name, and calls its original implementation before the
after-exec fault. ExtensionFileLoader faults for NumPy/Torch are unchanged.
No second loader, import implementation, source exemption or relaxed assertion.

All11 workers/supervisors/pytest identities were subsequently absent by
PID/start-time checks, and their worker PIDs were absent from the GPU process
listing. Raw attempts, pytest output, identity records and scoped residual
snapshot are retained in
`/data2/shared/ang.gao/c2d3-8c794db-runtime-fault-matrix-v1.tar.gz`,403056bytes,
SHA256 `0f04442d37a8d9477f0a0212b10fc4928e8788cc18c002a110ef6ad5453dc34e`.
The failed normal Warp result is not counted as fault-path validation.

The additional26 lane cases use existing real guarded segments and transport:

* Six phases per lane: setup definitions, environment, materialization,
  initialization, upload, readback. After the original segment receives ARMED,
  the child writes a CPU marker and stalls. The pytest parent then closes the
  actual inherited liveness pipe; the original supervisor observes EOF and
  bounds/reaps the child under the already-issued deadline.
* Initialization guard3 and terminal guard6: absent ACK, wrong-session ACK and
  write-half EOF. Only the selected real pending response is intercepted; earlier
  ACKs use the unchanged transport. Initialization DISARMED loss is likewise
  one dropped real response. No clock is extended or renewed.

The test checks the original core evidence, observed exit/reap, no matching
`ack_written`, no terminal native sync/release after protocol loss, and exact
aggregate attachment. Parent EOF is explicit pipe closure, not a claim that
the pytest process itself died. Existing CPU/source driver modes preserve their
original fault scheduling. Local collection confirms74 cases; collection is
not execution. Fresh runtime validation and whole independent review remain.

C2d3/C4 remain unaccepted. Dual-device/ambient cases and other still-required
failure coverage must close before whole review. All downstream gates remain
closed as stated in the governing design.
