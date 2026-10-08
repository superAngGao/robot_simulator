# Native-scope diagnostic repair and C0 source-list compatibility

Initiative: Q54 / P12.4 / C4c2 / C2d3
Stage: implementation-note and fresh code/native preflight request
Author: codex
Version: v3
Date: 2026-10-08
Status: in_review
Related Files: native-scope-aggregate implementation-note/code-preflight-review v2;
native-scope-cache design-request/design-review v1;
benchmarks/results/q54_p12_4/closed_worker_runtime_contract.json
Owner Summary: Close F3's missing raw prerequisites and repair the independently
discovered old C0 bootstrap regression. These are bounded repairs under the
approved observer supplement and production CPU-owner extraction, with no new
native query, admission mechanism, protocol or product gate.

Base4da4ed271f20f847bea45ee005516c9d7cb21197; exact HEAD in review packet.
Full v2 report SHA7be029dba351e713c46d5fe904983b0f709235f25f29d55fde76133b463f36db
is archived unchanged at0e0b63e. It independently closes original F1/F2 but
requests F3 raw prerequisites before rejection. It does not authorize GPU tests.

## F3

Before checking raw target fields, the observer now records the pre/post
boundary, field presence, primitive context/type, stream nullness and current
cache flag. Post-return additionally records exact stream identity and the
returned raw handle's presence/type/value. Reads use vars and bounded fields,
without properties, repr, new object tokens, native queries or resource discovery.
Unexpected nonprimitive context/handle has its type recorded and no invented
pointer value. Actual returned owner stays held before this work. Failed
preconditions still do not emit a complete snapshot or query the error buffer.

Eight new CPU cases cover nonnull/missing raw context versus stream, post-return
identity/context/handle failures, and hostile property/repr objects. They read
the real persisted diagnostic and retain UNKNOWN/no release behavior. Local
observer/readback/parser now86passed6.46s, no skips with pinned source text.

## C0 regression found by required foundation checks

At exact4da4ed2, the primary-host foundation command covered C0 protocol/real
processes, guarded wait core/processes/lifecycle, native/CUDA arena CPU contracts
and cold custody-failure doubles. Result **303passed,5failed46.38s**. All five
failures are old C0 HELLO/bootstrap cases, not native observer calls. Their
transport-termination.bin names the exact cause: loaded repository module
optical_worker/lane_contract.py is outside the old C0 source allowlist.

Actual dependency is worker -> native_setup -> native_contract -> native_manifest
-> lane_contract. The last two CPU-only production owners were previously
extracted and approved, but C0's frozen literal source lists still named only
the original nine files. Add exactly those two paths to both the bootstrap
literal and its matching runtime contract. Keep exact full-list equality, hashes,
AST verification and actual import-ledger rejection. No discovery/prefix/name
bypass, native module, benchmark import, protocol/version or GPU preparation
change. Existing source catalog v12 repins only tools/optical_closed_worker.py;
raw catalog SHA8c8554e30a09319c8bc41d3eccefa0ef301e827341be6b626f91c7ba1687e2dc.

The existing real HELLO tests now require both actual owner paths in their
reported import ledger. Two CPU negative cases remove each path from the source
list and require bootstrap rejection before source loading. The original five
failures must be rerun against the clean revised commit, with both lanes,
disabled product PREPARE, raw-request rejection and control EOF still covered.
No successful revised C0 run is claimed before that command finishes.

Original failure archive: c2d3-4da4ed2-foundation-cpu-v1.tar.gz,978731bytes,
SHAb87d22bb0caf9fd1eefe8172c079645af3bf7e38cb538bb93490938ec8d78ae5,
local c4_assets and primary /data2/shared/ang.gao. It includes all test artifacts,
raw stdout/stderr,39 observed live descendant identities, summary and residual
check. All39 were absent afterward from OS and GPU PID records. Procfs polling
can miss very short-lived descendants; it is not complete attestation of every
spawn. No broad kill, reset or dependency mutation occurred.

Separately, exact4da4ed2 observer/parser/readback/codec **110passed9.39s** with
no skips; archive c2d3-4da4ed2-cpu-regression-v2.tar.gz,12217bytes,
SHA9c10b2f587ca13b70fec321272768331c08f81182455e3ae6bb7b5617f02f03f.
Its runner/pytest exited with no owned residual. These old-revision results and
failures remain distinct from forthcoming revised-HEAD checks.

## Requested review and limits

Fresh independent review must close F3 using the original two counterexamples,
recheck F1/F2/retention/order/ABI invariants, and inspect the exact two-path C0
compatibility repair and catalog-only repin. This is a bounded code/native
preflight plus C0 compatibility review; not whole C2d3 acceptance. No new GPU
observer has run. The16 cases remain gated until this preflight approves;
affected full fault/source/persistence regressions and whole independent C2d3
review still follow. C2e/f,C4c3/c4,C4d/B3,product GPU preparation,a3b,
result/baseline,host,R2,shipping remain closed. C4 stays active.
