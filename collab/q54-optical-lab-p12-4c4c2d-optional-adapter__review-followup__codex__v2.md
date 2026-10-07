# Optional-adapter design closure, not C2d3 code approval

Initiative: q54-optical-lab-p12-4c4c2d-runtime-adoption
Stage: review-followup
Author: codex
Version: v2
Date: 2026-10-07
Status: accepted
Related Files: optional-adapter design-request v1/v2 and design-review v1/v2;
runtime-adoption implementation-note v1; optional-import gap-scan v1
Owner Summary: Fresh independent re-review approved only the bounded optional
adapter design supplement. Implementation and full C2d3 acceptance are pending;
all product/result/baseline and later-stage gates remain unchanged.

Approved design revision:c60d6277ff94d31cdba9d10336b5a73fd78672ed.
Design v2 SHA-256:877da7e5476ff85e9273b591481447a5c76ba4636bcdee10100af8098ae675eb.
Reviewer:/root/c4_c2d3_optional_adapter_design_review_v2.
Unchanged report SHA-256:68dc15520875fa2cb0de36112891ea0937dcf303ca31a266e27d4fc6a58795da.
Prior report v1 SHA-256:aaee0d3ff0c67ee56a1e6f74771e946e6af1d8a06d0d5fc8c477119b24045b40.

The v1 design at65319cb was reviewed independently and received Request changes
for F1: it incorrectly required fallback_confirmed at the configuration boundary
before materialization. V2 permits unrequested rules there and on unused
synthetic paths, rejects pending denials at every successful boundary, and
requires both Go2 rules confirmed exactly once before materialization completes.
It does not move imports earlier or combine/renew guarded segments.

Both fresh sessions independently observed the pinned trimesh behavior and
verified the proposed exact built-in ImportError reaches each original fallback
once without executing its adapter. Each also ran the existing absent-versus-
started-failure distinction tests:2 passed in0.20s. These are source/import
compatibility observations, not implementation, custody or GPU acceptance.

No production code, tests, dependency/asset bytes, source catalogue, gate.json,
shared README/REVIEW_PRINCIPLES or C0 IPC was changed in this design round.
The implementation must continue to use the sole captured finder, canonical
materializer, guardian/controller and shared upload/readback mechanics. No
general ImportError exception, third-party patch, install or fake absence.

## Next implementation obligation

Implement the approved v2 environment schema, two exact disabled_optional rows,
identity-registered one-use rejection tokens, fallback confirmation and explicit
boundary validation. Add the same-session configuration counterexample and the
negative matrix. Repin the reviewed-source catalogue/policy and explicitly
version active fixtures. Continue actual both-lane/both-Go2 input parity and all
remaining whole-C2d3 real-device fault, multi-device and custody evidence before
requesting independent code approval. No CPU-subset approval substitutes for it.

## Read-only work during review

An author-only installed AST scan ran outside the frozen repository candidate.
Script:/data2/shared/ang.gao/c2d3-audit-static-dependencies-v3.py;
SHA-256:486dbdf50c3abac0352d076278ed0011feb0323dcddc845f126f69f077fc944e.
Artifact:/data2/shared/ang.gao/c2d3-torch-static-possible-edges-v1.json;
SHA-256:434d4ebcb97e0196abbfcb8122a69963ffb15ad84c196d49dad9ce0ca681dec5.
Local matching artifacts retained under c4_assets. It reports680 possible Torch
module edges from Python ASTs, not the true runtime closure. It skips function
bodies and cannot infer native-created modules; conditional branches may be
unused. No automatic allowlist entry, dependency import or GPU operation follows
from this inventory. It is not the CUDA acceptance proof.

Actual invocation:robot_sim Python -I -S -B with120s outer fail bound, target
argument torch and a new exclusive output artifact. Wrapper3368051, Python3368053,
parent3368049,start23:43:00+08; wait/SSH0. At23:51:25 both exact PIDs were absent.
No restart or outer-timeout-as-success. The design candidates remained frozen
and clean while each reviewer worked. Reports retain each reviewer's separate
probe/test limitations and scoped PID checks.

Only design scope is accepted. Whole C2d3/C2d, C2e/f, C4c3/c4, C4d/B3,
product GPU preparation, a3b, result/baseline, R2/host-payload and shipping remain
closed. C4 is not complete and the active goal is unchanged.
