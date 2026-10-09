# Named-array join: source-release pin repair and exact validation provenance

Initiative: q54-optical-lab-p12-4c4c2d-array-adoption-join
Stage: implementation and pending independent re-review
Author: codex, implementation coordinator
Version: v2
Date: 2026-10-09
Status: in_review
Related Files: array-adoption-join implementation-note v1; runtime-adoption whole-code-review v2/v3; native_source_policy.py; data/native_sources/v12.json
Owner Summary: Preserve the valid named-array repair and fix its newly found
cold-source regression. Exact committed-source verification follows this commit;
no source-preflight or whole-stage success is claimed before those tests run.

The independent v3 Request changes report is archived unchanged, SHA256
bff857b156afc8fe05778a9d399790b9869bf83530d02f4477ad390ced01e5d5,
in evidence-only commit51519ad. It found lifecycle's catalog digest updated
but its separate exact regex-source exception still bound to old fc86182 bytes.
Both-lane captured entry consequently fails before GPU work. The original
Git-LF reproduction and all failed cold-suite logs remain in c4_assets as
listed in that complete report. No failure is reclassified as a pass.

This delta changes only the existing lifecycle regex exception pin to its
actual committed LF SHA256
2ca76eac3b72fc99f22d6c671e51a9a86724609575b0d61c5b101d3f7a6334dc.
The derived AST-policy digest and source catalog's native_source_policy file
digest are refreshed consistently. All permission sets, exception owners,
verifier logic, schemas, lane/source lists and runtime code stay unchanged.
No byte normalization is added to production verification. No new verifier or
test-only implementation is needed: existing complete capture and cold tests
already detect the regression.

Correction to v1 validation provenance: its684pass/3skip run occurred on the
edited working tree before edba56a was committed, while Git HEAD was83cfef6.
Most tests exercised edited modules, but the regex-release test explicitly
read git show HEAD and thus inspected the old lifecycle bytes. That run is
not proof of exact edba release admissibility. Future release/cold checks in
this round run after the concrete commit is frozen. The v3 reviewer independently
observed five exact custody-preflight failures and a separate76fail/544pass/
29skip source suite; many Windows errors surfaced as non-UTF8 stderr decoding,
so their report carefully distinguishes those from direct pin diagnostics.

The edba parser repair itself has completed CPU actual-record validation:

- Original16 fc86182 records still roundtrip and attach; all8 original named
  array-token exchange attacks now reject, as do all8 old missing-completion
  attacks. The reviewer independently reproduced this bounded result.
- All113 historical originals (16native/85fault/12terminal) replay with their
  original source/trusted-input facts preserved. This is compatibility evidence,
  not runtime execution of edba.
- Actual16 old/new positive records times roundtrip/two exchange mutations:
  48passed/1056deselected in112.01s. The32 mutations exchange array tokens or
  guardian-sequence references while keeping the original lifecycle fixed.
- All336 original source artifact hashes remain unchanged. Runner3424192 and
  children3424198/3424954/3427808 are absent from OS/GPU residual checks.

Raw archive (local c4_assets and primary /data2/shared/ang.gao):
c2d3-edba56a-array-join-cpu-v1.tar.gz,13577bytes,SHA256
1704801cde136b1cfb2200a925e0f0556f814006c65b38e0700f89e9d2ba94ba.
Its runner script is run_array_join_cpu_v1.py. The record replay imported no
NumPy/Torch/Warp. This archive does not close the source-pin finding.

Next required evidence is committed-source validation for every catalog blob,
both-lane actual capture/trusted preflight, affected complete source/bootstrap/
alias/failure suites, and fresh independent whole review. Linux cold importer
fixtures run separately from arena collection that imports real Torch. The
fc86182 full85 matrix continues on its unchanged clean GPU tree; its full
result and approved subsequent12terminal remain pending. The isolated CPU
tree can receive this repair only after reviewer jobs have finished; no running
GPU checkout is switched. Existing runtime provenance is never relabelled.

C2d3/C4 remain unaccepted. Compilation/warmup/final graph/readiness, C2e/f,
C4c3/c4, C4d/B3, product GPU PREPARE, a3b, result/baseline, host/R2/shipping,
performance comparison and backend selection remain closed.
