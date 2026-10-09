# Independent R0 fixed render storage code review v1

Initiative: Q54 / P12.4 / C4 remaining render integration
Stage: independent code review, bounded R0 fixed render storage integration
Author: codex, independent reviewer /root/fixed_render_storage_code_v1
Version: v1
Date: 2026-10-10 (Asia/Shanghai)
Status: accepted
Related Files: runtime-integration design-request codex v3 sections 4, 5, 8; remaining-setup gap-scan codex v1 G2; warp-slot-construction code-review codex v2; fixed-render-storage implementation-note codex v1; candidate paths below
Owner Summary: Approve the bounded R0 storage integration at 12c161969a6393722510cb3c373e2a0e3c25a637. No substantive finding. The original fixture math and CUDA tensor ordering are independently verified unchanged. Fresh CPU regressions: 572 passed, 13 skipped across two separate invocations. This is code/CPU approval only; real allocation/scope/failure acceptance, complete rendering and C4 remain unapproved.

## Identity, independence and scope

Reviewer session: `/root/fixed_render_storage_code_v1`. I received a fresh review packet without the implementation conversation, authored no candidate code, and delegated no work. I read AGENTS.md, collab/README.md and collab/REVIEW_PRINCIPLES.md, the required parent sections and G2, the predecessor allocation review, and the current implementation note. Author notes and prior approval were treated as claims rather than evidence for this delta.

Repository: `C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design`.
Branch: `implementation/c4-supervised-waits`.
Base: `83013616e4df5cf6bae13094e42ecae9cb529bf3`.
Reviewed HEAD: `12c161969a6393722510cb3c373e2a0e3c25a637`.
HEAD tree: `10da21949b3cd19edb928fa0ee885e3ffbb1fb16`.
Initial and final working-tree status were empty, and HEAD remained unchanged.

**Verdict: Approve**, exclusively for this nine-file R0 fixed render storage delta. There are no substantive unresolved findings in that bounded scope. This does not approve G2 as a whole or replace the mandatory native validation described below.

The delta comprises native_bootstrap.py, native_input.py, native_setup.py, native_source_policy.py, the v15 catalog, cuda_direct_light.py, test_native_contract_extraction.py, test_prepared_fixed_render_inputs.py, and the implementation note. Approval covers retaining original host BVH statistics, wrapping exact uploaded fixed inputs, and allocating/retaining one output bundle through the existing guarded mechanics. The predecessor allocation implementation was reread where this integration depends on it; its earlier approval is not being expanded.

## Findings and inspected behavior

No blocking or substantive non-blocking code defect was found.

1. **Original math and compatibility.** I independently parsed the base `_derive_fixed_fixture_arrays` and the candidate `_derive_fixed_fixture`. Every statement before the return and every expression in the returned array dictionary are AST-identical. The only added return is the original `host_bvh['stats']`. The array-only API delegates to that implementation. Fixture hashing still uses the same payload, without statistics, and the new test observes one real host build and retains its actual stats object. Frozen synthetic fixture checks execute in the regression suite. No second BVH implementation or fabricated statistics were introduced.

2. **Exact input storage and ownership.** Both wrappers check the exact backend prepared type and existing registered physical storage before constructing views. The original fixture must be the canonical type, own the same packed scene, and carry the selected fixture identity. Warp world/BVH fields borrow the prepared arrays directly; primitive metadata, ready event and original statistics are retained. No topology-origin/refit authority is supplied. CUDA first-hit/shade views likewise borrow arrays, and the local-scene owner retains the original prepared source. The session retains the complete view tuple. There is no native upload, copy, transform or reconstruction in these wrappers.

3. **State, scope and failure flow.** The source-session method checks same-session compilation phase, exact diagnostic bool, fixed-native input, original uploaded prepared identity, runtime binding and compiled owners. One existing setup segment surrounds validation, scoped construction and allocation. The new runtime leaf precedes device/stream context construction and the CUDA availability preamble. The existing single-use allocation context ties runtime, target, stream and original guard; each returned array is held before metadata inspection/adoption or the next allocation. Inputs, output aggregate and final storage tuple are retained through existing custody. Exceptions leave the segment through the existing sticky failure/quarantine path without normal completion or emergency resynchronization. The ready phase is set only after normal segment completion. No replacement lifecycle or callback framework is added.

4. **Single source of truth and minimality.** Existing production scene constructors, BVH container, render input types, output layouts and allocators are reused. The CUDA accessor name tuples are now shared with the new constructors; an independent comparison confirmed identical ordering for all 21 first-hit and 5 shade names. The orchestration helper and prepared wrappers serve distinct boundaries, with no duplicate renderer, uploader or allocation algorithm. The nested runtime scopes preserve the standalone allocator contract while covering the outer constructor/preamble work. No additional framework or refactor is necessary to accept this delta.

5. **Source closure and gates.** The two new phases extend the existing literal lane-specific closure; they do not admit tests/tools/benchmarks or the opposite renderer lane. A parsed catalog comparison found unchanged schema and lane inventories, with changes limited to the AST policy digest and the five changed production-source digests. The policy change updates source/class pins without weakening AST rules. Product dispatcher code is unchanged. There are no command recordings, kernels, launches, events, warmups, final readiness/origin issuance, resource seal/barrier or sample capability in this delta. CUDA's initial first-hit view takes frame-zero query scalars; it is not final per-frame binding.

## Test quality and fresh verification

Environment: Windows PowerShell, `C:/Python314/python.exe`, CPU tests and native/intake doubles. All invocations were fresh, with `-B` and pytest's cache provider disabled.

```text
C:/Python314/python.exe -B -m pytest -q -p no:cacheprovider tests/unit/optics/test_prepared_fixed_render_inputs.py tests/unit/optics/test_native_contract_extraction.py tests/unit/optics/test_render_parameters.py tests/unit/optics/test_native_output_allocation.py tests/unit/optics/test_cuda_artifact_runtime.py tests/unit/optics/test_cuda_scene_native.py tests/unit/optics/test_warp_module_preparation.py
291 passed in 19.43s

C:/Python314/python.exe -B -m pytest -q -p no:cacheprovider tests/unit/optics/test_native_source_policy.py tests/unit/optics/test_native_sources.py tests/unit/optics/test_native_bootstrap.py
281 passed, 13 skipped in 167.06s (0:02:47)
```

Both processes were polled to observed exit code 0. The source-suite skips are host-dependent POSIX/release-byte, pinned CPython 3.12, symlink, and unavailable real Go2 asset checks. They are not successful native acceptance. The author separately reported its source-suite result, but the result above is my own execution.

The 27 new cases exercise real fixed materialization, storage registration, production wrappers, output builders, session method and lifecycle while doubling native arrays/scopes and preceding intake/compilation checks. They reject copied prepared wrappers, replacement arrays, changed physical shape/device and lost fixture ownership before scene construction. Success and second-allocation failure are exercised in both lanes with diagnostics on/off; tests check original owners, scope restoration, exact failure identity, retained partial arrays, quarantine and zero normal completion/emergency synchronization on failure. The pre-existing allocation tests cover additional metadata/adoption/scope failures, caught retries and cross-guard reuse. These checks examine consequences, not merely exception types.

The relocation-test adjustment is not proof of equality for its explicitly exempted definitions. I independently confirmed that the two named CUDA factory bodies already differed from the original a1f6c8c relocation baseline at this candidate's base; this delta does not change those methods. The new derivation's full math was independently compared as described above. Remaining baseline bodies and legacy alias identities are still checked by the test. Thus the changed allowlist does not conceal an observed math regression in this revision; future edits to exempted bodies continue to need direct review and behavioral coverage.

`git diff --check 83013616e4df5cf6bae13094e42ecae9cb529bf3 12c161969a6393722510cb3c373e2a0e3c25a637` passed with no output. Independent inline Python AST checks and parsed catalog comparisons all passed. Initial Chinese text decoding and one large truncated read were corrected with UTF-8 targeted reads. An unquoted PowerShell `HEAD^{tree}` inspection failed; quoting the revision produced the recorded tree successfully. These were inspection-tool issues, not candidate failures.

## Limitations, remaining gates and process hygiene

No real Warp/CUDA allocation, GPU probe, remote command, compilation or rendering was executed by this reviewer. CPU doubles do not establish real target1/ambient0 scope/restoration, allocation lifetime, native partial-failure behavior, or complete source-frozen runtime integration. The new session tests deliberately double runtime binding, compile validation and source audits; the separate source suites verify source-policy/bootstrap behavior but do not execute the full native path. Real allocation/scope/failure acceptance remains mandatory before claiming this integration works on the selected native environment. The separate remote batch at 4fe6b3a is neither evidence for this delta nor part of this review.

R1 mutable storage, the complete slot/resource graph, command/ABI and per-frame binding, terminal/diagnostic events, actual warmups, final readiness/origin, final recording/seal/barrier, C2e/f acceptance and activation, whole C4/refactor completion, product PREPARE/native sample dispatch, R2/S/host delivery, performance/results/baselines, backend selection and shipping remain closed under their own requirements and independent reviews. This code approval does not waive or satisfy those gates.

I changed no candidate or tracked file. Only this report was written outside the repository. Tests created their normal temporary release fixtures outside the candidate. Final Git status was clean. A scoped Win32_Process check after both test exits found no process matching this review's test invocations, observed source-suite PID 34140 and children, or its pytest-1046 temporary fixture path. Peer teardown reported no failure. I did not signal or modify any process; the separately observed author tests and remote-regression SSH process were left alone. This is bounded owned-process hygiene, not exhaustive OS attestation.

This complete report may be archived verbatim as evidence for the exact revision and scope above.
