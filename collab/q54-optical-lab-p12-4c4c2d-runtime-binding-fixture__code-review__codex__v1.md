Initiative: q54-optical-lab-p12-4c4c2d-runtime-binding-fixture
Stage: independent code review, bounded test-fixture repair
Author: codex
Version: v1
Date: 2026-10-08
Status: accepted
Related Files: tests/unit/optics/test_guarded_runtime_binding.py; collab/q54-optical-lab-p12-4c4c2d-torch-upload-workloads__gap-review__codex__v1.md
Owner Summary: Approve the exact fixture-only repair at dcece18435e7719e87114ce8d95fa2e17ccb8519. Independently reproduced the prior 58-pass/2-fail result, then observed all 60 affected tests pass with the candidate. This closes C3's stale fixture counterexample and affected-suite rerun requirement only; it grants no whole-stage or GPU acceptance.

## Identity, revision and verdict

Reviewer session: `/root/runtime_binding_fixture_review_v1`, fresh independent review agent, without the implementation conversation. I did not author or modify the candidate. Read the full AGENTS.md, collab/README.md and collab/REVIEW_PRINCIPLES.md; reread the Chinese principles explicitly as UTF-8 after the shell's initial decoding was garbled.

Verdict: **Approve** for this bounded test-fixture repair. No substantive findings or blocking evidence gaps were found within this scope.

Repository: `C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design`.

- Base: `c683c9647b52b1d072b71a6dd8fb7e273e071648`.
- Reviewed HEAD: `dcece18435e7719e87114ce8d95fa2e17ccb8519`.
- Initial and final inspected tracked/untracked status: clean (`git status --short` / `git status --porcelain=v1` returned no rows).
- Entire base-to-HEAD delta: one file, 7 insertions and 1 deletion, confirmed with the full diff and `git diff --numstat`.
- Candidate test file SHA256: `064b037c5c146e58ec1d90730b1c27e223d074c0aaa0f6f2f8fe9bf108e611be`.
- Prior gap-review report SHA256 independently matched `0b7361db53054e11951fcd746b78fef239189dd60ce31f475a1327916823313b`.

The prior report and its actual CPU failure log were inspected as historical evidence. Their results were not reused as proof of this review's execution.

## Exact candidate diff

```diff
diff --git a/tests/unit/optics/test_guarded_runtime_binding.py b/tests/unit/optics/test_guarded_runtime_binding.py
index c07a388..a00365a 100644
--- a/tests/unit/optics/test_guarded_runtime_binding.py
+++ b/tests/unit/optics/test_guarded_runtime_binding.py
@@ -35,8 +35,12 @@ def runtime_session(monkeypatch):
     session.failed, session.failures = False, []
     session.runtime_owners = []
     session.canonical_input = object()
+    materialization_steps = []
     session.external = SimpleNamespace(
-        audit=lambda: None, phase="materialization", records={"torch": {"module": module}}
+        audit=lambda: None, phase="materialization", records={"torch": {"module": module}},
+        materialization_steps=materialization_steps,
+        bind_materialization_case=lambda case: materialization_steps.append(("bind", case)),
+        finish_materialization=lambda: materialization_steps.append(("finish", None)),
     )
 
     def check():
@@ -334,6 +338,7 @@ def test_session_materializes_actual_canonical_source_without_readiness(runtime_
     manifest = session.materialize_native_source(case_id)
     expected = frozen_workload_content(case_id)
     assert session.phase == "materialized"
+    assert session.external.materialization_steps == [("bind", case_id), ("finish", None)]
     assert manifest is session.canonical_input.manifest
     assert manifest.scene_contract_sha256 == expected.scene_contract_sha256
     assert session.render_source.packed_scene is session.canonical_input.packed_scene_owner
@@ -352,6 +357,7 @@ def test_shipping_or_unknown_case_cannot_enter_materialization(runtime_session,
     with pytest.raises(bootstrap.SourceBootstrapError, match="outside C2d3"):
         session.materialize_native_source(case_id)
     assert session.canonical_input is original_source
+    assert session.external.materialization_steps == []
     trace = controller.evidence().to_payload()["transition_trace"]
     assert not any(event["operation"] == "descriptor_observed" for event in trace)
```

## Behavioral inspection and preserved invariants

Inspected the complete changed test module, `_SourceSession.materialize_native_source` and adjacent ownership/runtime-binding flow, the real `_ExternalAdmission.bind_materialization_case` / `finish_materialization` methods, and the CPU render-source/fixed-fixture/replay builder chain.

The fixture creates a fresh list for each fixture instance. Both protocol callbacks record to that list. The success assertion checks exactly one bind for the supplied case followed by exactly one finish, rejecting missing, reversed, extra or wrong-case callback records. Shipping and invented cases require an empty list, in addition to retaining the original source and never entering a device descriptor. The added methods do not replace the production session method or canonical materializers.

The actual production method rejects disallowed cases before obtaining the setup segment; for accepted cases it binds while materializing, enters the recorded dependency leaf, constructs the actual CPU source, retains render/fixture-or-replay/source owners, calls finish, audits, and transitions to materialized. The fixture repair supplies the two methods this existing path requires. It introduces no alternate source-building implementation.

Both original success parametrizations remain present: asymmetric synthetic fixed input and synthetic dynamic rigid replay. Their assertions still check manifest identity, frozen scene/fixed-fixture/replay digests, exact packed-scene ownership, adopted render and source owners, absence of a runtime binding, and absence of a ready event. The existing single-use, identity-drift, copied-binding, upload/readback ordering, adoption-failure custody and device-before-stream restoration tests are unchanged. Fixture cleanup still terminates unfinished controllers, removes runtime/upload registry records and closes the peer.

No production, runtime profile, GPU helper/support, policy or acceptance-gate file changed. A two-callback test double is appropriate for this session-unit scope; it does not certify the real external admission's import-policy enforcement.

## Independent reproduction and verification

Environment: Windows, PowerShell; `C:/Python314/python.exe`, CPython 3.14.0, NumPy 2.5.2, pytest 9.1.1. All fresh runtime verification in this review was local CPU execution. Torch/runtime leaves are test doubles; canonical NumPy materialization is real.

Scratch runner: `C:/Users/AG/新建文件夹/c4_assets/runtime_binding_fixture_review_runner_v1.py`. It extracts the original test bytes directly with `git show c683c9647b52b1d072b71a6dd8fb7e273e071648:tests/unit/optics/test_guarded_runtime_binding.py` into `test_runtime_binding_fixture_review_original_v1.py`, outside the repository. The production/support files are identical across the reviewed delta. It then runs the original and repaired test sets in separate fresh subprocesses, captures each subprocess return code, writes raw stdout/stderr logs and asserts return codes `[1, 0]`. No test-source text substitution or canonical builder monkeypatch was used.

Executed runner command from the repository:

```text
C:/Python314/python.exe -B ../c4_assets/runtime_binding_fixture_review_runner_v1.py
```

Its original-suite command was:

```text
C:/Python314/python.exe -B -m pytest tests/unit/optics/test_guarded_native_readback.py C:/Users/AG/新建文件夹/c4_assets/test_runtime_binding_fixture_review_original_v1.py -q -p no:cacheprovider
```

Observed: **58 passed, 2 failed in 5.13s**, pytest exit 1. Both failures were `test_session_materializes_actual_canonical_source_without_readiness` for the two original parametrizations. Both raised `AttributeError: 'types.SimpleNamespace' object has no attribute 'bind_materialization_case'` at `optical_worker/native_bootstrap.py:2258`, before canonical construction. This independently reproduces C3 as a fixture drift defect, not an environment/path failure.

The candidate command was:

```text
C:/Python314/python.exe -B -m pytest tests/unit/optics/test_guarded_native_readback.py tests/unit/optics/test_guarded_runtime_binding.py -q -p no:cacheprovider
```

Observed: **60 passed in 4.82s**, pytest exit 0. Both previously failing real CPU construction paths now run through their original assertions and the new bind/finish assertion. Both rejected-case parametrizations pass the new no-binding assertion. The full affected pair of test modules passed without skips or xfails.

Fresh logs, relative to `c4_assets`:

| Evidence | SHA256 |
| --- | --- |
| runtime_binding_fixture_review_original_v1.log | 255593191ca17436e9da028c0d01da1947f320243c066a8ea64b2a62b362d4d4 |
| runtime_binding_fixture_review_candidate_v1.log | 694152fa0dbf3707b62802c46a518a7e1992f4cc3b91ce74d9a643bdbc23826b |

Additional verification: `C:/Python314/python.exe -B -m ruff check tests/unit/optics/test_guarded_runtime_binding.py` printed `All checks passed!`; `git diff --check c683c9647b52b1d072b71a6dd8fb7e273e071648 dcece18435e7719e87114ce8d95fa2e17ccb8519` returned no diagnostics. Final inspected HEAD remained the reviewed revision and status remained clean.

Tests used `-B`, `PYTHONDONTWRITEBYTECODE=1`, and disabled pytest's cache provider. Persistent review artifacts were written only outside the tracked repository in `c4_assets`. An optional host-routing decision file was absent during inspection; no fallback host or remote operation was needed. Initial shell rendering of Chinese paths/principles was an output-decoding limitation, not a test failure; result logs were captured as raw bytes.

## Limitations and retained gates

This approval closes the bounded C3 fixture defect and this affected-suite regression requirement. It preserves the original failure record: the prior 58-pass/2-fail suite remains a failed historical run. This review's fresh 60-pass result is separate evidence.

I did not execute the author's claimed remote 32-test run or independently revalidate its environment. No SSH command, remote checkout mutation/sync, GPU query, CUDA/Warp backend call, real external-admission test, Go2 asset materialization, native fault/hang case or full-project suite was executed. The locally observed real CPU paths are the two synthetic cases. The real external admission's checks and GPU lifetime/device properties require their own existing evidence; the double's callback-order log cannot substitute for them.

No whole C2d3 approval follows. Actual replay/Go2 CUDA attempts, ambient/logical/raw/scoped-stream and foreign/UNKNOWN observations, native fault/deadline/ACK/EOF/terminal coverage, complete process custody, final aggregate evidence and fresh whole-stage review remain governed by the prior report and other applicable decisions. This report makes no claim about the coordinator's concurrently owned attempts or their outcome. C2e/f, C4c3/c4, C4d/B3, product preparation, a3b, result/baseline, host and R2/shipping remain closed; no readiness, measured optical sample, benchmark or product authority is issued.

The complete unchanged report may be archived under the documented evidence-only exception. Archival cannot enlarge this approval's revision or scope.
