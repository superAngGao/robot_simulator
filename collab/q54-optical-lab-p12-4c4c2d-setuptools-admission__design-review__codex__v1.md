# Independent design review: pinned setuptools initialization v1

Initiative: Q54 / P12.4 / C4c2 / CUDA G1
Stage: independent design review
Author: codex, independent reviewer /root/setuptools_admission_design_v1
Version: v1
Date: 2026-10-09 (Asia/Shanghai)
Status: in_review
Related Files: setuptools-admission design-request/gap-scan/gap-review/test-evidence v1; compilation-admission design-request/design-review v3; systematic-refactor implementation-note v1
Owner Summary: Request changes. The bounded transaction is feasible in its proposed topology, same-object alias, resource and named-version interfaces, but its new evidence schema does not distinguish successful helper roles against fixed lifecycle facts. Freeze those joins before implementation. No dependency, compiler, native or product execution gate is opened.

## Verdict and reviewed identity

**Verdict: Request changes.** One blocking design finding, R1, remains. No implementation approval is granted for this amendment. The finding concerns the proposed exported evidence contract, not a demonstrated live-runtime bypass in an implementation that does not yet exist.

This is a fresh independent session, `/root/setuptools_admission_design_v1`, supplied the review packet without the implementation conversation. I did not author or change the candidate. I read AGENTS.md, collab/README.md and collab/REVIEW_PRINCIPLES.md, the candidate and its governing compilation design/review, the gap evidence and the owner-requested systematic-refactor direction. Author and prior reviewer claims were checked against source or explicitly kept as unrerun evidence. No sub-reviewer was used.

- Repository: `C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design`.
- Branch: `implementation/c4-supervised-waits`.
- Requested and actual base: `4ac1e0dd0b42a08c9490152e84d47969e3d73da0`.
- Requested and actual HEAD: `b05f707b24eadae688299c1685c3ee47ed720601`.
- Actual HEAD tree: `25fea428741c95192f8b732d1dec1e0df62825eb`.
- Sole base-to-HEAD change: the 361-line `collab/q54-optical-lab-p12-4c4c2d-setuptools-admission__design-request__codex__v1.md`.
- Candidate SHA256 independently matches `f3061a5d3319c5816e973fdc4ec0619172fd55e366e9f55a8b67453f95bf5d46`.
- Governing compilation design v3 SHA256 independently matches `028b52c466ef46433235d4ec2495cd47d00200822cc64dfaea95614fd5bf26a2`.
- Initial and final porcelain status empty; HEAD and tree unchanged; base-to-HEAD `git diff --check` passed.

## R1 — High, blocking: helper identity fields lack distinguishable lifecycle anchors

Locations: candidate sections 4 and 7, especially lines 167–168, 288–296 and 298–317; section 8 item 4, lines 349–351. Relevant current shared APIs are `optical_worker/runtime_custody.py:329` (`_RuntimeReplay.adopted`), `native_bootstrap.py:1356` (`_ExternalAdmission.hold`), `native_bootstrap.py:2171` (`_CompileCustodyRoot`) and `cuda_compilation_evidence.py:151` (`_CompilationReplay._role`).

The draft requires new successful helpers to use HOST_OWNER / external_import_owner and exact guard6, and requires actual lifecycle adoptions, prior retention and stable tokens. It does not freeze a distinct role/operation or independently anchored relationship for each helper field. The ordinary holder currently adopts all such objects with both role and operation_id equal to `external_import_owner`.

Concrete counterexample: retain and adopt the actual original insert hook as token101 and its replacement as token103, each with distinct real adoption sequences, generic external_import_owner role/operation, HOST_OWNER scope and guard6, before `hooked`. Keep all object rows, lifecycle events, retention slots, upstream evidence, guards and event positions fixed. Exchange only `hooked.insert_original_token` and `hooked.insert_wrapper_token`. Both still resolve to valid, earlier, distinct guard6 adoptions with the required generic role. There is no other serialized original-hook/wrapper relationship to distinguish the assignment; `hooks_restored` has no extra fields. The same ambiguity permits exchanging insert/remove originals, or exchanging alias delegate/wrapper tokens consistently across `alias_spec`, `alias_created`, `alias_executed` and `alias_published`.

Consequences: a replay conforming to the concrete checks can accept a coherent but false helper-role binding. Saying that the live owner checked exact objects does not specify the independent evidence join, and repeating the initially assigned tokens does not supply one. The immutable live collector can reject mutation of its own issued journal, but the exported parser is separately required to reject these mutations with independently supplied lifecycle/retention facts held fixed. This is not a demand to authenticate an entirely rewritten execution history or to make parsed data a live capability.

Why existing checks miss it: retention proves which token owns which held object; generic adoption proves host custody, guard and time. Neither records whether that object was the original insert function, its wrapper or another helper. The ordinary source events anchor imported modules/specs/delegates, but do not separately bind these new hook functions and alias helper relationships. Generic source hashes and exact event field sets do not close that distinction.

Requirement: REVIEW_PRINCIPLES design sections 3–5 and 11 require verifiable contracts, real identity joins and frozen evidence schemas. The approved compilation design v3 sections 7.2 and 8.1 explicitly solved this same class of ambiguity with role/operation-specific anchors and fixed-lifecycle permutation tests. The new draft itself promises coherent-token-permutation rejection in section 8.4 and says copied labels cannot replace the joins in section 7. Those promises need concrete discriminators.

Independent proof: `../c4_assets/setuptools_admission_design_review_v1_probe.py` executes the exact production `_RuntimeReplay.adopted` AST method on fixed CPU stand-ins and a clearly labeled schema-slice model of the proposed helper checks. All three exchanges above pass the draft's generic-anchor slice without changing lifecycle, objects or retention. An illustrative per-field operation anchor, checked by the SAME production method, rejects all three with `runtime adoption operation mismatch`. The slice is not a future parser or a live guardian execution; it demonstrates the missing information in the specified join.

Required repair direction: freeze one complete successful-helper binding table and its first-use rules in the amendment. For new independently adopted helpers, distinct fixed operation IDs under the existing external_import_owner role are sufficient where they identify the exact field; distinct roles are another bounded option. Define the source/ordinary-event joins for already-adopted source objects instead of adopting the same object twice. Cover originals, replacements, finder/producer, alias spec/delegate/wrapper and resource/metadata helpers whose meanings otherwise have only generic anchors. Specify allowed sharing, required distinctness, guard/time/retention order, repeated-read occurrence binding, and what remains root-only on partial failure. Use the existing registry/adopter/shared join rather than introducing a second ownership system. Operation spelling may be chosen in the revision, but its closed mapping must be frozen there, not left to implementation judgment.

Proof after repair: the revised design must make the unchanged counterexample reject with lifecycle/object/retention facts fixed, while legal source-object reuse and successful helpers pass. The later production collector/parser tests must coherently swap every repeated binding, substitute a valid unrelated same-guard helper, and test missing/wrong/late anchors and failed adoption. Failed adoption must retain the original object without manufacturing a successful role. A schema-slice model alone will not satisfy implementation acceptance.

## Other high-risk design assessment

**Bounded topology and entry.** The actual installed `setuptools/__init__.py:17–21` appends the vendor directory, removes backports and imports the override. The proposed one-way P to P+vendor transition, absence preconditions, fixed site-before-vendor order and singleton custodian address those actual effects without granting a foreign finder or prefix import permission. Freezing SETUPTOOLS_USE_DISTUTILS and the fresh cwd/pybuilddir absence removes the actual source branch in `_distutils_hack.is_cpython`. Standard-library top-level resolution must use P explicitly; the existing finder passes the incoming path to standard resolvers and therefore needs the specified change, including generated sysconfig and terminal imports. This is an implementable bounded change, not a claim the current implementation already supports it.

**Shim state and exact source objects.** The pinned `shim.__enter__` and `shim.__exit__` call the two module globals that the draft interposes. The held original `spec_for_distutils` imports the source package, closes over the actual module in its nested custom loader, changes its name in create_module and does nothing in exec_module. Holding compiled code trees in the ordinary source loader makes identity-based function/code checks feasible; matching filenames alone would not. Retaining each replacement before assignment and tracking partial installation gives the single restoration owner a defined obligation. Successful restoration cannot clear FAILED, and caught failures/late publications still extend custody. These requirements are materially stronger than a final topology check and should stay explicit in implementation tests.

**Same-object alias states.** Target `importlib/_bootstrap.py:733–820` confirms the candidate's matrix: custom create changes the already-held object's name; module_from_spec subsequently replaces __spec__, while existing __loader__, __package__, __path__ and __file__ survive. Thus ALIAS_CREATED is an intentionally narrow intermediate state, and alias exec must see the alias spec with the old source wrapper/package/path. The alias spec is not a package spec even though the returned module has a package path. The proposed explicit root relation and later child resolver can accommodate that; treating it as a normal source-file record or a subtree alias cannot. Ordinary source checks must delegate only this matrix and retain the immutable source spec. The allowed source-only, post-create and alias-exec/publication matrices must not become generally interchangeable on later audits. Original source-package identity and source-name publication remain required after aliasing.

**Resource and metadata interfaces.** The actual `jaraco.text` module-level expression at lines 232–234 needs a resource reader. Target ResourceReader compatibility code supports the proposed held reader; the resource archive contains the actual 1335 bytes with SHA256 `37fedcffbf73c4eb9f058f47677cb33203a436ff9390e4d38a8e01c9dad28e0b`. Target metadata Context stores only the supplied name and defaults path to sys.path. Distribution.metadata tries METADATA then PKG-INFO, so the described absence check and pinned PKG-INFO subclass can return the actual 83.0.0 without directory or entry-point discovery. The reader's read-only handle, exact active execution/thread checks, immutable bytes, denied path export and sticky misuse rejection remain production obligations; the toy interface probe does not test those authorities. The trusted package's ordinary monkey patches operate on admitted module attributes and need no invented module-replacement record.

**Failure and ownership.** One initialization owner under the existing compile root, retention before fallible registration/adoption/serialization, no unload/rollback, one restore pass, preserved earliest error and sticky source/controller failure form a coherent lifecycle direction. No retry guard or deadline reset is introduced. On failure, successful publication/COMPLETE must not be emitted simply because restoration succeeded. Existing late-owner retention remains necessary because the actual producer and setuptools.version catch exceptions. The new journal must keep the predecessor's incomplete-prefix rules and closed old grammars; R1 concerns the additional success joins, not a need for another failure stack.

**Maintainability.** Locating the dependency-specific transaction in one captured module while reusing ordinary file checks, token collection, guardian ownership and failure propagation is appropriately bounded. Complete exact import/namespace/absent/resource rows are explicitly deferred to a frozen implementation preflight, not learned at runtime. That deferral is acceptable at this design checkpoint; this review has not proved the whole transitive closure. Fixed role bindings and action schema should likewise have one shared definition for collection/replay. The owner-requested whole-C4 structural refactor remains mandatory after functional and real evidence; it is not grounds to postpone R1 or tolerate duplicate importers now. The stated removal condition avoids retaining both this transaction and a future replacement artifact route.

## Non-blocking clarification for the future CPU preflight

Section 8.1 can prove actual pinned setuptools/resource/version behavior through production dependency boundaries without importing Torch. It cannot, by itself, prove the entire production transaction's COMPLETE predicate: sections 3–4 place that transaction inside the real first compilation context and require the actual cpp_extension import to return before COMPLETE. In the dedicated future preflight, identify exactly which production entry/state is exercised, which outer context is a CPU stand-in, and where the dependency-only observation stops. Resource/version reads may occur after RESTORED but before COMPLETE. Do not synthesize a cpp_extension completion or present the CPU-only result as full transaction, native scope or compilation success. The later real guarded route still proves that integration. This is a scope clarification, not an additional blocking finding or permission for a test-only alternate importer.

## Non-blocking clarification: ordinary namespace descendants of the alias

The author supplied a late source pointer, which I independently checked against the pinned archive: `setuptools/_distutils/ccompiler.py:5–6` imports `.compilers.C` and `.compilers.C.base`; neither `setuptools/_distutils/compilers/__init__.py` nor `setuptools/_distutils/compilers/C/__init__.py` exists in that source archive. This path needs finite `distutils.compilers` and `distutils.compilers.C` namespace rows, followed by ordinary named file children. Directory/file inspection is not dependency execution or a proof of the complete target closure.

Section 2 already explicitly preserves ordinary namespace tables for every other permitted name and requires frozen namespace locations. I therefore do not interpret its mention of distutils file children as revoking that existing mechanism, and do not make this a second blocker. However, section 5's wording that every child uses file-wrapper verification should be tightened in the revision: dispatch declared descendants through their ordinary file OR namespace rules, with no counterpart alias inference. The existing `_ExternalAdmission.find_spec` namespace branch and `_paths` namespace-parent branch provide the necessary mechanism; no new namespace adapter is justified. Include the two finite namespace parents in the source-audited preflight and production child-resolution positive, and reject an extra namespace location or undeclared descendant. Omitting these mandatory namespace rows or treating them as source-file executions would fail initialization.

## Verification, commands and results

Environment: Windows PowerShell, local CPython 3.14.0 at `C:/Python314/python.exe`, always `-I -S -B` for the executed Python probes. No dependency/backend import occurred. Only source/stdlib/AST CPU verification was performed.

Read-only inspection used git status/rev-parse/branch/diff, UTF-8 Get-Content, rg, Get-FileHash and stdlib zipfile/ast/hashlib source reads. Both supplied archive digests independently matched:

- `cuda-compilation-helper-sources-v1.zip`: `8f5a7f25d4e7654cea2e157d44af708cc14b912edcdedbcc68e7852dcffdaac3`.
- `cuda-compilation-target-importlib-sources-v1.zip`: `3a41be7920961a96cb36e0aa442ecbc0ec0ff2f805ceadae9fadba917fab8289`.

Inspected actual dependency initialization, shim/spec producer/custom loader, source package, version lookup, jaraco resource expression, backports path extension, monkey patches, ccompiler/compilers namespace layout and cpp_extension's mandatory imports. Inspected target importlib metadata initialization, distribution Context/discovery/version and resource compatibility paths. Inspected current source finder/loader, external resolution/audit/failure/adoption, compile root/observation registry and compilation/replay interfaces against the proposed changes.

Executed from the repository, each exit0:

```text
C:/Python314/python.exe -I -S -B ../c4_assets/setuptools_topology_gap_v1.py
C:/Python314/python.exe -I -S -B ../c4_assets/setuptools_admission_independent_probe_v1.py
C:/Python314/python.exe -I -S -B ../c4_assets/setuptools_reader_design_probe_v1.py
C:/Python314/python.exe -I -S -B ../c4_assets/setuptools_admission_design_review_v1_probe.py
git diff --check 4ac1e0dd0b42a08c9490152e84d47969e3d73da0 HEAD
```

Observed results:

1. The unchanged topology probe passes its baseline and reproduces both current `closed import topology changed` rejections from exact dependency AST nodes.
2. The unchanged independent gap probe reproduces same-object alias metadata, intrinsic-name and alias-child rejection, missing distribution resolver and the source wrapper's missing resource-reader failure, with its positive controls passing.
3. The supplemental author interface probe, inspected before execution, returns exact resource bytes and version83.0.0, observes Context `{'name': 'setuptools'}`, and reads METADATA then PKG-INFO. Traversal is denied by its toy reader. This establishes interface feasibility only, not production ownership or readonly-handle enforcement.
4. The new independent design-slice probe reproduces all three R1 permutations against unchanged generic adoption facts. Illustrative fixed operation anchors reject all three through the actual extracted adoption method.

Scratch probe SHA256s:

| File under c4_assets | SHA256 |
| --- | --- |
| setuptools_topology_gap_v1.py | 3ffa3fcc206ca87e15c06067bd37811c02e2ebe68005e6b01ec2db57f877eab0 |
| setuptools_admission_independent_probe_v1.py | 0eaf1d17da42049cf071963d9c527593f79cf0ff4765a560d5c298a7d6ea3631 |
| setuptools_reader_design_probe_v1.py | ab1c504fd4dc2e0743adeaf23ad6e17cbca9f8771e524fe0e61dc6b3ed8fa742 |
| setuptools_admission_design_review_v1_probe.py | 66e8cf6d484e92fc97cda9ea50b0ed6b5db65ebbb6b7339025351d184645f055 |

The target-3.12 probe/report is supplied historical evidence, not a new execution by this reviewer; target source bytes were independently inspected. The actual supplied target script is `setuptools_admission_target_probe_v1.py` (SHA256 `cd5205c1d9204febd75f98f48f5d16b06ffb8e2b061a7a2669a8c985932bad39`), rather than the shortened `target_probe_v1.py` in the task packet. I did not invoke its remote path locally or rerun it remotely.

Non-product inspection issues: the initial unquoted PowerShell `HEAD^{tree}` was misparsed; it was corrected to the quoted expression. Initial non-UTF-8 text display was reread with UTF-8. Some combined displays were truncated and relevant/full remaining sections were subsequently read in bounded ranges. These are tool/display limitations, not design or environment failures. No test code needed correction after the independent R1 probe was written.

## Limits, hygiene and exact scope

No setuptools/_distutils_hack/Torch/Warp import, dependency installation, remote command, compiler/Ninja invocation, native/device observer, process-supervision test, process signal or environment modification was performed. The existing probes' temporary metadata/resource directories are managed by TemporaryDirectory and their synchronous processes returned. No worker, supervisor, compiler or background service was launched by this review. Only the new independent scratch probe and this report were written outside the tracked repository; the frozen candidate stayed clean and unchanged.

Unexecuted and still required: complete frozen target closure and absence/path facts; reviewed exact dependency-only source/environment/runner preflight; actual production-boundary target dependency positive/failure tests; integrated collector/parser mutation tests; fresh compilation source/catalog/runner/observer/ABI preflight; real three cold builds, target1/ambient0, native partial-failure and process/quarantine evidence; independent whole-checkpoint review. Stand-in and interface results do not discharge any of these requirements.

This report approves no new implementation or execution scope. The bounded architectural direction is supportable, but v1 must be revised for R1 and independently re-reviewed before implementing this amendment. Existing predecessor approvals retain only their original scope. Dependency-only execution still requires its future exact preflight. Compiler/backend/native execution, CUDA checkpoint acceptance, Warp compilation admission, R_setup/R_final, one-slot construction, warmups, C2e/composition, full C2d/C4 acceptance, product PREPARE/submission, sample/result/measurement authority, performance/backend selection and shipping remain closed as previously specified. The whole-C4 systematic refactor remains unfinished and mandatory.

The author may archive this complete report verbatim as review evidence. Archiving must not change the verdict, reviewed revision/digests, R1, required proofs or remaining gates.
