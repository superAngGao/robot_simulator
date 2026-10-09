# Independent design re-review: pinned setuptools admission v2

Initiative: Q54 / P12.4 / C4c2 / CUDA G1
Stage: independent bounded design re-review
Author: codex, independent reviewer /root/setuptools_admission_design_v2
Version: v2
Date: 2026-10-09 (Asia/Shanghai)
Status: accepted — design only
Related Files: setuptools-admission design-request v2 and unchanged design-request/design-review v1; setuptools-admission gap-scan/gap-review/test-evidence v1; compilation-admission design-request/design-review v3; systematic-refactor implementation-note v1
Owner Summary: Approve the exact v2 bounded design. R1 is closed by literal per-field operation anchors, explicit ordinary-file joins without duplicate adoption, and fixed identity/occurrence/order requirements. No substantive finding remains. This authorizes bounded implementation; target dependency execution still requires its dedicated exact preflight, and compilation/native/product gates remain closed.

## Verdict, independence and candidate identity

**Verdict: Approve.** No substantive blocking or non-blocking finding remains in this design re-review. This is approval of the specified design, not an assertion that a setuptools transaction, new environment/record grammar or production collector/parser has been implemented or verified.

Fresh session `/root/setuptools_admission_design_v2` received the review packet and repository access without the implementation conversation. I did not author the candidate or delegate this review. I read AGENTS.md, collab/README.md and collab/REVIEW_PRINCIPLES.md, inspected the full candidate and its v1 delta, checked the previous R1 against its original reproduction, and inspected relevant governing contracts, actual production APIs and pinned source archives. Prior verdicts and author claims were not substituted for evidence.

- Repository: `C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design`.
- Branch: `implementation/c4-supervised-waits`.
- Actual frozen base: `b05f707b24eadae688299c1685c3ee47ed720601`.
- Actual reviewed HEAD: `d62da69a7d71ac1dafb92e1054630a39b934bd0a`.
- Actual HEAD tree: `7e49f58a8e21531f3e519545479ac71d59f41753`.
- Base-to-HEAD diff: only new design-request v2 (470 lines) and preserved design-review v1 (121 lines); no implementation delta.
- Candidate: `collab/q54-optical-lab-p12-4c4c2d-setuptools-admission__design-request__codex__v2.md`.
- Candidate working-byte SHA256: `e8343ed7c9217a47ad45da5f055801cf43537a67f8059240453ed14078e27db8`.
- Preserved prior report SHA256: `8eb4123e1e217dd7b248c899a4f4c3246dc0bbb538851b000520bc7496729165`.
- Governing compilation design v3 SHA256: `028b52c466ef46433235d4ec2495cd47d00200822cc64dfaea95614fd5bf26a2`.
- Initial and final `git status --porcelain=v1` empty. HEAD unchanged; `git diff --check` from the frozen base passed.

## R1 disposition: closed at design level

V1 allowed new helpers to share generic `external_import_owner` role and operation. Retention and exact object tokens did not independently distinguish insert-original from insert-wrapper, insert-original from remove-original, or alias delegate from alias wrapper. Coherent field exchanges could consequently pass the specified checks with lifecycle/object/retention facts fixed.

V2 section 7.1 closes that information gap. Every exported new helper field has one literal operation under the existing `external_import_owner` role and `HOST_OWNER` scope. Initialization itself retains its own literal role/operation. Generic fallback, alternative role sets and caller-selected labels are prohibited. The source-created finder and raw producer, both original hooks, both replacement hooks, alias spec/delegate/wrapper, distribution/metadata bytes, resource reader/bytes and per-read handles are all covered. Operations must be bound by the live receiving/creation boundary to the actual verified object, not accepted from serialized arguments.

The additional constraints are material: continuation origin, distinct helper/source/initialization slots, unique guardian sequence, actual guard6, prior root retention in both relevant index domains, adoption no later than first use, stable later alias bindings, and singleton reuse limited to the declared reader/bytes/distribution slots. Resource opens get distinct handles with canonical zero-based occurrence operations. Metadata and resource read indices progress independently, and metadata observations count successful PKG-INFO reads rather than the permitted absence query. Root-only partial objects remain held but cannot fill successful slots.

Ordinary source bindings are deliberately different. `_distutils_hack` must be its completed ordinary file owner; `_distutils_hack.override` and `jaraco.text` must be their currently executing ordinary owners under guard6. `setuptools._distutils` joins its completed module/spec/standard-delegate triple. The source wrapper Ls is retained and checked through the existing loader binding; it is not confused with the serialized delegate or new alias wrapper. Completed setuptools/build_ext/cpp_extension objects likewise join their named completed file records. These joins prohibit substituting an unrelated same-guard ordinary import without inventing a second source adoption.

This matches the actual APIs:

- `runtime_custody.py:329` checks direct guardian sequence, role, scope, operation and adoption-before-use; its ordinary file replay maintains name-specific spec/delegate/module and execution-call relations.
- `cuda_compilation_evidence.py:151` already uses the same role/operation join and separate lifecycle retention index check for generated roles.
- `native_bootstrap.py:1355` adopts through the generic holder. Calling it first and then trying to specialize the same object would be wrong; v2 explicitly forbids that sequence.
- `native_bootstrap.py:2171` and `:2222` provide root retention and the original observation registry. `freeze_adoptions` at `:2281` matches exact objects to existing guardian records.
- `lifecycle.py:306` rejects a second adoption of the identical object. The new helper strings fit the real identifier validation and can be adopted once using the existing mechanism. Ordinary source objects keep their original adoption.

The unchanged original v1 probe still reproduces all three old generic-anchor exchanges. A new independent v2 probe derives the literal table from the reviewed document and exercises the exact production `adopted` AST method against fixed CPU facts. Fifteen concrete slots (the thirteen singleton rows plus handles 0 and 1) pass; all 105 pairwise exchanges reject with `runtime adoption operation mismatch`, without changing lifecycle/object/retention facts. Root-only success, input origin, late adoption, wrong adoption/action guard, late lifecycle/journal retention and handle reuse also reject. Executing the real guardian `adopt`, `contains_exact_owner` and identifier-check AST on CPU stand-ins accepts the distinct helper objects and an ordinary source owner, then rejects a second adoption of that source without changing its existing custody.

These results demonstrate that the revised contract contains the missing discriminators and is compatible with the real API. The additional validation surrounding the extracted methods is an explicitly labeled schema-slice model. It is not a future production parser, a full live controller/guardian run, target execution, or proof that all implementation checks exist. Production collection/replay must later reject coherent permutations across every repeated action with independent lifecycle/retention fixtures fixed, including unrelated valid-token substitution and partial-adoption failures. V2 requires that proof; this review does not waive it.

## Affected overall design assessment

**Topology and authority.** The pinned setuptools source really appends `_vendor`, pops `backports`, and executes `_distutils_hack.override`. The one source-session-owned P-to-P+vendor transition makes the first two operations verified no-ops, with absence preconditions instead of deleting preexisting owners. Explicit rows, fixed site-before-vendor precedence, a singleton custodian, and original-P top-level interpreter resolution prevent the extra path from becoming generic import permission. Existing standard resolver/hook identities stay frozen, including generated sysconfig and terminal-after-poison paths. No generic finder, callback, runtime-learned allowlist or alternate deployment is introduced.

**Shim and alias behavior.** The exact installed `shim.__enter__` and `shim.__exit__` call the two module globals that the design interposes. The held original producer imports `setuptools._distutils`, creates a custom loader closing over that actual module, changes only its name in create_module, and has a no-op exec_module. The target `_bootstrap.py` confirms the specified intermediate and final matrices: `module_from_spec` overwrites `__spec__` while preserving the already set source loader/package/path/file. This is one module under two root names, not a new source execution or native creator-self alias. Restricting metadata delegation to the exact owner and permitted state, retaining the immutable original spec, checking actual publication after importlib return, and checking the final matrix thereafter are coherent requirements. The design explicitly retains and invokes the real custom methods rather than rewriting their behavior.

**Namespace clarification.** The archive's ccompiler imports `.compilers.C` and `.compilers.C.base`; the two namespace-parent `__init__.py` files do not exist in the archive. V2 sections 2 and 5 now explicitly route declared children through their existing file OR namespace rules. Current `_ExternalAdmission.find_spec` has the namespace branch and `_paths` already handles namespace parents. The special root alias only needs the bounded D parent-path relation; it does not authorize source-counterpart aliasing or invented namespace source execution. Finite locations and unlisted-descendant rejection remain required in the frozen preflight and production tests.

**Resource and metadata.** The actual jaraco expression is `files(__name__).joinpath('Lorem ipsum.txt').read_text(encoding='utf-8')`. The archived target compatibility interface supports a held ResourceReader; the actual resource is 1335 bytes with SHA256 `37fedcffbf73c4eb9f058f47677cb33203a436ff9390e4d38a8e01c9dad28e0b`. Target DistributionFinder.Context and Distribution.metadata confirm the exact name-only request/default path and METADATA-to-PKG-INFO fallback. A private Distribution subclass can expose the pinned 83.0.0 metadata without exposing the directory or adding general discovery. V2 freezes shared owner reuse and actual read occurrences as well as the original execution/thread/guard restrictions. Read-only handles, denied filesystem export/traversal, exact absence/file checks, copied-reader rejection and sticky caught failures remain production obligations. The interface probe uses a toy reader and does not prove those authority or write-denial properties.

**Failure and lifecycle.** The first compile root remains the one ownership base. Retention precedes fallible adoption/serialization; no failed object becomes a successful binding merely because it is held. Partial hook installation tracks restoration obligations individually. One restoration owner attempts each installed hook once, retains original and later errors, and cannot clear FAILED. Alias/publication and resource/metadata failures propagate through existing session/external/controller/compiler failure paths. Catching dependency exceptions or restoring originals does not permit COMPLETE or a new guard. No unload, rollback, retry deadline or second cleanup authority is introduced. The distinction between successful restoration and successful initialization stays explicit.

**Evidence and maintenance.** The immutable runtime-custody input prefix remains unchanged; v6 environment facts are frozen before that cut, while transaction events occur only in the guard6 continuation. New record v2 semantics do not widen old grammars or make serialized records live capabilities. The helper map belongs in one shared pure contract, while existing file, token, guardian and sticky-failure mechanics retain their owners. The dependency-specific module is a bounded extension, not a second importer. Full frozen source/namespace/optional/absence/data closure is still an implementation preflight deliverable, not demonstrated by the archive or this review. The removal condition and mandatory later whole-C4 structural refactor remain explicit.

**CPU proof boundary.** Section 8 now resolves the prior clarification: the future target dependency-only packet must identify its real production entry, any CPU outer-context stand-in, and its exact observation stop. It may reach RESTORED and exercise subsequent real resource/version reads, but cannot emit COMPLETE, synthesize cpp_extension completion, or claim the actual guarded device transaction. The full COMPLETE predicate still requires the actual cpp_extension import and later native route. That separation prevents stand-in evidence from approving integration it did not execute.

## Verification and observed results

Environment: Windows PowerShell and local CPython 3.14.0 at `C:/Python314/python.exe`; all executed Python used `-I -S -B`. Only stdlib, source reading and AST-extracted methods with CPU stand-ins were used.

Read-only inspection used `git status`, `git rev-parse`, `git branch`, `git diff`, UTF-8 `Get-Content`, `rg`, `Get-FileHash`, and stdlib zipfile/ast source reads. Both archive digests matched the packet independently:

| Archive | SHA256 |
| --- | --- |
| cuda-compilation-helper-sources-v1.zip | 8f5a7f25d4e7654cea2e157d44af708cc14b912edcdedbcc68e7852dcffdaac3 |
| cuda-compilation-target-importlib-sources-v1.zip | 3a41be7920961a96cb36e0aa442ecbc0ec0ff2f805ceadae9fadba917fab8289 |

Executed from the repository; all completed successfully:

```text
C:/Python314/python.exe -I -S -B ../c4_assets/setuptools_admission_design_review_v1_probe.py
C:/Python314/python.exe -I -S -B ../c4_assets/setuptools_admission_design_review_v2_probe.py
C:/Python314/python.exe -I -S -B ../c4_assets/setuptools_topology_gap_v1.py
C:/Python314/python.exe -I -S -B ../c4_assets/setuptools_admission_independent_probe_v1.py
C:/Python314/python.exe -I -S -B ../c4_assets/setuptools_reader_design_probe_v1.py
git diff --check b05f707b24eadae688299c1685c3ee47ed720601 HEAD
```

Observed results beyond the R1 checks above:

- Topology baseline passes; actual pinned path-extension and insert_shim AST nodes each reproduce `closed import topology changed` in the existing implementation.
- Same-object alias metadata is reproduced; existing production checks reject intrinsic-name drift and an alias child parent. The current source wrapper's missing reader produces `Can't open orphan path` after its positive control.
- Exact PKG-INFO bytes yield 83.0.0 through ordinary stdlib discovery, while a sole finder without distribution discovery yields PackageNotFoundError.
- The toy interface probe returns exact resource bytes and version83.0.0, observes `{'name': 'setuptools'}`, and reads METADATA then PKG-INFO. It rejects its traversal example. These are interface-feasibility observations only.

| Probe under c4_assets | SHA256 |
| --- | --- |
| setuptools_admission_design_review_v2_probe.py | 335242984119409a5cb3ab1f0883357718cec0f4ee9fb81429d5e5b46e8a135c |
| setuptools_admission_design_review_v1_probe.py | 66e8cf6d484e92fc97cda9ea50b0ed6b5db65ebbb6b7339025351d184645f055 |
| setuptools_topology_gap_v1.py | 3ffa3fcc206ca87e15c06067bd37811c02e2ebe68005e6b01ec2db57f877eab0 |
| setuptools_admission_independent_probe_v1.py | 0eaf1d17da42049cf071963d9c527593f79cf0ff4765a560d5c298a7d6ea3631 |
| setuptools_reader_design_probe_v1.py | ab1c504fd4dc2e0743adeaf23ad6e17cbca9f8771e524fe0e61dc6b3ed8fa742 |

The first default-encoding display rendered Chinese text incorrectly; it was reread with UTF-8. Some large combined displays were truncated and relevant sections were reread in bounded ranges. `git diff --no-index` produced the expected design delta and line-ending warnings; neither is a product failure. The new v2 scratch probe passed on its first execution. There was no candidate code, environment or toolchain failure during this review; current admission rejections are the expected gap controls.

## Limits, hygiene and exact approved scope

No setuptools/_distutils_hack/Torch/Warp package import, target interpreter run, SSH call, compiler/Ninja invocation, native/device observer, GPU query, process-supervision test, signal, installation or host-environment change occurred. No worker/supervisor/compiler/background service was launched. Probes completed synchronously; their temporary metadata/resource directories were managed by TemporaryDirectory. Only this report and the new v2 scratch probe were written outside the tracked repository. Candidate files remained unchanged and clean.

Approval binds only design-request v2 at the exact HEAD and SHA256 above. It permits bounded implementation under its frozen rules and proportionate allowed stdlib/AST work. It does not approve an implemented transaction or integrated collector/parser. Actual target dependency execution still needs the dedicated reviewed exact source/environment/runner preflight. No stand-in result here opens that gate.

Still required: complete frozen target import/namespace/absence/resource/metadata closure; the exact dependency-only CPU preflight and real production-boundary dependency positive/failure evidence; complete collector/parser mutation and partial-retention tests; renewed exact compilation source/catalog/runner/observer/ABI preflight; actual three cold builds and native target1/ambient0/partial-failure evidence; actual process/quarantine closure; and fresh independent whole-checkpoint review. Existing historical target stand-in results were inspected as supplied evidence, not rerun or promoted to full target success.

Compiler/backend/native execution, CUDA checkpoint acceptance, Warp compilation admission, R_setup/R_final, one-slot construction, warmups, C2e/composition, full C2d/C4 acceptance, product PREPARE/submission, sample/result/measurement authority, performance/backend selection and shipping remain closed within their existing gates. The whole-C4 systematic refactor remains unfinished and mandatory after functional and real evidence, with affected regressions and independent review before final acceptance.

The author may archive this complete report verbatim as review-only evidence. That archive must preserve this verdict, exact reviewed revision/digests, scope, limitations and remaining requirements.
