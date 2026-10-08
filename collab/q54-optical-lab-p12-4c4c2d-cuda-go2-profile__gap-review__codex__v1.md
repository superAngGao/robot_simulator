Initiative: q54-optical-lab-p12-4c4c2d-cuda-go2-profile
Stage: independent deployment / code gap review
Author: codex, fresh independent reviewer /root/cuda_go2_profile_gap_v1
Version: v1
Date: 2026-10-08 (Asia/Shanghai)
Status: accepted for the bounded descriptor scope only
Related Files: cuda_go2_profile_review_request_v1.md; 183_cuda_go2_author_proposed_v1.json; runtime-adoption design v2; native-global-publication design v1 and implementation-gap-review v4; torch-publication design v1 and implementation-gap-review v3; optional-adapter design v2; absent-resolution design v1
Owner Summary: Approve this exact finite deployment descriptor as appropriate for first supervised CUDA Go2 attempts with the existing driver, target1, ACK1000ms and operation120000ms. The union preserves the exact established permission relations and passes fresh production schema/file verification. This is neither native upload scope approval nor a claim that the combined CUDA Go2 execution has succeeded. Full runtime, ambient/fault/dual-GPU and aggregate custody obligations remain open.

VERDICT: Approve — bounded deployment descriptor only.

1. Independent identity and exact snapshot

I did not author or edit the candidate, inherit the implementation conversation, delegate this review, or execute any backend/GPU workload. Read AGENTS.md, collab/README.md and collab/REVIEW_PRINCIPLES.md fully; reread Chinese content with explicit UTF-8 after initial PowerShell decoding was garbled. Read the governing designs and bounded predecessor reports above, the captured-import review, captured-stream evidence, environment fixture documentation, builder/checker and relevant actual production code. Oversized output was narrowed before relying on operative sections. Prior verdicts were scope/history, not proof of the proposed union.

Repository: C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design
Frozen local HEAD: a30a761da355ca4cd884b202428e29d8f5418285
Production baseline, independently observed on primary tileops-dev: a3d750bbaf79d760a6727e56146d277e1cb3b770
Remote checkout: /data2/shared/ang.gao/robot_sim_c2d3_binding_wip
Remote interpreter: /home/ang.gao/miniforge3/envs/robot_sim/bin/python, CPython3.12.13
Local interpreter: C:/Python314/python.exe, CPython3.14.0

Both working trees were clean before and after checks. The local a3d750b-to-HEAD delta has only documentation/evidence and the two previously frozen CUDA fixtures; optical_worker code is unchanged. No tracked file, installed dependency, private context, guard budget, device setting, process signal, push/reset or GPU state was changed. Scratch reviewer script/report/logs are outside the candidate.

Review packet raw SHA256, independently matched:
6e3e7ef519e27381b281a3d5c200a3677d91d98109bef0c471ab54f3c6f90102

Reviewed candidate: C:/Users/AG/新建文件夹/c4_assets/183_cuda_go2_author_proposed_v1.json
Raw SHA256: 47f755f6dc229858a9bf0b59673010a6aeac30a92a9afe9fbd5487300e7e8f71
Canonical SHA256: 79c3580baff1ef3a019c97a07177558808918b7e133caa430239e25b82853919

Source descriptors, independently canonical-hash checked:
- 183_cuda_stream_author_v1.json: b4e47505a751377d48336beba5508cdbe2a2adcccdd10ae0e463e0767a537a47
- 183_warp_go2_author_v3.json: 88d295b90bc4047b2f2cd4c3d0e48c33ead1a1050919b47d526b4576159146e8

2. Findings and actual permission assessment

No substantive blocking finding or descriptor-specific required-evidence gap was established for the requested first-attempt scope. No non-blocking repair is requested. This does not approve an unexecuted combined path by subtraction.

Independent reconstruction, rather than execution of the author's builder, establishes the exact union. All 86 shared file-module rows, 135 shared file rows, one shared distribution and one shared absent-probe row agree in every field. Interpreter, roots, schema and settings agree exactly. The only removals from the Warp source are its 25 runtime file-module rows, all under warp; 31 site files under warp/ or warp_lang-; and the warp-lang distribution. No Warp runtime module or native library permission remains in the candidate. No phase, requester, creator, publisher, attribute path, namespace location or replacement rule is rewritten by union.

The resulting descriptor contains 1704 files, 1573 file-module rules, 23 distributions, 48 publications, seven replacements, three native globals, two namespaces, 35 absent-probe rows and two disabled optional modules. File-module phases are 900 materialization and 673 runtime. The 814 module rows added to the CUDA source all belong to materialization. All 1645 site-root file rows independently agree in size/hash with the original offline wheel inventory; the other 59 file rows retain exact interpreter-native/OS-native provenance from the source descriptors. Fresh production RuntimeEnvironmentContract.verify_files checked every actual file, interpreter identity, contained path, hash/size, distribution METADATA Name/Version and namespace directory. Inventory membership itself is not runtime permission.

The existing runtime_environment parser enforces exact schemas, sorted unique keys, file references, qualified parents, collision rules, fixed optional-adapter mappings and all-seven-or-none literal Torch replacements. The candidate introduces no wildcard/prefix rule, importer callback, owner invented from a name, creator cycle or discovery-to-admission mechanism. The trusted immutable-deployment/native-dependency boundary remains explicit; hashes do not prove native exception safety or sandbox an admitted binary.

3. Requester, publication, phase and interval interactions

Inspected production runtime_environment.py validation/verify_files and native_bootstrap.py finder, external loader, configuration/materialization/runtime sequence, requester resolution, publication reconciliation, native-global frames/checkpoints, final audit and the existing supervisor/child path.

The existing sequence completes canonical Go2 materialization and its audits before changing external admission from materialization to runtime. The CUDA source's 86 shared NumPy modules retain their materialization phase; the Go2-only NumPy/SciPy/Trimesh/Pillow/networkx closure is also materialization. Both historical actual Go2 ledgers independently contain precisely all 900 materialization modules, each with its complete five-step file lifecycle and a well-nested exec stack. The materialization code is common to the two lanes. This supports the proposed phase split; it is not an assertion that adding warmed dependencies cannot expose another interaction in the first combined run.

At find_spec the actual active external exec stack supplies the requester. Exact (name, requester, phase) absence keys precede fixed-root resolution; the sole finder preserves normal no-spec behavior without a tail finder. I reconstructed that stack for all historical absent events in both Go2 records and the stream record. Every row matches the candidate. Both Go2 records have 29 absence events representing all 28 distinct materialization rows, including repeated xxhash; the stream record has eight rows, with the NumPy probe shared. All 35 candidate probes independently resolved to no spec on the current remote filesystem without executing any delegate. Both disabled prerequisites also resolved absent. The two present adapter files remain inventory entries with no file-module execution permission, and the historical ledgers record two deliberate denials plus two requester-bound fallback confirmations, rather than reporting executed adapters or ordinary absence.

All six Go2 materialization publications and 42 Torch runtime publications have their declared actual publisher on the reconstructed exec stack, a created declared root and matching kind/phase. The six Go2 rows remain four native self-aliases with empty paths and two explicit _highspy._core attribute publications. The 42 Torch rows preserve the reviewed distinction between torch._C native publication, torch's recursive publication helper, later native initialization callers, and separately created ops/classes objects published by torch. Runtime publication audit validates the previous record's original phase, so materialization publications do not become new runtime publications. No fake file origin is assigned to a fileless publication. Seven replacement rows retain original file ownership and separate exposed-wrapper relations.

The three native-global rows preserve these precise materialization/exec intervals:
- cython_runtime: numpy.random._bounded_integers
- _cython_3_2_8: numpy.random.bit_generator
- _cython_3_2_5: scipy._lib._ccallback_c

Both actual historical Go2 records contain the same six pending/admitted events. The first cython_runtime observation occurs at a request for numpy.random.bit_generator while _bounded_integers is current; _cython_3_2_8 occurs at a request for numpy.random._common while bit_generator is current; _cython_3_2_5 occurs at a request for scipy._cyutility while _ccallback_c is current. Each admits at its own successful return, after its recorded absence checkpoint. The proposed rows name those observed intervals, not the child about to execute, a fictitious Python attribute owner or an exact native creator instruction. Later runtime checkpoints reuse the held identities; they cannot first admit an object outside its frozen own interval. Audit does not discover permission, and a changed first interval must fail closed and receive a reviewed table revision.

Historical records examined locally and remotely had identical raw hashes:
- c2d3-3646630-go2-960-guarded-author-v11/canonical_materialization.json: e111d7a358db1a922660a1a55c7a653f13ac723fb719982651935fce0f38ee7b
- c2d3-3646630-go2-1080-guarded-author-v12/canonical_materialization.json: 3990e3e93d685e28967c6c2fbb8d53d185964c7dcba01a3b3a076e88a539f751
- c2d3-a3d750b-captured-torch-stream-diagnostic-v3/runtime_initialization.json: cc49030f879519ab6d69910d28d42dc6253b18cd5f6085d6212fe45549870c21

Original stream native-call/replacement/alias components replayed unchanged against their original v5 source descriptor. Its ledger has 758 completed modules and one extra Triton resolution without create/exec, not 759 completed modules. The stream diagnostic's intentional pre-upload quarantine is not a successful runtime/upload result.

Historical Go2 used schema-v4 environment digest e1c1a6b02128f329f5639af0c41b8bc6317002fd6a8c33eab469bf289f948ece. I did NOT claim an unchanged v5 replay of that history. For an explicit compatibility test only, the reviewer script made in-memory copies of the two native observation components and relabeled their environment digest to the proposed v5 union, whose relevant exact module/global rows were independently compared. Current complete-component replay accepted each copy; a wrong-observer mutation rejected. These are compatibility tests, not newly issued evidence, authentic execution under the union digest or aggregate runtime-custody proof. Original artifacts were not edited.

4. Executed verification and limitations

Retained independent script: c4_assets/cuda_go2_profile_independent_v1.py

Local execution from the frozen repository:
python -I -S -B ../c4_assets/cuda_go2_profile_independent_v1.py
Result: exact union, source digests, inventory comparison and all three historical ledger checks passed. The script wrote no candidate and imported no backend. Fresh output: cuda_go2_profile_local_checks_v1.log.

Remote read-only execution:
Get-Content -Raw -Encoding UTF8 ../c4_assets/cuda_go2_profile_independent_v1.py | ssh tileops-dev '/home/ang.gao/miniforge3/envs/robot_sim/bin/python -I -S -B -'
Result: all checks above plus fresh 1704-file/23-distribution verification, 35 current no-spec resolutions, both disabled prerequisites and the explicitly distinguished component replays passed. The script asserted no torch/numpy/scipy/trimesh/warp import. Fresh output: cuda_go2_profile_remote_checks_v1.log.

Remote CPU regression from the exact clean production checkout:
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/ang.gao/miniforge3/envs/robot_sim/bin/python -B -m pytest tests/unit/optics/test_runtime_environment.py -q -p no:cacheprovider
Result: 52 passed in 1.58s, no skips. Fresh output: cuda_go2_profile_environment_tests_v1.log. These tests do not execute actual Go2 or GPU setup.

Other executed checks: git rev-parse HEAD; git status --porcelain/--short before and after; git diff --stat and diff --check against a3d750b; production-code diff; raw/canonical hashlib and Get-FileHash checks; bounded UTF-8 source reads and rg. One exploratory rg used nonexistent tests/helpers paths, then the actual tests/support paths were inspected. This was a reviewer path error, not a candidate failure. No failed exploration was counted as a passing check.

Remote reviewer process PID414130/parent414129 completed successfully. Final scoped ps for those PIDs plus the author's prior checker399635/399629 returned header only. Other short-lived git/test/SSH utility PIDs were not exhaustively recorded. No GPU query or execution was made and no GPU residual claim follows. No fallback host was needed.

The parent supplied subsequently frozen mechanical Go2 invocation scripts. I inspected, hash-checked, but did not execute them:
- run_c2d3_cuda_upload_author_v3.py: b6bf6741b2c41c864d904e13055687f38ddbbc603498878b5513573123b8f511
- run_c2d3_cuda_upload_author_v4.py: 60de9fba981c2f1007b9d80e66ac9961322bd8b612ce0a9a6efb1d8d61e823ae

They select the exact proposed digest, frozen real960/1080 case/asset paths, a3d750b, target1 and ACK1000/operation120000, then invoke the existing runtime_success supervisor/child. The actual child starts -I -S -B, sequences the production session's materialization/init/shared upload/readback and does not replace the importer or upload loop. The launcher's outer timeout does not renew an armed operation's deadline. This inspection does not extend this review to native upload scope; that separate review remained pending and was not used as approval. The parent's newly generated ordinary CPU reference claim was not needed for this descriptor verdict and was not independently rerun here.

5. Exact approval boundary and remaining gates

This exact descriptor is appropriate for first actual supervised CUDA Go2 attempts with the existing driver and the specified limits, once the separately required native upload scope prerequisite is satisfied. It is a finite launch candidate, not proof of successful combination. It must not be used to infer permission from a later observed module, silently relax a requester/phase/interval, waive UNKNOWN/foreign-device handling or import an unlisted dependency after a failed attempt. A necessary descriptor change needs a new frozen version and independent review.

No new actual captured materialization, Torch import, Stream, a2 allocation/upload/readback, normal terminal release, native side-effect/exception-safety or ambient restoration was demonstrated in this review. Full native loading/scope and future operation requirements are not established merely because all listed files exist. Potential new combined-order dependencies remain a runtime test obligation under the existing fail-closed guard.

Whole C2d3 still requires fresh actual CUDA Go2 evidence; both lanes' canonical asymmetric/dynamic/both real Go2 upload/storage/readback; all native import/allocation/copy/event/wait/registration/audit failure and original-deadline hang cases; real logical/raw ambient and dual-GPU target1/foreign/UNKNOWN histories; supervised terminal sync/release and terminal failure/exit/reap/residual checks; the final versioned aggregate runtime-custody record with cross-bound import/guard/storage ordering and forged-success negatives; required regressions and fresh whole-stage independent review. Historical 3646630 Warp evidence and a3d750b intentional Stream quarantine do not satisfy those new runs.

No final optical readiness, active sample lease, renderer execution, benchmark/result acceptance or whole C2d3/C2d/C4 completion follows. C2e/f, C4c3/c4, C4d/B3, product GPU preparation/activation, a3b, result/baseline, host payload, R2 and shipping remain closed. P12.4e cleanup obligations for losing-lane and temporary author fixtures remain unchanged. The report may be archived verbatim as evidence only; archival cannot broaden the verdict.
