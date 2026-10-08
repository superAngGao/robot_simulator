Initiative: q54-optical-lab-p12-4c4c2d-torch-captured-import
Stage: independent deployment / code / evidence gap review
Author: codex, fresh independent session /root/torch_cold_profile_gap_v1
Version: v1
Date: 2026-10-08
Status: accepted for the bounded cold-import candidate and evidence scope only
Related Files: runtime-adoption design-request/design-review v2; torch-publication design-request/design-review v1 and implementation-gap-review v3; torch-native-scope gap-review v1; cuda-ambient design-request/design-review v1; captured-import test-evidence v1; runtime environment fixture README; PROGRESS.md
Owner Summary: Approve this exact cold-import deployment/evidence delta. Independent CPU checks substantiate the completed 758-file prefix, 42 exact publisher-context relationships, eight exact absent requesters, seven replacement relations and interpreter alias. The actual diagnostic ends in intentional UNKNOWN quarantine before Stream. Neither complete Torch implementation nor C2d3/runtime/upload acceptance follows.

VERDICT: Approve — bounded cold-import profile and truthful captured-prefix evidence only.

1. Identity and isolation

Repository: C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design
Branch: implementation/c4-supervised-waits
Review base: 1dc27d79485a80560c9a057b3a330868c584dc4f
Actual reviewed HEAD: e3e6d791cb30a9c2677b850f33c56f51971ccebf
HEAD tree: 4cd7d865435207c759a3045c8b88c4d280c68f4a
Production implementation: a3d750bbaf79d760a6727e56146d277e1cb3b770
Candidate: tests/data/runtime_environments/183_cuda_cold_import_author_v1.json
Canonical environment SHA256: 2ffb9a81595b60855db8768f2892409c49d32d8ec97d41a3673b18e456492e07
Raw fixture SHA256: 4d77db05c845d150c5f49b7dd888f55b221e080c1f635c6b1d0bc8d813a7b74b

Initial and final local status --short were empty and HEAD remained exact. Base-to-HEAD changes exactly four files: this fixture, its README, the captured-import evidence note, and PROGRESS.md (171 added lines). git diff --check passed. The complete a3d750b-to-HEAD diff contains only these changes and archived reviews/ambient design documents: no production, tests, policy/catalog, watchdog budget or product activation change.

Primary tileops-dev was reachable. Independently observed /data2/shared/ang.gao/robot_sim_c2d3_binding_wip clean, detached, at exact a3d750b before and after verification. Pinned executable /home/ang.gao/miniforge3/envs/robot_sim/bin/python reports CPython 3.12.13 and resolves to the fixture's python3.12 file. No fallback, synchronization/reset, installed-package change, candidate edit, private-context edit, GPU operation or process signal was performed. Reviewer outputs are only in sibling c4_assets. No delegation occurred.

Read AGENTS.md, collab/README.md and collab/REVIEW_PRINCIPLES.md fully; reread the Chinese principles with explicit UTF-8 after PowerShell's initial decoding was garbled. Read the governing designs and bounded approval scopes listed above. Prior author conclusions and approvals were requirements/history, not proof of this candidate's behavior. The parent subsequently identified a new deferred-NVML scratch audit; I did not use it to extend this review or cure the explicitly missing full-init closure.

2. Findings and assessment

No substantive blocking finding or missing evidence requirement for this precisely bounded cold-import profile/evidence delta was established. No new non-blocking repair comment is needed. This conclusion does not approve later omissions by subtraction.

Deployment provenance and finite authoring: independently byte-compared the tracked fixture with external draft v7. Inspected all seven builder scripts and compared adjacent draft payloads. v2 adds dill; v3 adds torch._rocm_init; v4 corrects sixteen native publisher rows; v5 corrects the two ops/classes publishers; v6 adds monkeytype; v7 adds tqdm/pydot. No executable worker path learns permissions from a trace. Builders are offline tools, frozen drafts remain separate, and each finite revision still requires review. The fixture has no Warp module, namespace wildcard, disabled optional module, or speculative native-global permission. It does not select a product launch.

Fresh remote RuntimeEnvironmentContract parsing and verify_files checked all 857 contained files, sizes/digests, interpreter identity and 18 distribution METADATA identities. Independently compared all 804 site-root file rows to the original wheel inventory and all 81 mapped-file audit rows to the candidate. These are actual filesystem checks and provenance cross-checks, not an inference that a supplied checksum creates execution authority. The declaration is sufficient for the examined cold prefix; it is explicitly incomplete for deferred NVML and full CUDA/native loading. The retained ordinary mapped-file census is not an independently observed captured-process map and is not promoted to one.

Creator, publisher and path distinctions: read production reconcile_publications and its placement at successful own exec return, before file completion. It requires the executing declared publisher, exact held creator root, fixed raw attribute traversal and object identity against sys.modules; final audit rechecks the relationship and retained owners. Independently reconstructed the execution stack from all actual external events: all 42 publication events occur in their declared publisher's active exec interval, after creator creation. There are 24 _C-published rows, twelve native descendants published by torch's recursive helper, four later native initializer publishers, and two root-published aliases. Fresh installed-source reads confirm torch's helper at 1061/1076, the four Python callers of _autograd_init/_c10d_init/_dist_autograd_init/_rpc_init, and root setdefault publications at 2271/2272. ops/classes originate in their separately pinned creator files; their synthetic __file__ values are not file provenance. torch._C._dynamo.autograd_compiler correctly uses attributes [_dynamo, compiled_autograd], not the key's final spelling. The executed identity checks plus audited source establish the bounded relations; the census alone does not.

Absent requester provenance: independently reconstructed the active file-exec stack at every absent_probe and matched all eight exact name/requester/phase rows. There is one NumPy materialization probe and seven runtime probes. Fresh PathFinder resolution only within the declared exact roots/parent package path found no matching spec for all eight. Fresh installed-source excerpts show the optional branches for _distributor_init_local, _rocm_init, pynvml, opt_einsum, monkeytype, tqdm and pydot. dill is requested through cached dill_available/_check_module_exists during common.py's DILL_AVAILABLE assignment, so the actual importer is torch.utils.data.datapipes.utils.common, not the helper-definition module. Production uses the active import stack and rejects an unexpected requester/found module before delegation. No create/exec event exists for these absent names. TYPE_CHECKING imports, ROCm amdsmi, fallback _dummy_thread and monkeytype descendants receive no added permission. The raw event schema does not itself carry a requester field; the source/execution-stack cross-check here substantiates the claim without pretending the final aggregate already exists.

Original/exposed replacement and alias evidence: inspected original file ownership and native call frames, immutable replacement bindings, raw class/MRO/backlink/native-value checks, pending/admitted transitions, parent path resolution and final original/exposed parent-child checks. Actual component data has fourteen ordered events for all seven relations. VF's original/replacement/class/native-root/native-value tokens are distinct; native value is not asserted to be ModuleType. torch.backends first becomes pending on the request for torch.backends.cpu and admits last after child returns. Fresh pinned source agrees: backends replaces at line125 before the from-import list, opt_einsum writes original globals after replacement, and VF retains _C._VariableFunctions. The exact multiprocessing alias is observed once, separately from external file/publication/replacement events. Its pinned stdlib source digest is a5a42976033c7d63ee2740acceef949a3582dcb0e0442845f9717e1be771c68b; the actual source has the exact conditional assignment to cold __main__. This remains an observation within the trusted interpreter boundary, not a witnessed constructor or process/spawn authorization.

Actual diagnostic and termination: read the entire child, supervisor substitution wrapper and v6 launcher. The child SHA256 independently matches locally and remotely: 327fae1e3f0ec722371b993345f9ef9cdf18afea87e21e52735dde3bb2abb638. Production SourceSession initialization performs the real import, adopts _C and the indexed CPU device descriptor, then reaches the patched begin_device_leaf checkpoint before the original indexed_runtime_stream begin or Stream call. The fixed hook checks raw _initialized=false, calls live external.audit(), emits observations, marks UNKNOWN and raises. It neither clears UNKNOWN nor bypasses the production running-state check. Binding/upload/readback code is unreachable in this run. The wrapper substitutes only the fixed child path; supervision and captured session are the existing implementations.

The actual external ledger has 3840 events: 758 each file_resolved/create_started/created/exec_started/completed, 42 creator_publication and eight absent_probe. My independent state-machine walk verifies each file's five-step order, nested exec stack, exact phase, complete declared file set and exact publication set. Existing component validators pass with require_complete=True. This means complete import/replacement components at the checkpoint; it does not mean successful runtime initialization.

Supervisor and worker artifact copies are equal across raw files. Read transition traces as well as summaries: normal COMPLETE occurs for sequences0-2 only; sequence3 arms runtime initialization, enters indexed_runtime_import, then sets UNKNOWN and fails. The two entered runtime_import_or_init leaves are canonical_dependency_import and indexed_runtime_import; no Stream leaf enters. Observed final facts are four ARM, three normal COMPLETE, worker return74, supervisor EXITED/process exit observation, no kill/deadline expiry, UNKNOWN, dirty upper bound [1], full quarantine, zero emergency sync/terminal ARM, release not_authorized. The checkpoint reports no initialized CUDA and sample_submission_count0. No ordinary_atexit artifact exists. The child hard exit and production guard, rather than a parsed summary or destructor, govern termination.

Historical failed attempts remain failures. Independently read v2-v6 failure artifacts: unlisted import, missing creator publication, unlisted import, unlisted import, then the intentional pre-Stream RuntimeError. They are not accumulated as successful runtime runs. The note, fixture README and PROGRESS accurately preserve the narrow scope, prior bounded F1-F4 approval, ambient logical/raw distinction, missing NVML/full-init closure and closed gates.

3. Independent verification, commands and results

Local commands: Get-Content -Encoding UTF8, rg/rg --files, git status --short, git branch --show-current, git rev-parse HEAD, git rev-parse 'HEAD^{tree}', git diff --stat and git diff --check against the exact base; complete adjacent draft comparisons and raw JSON/hash checks using C:/Python314/python.exe -I -S -B -. Local Python is CPython3.14; the Linux contract was intentionally parsed on the pinned POSIX interpreter instead of patching production path rules.

Retained independent script:
C:/Users/AG/新建文件夹/c4_assets/torch_cold_profile_independent_v1.py

Executed from local repository:
Get-Content -Raw -Encoding UTF8 ../c4_assets/torch_cold_profile_independent_v1.py | ssh tileops-dev '/home/ang.gao/miniforge3/envs/robot_sim/bin/python -I -S -B -'

Result: completed successfully; fresh 857-file/18-distribution verification; complete 758-file ordering, 42 publisher-context checks, eight exact requester/absence checks; all native-global/replacement/alias/guarded component validators pass. Four independently mutated copies reject: missing final replacement admission, replacement token equal to original, parent admission reassigned to a child's call, and missing required alias observation. The script asserts no torch/numpy/warp module was imported. Remote verification PID301962/PPID301945.

Executed fresh remote CPU regression:
ssh tileops-dev 'cd /data2/shared/ang.gao/robot_sim_c2d3_binding_wip && /home/ang.gao/miniforge3/envs/robot_sim/bin/python -B -m pytest tests/unit/optics/test_runtime_environment.py -q -p no:cacheprovider'
Result: 52 passed in1.58s, no skips. No unrelated whole-suite/GPU rerun was needed for this data/document delta. These unit tests are not claimed to execute the new actual cold import.

Retained logs in c4_assets:
- torch_cold_profile_independent_v1.log — fresh deployment, replay, exact absent/source checks.
- torch_cold_profile_environment_tests_v1.log — fresh52-test output.
- torch_cold_profile_local_checks_v1.log — source/raw-artifact digest checks, draft diffs, inventory/map agreement and actual lifecycle trace summaries.
- torch_cold_profile_final_source_v1.log — fresh installed publisher/replacement/stdlib source excerpts, exact child digest, final clean detached remote revision and scoped PID check.

All eleven JSON hashes printed in the candidate note independently match their local raw artifacts; the five final raw artifacts also match fresh remote file hashes. The six source-evidence digests are inventory5e7e954d..., frozen-dependency fba561913..., mapped-files93f03713..., dill80a77feb..., publication4b28c1de..., eager-optional2548c01c...; their full values remain in the unchanged reviewed note. Checks used those actual files rather than trusting the note's verdict. The canonical/raw fixture digests above distinguish canonical encoding from the newline-terminated byte copy.

One exploratory local JSON inspection assumed sources was a dictionary and raised AttributeError; a corrected inspection handled the list. Some oversized source/document output was truncated; the operative designs, source excerpts and high-risk code were subsequently read in bounded chunks. These were review-tool issues, not candidate failures. No failed exploration is counted as a passing check.

Final scoped ps for268306/268307/268312/268314 and reviewer301962/301945 returned only its header. This corroborates current absence of the named processes, not a new replay of historical wait/reap. No current NVML query was made; the note's original MPS-only GPU residual assertion remains historical author evidence. No new GPU workload or sacrificial worker was launched by this review, so no independent new-device-run residual claim is made.

4. Exact approval boundary and remaining obligations

Approve the exact fixture and four-file delta at e3e6d791cb30a9c2677b850f33c56f51971ccebf as a finite cold-import deployment candidate and an accurately described actual captured pre-Stream prefix under a3d750b. This fulfills the bounded evidence claim that captured Torch reached that checkpoint with the specified relations. It is not a whole implementation review, native sandbox guarantee, proof of transient native side-effect absence, successful CUDA initialization, binding, a2/upload/readback, ordinary release or complete Torch supplement/C2d3 approval.

No reviewer rerun of actual GPU/captured import occurred. The real-run facts above are inspected historical artifacts tied to the checked exact driver/code/profile and source behavior, with fresh independent CPU/OS/file/replay checks. Native exception safety and installed native dependency trust remain the accepted obligations; file hashing or snapshots cannot prove them. Full deferred-native audit, exact native-loading closure including NVML, allocator configuration, real captured pristine logical/raw ambient observations and nested device/stream regression remain required. The ambient design's logical0 restoration must never be reported as raw CUDA0 restoration.

Still mandatory: both-lane actual canonical asymmetric and dynamic replay, both real Go2 inputs, guarded allocation/upload/readback/storage proof, failure/hang/ACK/EOF/foreign/UNKNOWN and dual-GPU matrix, supervised terminal sync/release and failure/hang cases, actual exit/reap/residual evidence, the versioned final runtime-custody aggregate with cross-bound ordering/negative replay, and fresh whole-stage independent review. Component validators and this independent stack reconstruction do not replace that aggregate or grant live authority.

C2e/f, C4c3/c4, C4d/B3, product GPU preparation, a3b, result/baseline, host payload, R2 and shipping remain closed. C4 remains active. The unchanged report may be archived under the evidence-only exception without changing its verdict or approval scope.
