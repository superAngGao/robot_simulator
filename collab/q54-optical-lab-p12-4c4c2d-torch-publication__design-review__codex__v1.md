Initiative: q54-optical-lab-p12-4c4c2d-torch-publication
Stage: design-review
Author: codex, independent session /root/torch_publication_design_v1
Version: v1
Date: 2026-10-08
Status: accepted for bounded design scope only
Related Files: torch-publication design-request/gap-scan/test-evidence v1; runtime-adoption design-request v2; native-global-publication design-request v1; generated-stdlib design-request v1; optical_worker/native_bootstrap.py; optical_worker/runtime_environment.py
Owner Summary: Approve the finite seven Torch file-module replacement relations and separate exact CPython __mp_main__ observed alias design. No substantive blocking design finding. This permits bounded implementation, not implementation acceptance, captured Torch success, a2 upload success, C2d3 acceptance or any later gate.

Verdict: Approve

1. Exact candidate and isolation

Fresh independent design reviewer, not the candidate author. Read AGENTS.md, collab/README.md and collab/REVIEW_PRINCIPLES.md fully. Reread the Chinese principles with explicit UTF-8 after the first PowerShell read displayed mojibake. Read candidate sections 1-7, the complete adjacent G1/G2/G3 findings and report limitations, adjacent evidence including v2, and governing runtime-adoption v2, native-global-publication v1 and generated-stdlib v1. The gap report was input evidence, not an approval.

Local repository: C:\Users\AG\新建文件夹\p12_4_c4_warp_timeline_design
Branch: implementation/c4-supervised-waits
Base: b88dbf1d5bf0a94951f3d0ecaa2bb4bdefd3ddc9
Reviewed HEAD: 292811101a48f5c43e519db782604111f1e4c548
HEAD tree: 2fde0bc43685fbaa41d6961e08d64a46506c85e8
Implementation parent: 8bc53bfe25acff9d384a3536d249c0e3268538d0
Candidate: collab/q54-optical-lab-p12-4c4c2d-torch-publication__design-request__codex__v1.md
Raw SHA-256: 2dace48ad6002db9cb6523a6852ef2ec64314130c59563762c17dd95371f7186

git status --short was empty on initial and final local checks; HEAD remained exact. The base-to-HEAD delta contains only three collab documents, 597 added lines. git diff --check passed. Remote primary tileops-dev was reachable; /data2/shared/ang.gao/robot_sim_c2d3_binding_wip reported exact code parent 8bc53bfe25acff9d384a3536d249c0e3268538d0 with empty status. No fallback host was needed.

No candidate, private instructions, installed package or remote repository was changed. The only persistent new files are this report and its CPU mechanics probe under local c4_assets. No Torch/NumPy/Warp import, GPU diagnostic, install, process kill/reset or supervised runtime matrix was executed by this reviewer.

2. Findings and design assessment

No substantive findings requiring repair. No unresolved blocking design or required design-stage evidence gap was identified. Approval is tied to the exact candidate digest above.

Finite permission and provenance: section 1 adds a versioned exact-key table referencing existing admitted Python file rows and the existing inventory. Seven-or-none membership, runtime phase, exact package flags, literal class/relation paths and collision exclusions prevent this from becoming a same-name or prefix exemption. The eventual CUDA dependency profile still needs its own complete freeze and review. VF's native value is correctly not required to be ModuleType; its identity is bound to the exact held torch._C root and raw _VariableFunctions value. No native ownership claim or extra permission follows from its type annotation.

Original versus exposed object: sections 2-3 preserve the execution original, spec, loader, delegate and actual own frame. This fits _ExternalSourceLoader.create_module/exec_module and the existing _check_native_call root binding, which would be broken by overwriting records[name][module]. The separate immutable replacement binding and mutable evidence projection are explicit. Class, bases/MRO, constructor/forwarder functions and defining globals are checked by identity, not name/hash. The source inspection confirms the exceptional direct ModuleType inheritance of QuantizedEngine and VF and the five PropModule-derived classes. The design does not misstate first observation as a witnessed constructor instruction.

Pending execution and topology: the common finder entry before branch dispatch and the own exec return are existing reachable boundaries. The current marker/frame topology distinguishes an executing delegate from wrapper validation and suspended ancestors. Sections 2-3 require that exact context and do not lend the parent's attribution to a child. A pending replacement parent can supply its original frozen package path without completing the parent. The original remains the global namespace for the rest of the body, including opt_einsum assignments after replacement. Admission is before original file completion but after the own successful return and required checks. Final audit cannot repair missing replacement observations, and sticky nested failure prevents later success.

Introspection and parent semantics: section 4 correctly separates raw file metadata from raw replacement metadata. Built-in module/type dictionary access avoids invoking the pinned forwarders or ContextProp descriptors for provenance. CPython's ordinary use of forwarding remains allowed. Child-to-parent assignment is deliberately checked after importlib has had the opportunity to perform it, rather than inside child exec-return. Freezing the source-defined from-import list also covers original backend globals, while raw wrapper bindings prove the exposed path. The actual pinned list is cpu, cuda, cudnn, cusparselt, kleidiai, mha, miopen, mkl, mkldnn, mps, nnpack, openmp, opt_einsum, quantized. This is a source fact to freeze in implementation, not permission to enumerate and admit children dynamically.

Alias permission/boundary consistency: section 5 is a separate observed identity relation within the already trusted interpreter/stdlib boundary. It is consistent with runtime-adoption v2's sealed trusted worker model and generated-stdlib's topology-based observation model. Cold main custody and cold absence, fixed standard resolution/source structure, exact completed publisher metadata and stable main/alias identity provide the stated observed relation. They do not prove a particular Python instruction created the alias, and the design explicitly refrains from that stronger claim. No native observer, file execution record, wildcard __mp_* classifier or process/spawn capability is minted. Deferring first observation until a supported boundary where the held publisher spec is no longer initializing is compatible with ordinary standard loaders; a general stdlib exec wrapper is not needed for this claim. The conditional audit bypass is scoped to the exact successfully checked alias. Within the stated trusted dependency model, this is sufficient and appropriately narrow; it is not a hostile same-process sandbox guarantee.

Failure custody: sections 3 and 6 require intake before fallible validation/adoption/serialization, including wrong values and the original objects and frames. The existing adopted admission/session graphs and fail() delta retention provide the integration points. This design requires extending those points; current unchanged code does not already implement the new custody. Original and replacement survive independently, pending cannot become admitted after poison, and terminal stdlib diagnostics retain without issuing success. Existing UNKNOWN/foreign-device quarantine, no retry/new ARM/COMPLETE/normal release, and native exception-safety limitations remain intact.

Evidence and validation: sections 6-7 preserve actual original file calls, separately describe pending/admitted replacement events and observation-only alias events, bind live identity before issuance, and require replay checks against import and guard history. Freezing final exact field layout in implementation is explicitly permitted by the governing runtime-adoption schema contract; it is still mandatory before code approval. Descriptive parsing grants no live authority. Both normal and adversarial production-loader tests are required, including early pending parent behavior, wrong identity/class/backlink/native value, copied mutable projections, fault custody and poison-time observation. The complete actual C2d3 matrix and aggregate schema remain required rather than being replaced by these component records.

3. Independent verification

Local commands included:
  Get-Content -Raw -Encoding UTF8 <governing and candidate documents>
  rg -n '^(class |def |    def )|stdlib|generated' optical_worker/native_bootstrap.py
  Get-Content -Encoding UTF8 <code> | Select-Object -Skip <offset> -First <count>
  git status --short
  git branch --show-current
  git rev-parse HEAD
  git rev-parse 'HEAD^{tree}'
  git diff --stat b88dbf1d5bf0a94951f3d0ecaa2bb4bdefd3ddc9 HEAD
  git diff --check b88dbf1d5bf0a94951f3d0ecaa2bb4bdefd3ddc9 HEAD
  Get-FileHash -Algorithm SHA256 <candidate and v2 census>

Inspected high-risk actual code: loader create/exec custody and execution globals; common finder dispatch; native frame/marker identity, entry/return/completion and immutable event bindings; failure retention; parent path resolution; publication reconciliation; module identity/metadata and final audit; generated-stdlib capture/resolution/audit; cold singleton topology installation; runtime environment row validation/collisions/versioning. This was design feasibility inspection, not a whole-file code approval.

Remote read-only source command:
  ssh tileops-dev 'cd /data2/shared/ang.gao/robot_sim_c2d3_binding_wip && git rev-parse HEAD && git status --short && /home/ang.gao/miniforge3/envs/robot_sim/bin/python -I -S -B /data2/shared/ang.gao/inspect_torch_replacements_c2d3_v1.py'
Exit 0. The inspector verifies inventory SHA-256 5e7e954d4c79ce323c9712d97c883d9311dc7182f9fb72891912ab007ab9099a and each installed replacement source before printing source excerpts. I read its script before execution; it does not import Torch. The seven source relations match the design. In particular backends replaces itself at line 125 before lines 127-142 imports, opt_einsum writes original globals at lines 118-119 after replacement, and VF raw vf is sourced from _C._VariableFunctions.

Retained historical v2 census independently hashed:
  c4_assets/c2d3-8bc53bf-torch-dependency-diagnostic-v2/ordinary_torch_dependency_observations.json
  SHA-256 0d17823a3b78c5fe87c4d307db9afc1775443c17b08661a383abc77e4ce8b6a1
Read its replacement_relations and the complete child probe source. Six exact original/class observations and VF native-value identity/non-ModuleType agree with the source. This is inspection of historical ordinary-import evidence, not a new run and not captured execution proof. The ordinary probe explicitly lacks the original VF object; future captured proof remains required.

Fresh CPU/stdlib-only mechanics probe, retained for reproducibility:
  C:\Users\AG\新建文件夹\c4_assets\torch_publication_design_probe_v1.py
  SHA-256 36037d66396d2c72ea6c4e2ea874ac68623594c6a5bcd4c4cef6b08681765e5a
Executed command:
  Get-Content -Raw -Encoding UTF8 C:/Users/AG/新建文件夹/c4_assets/torch_publication_design_probe_v1.py | ssh tileops-dev '/home/ang.gao/miniforge3/envs/robot_sim/bin/python -I -S -B -'
Exit 0. Asserted results:
  multiprocessing source SHA-256 a5a42976033c7d63ee2740acceef949a3582dcb0e0442845f9717e1be771c68b;
  alias initially absent, then exact __main__ identity, publisher spec _initializing false;
  AST is the exact conditional '__main__' in sys.modules and assignment of that object to __mp_main__;
  synthetic package child request sees the already replaced parent and exact forwarded original path;
  child exec-return still has no raw child binding on parent;
  parent return preserves original execution globals and has exact raw original and wrapper child bindings;
  replacement metadata has null loader/spec/package/doc and absent file/path/cache;
  ModuleType.__getattribute__(wrapper, '__dict__') bypasses an injected __getattribute__ override with zero trap calls;
  backend_modules [] at process completion.
The synthetic loader tests CPython mechanics only; it is not the future production-loader test or captured Torch proof.

Two exploratory/tool failures were resolved and are not candidate findings: an unquoted PowerShell HEAD^{tree} argument was mangled by the shell, then succeeded quoted; an exploratory attempt to assign a property to an existing class's __dict__ raised AttributeError before the proposed adversarial read. Thus that attempted descriptor mutation did not establish a bypass. No false finding was retained from it.

4. Limits, required implementation obligations and exact approved scope

No new implementation exists in this candidate. I did not run a production replacement/alias path, whole unit suite, Torch/GPU execution, fault/hang matrix, captured native map audit or C2d3 aggregate replay. The fresh CPU checks demonstrate interpreter semantics and pinned source compatibility, not guardian/device/resource success. Historical process exits, residual checks and ordinary diagnostics remain author evidence as labeled, not this reviewer's independent re-execution.

Approved: implement exactly the design's seven finite runtime file-module replacements with preserved originals, immutable exposed relations and pending-parent semantics, plus the separately scoped exact CPython observation-only __mp_main__ alias. Required schema freeze, policy/catalog repin, production-loader positive/adversarial tests, custody faults, regression tests and captured actual Torch demonstration are implementation obligations already stated by the candidate, not optional follow-ups. No unresolved non-blocking finding needs a separate repair disposition.

Still required for C2d3: both lanes, actual canonical asymmetric and dynamic replay cases, both real Go2 materializations, real guarded upload/readback, exact environment/native dependencies, failure/hang/ACK/EOF/terminal behavior, real dual-GPU scope/restoration and foreign/UNKNOWN cases, actual exit/reap and residual evidence, complete aggregate runtime-custody schema/replay, and fresh whole-stage independent code review. Design approval and CPU component success cannot replace these gates.

C2e/f, C4c3/c4, C4d/B3, product GPU preparation, a3b, result/baseline, host payload, R2 and shipping remain closed. No renderer, benchmark, process/spawn product capability, package fork or second upload/import implementation is authorized. P12.4e cleanup/selected-backend obligations remain unchanged. C4 is not complete.
