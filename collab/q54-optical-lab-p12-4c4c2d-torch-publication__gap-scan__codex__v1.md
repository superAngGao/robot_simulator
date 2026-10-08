Initiative: q54-optical-lab-p12-4c4c2d-torch-module-replacement
Stage: gap-scan
Author: codex, independent session /root/torch_replacement_gap_v1
Version: v1
Date: 2026-10-08
Status: completed bounded findings/evidence report; no implementation or acceptance verdict
Related Files: runtime-adoption design-request v2; native-global-publication design-request v1; generated-stdlib design-request v1; optical_worker/native_bootstrap.py; optical_worker/runtime_environment.py
Owner Summary: The pinned Torch sources require a distinct, finite file-module replacement relation with two retained identities. The evidence also establishes a separate CPython __mp_main__ alias prerequisite. Neither the existing native-global outcome nor a generic same-name replacement exemption honestly describes these facts. No candidate changes were made and no C2d3/GPU acceptance is claimed.

1. Revision, isolation and scope

Read AGENTS.md, collab/README.md and collab/REVIEW_PRINCIPLES.md completely. The Chinese principles were reread with explicit UTF-8 after the initial PowerShell default decoding produced mojibake. Read the governing runtime-adoption v2 and native-global-publication v1; inspected the relevant generated-stdlib design and actual implementation.

Actual local HEAD: b88dbf1d5bf0a94951f3d0ecaa2bb4bdefd3ddc9.
Actual implementation parent: 8bc53bfe25acff9d384a3536d249c0e3268538d0.
Local candidate: C:\Users\AG\新建文件夹\p12_4_c4_warp_timeline_design.
git status --short was empty before and after inspection. The parent-to-HEAD diff contains only PROGRESS.md and the archived native-global implementation gap review v4, 126 added lines; no implementation delta.
Actual remote HEAD: 8bc53bfe25acff9d384a3536d249c0e3268538d0; remote git status --short empty.
Remote candidate: /data2/shared/ang.gao/robot_sim_c2d3_binding_wip via tileops-dev.
Remote interpreter: /home/ang.gao/miniforge3/envs/robot_sim/bin/python.

This is a fresh independent bounded gap scan, not a review of all C2d3 code, a new supplement approval, or implementation authorization. Scratch output is outside the candidate. No candidate/private instruction changes, install, reset, process kill, GPU execution, Torch import rerun, or diagnostic census rerun was performed. The new executable checks used CPU/stdlib only with -I -S -B.

2. Checked evidence and its limits

The retained ordinary diagnostic census is byte-identical locally/remotely to SHA256 4f76956c9baf2ad319b145f46b9efb5e39cca1541b25f84cd1bf62f44325a7a6. I read its script, supervision wrapper, runner, census, diagnostic failure and lifecycle/supervisor artifact. It records worker 42034, supervisor 42027, parent 42024, one ARM, zero normal COMPLETE, zero emergency sync, release not_authorized, UNKNOWN scope, intentional RuntimeError and observed worker exit 74. The guardian retained a list containing exposed import roots and observations. The script sets UNKNOWN before importing Torch and deliberately hard-exits. Its raw torch.cuda._initialized flag is false; that flag is not proof of zero driver work, target-only scope or completed initialization. I did not independently rerun/reap these historical processes or perform GPU residual inspection; the parent packet reports their exit/reap. This report validates the retained diagnostic evidence, not a new execution.

The census records 956 newly observed modules and the seven replacement keys with no raw file/spec/package-path provenance, plus exact final ops/classes alias checks and 40 fixed raw native attribute paths. It does not retain/serialize the exact original-to-replacement/class relations or publication timing as proof. Those relations below are independently grounded in the pinned source and must still be demonstrated by the future captured path.

Independently reran inspect_torch_replacements_c2d3_v1.py remotely, with no Torch execution. It verified inventory SHA256 5e7e954d4c79ce323c9712d97c883d9311dc7182f9fb72891912ab007ab9099a and each of the seven installed source hashes before printing source excerpts. Local and remote inspector hash: f254835c9fd98f427739b6a6676bccb6a45bc4bd0b5a1702bc57808b99555e33.
Diagnostic child hash: 5f953ae46216dc69b5fe3012253c13cbe02a1014e7dd43d9f266080ceb43c55e.
Diagnostic supervisor hash: d455e7f3f21fc86bd19df62392cb2dc49a1c1b916368fcb4c789cd21f23aba81.
Diagnostic runner hash: ea62e04f39344fdf154ac95257d02331627159c13da911174d27caa04895b268.

The subsequently supplied c2d3-torch-frozen-dependency-audit-v1.json has checked local SHA256 fba561913dceecbb5986debdff9de42d768b64fee0ffdb17c2ea65bdbf100951. I inspected its provenance and sample linker evidence, not every source edge or linker result. It describes offline source/linker facts, not runtime loaded-map closure or admission permission. I do not independently certify its entire closure or convert its observed/possible imports into policy.

3. Finding G1 — P1 prerequisite: the current same-object file contract cannot represent these seven normal replacements

Locations: native_bootstrap.py:497-576 (create/exec custody), :647-674 (frame-to-file-module binding), :1495-1553 (identity and metadata), :1560-1580 (audit); runtime_environment.py:214-228 and :256-304 (file/publication/global row kinds and collisions); runtime-adoption v2 sections 3, 4 and 4.2.

The loader retains the initial module in records[name][module], executes that object and requires sys.modules[name] is that same object on return/audit. An isolated CPU probe against the actual production check_module method confirmed a same-metadata distinct-object substitution raises 'external sys.modules identity drift'. This rejection is correct under the present contract; supporting pinned Torch requires a reviewed addition, not removal of the identity check.

Pinned relations (class names here locate raw objects, never independently prove identity):

* torch._VF, torch/_VF.py:20-31: original raw VFModule is the exact replacement class; published = VFModule(__name__); published raw vf is torch._C._VariableFunctions. There is no m backlink and no reason to require vf to be a ModuleType merely because an annotation says so. Require actual vf identity to the raw attribute on the exact held native root. The original file module remains separately held for source/class provenance. Do not invent an original->replacement attribute or infer the native value's creator from its name.
* torch.backends, torch/backends/__init__.py:54-60,115-142: original raw GenericModule is the exact class, with pinned PropModule inheritance; published raw m is the exact original. Replacement happens at line125 before child imports at127-142.
* torch.backends.cudnn, source lines194-224: exact original raw CudnnModule; raw m is original; PropModule comes from the held torch.backends original definition.
* torch.backends.miopen, lines39-50: exact original raw MiopenModule; raw m is original; same PropModule origin.
* torch.backends.mkldnn, lines106-134: exact original raw MkldnnModule; raw m is original; same PropModule origin.
* torch.backends.opt_einsum, lines102-120: exact original raw OptEinsumModule; raw m is original; same PropModule origin. Source continues assigning original globals after replacement; replacing the execution globals would change behavior.
* torch.backends.quantized, lines49-63: exact original raw QuantizedEngine, directly derived from types.ModuleType, with its own raw m backlink and __getattr__. It does not inherit PropModule. Do not force all six backend wrappers into a false common base-class relationship.

All seven classes are defined by the original Python file body. This is not native-global publication, an extension self-alias, or a fileless child whose Python owner path already exports the replacement. Existing publications forbid collision with a file key; native_globals also forbid it and require a native extension observer. The native-global supplement explicitly preserves other ownership rules and cannot authorize these Python replacements.

Minimal required supplement:

a. Add one finite, versioned replacement relation referencing existing admitted Python file rows, not independent file provenance or wildcard Torch permission. Freeze exact names, phase, own exec interval, class location on the original, and one of the two actual relations: raw m->original, or raw vf->held native root's fixed _VariableFunctions attribute. State the exact class/base constraints, including QuantizedEngine's difference. Reuse the one inventory for source bytes. A row cannot simultaneously be an unrelated publication/global/namespace/absence/disabled rule.
b. Keep execution_original and published_replacement distinct in custody, frames and descriptive evidence. Original create/exec/spec/loader facts remain attached to the object actually created/executed; only one actual file body ran. The replacement receives an observed/admitted replacement event, never fabricated create_module/exec_module/file metadata. Do not overwrite records[name][module] in a way that changes the original bound execution frame or loses the original.
c. Bind the exact class object, original object, replacement object, native value/root when relevant, spec/delegate and exec frame by live identity. Require type(replacement) is the exact class from the held original, not a type-name, __module__, isinstance-only, matching dictionary or matching hash test. Freeze the observed relation so another object of that exact class with copied attributes fails after first observation. This proves an observed relation within trusted pinned source execution, not an uninstrumented constructor instruction.
d. Reject replacement of any unlisted file, pre-create/pre-exec replacement, early preloading, second replacement, disappearance/None, cross-module/session reuse, wrong phase and later identity/class/owner drift. An unused row grants no successful execution evidence; an actually executed required row must finish with its replacement.

Proof needed: synthetic production-loader positives for each relation plus actual captured Torch showing all seven, exact held originals/classes and final published objects. Negatives must include copying the replacement raw dictionary, a fresh same-class instance, a same-named different class/subclass, wrong m or vf object, copied original, changed native root, unlisted eighth replacement, and repeated/cross-session imports. No positive GPU acceptance follows from these CPU mechanics.

4. Finding G2 — P1 design obligation: pending parent replacement, raw metadata and failure ownership must be designed together

Locations: native_bootstrap.py:677-732 (entry/return frames), :1015-1043 (parent path), :1417-1493 (owner-path reconciliation), :1502-1553 (metadata), :977-1013 (failure retention). Trigger: torch.backends replaces itself before importing torch.backends.cpu and other children. A return-only rule either misses the live pending object at a nested boundary, breaks parent handling, or accidentally admits it too early.

Required boundaries/state:

* Before own exec require the exact original in sys.modules. On the existing common request-entry boundary, before suspending/starting a nested request, inspect the finite replacement keys and retain newly available values before fallible validation/adoption/event writing. Own successful exec return is the other first-observation boundary. First observation belongs to the exact currently executing Python source frame for that row; a matching ancestor, create frame, child frame or wrapper validation is insufficient. Do not repurpose the native-global observer enum to claim native execution.
* State must distinguish original/executing, observed replacement pending, admitted replacement and sticky failure. A pending replacement remains tied to its original live frame while child imports run; nested checks can reuse its identity and required parent paths but cannot complete the parent. Complete it only at its own normal return after nested failure checks and other required audits. At final audit no pending frame may remain, and audit cannot first discover/admit a replacement or repair erased evidence. Repeated audits are read-only.
* Preserve the current single-thread, actual-frame and wrapper-suspension protections. Ordinary cached imports may bypass the finder; supported return/segment boundaries must still catch observed drift. This is not instruction tracing, and transient unobserved mutations inside trusted code remain outside the claim.

Raw metadata/no-hook rule:

ModuleType(name) constructs replacement metadata different from the original. The census shows all seven replacements with no raw __file__ or __path__, null spec/origin and no recorded spec loader. It did not record raw __package__; its expected null value follows from source construction and must be checked in actual captured evidence. The file source, origin, loader, package flag and contained paths belong to the original. A backend's normal __getattr__ forwards to original m; VF forwards to a native object. An audit must not call those methods, ContextProp descriptors, arbitrary __getattribute__, dir, hasattr or getattr to obtain provenance or owner paths. Read raw module dictionaries through the known built-in mechanism and class dictionaries/MRO without invoking custom hooks; validate supported exact types/classes before traversing. Validate raw original metadata and raw replacement metadata as different explicit profiles. Do not copy spec/file/path onto the replacement to make it look like a loaded file module.

The six backend sources are packages even though the exposed wrappers lack raw __path__. The admission parent resolver must use the frozen original package descriptor and validate the actual incoming search path against it. Ordinary CPython import machinery may legitimately invoke the pinned package wrapper's forwarding behavior; the no-hook requirement is for audit/admission introspection, not a ban on normal dependency behavior. Validate the original path, spec search locations, wrapper m relationship and any unexpected raw shadowing that would change effective resolution. VF is a non-package and must not gain child-search permission.

Define parent attribute checks at a boundary where importlib has actually published them: its normal child-to-parent assignment happens after the child's exec wrapper returns. Requiring torch.backends.<child> or torch.backends on torch too early would be false; neglecting their eventual exact identity would miss a rebound import path. For the active backend original and wrapper, freeze which raw child bindings are expected once the from-import assignments complete. Require stable identity through subsequent relevant boundaries, not synthesized properties.

Failure ownership:

Retain original, class, all observed replacements including wrong/None values, native root/value, original spec/delegate, pending frame, nested owners and original/late exceptions before any validation that may fail. The already adopted admission graph must own new values before optional controller adoption/event emission. On swallowed inner error, later outer publications can enlarge failure custody only; they cannot turn pending into admitted or complete a normal segment. Current fail() retains available sys.modules deltas, but that alone cannot recover an original that an implementation discarded or a replacement lost before intake. Fault each retention/adoption/record/reconciliation boundary with sentinels to prove both identities survive importlib cleanup through terminal disposition.

No retry, second ARM, successful normal COMPLETE, target sync or ordinary release after the failed normal segment. UNKNOWN and native exception-safety obligations remain unchanged. Test failure just after publication and before nested import, nested exception swallowed by parent, return-audit failure, event/adoption failure, deleted registry/frame, and terminal stdlib reporting after poison. This is a prerequisite to implementing the relation, not a claim that the frozen candidate violated a promised replacement feature.

5. Finding G3 — P2 separate prerequisite actually established: trusted-stdlib __mp_main__ alias

Locations: native_bootstrap.py:184-188 (_is_interpreter_name), :1564-1578 (external audit), :2438-2445 (cold stdlib names); generated-stdlib design v1 sections1-3. Census row __mp_main__ has intrinsic __name__ '__main__', no spec and the diagnostic child's __file__. Its script path is not a third-party deployment origin.

Independent fresh CPU-only remote observation, no Torch/NumPy/Warp imported:
  '__mp_main__' in sys.stdlib_module_names -> False
  '__mp_main__' in sys.modules before importing multiprocessing -> False
  after import multiprocessing: sys.modules['__mp_main__'] is sys.modules['__main__'] -> True
  actual production _is_interpreter_name('__mp_main__', stdlib|builtin) -> False

Pinned interpreter source /home/ang.gao/miniforge3/envs/robot_sim/lib/python3.12/multiprocessing/__init__.py SHA256 a5a42976033c7d63ee2740acceef949a3582dcb0e0442845f9717e1be771c68b:
  36 if '__main__' in sys.modules:
  37     sys.modules['__mp_main__'] = sys.modules['__main__']
Inventory-verified Torch multiprocessing source SHA256 3b00cb1453bdb5c280091caecb4a9ecab95c2ca5b50a5a9335ec64b70958897c imports multiprocessing at line18 and from-imports it at29. The diagnostic census demonstrates the resulting name on the actual ordinary Torch path.

The current generated-stdlib exception covers one exact constructed _sysconfigdata name, not this alias. With an initially absent alias, current external audit treats its later appearance as unlisted: no stdlib classification and no external record. This is a distinct supported-path dependency prerequisite, not another Torch replacement. I did not execute the full captured Torch path and do not claim it reaches this rejection before earlier missing rules.

Direction: separately specify the exact trusted-interpreter alias to the already held bootstrap __main__ object, the permitted stdlib publication/observation boundary and absence/preload policy, later unchanged identity, and descriptive evidence. Do not admit arbitrary __mp_* prefixes, accept arbitrary values under this name, treat its __file__ as a site dependency, or invent a native observer/creator. If only observation is supported, say observation; source plus a final equality is not runtime proof of an instruction creator. Keep classifier and ledger/audit handling consistent while preserving all current file/generated-stdlib checks. The implementation choice for a reliable boundary must be settled in the supplement, not improvised by merely adding the string to stdlib names.

Proof needed: cold supported production import showing exact baseline __main__ alias, stable repeated use, wrong same-metadata object, preloaded/injected alias, original __main__ replacement, missing/removed alias after observation, and failure-path retention. Importing multiprocessing here created no Process and no backend work. This evidence grants no multiprocessing/spawn product capability.

6. Other implications that are supported, and claims that are not

The native-publication table still needs complete fixed names/paths/publishers from the pinned environment. The census is useful evidence for existing creator_attribute mechanisms, not permission to invent native origins from module names. Example: published torch._C._dynamo.autograd_compiler was observed at raw path ['_dynamo','compiled_autograd']; matching suffix strings would be wrong. ops/classes have checked final identities to their actual Python owners; their source owner and publisher remain distinct. These are existing table/completion obligations, not justification for another generic creator policy.

Source closure, possible AST imports and ldd output do not establish actual imported module closure for all canonical workloads, loaded-library maps, native exception safety, target scope, deferred CUDA initialization or successful custody. A false _initialized flag, exit74 and no normal COMPLETE cannot be relabeled GPU success. No evidence here establishes a need for a broad stdlib wrapper, general hook exemption, native interposer, new GPU synchronization policy or package modification.

The supplement must freeze descriptive replacement evidence with original/replacement/class/native-root tokens, own actual file-call reference, phase, observation checkpoint, pending/admitted ordering and exact rule/environment binding. Replayer rejects invented file execution for the replacement, token/object reuse, missing or duplicate completion, changed relation, completion after poison and audit-based late admission. Parsed evidence has no authority. Integrate it into the still-required final C2d3 runtime-custody schema; do not create a separate lifecycle authority or postpone that schema indefinitely.

7. Commands and verification record

Executed read-only local commands using powershell: Get-Content -Raw/-Encoding UTF8 for the governing documents and scripts; rg --files and rg -n for code locations; git status --short; git rev-parse HEAD; git log -3 --format=oneline; git diff --stat 8bc53bf b88dbf1; Get-FileHash -Algorithm SHA256 on the census/scripts/audit artifact; ConvertFrom-Json queries on the retained evidence. A search initially used nonexistent tests/fixtures/optics and was corrected by rg --files to tests/data/runtime_environments. No result was inferred from the failed search. One command printed an overly long one-line fixture and was truncated; conclusions use targeted code/evidence instead.

Executed remote via tileops-dev:
  git -C /data2/shared/ang.gao/robot_sim_c2d3_binding_wip status --short
  git -C /data2/shared/ang.gao/robot_sim_c2d3_binding_wip rev-parse HEAD
  /home/ang.gao/miniforge3/envs/robot_sim/bin/python -I -S -B /data2/shared/ang.gao/inspect_torch_replacements_c2d3_v1.py
All succeeded; inspector independently validated the seven source hashes before excerpts.

CPU alias probe was piped via a PowerShell literal here-string to:
  ssh tileops-dev '/home/ang.gao/miniforge3/envs/robot_sim/bin/python -I -S -B -'
It read/hashed multiprocessing/__init__.py, printed membership/absence, imported multiprocessing, checked exact __main__ identity and printed source lines36-37. A first attempt using nested -c quoting failed in bash syntax before Python execution; literal stdin avoided that shell issue.

A second stdin probe loaded native_bootstrap.py with importlib.util.spec_from_file_location under a scratch module name (no _install call), checked the production classifier, and constructed two ordinary ModuleType objects with equal raw metadata. With the original in a synthetic _ExternalAdmission.records entry and replacement under the same sys.modules key, actual check_module rejected before metadata processing. It removed that scratch key afterward. It also checked all supplied remote hashes, inventory-verified torch/multiprocessing/__init__.py, and printed backend_modules [] at completion. This direct CPU probe demonstrates existing local rejection/classification only, not an actual captured-session or guardian execution.

No full unit suite was rerun: no candidate implementation changed, and the scan targeted the contract gap. No new real Torch import, captured admission, GPU matrix, process-hang/fault matrix, native-map audit or C2d3 aggregate-schema validation was performed. Required future validation includes actual captured cold Torch with these relations, both frozen Go2 inputs, both lanes, canonical synthetic/dynamic cases, dual-device/scope cases, failure/hang and terminal disposal cases, and fresh whole-stage independent review.

8. Scope disposition

Findings G1/G2 require a narrowly reviewed replacement supplement before implementation; G3 requires an explicitly distinct stdlib-alias decision. The existing candidate correctly fails closed for unsupported replacement; this scan does not propose weakening that behavior in place. No implementation or design approval is issued by this report. Native-global v1 remains bounded to its approved outcome. All later gates remain closed: C2e/f, C4c3/c4, C4d/B3, product GPU preparation, a3b, result/baseline, host payload, R2 and shipping. C2d3 still requires its actual matrix, complete schema and whole independent code review. C4 overall work continues.

9. Late supplementary evidence inspected before final delivery

The author supplied ordinary diagnostic v2 while this report was being finalized. I checked the local census hash 0d17823a3b78c5fe87c4d307db9afc1775443c17b08661a383abc77e4ce8b6a1 at c4_assets/c2d3-8bc53bf-torch-dependency-diagnostic-v2/ordinary_torch_dependency_observations.json, read its replacement_relations, and inspected the corresponding child script's raw identity observations and lifecycle/supervisor summaries. I did not execute this diagnostic or independently check its remote copy/process residuals.

This adds actual ordinary final observations: for each of the six backend wrappers, raw m points to a distinct object with the expected original source/name and type(exposed) is the exact original raw class attribute. Their recorded raw __package__ values are null. For VF, raw vf is the exact held native root's _VariableFunctions value; isinstance(value, ModuleType) is false, and its reported type is torch._C._VariableFunctionsClass. The VF original file module is still not captured by this diagnostic; the script explicitly says so. This confirms the G1 warning against treating the vf value as a module and strengthens the six backend relation evidence; it does not establish the captured-session pending/return sequence or recover VF original/class provenance.

V2 also records 81 .so-like pathnames from one /proc/self/maps snapshot. That is an observed ordinary-process snapshot, not a complete captured lifetime native map audit, permission inventory or target-only scope proof. The retained summaries again show one ARM, no normal COMPLETE/sync/release, UNKNOWN, exit74, no deadline expiry or kill. The parent reports parent58788/supervisor58792/worker58794 exit/reap and scoped residual checks; those remain author observations, not checks performed by this reviewer. Zero tracked device leaves in either diagnostic means no device-leaf instrumentation was claimed; it does not prove no native device work. Findings, required follow-up and closed gates are unchanged.
