# Independent review: installed setuptools admission gap

Initiative: Q54 / P12.4 / C4c2 / CUDA G1 compilation
Stage: independent source-only gap/design assessment
Author: codex independent reviewer /root/setuptools_admission_gap_v1
Version: v1
Date: 2026-10-09
Status: accepted
Related Files: setuptools-admission gap-scan v1; compilation-admission design-request v3; systematic-refactor implementation-note v1; optical_worker/native_bootstrap.py; optical_worker/runtime_environment.py
Owner Summary: Existing admission cannot faithfully admit the pinned setuptools package through added module rows. The scan correctly identifies a policy gap. Three mandatory import blockers are frozen topology/vendor resolution, same-object package alias custody, and an additional package-resource read unsupported by the current loader. Draft one narrowly scoped setuptools initialization transaction before implementation. Metadata version discovery requires an explicit bounded choice, but no general distribution/entry-point platform is justified.

Verdict: **Approve with comments ? gap assessment only.** The finding that a bounded design amendment is required is supported. This does not approve an adapter, implementation, native execution, or any higher gate. The additional resource requirement below must be included in the next design packet; approval here is not closure of that implementation blocker.

## Revision and independence

- Fresh review session `/root/setuptools_admission_gap_v1`; no implementation conversation inherited and no candidate authorship.
- Repository: `C:/Users/AG/?????/p12_4_c4_warp_timeline_design`.
- Requested and actual base: `6f253096537711165aca26345233ea831e0662aa`.
- Requested and actual HEAD: `758950c93d4d8aa0c3666098fbd36ea7e29dc31f`.
- Actual HEAD tree: `8db538f96da26d499ff66fe7d3feab8901345223`.
- Working tree clean at initial and final inspection. Base-to-HEAD diff: exactly the 100-line gap-scan document; no code delta.
- Approved compilation design v3 SHA256 independently matches `028b52c466ef46433235d4ec2495cd47d00200822cc64dfaea95614fd5bf26a2`.
- Read AGENTS.md, collab/README.md, collab/REVIEW_PRINCIPLES.md, the task packet, compilation design v3 and whole-C4 refactor direction. The root-level REVIEW_PRINCIPLES.md mentioned in the handoff does not exist; the authoritative file is under collab as AGENTS.md specifies.

## Evidence inspected

Archive `../c4_assets/cuda-compilation-helper-sources-v1.zip`: 336 entries; SHA256 independently matches `8f5a7f25d4e7654cea2e157d44af708cc14b912edcdedbcc68e7852dcffdaac3`. Inspected installed setuptools initialization, _distutils_hack, _distutils, monkey patching, vendored jaraco/backports imports, package metadata and pinned Torch cpp_extension entry/import path.

Additional author-supplied facts `cuda-compilation-setuptools-conda-facts-v1.json` independently hash to `45203dcfcd97a7b925b8c72061b71b38187328e7205e08ced3f28c98a097f403`. Archived installed Conda record independently hashes to `7f0b65f43d8dc9d7392e6a57b986b33d010da754c679ea82db83cd441fbbd118`. All 441 non-bytecode fact rows match the corresponding recorded digest field; their union with the 319 explicit omissions matches the record's paths. Independently recomputed size/digest for the 326 fact files present in this archive. Unarchived remote installed bytes were not reread; the remainder is record/fact consistency, not a new installed-file observation. Neither record nor inventory grants import permission.

## Findings and precise gaps

### G1 ? High, blocking admission: topology and vendor lookup are incompatible with the current contract

Locations: archive `setuptools/__init__.py:17-21`; `_distutils_hack/__init__.py:63-77,100-142,218-234`; production `native_bootstrap.py:448-466,666-673,1439-1539,3152-3177,3410-3418,3509-3577`.

Pinned `torch/utils/cpp_extension.py:10,29` imports setuptools and setuptools.command.build_ext, so initialization is mandatory even though the chosen build uses Ninja/load_inline. Setuptools appends its `_vendor` directory to sys.path before importing `_distutils_hack.override`. The override normally selects local distutils and temporarily inserts the dependency finder at index zero. The production session requires the exact singleton chain, finder identity and frozen sys.path at each boundary. The ordinary next import after the append therefore rejects; independently testing the finder insertion also rejects.

Adding a vendor directory to environment inventory is insufficient by itself: `_ExternalAdmission` searches its own ordered site roots and package-parent paths, not arbitrary newly appended sys.path. Top-level `jaraco`, `more_itertools`, `backports`, etc. need explicit resolution/origin rules and precedence. `jaraco` is a namespace in this archive; `backports` is a file package whose body calls pkgutil.extend_path, not a namespace-only substitute. Existing namespace/file machinery may cover unchanged finite path results, but the actual result must be frozen and checked.

Allowing a dependency finder to sit ahead of custody and checking topology only when the main finder happens to run is not a repair. That earlier finder can return a loader without the main finder observing the request; even its nested imports execute while the leading foreign resolver exists. Existing checks correctly reject this state. Required proof after amendment: each request remains under the source session's authority before any dependency spec/loader execution; arbitrary insertions, vendor shadows, extra namespace locations and reordered/duplicated path entries fail closed.

### G2 ? High, blocking admission: this is a Python package alias with metadata transition and child resolution

Locations: archive `_distutils_hack/__init__.py:111-142`; production `runtime_environment.py:124-134,269-300`; `native_bootstrap.py:1439-1454,1830-1985`.

The dependency's finder first imports `setuptools._distutils`, then returns a custom loader whose create_module changes that SAME object's __name__ to `distutils`. It does not create a second independent source package. The alias is published by importlib, and child imports under `distutils.*` read the vendored directory.

An independent exact-source-node probe using stdlib importlib and a stand-in source package observed: the source object's __name__ becomes `distutils`; __spec__ becomes the alias spec; existing __loader__, __package__ (`setuptools._distutils`) and __path__ survive. The generated alias spec has no submodule_search_locations despite the returned object's existing package path. This is a Python 3.14 probe, not a target-3.12 execution result; the exact target metadata matrix must be proved before freezing implementation.

The production intrinsic-name check rejects the changed source object. Merely adding a publication does not solve this: `_paths('distutils.core')` requires a file-package/namespace rule for its parent, and a publication cannot collide with such a module rule. A fabricated creator-self alias is expressly prohibited for Python packages. An attribute-path publication likewise provides no alias child resolver, cannot excuse the source metadata transition, and is observed too late if the child is imported before publisher completion. No existing approved native-global, Torch module-replacement, or __mp_main__ mechanism supplies this relation.

Required design: one exact two-name/one-object root relationship, actual producer/loader/spec identities, bounded metadata transitions and timing, and a finite child module map. Do not infer that every `distutils.x` aliases `setuptools._distutils.x`; they can be separate file executions under different names, and the source expressly asserts `setuptools._distutils.log` is absent. Prove actual identity and requested-name semantics rather than inventing a subtree alias by prefix.

Required proof: same-object positive under both root publications; source-name/spec drift outside the transition rejected; copied/foreign alias object, swapped spec/loader, early/repeated alias, unlisted child, changed package path and duplicated source execution rejected before successful admission. Retain the returned alias object/spec/loader even when validation or adoption fails.

### G3 ? High, additional mandatory import blocker: package data reads are not supported by the source wrapper

Locations: archive `setuptools/dist.py:20-25`, `setuptools/_reqs.py:7`, `setuptools/_entry_points.py:5-6`, `setuptools/_vendor/jaraco/text/__init__.py:232-234`; production `_ExternalSourceLoader`, `native_bootstrap.py:512-565`.

Both imported setuptools helpers lead to top-level `jaraco.text`. Its module body unconditionally calls `importlib.resources.files(__name__).joinpath('Lorem ipsum.txt').read_text(encoding='utf-8')`. The existing wrapper only reads the exact Python source through get_data and provides no get_resource_reader. In a stdlib CPU probe, a standard SourceFileLoader read a temporary positive-control resource, while the exact production wrapper class for the same stand-in module failed with FileNotFoundError: `Can't open orphan path`.

The source-only ZIP does not contain that text file; this is consistent with its stated collection scope, but means the ZIP is not the complete runtime input closure. The supplemental Conda facts list `setuptools/_vendor/jaraco/text/Lorem ipsum.txt`, 1335 bytes, SHA256 `37fedcffbf73c4eb9f058f47677cb33203a436ff9390e4d38a8e01c9dad28e0b` (relative to site-packages). Those resource bytes still need direct collection/checking before code preflight.

Adding only module rows or topology/alias exceptions still cannot complete this real import path. Required direction: a source-session-owned, read-only resource view bound to exact admitted package, inventoried relative name and verified bytes. Reuse file containment/hash ownership; do not delegate arbitrary filesystem reads or implement an unrestricted packaging resource service. Prove the actual jaraco expression on target Python, changed/missing resource rejection, traversal/symlink/outside-root rejection and no capability expansion through copied reader objects. No claim of complete resource closure is made here.

### G4 ? Medium, bounded semantic decision: metadata inventory is distinct from metadata discovery

Locations: archive `setuptools/_importlib.py:1`, `setuptools/version.py:1-6`; production `_CapturedSourceFinder` at `native_bootstrap.py:430-510`; `runtime_environment.py:212-222,788-795`; `tools/optical_runtime_inventory.py:77-104`.

The wheel scanner intentionally requires *.dist-info/METADATA plus RECORD and cannot collect this Conda egg-info layout. Runtime environment validation already accepts an inventoried metadata file and verifies Name/Version headers, so PKG-INFO needs no format exemption to runtime admission. Extend/reuse the offline collector's format-neutral file checks if needed; do not reinterpret scanner success as authority.

Separately, the singleton finder implements find_spec but no find_distributions. Restoring a sys.path entry alone therefore does not restore normal importlib.metadata discovery. Independent stdlib CPU positive control with the exact PKG-INFO bytes returned 83.0.0 under normal finders and PackageNotFoundError under a sole finder lacking that method. The actual source catches version lookup failure and reports `0.dev0+unknown`; this is silent semantic degradation, not by itself a proven compilation failure.

Do not use this as justification for a general metadata/entry-point facility. The only unconditional metadata-distribution query established by this review is version('setuptools'). `setuptools.dist` has queries in Distribution construction/finalization/command lookup (lines 315,786,807,850,858,865) and in the cached `_setuptools_commands` function (1110-1119); their presence in source does not prove the fixed load_inline recipes execute them. The chosen Torch path imports BuildExtension/build_ext classes, but is not a general setuptools.setup build. Likewise the cpp_extension intel-sycl-rt query is conditional on a SYCL branch excluded here.

For faithful installed identity, the smallest supported direction is a named setuptools metadata lookup backed by its pinned PKG-INFO, integrated with the sole custodian. Freeze its exact request/response scope after target-stdlib inspection; do not enable unnamed enumeration or EntryPoint.load. Alternatively, a design may explicitly accept the source's unknown-version fallback if it proves that behavior is harmless to every fixed recipe and does not label that fallback an observed installed version. That is a policy choice to review, not an implicit exception. Version fallback does not weaken G1-G3.

## Recommended single bounded direction

Draft a **pinned setuptools initialization transaction owned by the existing source session**, keeping same-process load_inline, the existing generated-extension direct-loader boundary, and unchanged guard/supervisor ownership. The transaction supports this installed version and the fixed import closure only. It is an explicit amendment to design v3 section 7.1's previously unchanged external/publication semantics; do not classify it as catalog-only maintenance.

An implementable narrow shape for that draft is:

1. Before the first setuptools import, within the actual first compile guard and after retention root issuance, verify cold absence of setuptools, _distutils_hack, distutils subtree and backports. Fix SETUPTOOLS_USE_DISTUTILS=local (or require exactly equivalent frozen absence/default), isolate/freeze the relevant cwd behavior (`pybuilddir.txt` must not select the CPython-build branch), and freeze optional `_distutils_system_mod`/Cython behavior from actual installed facts. Execute no .pth file. Reject package preloads, not silently clear owners.
2. Use one explicit source-session topology transition to install the exact vendor search layout before setuptools executes, then freeze it. Its own sys.path.extend should be a verified no-op. Maintain exact external resolution precedence separately; the presence of a vendor path is not permission for every file in that directory. Avoid temporarily adding all site-packages merely to satisfy metadata. Recheck interpreter resolver behavior so vendor additions cannot shadow trusted interpreter imports.
3. When the exact admitted _distutils_hack module finishes and before override executes, interpose only its pinned insert_shim/_remove_shim pair for this transaction. These fixed, identity-checked wrappers enter/leave an internal distutils resolution interval instead of inserting a foreign leading finder. The sole custodian handles the exact distutils request, may call the held original dependency spec producer under its own checks, and retains/wraps the actual returned custom spec/loader and same-object alias result. All ordinary source children continue through the single existing file-admission engine. This deliberately changes shim installation behavior; it must be explicitly designed and reviewed, not hidden as an unmodified-import claim or generic custom-loader permission.
4. Bind the root alias relationship and legal metadata states from G2 to that transaction. Restore the exact helper functions once override completes; require the shim interval closed, dependency attributes unchanged, all expected publication/path identities verified, and no extra resolver. Partial import, caught inner error, restoration failure or out-of-order call poisons the existing source/controller permanently. Successful dependency globals and vendor lookup state remain owned for subsequent compilation; do not attempt unsafe rollback/unload of executed modules.
5. Provide only the required finite package-resource view (G3) and the explicitly chosen named metadata behavior (G4). Freeze the additional files and source-audited transitive imports in the environment before the original input cut, as design v3 already requires. No learned import allowlist, metadata-wide enumeration, package installation, arbitrary hooks or backend-selected importer is introduced.

This is a design direction, not a complete or approved API/state machine. The next draft must specify exact hook installation/return checkpoints, original function identity/source checks, wrapper ownership, spec-producer identity and allowed behavior, retained object graph, observation schema, failure state, and target-Python metadata invariants. In particular, an already running parent may catch an exception and keep publishing: retain late owners on every failure without clearing poison or repeating destructive disposal.

Why this direction: it localizes unavoidable dependency adaptation while preserving one finder, one source-file policy, the existing guardian and the accepted direct native loader. The existing design already uses a tightly bound runtime interposition for Torch's final loader; that is architectural precedent, not automatic permission for these new hooks. Avoid an extensible registry of arbitrary dependency adapters.

Alternatives evaluated:

- More module/publication rows alone: insufficient by G1-G3.
- Whitelist the actual finder ahead of custody or delegate arbitrary package loading: weakens the central pre-execution boundary; not recommended.
- Preimport setuptools before custody: loses cold provenance/partial-owner coverage and violates current install checks.
- Patch installed files or execute transformed vendor source: changes the deployment/source provenance contract and duplicates upstream behavior; broader than the proposed held-original-function integration.
- Move packaging/build import into another process or use prebuilt artifacts: potentially valid future architecture, but changes accepted same-process build/direct-load provenance and process ownership. Not a minimal fix to this checkpoint.
- Use stdlib distutils or change package version: not an authorized faithful admission of the installed target. The pinned Python-3.12 deployment and dependency source do not supply an approved alternative this way.

## Ownership, evidence and maintainability requirements

Keep environment file identities and finite import/resource/alias rules as one immutable source of truth; observations describe what actually executed and cannot issue live authority. Reuse existing session/runtime token registry, guardian retention and sticky failure path. If new observations are necessary, freeze their fields and validator joins in the amendment, preserve the six-phase input cut, and retain failed prefixes without inventing successful publication/adoption.

Distutils monkey.patch_all (archive setuptools/__init__.py:256; monkey.py:72-105) mutates classes/functions on already-held module objects. Current module-replacement policy is a fixed Torch policy, not permission for unrelated package replacements. Ordinary attribute mutation is not itself an additional current metadata rejection. Source-audit the finite actual patched targets used by this dependency path, hold relevant original/result identities where the new transaction depends on them, and audit postconditions. Do not invent replacement-module evidence for ordinary attribute patches, or claim every callable in a Python package needs a new generic integrity mechanism.

The bootstrap is already 3647 lines. Do not add independent alias/file/resource validation stacks or a second failure owner to it. Give dependency-specific transition knowledge one narrow owner under SourceSession; shared file checks and observations remain shared production mechanics. This early structural requirement does not replace the owner-requested whole-C4 systematic refactor. That later pass still covers all C4 responsibilities and affected C2d3 interfaces, full functional success/failure evidence, removal of superseded paths and fresh independent review before final acceptance. Remove this compatibility transaction if the pinned dependency behavior or accepted artifact route eliminates the need; avoid preserving both routes indefinitely.

## Verification performed and limitations

Executed with `C:/Python314/python.exe -I -S -B`, Windows CPython 3.14.0. No setuptools, _distutils_hack, torch or warp module imported. No dependencies installed; no compiler, GPU/native, SSH/process-supervision test or host environment change. Scratch writes only under ../c4_assets; temporary metadata/resource directories were removed by TemporaryDirectory.

Commands/checks:

- `git rev-parse HEAD`, quoted `git rev-parse 'HEAD^{tree}'`, `git status --porcelain=v1`, `git diff --stat 6f253096537711165aca26345233ea831e0662aa HEAD`.
- Get-Content/rg source and schema inspection plus stdlib zipfile/ast/hashlib/json checks of exact archive/record bytes.
- `C:/Python314/python.exe -I -S -B ../c4_assets/setuptools_topology_gap_v1.py`: exit 0. Baseline passes; actual path-extension and insert_shim nodes both reject with `closed import topology changed`. Production bootstrap raw hash matches `d7f9048763ff1e2cd810d72416fb33f57acd6bdb714c33524a76c53970d84775`.
- `C:/Python314/python.exe -I -S -B ../c4_assets/setuptools_admission_independent_probe_v1.py`: exit 0. Exact installed alias producer with stand-in package observes the metadata facts above; production check_module_name rejects `external intrinsic module name drift`; production _paths rejects `external parent is not a file package`; stdlib metadata positive/rejection and exact-wrapper resource positive/rejection controls pass.
- Inline stdlib check reconciled all 441 fact digests/319 omitted rows to archived Conda record and recomputed the 326 overlapping archived file byte digests/sizes.

Non-substantive tool issues: initial unquoted PowerShell HEAD^{tree} was parsed incorrectly and was rerun quoted; initial root REVIEW_PRINCIPLES.md read was corrected to collab/REVIEW_PRINCIPLES.md; one source display hit console GBK encoding on a Unicode documentation character and was rerun with escaped output. These are inspection-tool failures, not product failures or successful runtime evidence.

The probe deliberately executes AST-extracted production/dependency nodes on stand-ins; it is not an actual setuptools import or a whole-session integration test. Local Python 3.14 metadata/resource/importlib details require confirmation against pinned target Python 3.12.13 before implementation acceptance. No complete import/optional-probe/data/resource closure, real installed target success, native restoration, compiler membership or process cleanup is proved. Unarchived data bytes and target stdlib source inspection remain needed before exact code preflight.

After independent approval of the amendment, require production-boundary CPU positive/counterexample tests for topology, exact alias transition/children, actual resource expression, bounded metadata choice, restoration and sticky failure/late-retention; then fresh exact source/policy/catalog/runner preflight before any newly authorized target dependency/native execution. Continue the previously required real build/guard/process/target-device evidence afterward. This report opens none of those execution gates.
