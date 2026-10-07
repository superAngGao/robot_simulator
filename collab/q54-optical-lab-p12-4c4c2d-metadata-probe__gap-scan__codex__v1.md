Initiative: q54-optical-lab-p12-4c4c2d-runtime-adoption
Stage: bounded metadata-probe gap scan
Author: codex, independent reviewer
Version: 1
Date: 2026-10-08
Status: in_review
Related Files: runtime-adoption design-request v2 section 4.1; optical_worker/native_bootstrap.py; tests/unit/optics/test_runtime_import_admission.py
Owner Summary: Request changes for the bounded absent-probe contract: an unconditional ModuleNotFoundError breaks ordinary find_spec-based optional checks. No whole-C2d3, Go2, GPU, release, or product approval is given.

## Identity and frozen candidate

Reviewer session: /root/c4_c2d3_metadata_probe_gap_review_v1. This reviewer did not author the candidate. Read AGENTS.md, collab/README.md, and collab/REVIEW_PRINCIPLES.md.

Both local C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design and remote /data2/shared/ang.gao/robot_sim_c2d3_binding_wip on tileops-dev had clean working trees and HEAD 306842c994228c80cae31dc79ce01f0748d4ee1b before and after verification. No candidate files were changed. No comparison base/diff acceptance was requested; this is a snapshot gap scan.

Design: collab/q54-optical-lab-p12-4c4c2d-runtime-adoption__design-request__codex__v2.md, SHA256 1a0d3eed4dd182ddd5ea4ad0c19344cad735aabbdd317d77a14aad1932cc0224.

## Finding F1 — High, blocking the mandatory dependency path

Locations: design section 4.1, lines 179–183; native_bootstrap.py _CapturedSourceFinder.find_spec lines 313–318; _ExternalAdmission.find_spec lines 598–623.

The design requires every admitted frozen absence to raise ModuleNotFoundError. The external admission correctly verifies pre-execution absence and returns _AbsentExternalSpec, but the captured finder unconditionally translates that outcome into an exception. This is compatible with a normal import inside a missing-import catch; it is incompatible with importlib.util.find_spec, whose normal top-level absence result is None.

Installed trimesh 5.0.0 actually uses both forms. Directly inspected on primary host, using robot_sim Python 3.12.13:

- trimesh/util.py:107–124 defines has_module; it imports the ordinary importlib.util.find_spec and returns find_spec(name) is not None, with no catch.
- trimesh/creation.py:35–40 calls this helper at module scope for mapbox_earcut, manifold3d, triangle.
- trimesh/exchange/dae.py:447 calls it for collada before choosing the original unavailable-format wrappers.
- trimesh/__init__.py eagerly includes creation in its imports. These are not merely unused later function bodies.

Hashes of the inspected installed sources match the supplied CPU observation evidence:

| Source | SHA256 |
| --- | --- |
| trimesh/util.py | 251f2cb47c5e3547d34141aa5839b02638d4b4921f857108a5ed685af6213f76 |
| trimesh/creation.py | e43a84d731e4f64a8dfb5e49beab4e691d3a5aff707ebbdaf4b053605ccc32ea |
| trimesh/exchange/dae.py | 8d80c3a3835c8c7aca94c149624f1b4d0fcbec2800cb4ef857cc885ce87ca453 |

Narrow fresh CPU reproduction used the existing test importer fixture without changing production finder/loader methods, with the exact installed has_module function extracted by AST source coordinates and admitted as the helper body. Requester modules were minimal fixtures, not complete installed trimesh modules. Source-session/guardian prerequisites were the existing test doubles. Each target was independently confirmed absent by PathFinder against the actual installed site root. Each run used ordinary importlib.import_module and ordinary importlib.util.find_spec.

Observed results for all four names:

| Name | Active requester | Result |
| --- | --- | --- |
| mapbox_earcut | trimesh.creation | ModuleNotFoundError; session and admission failed |
| manifold3d | trimesh.creation | ModuleNotFoundError; session and admission failed |
| triangle | trimesh.creation | ModuleNotFoundError; session and admission failed |
| collada | trimesh.exchange.dae | ModuleNotFoundError; session and admission failed |

Every target produced exactly one absent_probe event and no target loader or module record. Failure context named the executing requester, not the absent module: the unhandled metadata-probe exception escaped its already-started body. Thus sticky failure handling is behaving conservatively; weakening that handling would be the wrong repair.

The supplied c2d3-go2-960-cpu-dependencies-v3.json corroborates reachability in its instrumented observation: these four resolves have spec_found=false, immediate requester trimesh.util, and active_importer trimesh.creation or trimesh.exchange.dae. manifold3d also has a distinct ordinary import probe from trimesh.boolean. The source audit v4 contains the corresponding absent branch/source records. These older artifacts are corroboration, not fresh production-admission or GPU acceptance proof.

Why existing checks miss it: test_optional_absence_executes_no_loader_and_is_not_poison exercises `try: import c2missing` with a ModuleNotFoundError catch, not an uncaught metadata query. The focused existing subset still passes: 5 passed, 77 deselected. The frozen table alone cannot repair the protocol mismatch; even a correct admitted row reproduces it.

## Finding F2 — A plain None substitution is not a complete repair

Current _install inserts the captured finder before existing finders (line 1862), while check_live checks only sys.meta_path[0] identity (line 1746). CPython 3.12 importlib/_bootstrap.py:1255–1282 iterates later finders whenever one returns None. Fresh normal-importlib experiment with a first finder returning None and a witness tail finder returned a non-None spec and recorded both calls. A singleton first finder returned None for util.find_spec and generated the expected ModuleNotFoundError for import_module.

Therefore returning None with the current chain permits a later finder to resolve the frozen-absent name and violates the no-fallback requirement. The appended witness demonstrates protocol behavior, not a presently exploitable production bug: current production raises before the tail runs. This is a blocking constraint on the proposed fix, not a second assertion that current code already falls through.

## Minimal design direction and bounded risk assessment

The proposed single captured-finder authority is a viable direction for a separate design review. It preserves standard Python semantics without modifying installed dependencies or monkeypatching util.find_spec:

1. Preserve the same exact requester/phase/source admission and fixed-root absence check. Translate only the validated absent sentinel to None at the sole meta-path authority boundary. Do not infer a harmless outcome by catching arbitrary ImportError later. Import machinery then raises for real imports, while metadata probes obtain None.
2. Freeze/validate the cold topology before replacement; the observed -I -S -B interpreter starts with exactly BuiltinImporter, FrozenImporter, PathFinder in that order. Replace the chain with the one captured finder and verify the complete chain identity/count, not just element zero. Appended, prepended, reordered, or replaced finders must be rejected before their callbacks execute on the supported path.
3. Delegate only recognized standard-library/builtin roots internally to those exact captured standard resolvers, in their original order. Never delegate a nonstdlib frozen absence to them. Current cold admission recognizes sys.stdlib_module_names union sys.builtin_module_names; retain that distinction explicitly. On this interpreter builtin_module_names minus stdlib_module_names is empty, but do not silently assume that across interpreter contracts. Frozen status alone is not an admission rule: __hello__ and __phello__ have frozen specs but are outside both observed name sets.
4. Preserve the fixed external package-path rules in _paths/find_spec, including explicitly admitted namespace parents. Do not use stdlib delegation to resolve external children. util.find_spec for dotted names can import a parent; none of the four reproduced target names is dotted, and this scan does not prove new parent/namespace behavior.
5. Capture/pin standard lookup inputs, not merely finder class names. PathFinder uses sys.path or a package path and consults sys.path_hooks/sys.path_importer_cache. The cold observed sys.path is python312.zip, python3.12, lib-dynload under the pinned robot_sim interpreter, with no site root. A design must state the frozen-path/hook/cache trust invariant and tests that prevent accidental alternate-root or arbitrary-hook admission. This is an input to design review, not a finding of a demonstrated extra current exploit.
6. Keep standard-library resolution usable after session poison for bounded error serialization/termination, as current admission does by its early stdlib branch. Topology checks can be independent of session liveness. Do not make every stdlib lookup require not_failed or reopen repository/external admission on a poisoned session.
7. Keep actual started create/exec/body failures sticky, retained, and non-retryable even when dependencies swallow them. Keep zero optional target create/exec/body work, no fabricated sys.modules identity, and repeated identical absence behavior. Reject found modules, wrong active requester/phase, unlisted targets, resolver failures, and changed outcomes.

Freeze the active importing file as requester for these helper-mediated probes, with its admitted source identity, rather than granting every call originating somewhere in trimesh.util. The old observation distinguishes the helper frame from the active creator/importer; collapsing them would broaden authority. Do not introduce stack-inspection heuristics just to distinguish import from metadata probing: singleton authority can supply both outcomes through normal importlib semantics.

## Required repair proof

- Production finder plus unmodified stdlib util.find_spec returns None for each admitted absent target; the unchanged has_module helper returns False and the requester completes without poison.
- Ordinary admitted missing imports still raise exact-name ModuleNotFoundError, execute no target create/body, and allow only the original optional catches.
- Repeated probes remain absent without module identity; all three creation targets and the dae target are covered, with distinct manifold3d/boolean import coverage.
- Appended/prepended/replaced finder witnesses and alternate-root/path-hook witnesses cannot execute for frozen-absent external targets. An extra actual module in the frozen root rejects before its loader/body.
- Wrong requester, wrong phase, unlisted absence, preloaded target, resolver exception, and swallowed started failure remain sticky rejections with actual execution counters and retained custody checked.
- Cold -I -S -B source bootstrap and builtin/frozen/file stdlib/package imports still work under singleton topology; needed stdlib imports work after poison without reopening external/repository imports.
- After design/implementation review, both mandatory real Go2 positives are still required under production admission. This bounded CPU result does not substitute for those or any GPU/device gate.

## Verification commands, reproduction, and limitations

Read-only checks included git status/rev-parse on both repositories, installed-source inspection via sed/grep (remote rg is unavailable), installed-source sha256sum, and JSON-field inspection of the two supplied evidence files.

Focused existing test command executed on tileops-dev:

```sh
cd /data2/shared/ang.gao/robot_sim_c2d3_binding_wip
PYTHONDONTWRITEBYTECODE=1 /home/ang.gao/miniforge3/envs/robot_sim/bin/python -B -m pytest -q -p no:cacheprovider tests/unit/optics/test_runtime_import_admission.py -k 'optional_absence or optional_probe or swallowed_started'
```

Result: 5 passed, 77 deselected in 0.37s.

The fresh narrow repro was piped over SSH to robot_sim/bin/python -B -. Its essential complete failure loop is reproducible as follows:

```python
import ast, importlib, pathlib, runpy, sys, tempfile
import pytest
repo = pathlib.Path('/data2/shared/ang.gao/robot_sim_c2d3_binding_wip')
sys.path.insert(0, str(repo))
suite = runpy.run_path(str(repo / 'tests/unit/optics/test_runtime_import_admission.py'))
site = pathlib.Path('/home/ang.gao/miniforge3/envs/robot_sim/lib/python3.12/site-packages')
text = (site / 'trimesh/util.py').read_text()
node = next(n for n in ast.parse(text).body
            if isinstance(n, ast.FunctionDef) and n.name == 'has_module')
function = ast.get_source_segment(text, node)
for requester, name in [
    ('trimesh.creation', 'mapbox_earcut'),
    ('trimesh.creation', 'manifold3d'),
    ('trimesh.creation', 'triangle'),
    ('trimesh.exchange.dae', 'collada'),
]:
    assert importlib.machinery.PathFinder.find_spec(name, [str(site)]) is None
    monkey = pytest.MonkeyPatch()
    with tempfile.TemporaryDirectory(prefix='c4-meta-probe-') as folder:
        fixture = suite['importer'].__wrapped__(pathlib.Path(folder), monkey)
        create = next(fixture)
        sources = {'trimesh': '', 'trimesh.util': function + '\n',
                   requester: 'from trimesh.util import has_module\nvalue = has_module(' + repr(name) + ')\n'}
        packages = ['trimesh']
        if requester.startswith('trimesh.exchange.'):
            sources['trimesh.exchange'] = ''
            packages.append('trimesh.exchange')
        admission, root = create(sources, absent=[{'name': name,
            'requester': requester, 'phase': 'materialization'}], packages=packages)
        try:
            importlib.import_module(requester)
        except BaseException as error:
            print(name, type(error).__name__, str(error), admission.failed,
                  admission.session.failed,
                  [e for e in admission.events if e['name'] == name],
                  name in admission.loaders, name in admission.records)
        try:
            next(fixture)
        except StopIteration:
            pass
        monkey.undo()
```

This loop runs actual production finder/admission/loader code with unit prerequisite doubles and temporary miniature dependency files. It does not run full installed trimesh, the real cold guardian, complete canonical Go2 materialization, or CUDA. Separate -I -S -B inspection checked cold interpreter topology and stdlib sources. No package installation, GPU calls, dependency mutation, implementation repair, or broad test acceptance occurred. Temporary fixture directories were automatically removed; no user data was deleted. The report alone is saved outside the frozen candidate for author archival.

## Verdict and scope

Request changes for the bounded metadata-probe contract/implementation gap. F1 is substantiated; F2 constrains its repair. The singleton-authority direction is not implementation approval and still requires an explicit scoped design and independent review. All parent C2d3, canonical Go2, GPU, performance, deployment, and product gates remain closed by this report.
