# C2d3 Go2 literal deployment-table review v1

Initiative: q54-optical-lab-p12-4c4c2d-runtime-adoption
Stage: review
Author: codex
Version: v1
Date: 2026-10-08 (Asia/Shanghai)
Status: accepted
Related Files: 183_warp_go2_author_v1.proposed.json; runtime-adoption design v2; optional-adapter design v2; native-self-alias design v1; absent-resolution design v1; go2-dependencies test-evidence v5
Owner Summary: Approve with comments for the exact outside-repository schema-v3 table as a bounded input to the next guarded author Go2 validation attempt. This is not approval of captured closure, C2d3 implementation, GPU execution, product deployment, or any later gate.

Reviewer session: `/root/c4_c2d3_go2_table_scope_review_v1`.
Independent reviewer did not author the candidate. Review was assigned as an
isolated bounded data-snapshot review, without the author's implementation
conversation. AGENTS.md, collab/README.md and collab/REVIEW_PRINCIPLES.md were read.

## Verdict and exact scope

**Approve with comments.** No substantive blocking finding was established in
the literal permission/inventory snapshot within this scope. The comments below
are non-blocking for using this immutable table in the next already-authorized,
guarded author validation attempt; they are not waivers of mandatory later proof.

Candidate code base/HEAD, unchanged during review on both hosts:
`4f82a87d225f5d5bbb08d5dddb8bfcef855b2983`.
This is a data snapshot review, not a code-diff approval; no alternate code base
or merge range is approved.

Local repository:
`C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design`.
Remote repository:
`/data2/shared/ang.gao/robot_sim_c2d3_binding_wip`.
Both working trees were observed clean before and after inspection.

Only approved table snapshot:
`C:/Users/AG/新建文件夹/c4_assets/183_warp_go2_author_v1.proposed.json`,
identical basename under `/data2/shared/ang.gao/` on primary `tileops-dev`
(`hd03-gpu2-0031`).

- Raw SHA-256: `747e680530d2a02b8de372f7f0c660840b22b4cd3a0e375656a84e82a0898d60`.
- Canonical SHA-256: `d2b1edc48c5a28a4ef71237fdf16d92d089f02f3fe7926c858d3e0c651146d3b`.
- Schema: `q54.p12.4.runtime-environment.v3`.

Approval attaches to these literal rows/bytes, not to an assembler, an observed
module count, arbitrary recomputed digests, future additions, or a wildcard.
Any new required permission or changed table requires a versioned review.
No runtime observation may automatically extend this table.

## Independently checked facts

### 1. CPU contract and physical inventory

Used the actual candidate `RuntimeEnvironmentContract.from_payload` and
`verify_files(expected_sha256=...)`, supplying the fixed canonical digest above
independently of the object's computed digest. This passed on robot_sim Python
3.12.13 in approximately 0.741 seconds. Interpreter path, interpreter digest,
all 1,010 file hashes/sizes, contained paths, six distribution identities, and
namespace directories passed the implementation's CPU verification.

The process ran with `-I -S -B`; repository path was explicitly added only to
import the CPU verifier. It was not presented as a captured worker or an
admission test. Assertions confirmed no NumPy, SciPy, trimesh, Pillow, NetworkX,
Warp, Torch or benchmarks import. No dependency loader create/exec was invoked.

Inventory counts independently computed:

| Category | Count |
| --- | ---: |
| File-module rules | 925 |
| Materialization file-module rules | 900 |
| Runtime file-module rules | 25 |
| Inventory files | 1,010 |
| Python source files | 853 |
| Extension files | 74 |
| Other native files | 72 |
| METADATA files | 6 |
| Data files | 5 |
| Absent-probe rows | 28 |
| Disabled adapters | 2 |
| Namespace rows | 2 |
| Publication rows | 6 |

The 853 Python files include the two present-but-disabled sources. Thus 851
executable Python source identities plus 74 extension identities equal the 925
file-module rules. Native file records are inventory, not import permissions.

Module top-level counts are PIL 18, NetworkX 285, NumPy 146, SciPy 349, trimesh
102, and Warp 25. All 111 rules in the existing synthetic schema-v3 fixture are
retained identically; the Go2 table adds 814 file-module rules. The 25 Warp
runtime rows are retained, not inferred from the ordinary CPU observation.
There is no Torch module permission in this Warp author profile.

Distributions match actual installed METADATA: networkx 3.6.1, numpy 2.5.1,
pillow 12.3.0, scipy 1.18.0, trimesh 5.0.0, warp-lang 1.12.0.
The five data records are the two trimesh JSON resources plus three previously
inventoried Warp native source files, not five observed trimesh resource reads.

### 2. Exact module paths and namespaces

Independently resolved all 925 file-module names without executing them using
the standard PathFinder and exact declared parent/site paths. Each returned the
declared physical origin, package/non-package status, and a standard
SourceFileLoader or ExtensionFileLoader. The final name set and paths are not
merely accepted because the author counted them.

For `scipy._external` and `scipy._external.packaging_version`, inspected the exact
parent directories with FileFinder, obtaining null-origin, no-loader namespace
specs with precisely the single declared location each. No synthetic parent
module or namespace publication was installed into sys.modules.

The ordinary v4 observation's 908 materialization names equal exactly the union
of the table's 900 materialization file modules, two namespaces and six
publications. The 900 audited file identities also agree with table paths and
package flags. This is consistency/provenance evidence only, not proof that the
captured importer can execute that closure.

### 3. All 28 absent probes and the two disabled adapters

For every absent row, resolved its exact top-level or declared parent path using
the standard resolver without importing the dependency; every outcome was no
spec. Checked the actual installed source for each requester, using parsed
import/call locations rather than accepting an arbitrary substring match.
All rows have matching real source branches. Representative/full branch index:

| Probe | Requester and installed source line |
| --- | --- |
| cascadio | trimesh.exchange.cascade:69 |
| charset_normalizer | numpy.f2py.crackfortran:153 |
| collada | trimesh.exchange.dae:447 (metadata query) |
| cython | scipy._lib._testutils:22 |
| defusedxml | PIL.Image:62 |
| fcl | trimesh.collision:7 |
| lxml | trimesh.exchange.threedxml:413; threemf:517; xaml:148; trimesh.path.exchange.svg_io:801 |
| manifold3d | trimesh.boolean:16; trimesh.creation:36 (metadata query) |
| mapbox_earcut | trimesh.creation:35 (metadata query) |
| meshio | trimesh.exchange.misc:149 |
| numpy._distributor_init_local | numpy._distributor_init:13 |
| openctm | trimesh.exchange.misc:158 |
| rtree | trimesh.base:80 |
| scikits | scipy.optimize._linprog_ip:39; scipy.sparse.linalg._dsolve.linsolve:17 |
| scipy._distributor_init_local | scipy._distributor_init:16 |
| shapely | trimesh.creation:23; trimesh.path.path:69 |
| sksparse | scipy.optimize._linprog_ip:31; scipy.optimize._trustregion_constr.projections:8 |
| svg | trimesh.path.exchange.svg_io:793 |
| triangle | trimesh.creation:37 (metadata query) |
| uarray | scipy._lib.uarray:13 |
| xxhash | trimesh.caching:52 and :56 (repeated same row) |

Installed `trimesh/util.py:122-124` calls find_spec and tests for None without
catching an exception. The four metadata-query rows therefore genuinely need
the approved absent-resolution behavior; they are not ordinary caught-import
exceptions. No additional metadata-discovery permission is granted here.

The two disabled mappings match optional-adapter v2 literally. The actual
`trimesh/path/path.py:60-63` and `trimesh/ray/__init__.py:4-12` retain their
ExceptionWrapper fallbacks; the latter also sets has_embree=False. Present
adapter bytes match the inventoried hashes, and their own source imports the
missing shapely/embreex prerequisite before useful adapter work. Both
prerequisites resolve absent. The disabled sources are not executable module
rows and their bodies' missing imports are deliberately not absent-probe rows.

Cross-check of all ordinary observation missing-resolve events found exactly
two not covered by absent rows: embreex requested by the ray_pyembree body, and
shapely requested by the polygons body. These are exactly the two bodies the
approved disabled-adapter mechanism must prevent from running. Repeated xxhash
absence is consistent with the source's two fallback import attempts.

### 4. Publications, resources and native inventory

The four empty-attribute module_alias rows match native-self-alias v1 exactly:
_csparsetools, _cyutility, _moduleTNC and _ni_label, each with its own qualified
SciPy extension as creator and publisher, in materialization. The two nonempty
created_module rows select `cb` and `simplex_constants` on the exact
`scipy.optimize._highspy._core` native root. No fictitious alias file loader or
second native create/exec permission is present.

Inspected the ordinary observer source and its actual stored per-creator events:
each of the six publications is absent at create-return and exec-entry and
present at exec-return; self-alias events report root identity, and child events
report exact raw-attribute equality plus distinctness from the root. These are
internally consistent pre-existing ordinary observations, not a new execution
or captured-admission proof. This review did not execute the native creators.

Checked the raw source of `trimesh/resources/__init__.py:88-99`: resource reads
are fixed relative file reads and cache insertion. The v4 recorded cache contains
only creation.json and units_to_inches.json; both are individually hashed data
records in this table. Ordinary resource-cache observations do not establish a
future captured worker's complete resource closure.

Verified that all 139 author-observed native-path records and all 28 separately
resolved linker-path records are represented with identical digest/size in the
literal inventory. These sets overlap with extension and other native entries;
their counts must not be added to derive independent file-module permissions.

The audit's standalone libgfortran-to-libquadmath unresolved diagnostic remains
visible. Existing process-map evidence and the importing extension's RPATH are
different evidence from standalone ldd output. This review neither reran every
linker command nor inferred successful native runtime execution from that audit.

## Non-blocking comments and retained proof obligations

C1 — Captured closure is explicitly unproved. The source/file checks make this
table a defensible literal candidate for guarded author validation, not a
complete-deployment certificate. Actual captured cold Go2 execution may discover
a missing required rule/resource/native dependency. It must fail closed; do not
append observed names at runtime. Before whole C2d3 acceptance require both real
Go2 workloads through the production captured importer, correct outcomes and
custody, canonical arrays/digests, and actual guarded upload/readback. This table
does not cover the separate complete Torch/CUDA lane.

C2 — NetworkX plugin absence must be demonstrated in the sealed worker.
Installed `networkx/utils/backends.py:69,86,102,207` can discover/load entry points.
The ordinary observer records an empty backend map, which does not prove the
no-plugin property under every launch state. The approved singleton finder,
fixed -I -S search topology, no site initialization, and no find_distributions
forwarding remain mandatory. There is no table permission to import arbitrary
plugins, execute .pth files, broaden roots, or enable metadata discovery. Retain
this as a specific captured-worker test obligation.

C3 — Publication and native-runtime identity/lifetime remain runtime obligations.
Stored ordinary events support these six literal mappings but cannot establish
the production guardian's ownership or once-only native execution. Require the
real captured create/exec/publication checks and failure custody, no late audit
admission, and immutable external deployment assumptions specified by the parent
designs. Inventory hashes do not sandbox native code or prove native exception
safety, target-only device scope, successful readback, or clean process reap.

No comment changes the parent acceptance criteria. They are non-blocking only
because the requested decision is permission to try this exact table in the
already-guarded validation workflow, not approval of those uncompleted results.

## Verification commands, provenance and limitations

Local read-only checks used Get-Content with UTF-8, rg, Get-FileHash, git status
--short and git rev-parse HEAD. Remote commands used primary tileops-dev with
BatchMode and a 10-second connection timeout. No fallback host was used.

Remote inspection scripts were passed over stdin, without writing a script or
output into the candidate, using:

```text
ssh -o BatchMode=yes -o ConnectTimeout=10 tileops-dev \
  'timeout 30s /home/ang.gao/miniforge3/envs/robot_sim/bin/python -I -S -B -'
```

The first verifier used timeout 40s. It imported only the CPU contract and
stdlib, parsed the exact table, checked the fixed digest and verify_files,
computed counts, and resolved file specs without executing their loaders.
Verifier PID 3533845, parent 3533844. A later scoped ps check showed it absent.
The final independent source/artifact consistency script completed normally as
PID 3538662, parent 3538661. No background reviewer process was started.

One initial namespace diagnostic stopped with KeyError('scipy') because direct
PathFinder namespace construction requires the parent in sys.modules. This was
a reviewer harness prerequisite error, not a deployment-table failure. Without
importing or fabricating scipy, the subsequent FileFinder directory-level check
verified both namespace specs. All 925 file specs and verify_files had already
passed before that diagnostic stopped. No failed check is relabeled a runtime
pass.

Pre-existing evidence independently hash-checked and structurally inspected:

- c2d3-go2-960-cpu-dependencies-v4.json:
  `ee48dc09dfb8ddd06f1b171148dd923aa19ee7a22f8186243a36bea7b786ce5a`.
- c2d3-go2-frozen-dependency-audit-v4.json:
  `85b0d6e30df3f45e341e40401e9a1ada3d6734308ea607006cc1341f8f12928c`.
- c2d3-offline-wheel-inventory-v1.json:
  `5e7e954d4c79ce323c9712d97c883d9311dc7182f9fb72891912ab007ab9099a`.

Author evidence was treated as historical observation, not fresh independent
execution. No duplicate CPU regression, canonical materialization, dependency
import, backend initialization, GPU test, package install, environment change,
process signal, or candidate edit was performed. The parent reported its
separate CPU regression had 433 passes, two skips, and one stale policy-version
expectation failure; that report was not independently reproduced and is not
part of this table approval. The frozen candidate was not modified to fix it.

Only this requested report is written, with apply_patch, outside the candidate.
No GPU residual query was needed: the reviewer never imported a backend or
started device work. Other users' GPU processes were not inspected or touched.

## Gates that remain closed

Whole C2d3 code acceptance, complete runtime-custody evidence, full captured
dependency/resource closure, both-lane real upload/readback, multi-device scope,
fault/hang/quarantine/reap matrix, and independent final code review all remain
pending. No stage gate is opened by the bounded table verdict.

C2e/f, C4c3/c4, C4d/B3, product GPU preparation, a3b, results/baselines, R2,
host payload, shipping and full C4 completion remain closed. No benchmark,
performance baseline, product activation, executable sample authority, final
optical readiness, or approval of observational counts as product evidence is
created by this report.
