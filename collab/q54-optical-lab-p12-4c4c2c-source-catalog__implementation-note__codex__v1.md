# C2c6: literal component source catalog and actual import ledger

Initiative: q54-optical-lab-p12-4c4c2c-source-catalog
Stage: implementation-note
Author: codex
Version: v1
Date: 2026-10-03
Status: in_review
Related Files: runtime-integration design codex v3 section4;
native_sources.py; native_source_policy.py; data/native_sources/v1.json;
source-policy review codex v2; cleanup_ledger.json
Owner Summary: Freeze the current production component source graph, capture
verified bytes and inspect actual loaded repository modules. Do not activate
the unfinished integrated worker or characterize a post-import check as a loader.

Base: `73b4172caeb155158acdba2a3e4f3d98799fdcd6`.
The frozen candidate is the commit containing this note, reported in the review.

## Delivered behavior

`native_sources.py` owns a literal common source list and explicit Warp/CUDA
additions. The tracked LF canonical JSON catalog freezes each exact source
digest, policy digest and both ordered lane lists. No filesystem glob, caller
path list, runtime source discovery, benchmark package, example module or test
module is an admission input. Only the two already named production launchers
are listed under tools. Eager physics/sensing package dependencies are included;
unused deferred optical exports do not grant access to the losing renderer.

Source catalog scope is the **currently existing C2 components**: CPU inputs,
parameters, setup/guard contracts and native scene/acceleration/render/arena
sources. Future C2d/e modules and actual integrated dispatch do not exist yet and
cannot honestly be declared part of a complete executable-worker closure. Each
such addition must update this literal inventory and frozen artifact in its
reviewed candidate. Existing C0 source allowlist and protocol are not extended.

Capture takes a trusted repository root, literal lane and complete configured
Git revision, never values admitted from IPC. It checks actual HEAD, reads the
catalog from that revision's Git object, requires working catalog bytes to equal
that object, validates the strict canonical schema/ordered inventory, then reads
and AST/hash-checks every actual lane source. Symlink components and escapes
reject. The immutable snapshot retains the actual checked source bytes, rather
than returning only caller-filled hashes. Root/revision/catalog/ordered sources
are bound; a self-consistent edited working catalog cannot replace Git release
data. `verify_unchanged()` repeats actual checks and compares the full snapshot.

Trust is explicit: the configured Git revision is trusted bootstrap configuration,
not a signature proving owner approval. Creating a different commit does not
automatically authorize it. The later runtime manifest must bind the reviewed
revision and source snapshot. A directly constructed descriptive dataclass is
not a C4 capability; copying it grants no lease, runtime admission or execution.

`check_loaded_modules()` consumes actual `sys.modules` itself, not a supplied
module list. It revalidates the source snapshot and checks repository module
name, __file__, spec origin, root, source suffix and lane membership. It rejects
foreign checkout imports, unlisted imports, missing or changed metadata, and
repository names masquerading as non-module runtime namespace objects. Installed
third-party runtimes remain trusted dependencies, not Python-sandboxed code.

This is a **post-import boundary audit**. It does not claim to intercept imports
before code executes or establish which earlier .pyc bytes an import consumed.
Source consumption/cache policy and wiring checks to startup/HELLO/request
boundaries remain mandatory C2d integration work before dispatcher admission.
The captured bytes provide input to that work, not an invented execution proof.
This distinction is explicit in the public method documentation.

## Review follow-up and integration-discovered regression

C2c5 C1: both write/escape mutations now match the exact four-space line in the
snapshot function and assert `ast.parse` succeeds before expecting rejection.
They no longer accidentally modify an earlier eight-space-indented block.

Checking the full inventory exposed a C2c5 false rejection missed by representative
source tests: actual C0 protocol/lifecycle call `re.compile`, which is regex
construction, not dynamic Python compilation. Permit only exact `re.compile`
load references in these two unchanged, whole-source-pinned reviewed files.
The literal policy includes both release digests. Dynamic builtin compile,
arbitrary obj.compile, rebinds/aliases/changed regex source and source relocation
still reject. No C0 code or algorithm is changed; its own v1 policy stays intact.

## Validation and exclusions

Tests use explicit LF source-release packaging on Windows; production verification
never normalizes bytes. On POSIX the additional test checks the actual checkout
against its actual Git HEAD, not a reconstructed fixture. Cold subprocess tests
exercise the real CPU materializers and actual sys.modules checks in each lane
without importing Torch/Warp. Real Go2 needs the existing asset directory via
ROBOT_SIM_TEST_GO2_MODEL_DIR; local absence is a skip, not proof, and the remote
review/author run must supply it. GPU modules are inspected as pinned source,
not imported outside a supervised guardian. Actual guarded runtime import/launch
and final complete-worker closure remain C2d/e/f acceptance obligations.

Negative coverage: working source plus rehashed catalog substitution; wrong or
symbolic revision; missing/extra/reordered/duplicate source entries, wrong policy,
unknown fields, even in a newly committed invalid catalog; symlink file/directory;
snapshot drift and copied incomplete snapshots; actual unlisted-module import,
foreign-root import, and post-import name/file/spec/bytecode metadata changes.
All this is untimed CPU preflight, not measured acquire work or benchmark results.

The frozen lists contain65 Warp sources,70 CUDA sources and71 distinct files.
Development findings are not hidden: the first catalog shape used a71-field
digest object and correctly hit the existing64-field canonical-codec limit
(15failed/179passed/4skipped); it was replaced by ordered two-field digest records,
not by widening the old protocol. A subsequent local run had2missing-NumPy
failures under -I and192passed/4skipped. Materializer tests now explicitly declare
the known installed user-site dependency path rather than inherit arbitrary
parent sys.path; cold CPU-only checks still run -I -S without it. No installation.
Final frozen commands/counts and process evidence will be recorded separately.

Before freezing, local source-catalog/source-policy/C0-protocol collection
observed214passed/4skipped in42.72s. The four skips are the two lane-specific real
Go2 tests and two actual POSIX-checkout tests, all mandatory on the remote run.
Ruff and diff check passed. Observed local pytest PID7548 exited; it was absent
at the subsequent scoped process check, with no timeout or kill.

Existing C0, GPU preparation, C4 completion, B3/host delivery, shipping, a3b and
result/baseline activation remain unchanged and closed. No numeric performance
or memory threshold. Cleanup ledger now covers the source catalog/snapshot/ledger
and tests: collapse to the selected worker closure at P12.4e or remove if unused.
Primary183 timed out; authorized fallback101 at the clean base is used. H200 is
visible but this CPU slice does not claim real-device lifecycle validation.
