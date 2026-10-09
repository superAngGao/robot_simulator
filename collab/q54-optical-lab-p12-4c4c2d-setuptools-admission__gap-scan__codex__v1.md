# Installed setuptools import topology conflicts with compilation admission

Initiative: Q54 / P12.4 / C4c2 / CUDA G1 compilation
Stage: source-only gap scan and independent review request
Author: codex
Version: v1
Date: 2026-10-09
Status: draft
Related Files: compilation-admission design-request/design-review v3; systematic-refactor implementation-note v1; optical_worker/native_bootstrap.py; optical_worker/runtime_environment.py
Owner Summary: The installed setuptools closure cannot be admitted by adding module rows alone. Its real import-time topology and package-alias behavior conflict with existing closed import custody. Resolve this bounded architectural gap before implementation; native compilation remains closed.

## Candidate and evidence

Code base is clean local `6f253096537711165aca26345233ea831e0662aa`, branch
`implementation/c4-supervised-waits`. Remote source-only inspection used the clean
detached tree `52fb36a8960596ddb2ac3decd8d129188e4b1eb4` at
`/data2/shared/ang.gao/robot_sim_cuda_compilation_wip`, via primary `tileops-dev`.
No tracked implementation changes accompany this scan.

The existing wheel scanner was invoked with `-I -S -B` and explicit distribution
`setuptools`. It rejected `requested wheel metadata missing: ['setuptools']`.
This is a packaging-format mismatch, not a missing dependency: the installed
package is Conda setuptools83.0.0, with
`setuptools-83.0.0-py3.12.egg-info/PKG-INFO`, not wheel METADATA/RECORD. The runtime
environment distribution schema already accepts an inventoried metadata file;
no format exception to runtime admission is justified by this scanner failure.

Source-only archive, both at `/data2/shared/ang.gao/` on the primary host and
`../c4_assets/` locally: `cuda-compilation-helper-sources-v1.zip`, 991312bytes,
SHA256 `8f5a7f25d4e7654cea2e157d44af708cc14b912edcdedbcc68e7852dcffdaac3`.
It contains336 files: installed setuptools/_distutils_hack Python sources,
egg metadata, Conda installation record, distutils-precedence.pth, pinned Torch
cpp_extension/hipify/file_baton/versioner/appdirs sources. The collector read
bytes only, used exclusive output creation, and asserted none of torch, warp,
setuptools or _distutils_hack was imported. It completed with exit0. Archive
transfer hash matched. No compiler, backend or process-supervision test ran.

## Observed conflicts

1. `setuptools/__init__.py:17` extends `sys.path` with its `_vendor` directory;
   line19 removes a preexisting `backports` publication. Exact source SHA256:
   `f94b531bda5d7846646222054faf51724302ab151d5482e38f6e20dea789feb7`.
2. `_distutils_hack.override` calls `do_override()`. With its normal `local`
   selection, `ensure_local_distutils()` temporarily inserts DISTUTILS_FINDER at
   `sys.meta_path[0]`, imports distutils, and removes the finder. The actual
   finder imports `setuptools._distutils`, returns that SAME package from a
   custom loader, and changes its `__name__` to `distutils`. Source SHA256:
   `df81e6bcba34ee3e3952f776551fb669143b9490fdd6c4caeb32609f97e985b4`.
3. The existing `_SourceSession.check_import_topology()` requires the exact
   one-element finder chain and frozen sys.path. `_ExternalAdmission.find_spec`
   requires the standard file loader and the declared origin, and `_paths`
   expects package parents to be admitted file modules or namespaces. Existing
   publication aliases alone do not grant a package-alias subtree resolver or
   allow the source package's metadata to change.
4. `setuptools.monkey.patch_all()` changes distutils classes/functions. Its
   vendor closure also includes `backports` with `pkgutil.extend_path`.
   Transitive imports, metadata lookup and these modifications need a coherent
   source-audited contract, not an observed-name allowlist. No closure is claimed
   complete by this scan.

Local scratch reproduction `../c4_assets/setuptools_topology_gap_v1.py` parses the
actual production `check_import_topology` method and exact two installed source
nodes (the sys.path extension and `insert_shim`). It executes them only against
stand-in sys objects. Baseline topology passes; each mutation independently
raises `closed import topology changed`. Command:
`C:\Python314\python.exe -I -S -B ../c4_assets/setuptools_topology_gap_v1.py`.
Exit0, two expected rejections. Production source raw SHA256 was
`d7f9048763ff1e2cd810d72416fb33f57acd6bdb714c33524a76c53970d84775`.
This proves the local contract conflict; it is NOT an actual setuptools import,
complete import-trace proof, native evidence or an accepted replacement policy.

## Independent review requested

Review the exact installed sources and production admission path, and determine
the smallest coherent direction for a bounded design amendment. Check whether
any existing approved mechanism can support the dependency faithfully without
changing policy. If not, identify precisely which authority, identity, topology,
publication, metadata or failure rules must change and why. Consider source
admission, runtime ownership and maintainability together. In particular, a
temporary finder must not bypass custody while it sits ahead of the captured
finder, and a Python package alias must not be mislabeled as an existing native
creator-self alias. Do not invent relationships just because names coincide.

Compare viable approaches and their impact on the already approved same-process
load_inline/direct-loader/guard boundary. A new process boundary, patched vendor
source, preimport outside custody or delegated importer authority would itself
need explicit design, not an implementation shortcut. Do not recommend a generic
plugin/import exemption merely to accommodate this one pinned dependency.

Review outcome is a gap assessment and design requirements, not approval of an
as-yet unspecified adapter. Do not edit implementation, import these dependencies
on the GPU host, execute native tests, install/change dependencies or alter the
frozen environment. Scratch CPU reproductions are allowed outside tracked files.
Return a complete report with inspected revision, evidence, findings, limitations
and recommended bounded next step. The author will draft any required amendment
and obtain independent design approval before implementing it.

The owner-requested whole-C4 refactor remains required before final acceptance.
This early structural review does not replace it or complete C4. All compilation,
readiness, warmup, performance and product gates retain their prior status.
