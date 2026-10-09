# Target-Python confirmation of the setuptools admission gap

Initiative: Q54 / P12.4 / C4c2 / CUDA G1 compilation
Stage: source-only inventory and CPU gap reproduction
Author: codex
Version: v1
Date: 2026-10-09
Status: draft
Related Files: setuptools-admission gap-scan/gap-review v1; compilation-admission design v3
Owner Summary: The fresh independent reviewer confirmed that existing admission cannot support the installed setuptools behavior by adding module rows. Subsequent CPU-only stand-in probes on the actual Python3.12 deployment reproduce its alias, metadata and resource conflicts. No package/backend import or native compilation ran; the bounded design amendment remains next.

## Review disposition

Fresh reviewer `/root/setuptools_admission_gap_v1` reviewed clean exact HEAD
`758950c93d4d8aa0c3666098fbd36ea7e29dc31f`, base
`6f253096537711165aca26345233ea831e0662aa`. Full report is archived unchanged as
`q54-optical-lab-p12-4c4c2d-setuptools-admission__gap-review__codex__v1.md`;
raw SHA256 `462a69f35df836cdf8c1760178dc4639cb6854af8e967e604dd5caea514a9568`.
Its literal question marks in the repository path/punctuation are preserved;
actual repository is `C:\Users\AG\新建文件夹\p12_4_c4_warp_timeline_design`.

Verdict: Approve with comments, **gap assessment only**. It confirms mandatory
topology/vendor resolution, package-alias identity/child resolution and package
resource support gaps. It does not approve any implementation or native run.
The author accepts the additional resource finding and the distinction between
mandatory version lookup (caught unknown-version fallback) and unproven general
entry-point use. A bounded initialization transaction will be designed and
independently reviewed; no general packaging/plugin facility is authorized.

## File facts

Primary host only: `tileops-dev`, fixed robot_sim interpreter and clean detached
`/data2/shared/ang.gao/robot_sim_cuda_compilation_wip` at
`52fb36a8960596ddb2ac3decd8d129188e4b1eb4`. All artifacts below are in
`/data2/shared/ang.gao/` remotely and `../c4_assets/` locally.

- `cuda-compilation-helper-sources-v1.zip`:336entries,991312bytes,
  SHA256 `8f5a7f25d4e7654cea2e157d44af708cc14b912edcdedbcc68e7852dcffdaac3`.
  See gap-scan for collection scope and the deliberately rejected wheel scan.
- `cuda-compilation-setuptools-conda-facts-v1.json`:135186bytes,
  SHA256 `45203dcfcd97a7b925b8c72061b71b38187328e7205e08ced3f28c98a097f403`.
 441 non-bytecode installed files match the exact Conda record's
  `sha256_in_prefix` (or sha256 when absent);319 bytecode entries are explicitly
  omitted. Reused the existing offline inventory `_digest` through runpy under
  `-I -S -B`; no new inventory implementation or runtime permission was added.
  Conda record SHA256:
  `7f0b65f43d8dc9d7392e6a57b986b33d010da754c679ea82db83cd441fbbd118`.
- `cuda-compilation-target-importlib-sources-v1.zip`:11entries,33912bytes,
  SHA256 `3a41be7920961a96cb36e0aa442ecbc0ec0ff2f805ceadae9fadba917fab8289`.
  Contains target importlib/_bootstrap.py, metadata/__init__.py, resources Python
  sources, and the actual jaraco/text/Lorem ipsum.txt bytes. The latter directly
  matches1335bytes and SHA256
  `37fedcffbf73c4eb9f058f47677cb33203a436ff9390e4d38a8e01c9dad28e0b`.

All source collectors used exclusive output creation, read files without package
execution, asserted no torch/warp/setuptools/_distutils_hack imports, and returned
exit0. Transfers were hash-checked. Files are facts, not module admission rules.

## Target CPU reproduction

After review, copied the reviewer's scratch
`setuptools_admission_independent_probe_v1.py` to
`setuptools_admission_target_probe_v1.py`, changing only the repo assignment to
the fixed remote isolated tree. No production source changes. Executed:

`/home/ang.gao/miniforge3/envs/robot_sim/bin/python -I -S -B /data2/shared/ang.gao/setuptools_admission_target_probe_v1.py`

Exit0 on Python3.12.13. Exact AST nodes plus stdlib on stand-in objects reproduce:

- Source and alias are the same object; name changes to distutils, source spec
  is replaced by alias spec, source loader/package remain, alias spec is not a
  package spec. Production rejects `external intrinsic module name drift`.
- Alias child path rejects `external parent is not a file package`.
- Exact PKG-INFO is discoverable under standard finders; the singleton finder
  without a distribution resolver gives PackageNotFoundError.
- Standard source loader reads the positive-control resource; the production
  source wrapper has no resource reader and gives `Can't open orphan path`.

These results close the local3.14-vs-target3.12 uncertainty for these **stand-in
probes only**. No actual setuptools import/complete closure, compilation,
subreaper/sigaction/pidfd, native observer or backend test is claimed. Temporary
metadata/resource directories were removed by the probe's TemporaryDirectory.
Both synchronous SSH calls returned; no worker/supervisor/compiler/GPU process
was launched. Final remote working tree remains clean. The original review
report is not revised retroactively to include these later observations.

Next: explicit bounded design/state/ownership/schema and restoration rules for
the pinned dependency initialization, independent design approval, implementation
and required renewed exact-code preflight. Whole-C4 implementation, real evidence,
systematic refactor and fresh acceptance reviews remain unfinished.
