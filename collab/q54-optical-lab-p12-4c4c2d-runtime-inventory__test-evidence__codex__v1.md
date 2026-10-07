# C2d3 offline deployment facts, not runtime admission

Initiative: q54-optical-lab-p12-4c4c2d-runtime-inventory
Stage: test-evidence
Author: codex
Version: v1
Date: 2026-10-07
Status: draft
Related Files: runtime-adoption design v2; runtime-adoption implementation-note v1;
tools/optical_runtime_inventory.py; cleanup_ledger.json
Owner Summary: Observe installed file provenance without importing a backend or
executing site startup. This is incomplete C2d3 author work, not approval of a
smaller checkpoint, a runtime allowlist, GPU validation or product activation.

## Observed environment

Primary183 was reachable. At21:56:39 the host reported8 H200, driver595.71.05;
GPU0 UUID GPU-a4b256c9-c56a-0ef5-19fa-482196cc3274 and GPU1 UUID
GPU-3a3112ff-837c-3b87-38d1-cb203d656589. This observation does not freeze the
driver binary or prove cold target-only setup. No101 fallback.

The existing robot_sim site directory contains editable robot_simulator .pth,
coverage .pth and distutils .pth files. They were only listed, not executed.
The new tool runs directly under Python -I -S -B using only stdlib. It parses
explicit wheel METADATA/RECORD, hashes actual files, checks supplied RECORD
hashes/sizes, records excluded bytecode and external scripts/manuals, and rejects
missing/mutated/escaping files. It never imports a selected package, resolves or
installs dependencies, initializes CUDA, or emits import/probe/publication rules.
Output creation is exclusive; existing evidence is never overwritten.

The tool snapshot used remotely has SHA-256
65153895ece9a6c0d1625b581d8fb3ee57e106db6964425a3dce04d98fb61a9f.
Its remote copy is /data2/shared/ang.gao/c2d3-runtime-inventory-wip-v2.py.

## Reproduction

Use the tracked tool in a checkout with the above bytes and a NEW output path:

```sh
/home/ang.gao/miniforge3/envs/robot_sim/bin/python -I -S -B \
  tools/optical_runtime_inventory.py \
  --site-root /home/ang.gao/miniforge3/envs/robot_sim/lib/python3.12/site-packages \
  --output /data2/shared/ang.gao/c2d3-offline-wheel-inventory-new-run.json \
  --distribution numpy --distribution torch --distribution warp-lang \
  --distribution trimesh --distribution scipy --distribution pillow \
  --distribution pyyaml --distribution filelock --distribution typing-extensions \
  --distribution sympy --distribution mpmath --distribution networkx \
  --distribution jinja2 --distribution MarkupSafe --distribution fsspec \
  --distribution packaging --distribution triton \
  --distribution nvidia-cuda-nvrtc-cu12 --distribution nvidia-cuda-runtime-cu12 \
  --distribution nvidia-cuda-cupti-cu12 --distribution nvidia-cudnn-cu12 \
  --distribution nvidia-cublas-cu12 --distribution nvidia-cufft-cu12 \
  --distribution nvidia-curand-cu12 --distribution nvidia-cusolver-cu12 \
  --distribution nvidia-cusparse-cu12 --distribution nvidia-cusparselt-cu12 \
  --distribution nvidia-nccl-cu12 --distribution nvidia-nvshmem-cu12 \
  --distribution nvidia-nvtx-cu12 --distribution nvidia-nvjitlink-cu12 \
  --distribution nvidia-cufile-cu12
```

Observed successful run: Python3262389, timeout wrapper3262387, parent3262384,
sampled start2026-10-07 22:02:01+08, command date22:02:02. Explicit wait and SSH
exit0. The180s outer CPU-tool timeout did not fire. This timeout is not a CUDA
watchdog proof or a new product timeout policy.

Artifact:
/data2/shared/ang.gao/c2d3-offline-wheel-inventory-v1.json
and matching local c4_assets/c2d3-offline-wheel-inventory-v1.json.
SHA-256:5e7e954d4c79ce323c9712d97c883d9311dc7182f9fb72891912ab007ab9099a.
32 explicit distributions,18065 files,7512792622 bytes observed. Exclusions:
6555 bytecode files,8 external entry points,1 external manual. Actual byte hashes
and sizes matched every provided RECORD hash/size. Files with absent RECORD
hashes are hashed as observations, not misrepresented as wheel-authenticated.

The artifact stores individual names/versions, metadata dependencies and file
digests; it has no timestamp so unchanged bytes have a stable canonical encoding.
Its broad installed-file inventory is NOT the runtime import closure. Selected
runtime files and exact observed/source-audited import rules must still be frozen
in the reviewed RuntimeEnvironmentContract. E.g. Torch METADATA names setuptools,
whose installed egg-info is not among these explicitly selected wheels; OS/native
loader dependencies and the driver also need their own treatment. No transitive
closure completeness is claimed, and no unlisted dependency is auto-admitted.

## Failures and tests

- Initial local test rejected python.exe because Windows path-stat and fd-stat
  expose different ctime semantics. The tool now compares stable file identity,
  size and mtime across both APIs and compares ctime before/after within each API.
  A real mutation during hashing is rejected by a negative test.
- Initial remote run exited1, Python3261227/wrapper3261225/parent3261222,
  start22:00:55. SymPy's external manual path failed the containment check;
  no output inventory was written. The tool now explicitly records that
  non-runtime manual exclusion without reading outside site-packages. All other
  traversal remains rejected. Old script bytes were preserved as the separate
  c2d3-runtime-inventory-wip.py file, not silently replaced.
- Local tool tests:12 passed in0.76s. Includes malformed/duplicate records,
  changed bytes/size, missing identity, traversal, explicit selection, disabled
  package execution, exclusive output and real isolated CLI subprocesses.
- Tool plus repository cleanup-ledger validator:13 passed in1.34s. Ruff passed.
  The ledger now names the C2d3 runtime/readback code and tests for selected-lane
  collapse and preserves the CPU-only inventory tool without granting admission.
  Historical extraction tests lose old-revision dependencies only after equivalent
  selected single-core regression coverage exists. No gate or shared rules changed.

All tests here are author evidence. Complete C2d3 still requires its actual
captured-runtime GPU/fault/dual-device matrix, source-policy pinning, observed
runtime evidence replay, terminal disposal integration and fresh independent
code review. Full C4, C4d/B3, product preparation, a3b and result/baseline stay closed.
