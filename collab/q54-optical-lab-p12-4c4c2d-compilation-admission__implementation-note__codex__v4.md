# CUDA compilation draft: shared complete-coverage inventory scanner

Initiative: Q54 / P12.4 / C4c2 / remaining parent C2d
Stage: implementation note, pending host scan and independent preflight
Author: codex
Version: v4
Date: 2026-10-09
Status: draft
Related Files: compilation-admission design-request/design-review v3; implementation-note v1-v3; toolchain-layout test-evidence v1
Owner Summary: A fixed production scanner now defines inventory coverage for both offline generation and worker comparison. It handles nested headers, library roots, helper files, aliases and absent search roots. Local counterexamples pass; the first complete host scan and exact native preflight remain pending.

Base is c422add921df179712f6dbdcd40a3a00b15e36e1. The previous toolchain validator
only checked listed rows, allowing a partial input list to define its own coverage.
The fixed `_toolchain_roots` and `_toolchain_tools` definitions now determine the
closure. `_toolchain_inventory` observes it without executing tools or importing
Torch/Warp. The offline CLI calls that production function; the worker compares
its complete canonical projection before each guarded recipe, before the first
cpp_extension import. There is no independent generator-side list to drift.

Root rows record path, tree/files/file coverage kind, presence and resolved path.
File rows remain resolved path/size/SHA256; symlink rows retain each literal target
in the traversed chain. Resolution follows a directory link before processing a
subsequent `..`. Shared runtime-environment streaming hashing avoids whole-library
byte buffers, and pre/post metadata checks reject change during a read. Canonical
files are hashed once per scan even when reached through multiple roots/aliases.
Required roots and tools cannot be omitted by a supplied manifest. Optional search
roots retain explicit absence, so appearance changes the comparison. PATH is the
fixed Conda-bin:/usr/bin:/bin order, preventing an arbitrary manifest PATH from
silently selecting another Ninja/tool. Environment/PCH checks still run on entry
and exit; inventory scanning does not delete or mutate shared files.

The fixed profile follows the preserved layout evidence: CUDA/Python/Torch/system
headers, CUDA compiler/helper/profile/libdevice inputs, GCC private helpers and
startup files, compiler/link/runtime library search roots, loader configuration
and cache, and current system-CUDA loader roots. Including a search root does not
authorize using a different CUDA backend/toolchain. The exact generated snapshot
and actual compiler/native evidence still require the prescribed preflight and
same-run checks. Directory selection and cost must be checked on the real host;
no assertion of complete host coverage or fitting the setup budget is made yet.

Local Windows C:/Python314/python.exe -B, plugin autoload/cache disabled:
inventory/environment/compiler-unwind checks passed34 in0.74s. They exercise
actual scanner/comparison code with a fixed CPU filesystem fixture: recursive
header coverage, omitted roots/files, changed digest, duplicate rows, substituted
tool, added/modified/removed files, formerly absent root appearance, symlink chain
with `..`, symlink cycle, and a file changed while hashing. An initial test expected
plain Windows path spelling but readlink returned an extended-length spelling;
the corrected assertion compares the actual resolved file. Linux is the deployment
target. CLI --help also passed under -I -S -B without scanning or native execution.

This remains an unreviewed WIP snapshot with stale frozen source/catalog pins.
The host metadata-only inventory run is the next step, followed by the remaining
external closure, integrated collector/source/parser tests, fixed failures and
independent code/ABI/observer preflight. Real compiler/process/GPU execution is
still closed. Whole C4 and the requested systematic refactor remain active.
