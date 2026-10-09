# Source-only CUDA toolchain layout and environment follow-up

Initiative: Q54 / P12.4 / C4c2 / remaining parent C2d
Stage: source-only deployment evidence and bounded environment repair
Author: codex
Version: v1
Date: 2026-10-09
Status: draft
Related Files: compilation-admission design-request v3; implementation-note v1-v3; optical_worker/cuda_compilation.py
Owner Summary: The actual Conda target layout and compiler ELF metadata identify concrete inventory roots and indirections. They do not prove a build or dynamic library selection. Five missing Torch environment rejections were reproduced and fixed; the shared rejection table also covers GCC helper-path/dependency-output overrides. Complete inventory coverage remains unfinished.

## Provenance and method

Local base7280d7d7b5f187fe7f22d089d9ba2e9070b71fdd was clean at the start.
Primary tileops-dev was reachable. Its separate compilation CPU tree remained
clean at b42475e976c568b6a56a10a016a96e78bdf5311c. No remote checkout was changed.

Read-only commands used the existing interpreter with -I -S -B, pathlib/os
directory metadata, file reads/hashes and `/usr/bin/readelf -d` on13 fixed tool
binaries. Python subprocess.run waited for each metadata reader with check=True.
The script asserted no Torch/Warp imports and returned0. All SSH command handles
completed. No compiler, Ninja, CUDA observer, subreaper/signal/process-case driver
or GPU worker was executed. No package, shared file or PCH path was changed.

Full48-row raw source observation is preserved outside the repository at
`C:/Users/AG/新建文件夹/c4_assets/cuda_toolchain_layout_v1.jsonl`, 260777bytes,
SHA25617ca005142cb9d2b97ac4b7786812a25e809670f62c7d4d1b381aa4b5ec761a8.
It contains directory entries, nvcc.profile and loader-config bytes/digests, and
tool ELF NEEDED/RPATH rows. It is explicitly non-validating metadata, not the
complete future toolchain manifest. The initial broad terminal listing was
truncated; this retained capture completed without truncating its file.

## Facts that constrain the implementation

- Conda `include/cuda.h` and `include/cuda_runtime.h` are absent. CUDA headers are
  in `targets/x86_64-linux/include`; the ordinary include root contains Python3.12
  headers. The nvcc target bin contains a link back to the existing Conda nvcc.
- Conda `lib/libcudart.so` links to
  `../targets/x86_64-linux/lib/libcudart.so.12.8.90`. Conda `lib64` is absent as
  previously recorded; pinned Torch selects its existing lib fallback.
- nvcc.profile uses target-relative include/lib/stub, nvvm bin and libdevice paths.
  `bin/crt` holds link.stub/prelink.stub, `nvvm/libdevice` holds libdevice.10.bc,
  and `nvvm/bin` holds cicc. The inventory must include these actual inputs and
  aliases, not just nvcc/g++/Ninja executable hashes.
- CUDA tools' ELF RPATHs include Conda lib and target lib; cicc additionally names
  nvvm/lib64. Ninja has a Conda lib RPATH. cc1plus's NEEDED rows include isl, mpc,
  mpfr, gmp, z and zstd; as/ld include bfd and ld additionally ctf/jansson. These
  are static dependency names, not observations of loaded files.
- Existing `/etc/ld.so.conf.d` entries name system CUDA13 search roots as well as
  standard system library roots. This does not prove those CUDA13 libraries are
  used by the fixed CUDA12 build. It does mean version labels cannot substitute
  for concrete path/link provenance and the frozen loader search configuration.
- Several relevant library directories contain gigabytes of files. The scanner
  must hash incrementally instead of retaining a whole large library merely to
  compute its digest. Frozen setup budgets remain unchanged; no timing success
  or budget adjustment is inferred from this metadata.

The next inventory implementation must share one fixed root/coverage definition
between generation and validation, enumerate full header trees and actual link
search roots, record complete symlink chains and required absences, and reject
omitted/added/changed inputs. It must bind the fixed source/environment and private
run directories. A caller-supplied partial list cannot define its own completeness.
Exact target-macro/path closure and generated build rules still need the required
independent preflight followed by real same-run verification.

## Bounded environment repair

Pinned cpp_extension SHA25652ba4de5deedd894e2bdafc9b7f40526dbc2cc1e4c560bbae3b38d4a5dcab0c6
reads CUDA_INC_PATH, CUDNN_HOME/CUDNN_PATH, TORCH_DONT_CHECK_COMPILER_ABI and
TORCH_EXTENSION_SKIP_NVCC_GEN_DEPENDENCIES. The draft did not reject unlisted
values, allowing extra include/link inputs or changed build checks/rules while
the declared manifest stayed unchanged. Five CPU counterexamples all initially
failed to reject; the absent-versus-empty LD_LIBRARY_PATH control passed.

These variables now join the existing shared rejection table. GCC_EXEC_PREFIX,
COMPILER_PATH, C_INCLUDE_PATH, DEPENDENCIES_OUTPUT, SUNPRO_DEPENDENCIES and
GCC_COMPARE_DEBUG join it for helper/header search and output/option overrides.
Their semantics are documented in the official
[GCC environment-variable reference](https://gcc.gnu.org/onlinedocs/gcc/Environment-Variables.html).
The attempted version13.3.0 documentation URL was unavailable through the browser;
the retrieved general manual is not a version-pinned host execution claim.

Windows C:/Python314/python.exe -B, plugin autoload/cache disabled: the resulting
environment suite plus compiler-unwind suite passed21 cases in0.41s. The tests
invoke the production environment audit with CPU directory stand-ins; they do
not validate a full toolchain manifest or compile anything. git diff --check
passed. No independent code approval or native gate follows from these results.

The complete compilation checkpoint, parent C4 work and owner-required systematic
refactor remain unfinished and active.
