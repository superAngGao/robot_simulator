# Compilation v2 initialization journal replay WIP

Initiative: Q54 / P12.4 / C4c2 / CUDA G1
Stage: implementation progress, not checkpoint acceptance
Author: codex
Version: v4
Date: 2026-10-09
Status: draft
Related Files: setuptools-admission design-request/design-review v2; implementation-note v3; compilation-admission design-request/design-review v3; systematic-refactor implementation-note v1
Owner Summary: Compilation record/selection v2 now joins the initialization actions to ordinary file calls, continuation retention and exact guardian adoption operations. Local descriptive replay attacks pass. The full installed profile, captured-source integration, independent execution preflight and real dependency/compiler evidence remain unfinished.

Base: b18d50cae6ca8470e4d07fc218f414b0733bd93d. This note accompanies the
local WIP code commit on implementation/c4-supervised-waits. It is not an
independent approval or authorization for new native execution.

## Implemented

- New collector-facing record/selection constants name v2. Historical v1 remains
  descriptive parsing only, with its original seven-module continuation closure.
  A v2 record requires the environment's non-null pinned initialization descriptor;
  v1 cannot carry this transaction. Selection and record versions must match.
- Production continuation dispatch consumes the closed shared action fields,
  ordered actions, one initialization identity, metadata/resource read occurrences,
  same alias identities and irreversible failure. Failed partial owners may be
  held without successful adoption; they cannot satisfy a successful field.
- Every successful helper is joined to its exact operation/role, HOST_OWNER,
  guard6, new continuation token and prior compile-root retention. The retention
  journal index precedes the action, and its lifecycle index precedes adoption;
  adoption precedes use. Helpers cannot alias initialization or ordinary source
  objects, including ordinary source facts arriving after helper observations.
- Source fields join their exact named ordinary file owners/calls. The source
  spec/loader fields mean the existing spec and standard delegate. Executing
  requests require that exact active file call at the top of the import stack;
  completed fields require completed guard6 continuation imports. Metadata and
  resource file identities/digests join the frozen environment rows.
- Begin requires the first observed compile boundary and the exact one-vendor
  append shape. Completion requires restoration, resource and version reads,
  three actual named completed source records and closed imports. Later successful
  compiler steps/recipes require completed initialization. Original input records,
  journal prefixes, lifecycle cuts and source traces retain their existing joins.

## Local verification

Windows C:\Python314\python.exe; PYTEST_DISABLE_PLUGIN_AUTOLOAD=1:

```
C:\Python314\python.exe -B -m pytest -q -p no:cacheprovider tests/unit/optics/test_runtime_environment.py tests/unit/optics/test_setuptools_contract.py tests/unit/optics/test_setuptools_admission.py tests/unit/optics/test_setuptools_replay.py tests/unit/optics/test_runtime_custody.py tests/unit/optics/test_cuda_compilation_custody.py tests/unit/optics/test_cuda_toolchain_environment.py tests/unit/optics/test_cuda_compilation_unwind.py tests/unit/optics/test_cuda_compilation_codec.py
```

Final:179passed,11skipped in13.65s, exit0. The11skips are existing POSIX archived
input codec tests on Windows; they are not passes. git diff --check passed.
All local test processes returned.

New replay tests send a full descriptive initialization journal and ordinary
file-resolution/create/exec/return/completion events through production continuation
dispatch and inherited runtime file/custody joins. Guardian adoption records come
from the real guardian. The outer guard/native context and input cut are expressly
stand-ins: the tests do not invoke full CudaCompilationRecord acceptance or issue
any live initialization/native authority. They cover all105 coherent pairwise
exchanges of15helper slots with object/retention/lifecycle facts held fixed, named
source substitution, missing reads, wrong version/file/read index, skipped actions,
retention-after-adoption, reused input ownership, wrong executing requester,
unadopted failed prefixes, success after poison and version-mismatched selection.

Initial test failures were assertions expecting the wrong rejection point/type:
some cross-time swaps are rejected by prior-retention checks before operation joins,
and exact schema errors use GuardProtocolError. Corrected assertions preserve the
requirement that every mutation is rejected; no production validation was weakened.

## Next evidence and unfinished work

Primary-host read-only revalidation found a clean detached compilation tree at
52fb36a8960596ddb2ac3decd8d129188e4b1eb4 and the unchanged historical input archive
at /data2/shared/ang.gao/c2d3-fc86182-native-scope-v2.tar.gz. Target Python3.12 pure
CPU checks and historical full-record codec regression are next; no such result
is claimed by this note. Transfer preserves existing history and the isolated tree.

Still required: a complete installed v6 source/namespace/absence/data profile;
source policy/catalog refresh and complete production bootstrap/dependency CPU
integration; dedicated exact CPU preflight before actual setuptools import;
original full compiler/process/native preflight and real cold builds/faults;
fresh whole-checkpoint review, later C4 stages and the owner's systematic refactor.
The parser work does not prove real dependency initialization, actual worker cwd,
device context, process cleanup, CUDA performance or whole C4 completion.
