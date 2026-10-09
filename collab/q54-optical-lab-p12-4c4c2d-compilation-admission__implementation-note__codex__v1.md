# CUDA compilation implementation in progress

Initiative: Q54 / P12.4 / C4c2 / remaining parent C2d
Stage: implementation note, not a code review or acceptance
Author: codex
Version: v1
Date: 2026-10-09
Status: draft
Related Files: compilation-admission design-request/design-review v3; unchanged whole C2d3 review v4
Owner Summary: Implementing the approved bounded three-recipe CUDA prefix. Shared recipes, continued object retention, descriptive compilation replay and a dedicated compiler job owner are being connected. This is an incomplete work snapshot for preservation and CPU testing, not an executable release, native preflight approval or checkpoint acceptance.

## Scope and current work

The accepted design is b3c4110cbf34aaf7512aceb722609fade69f85f0, with its full
independent v3 report archived unchanged at 58d0db3. No implementation has received
independent approval yet. Whole C2d3 acceptance and the parent C4 objective remain
unchanged; no later gate opens.

Scene and direct-light kernel literals now have sole pure source modules. All
three compatibility loaders call the shared recipe kwargs factory. The six source
literal SHA256s match the pre-change AST literal baseline exactly. No entry ABI,
kernel algorithm, default backend or workload was intentionally changed.

The draft adds one fixed zero-based phase contract; compiler job deadline/reaping
records separate from the existing worker guard core; a private Linux subreaper/
pidfd owner and pre-COMPLETE transport join; a guardian-held compilation root using
the original runtime token registry; a separate downstream compilation grammar;
and fixed Torch helper/direct-loader integration with returned-module retention.
Old runtime-custody v1/v2 kinds remain closed. Common header/native-object/final
checks are factored without adding compiler kinds to the input parser.

## Actual local verification so far

Windows PowerShell, C:/Python314/python.exe -B, pytest plugin autoload disabled,
cacheprovider disabled:

- Shared recipe, original source-byte and existing scene/LBVH checks: 15 passed.
  The first test attempt had a test regex that missed a multiline pybind call;
  the whitespace-aware repair passed against the unchanged literal bytes.
- Compiler-job/core CPU checks: 136 passed. These include original operation and
  shorter pending-ACK deadlines across modeled worker exit, immutable exited-core
  evidence, post-exit clock failure, late adoption and duplicate attempts. Initial
  author mistakes used a nonexistent protocol constant and evidence method; both
  were corrected before the passing run. No real OS signal/reaping is claimed.
- Combined recipe/job/core/runtime-codec subset: 165 passed in 2.82s.
- Actual observation-registry/guardian-root tests: 2 passed in 0.66s, including
  failed direct adoption with the original object still held by the adopted root.
- Runtime codec/native-scope subset earlier: 23 passed, 32 POSIX skips.
- The unchanged independent v3 design probe still runs successfully. Its printed
  schema model is historical design evidence, not proof of the new parser.
- The new archived-input compilation codec suite initially produced 10 setup
  errors on Windows because the unchanged deployment contract rejects POSIX paths
  on that host. It is now explicitly POSIX-only, like the existing archive codec
  suite. It still requires actual execution on the isolated Linux CPU tree; skips
  are not a pass. Its continuation rows are explicitly synthetic descriptors,
  not newly executed native compilation or live-owner issuance.

No new compiler/Ninja, backend/native observer, libc subreaper/sigaction call or
Linux process test has run. No dependency or shared Torch PCH file was changed.
All local test peers joined. No GPU worker/supervisor was launched.

## Required work before native preflight and acceptance

This snapshot deliberately is not runnable through the frozen source launcher:
the source policy/catalog pins have not yet been regenerated for the whole final
implementation. The fixed compilation driver, expanded external manifest,
complete toolchain inventory and runner/observer matrix are unfinished. The
toolchain validator must still prove complete header/library/root coverage;
partial inventory rows cannot count as that proof. End-to-end collector/source/
parser and integrated transport counterexamples are still required. Linux ABI
and process behavior are unverified, and no three-recipe success is claimed.

Finish those pieces, update all held-source and AST pins coherently, commit a
concrete candidate, and obtain fresh independent exact-code/runner/ABI/observer
preflight before new native execution. Then perform actual Linux process and
three-cold-build/failure/native-scope matrices with owned residual closure and
fresh whole-checkpoint review. Continue Warp compilation definition and remaining
parent graph/warmup/composition work after this bounded checkpoint, until C4 is
complete. This work snapshot is not the endpoint of the active goal.
