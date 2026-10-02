# C2c2: capture the asset bytes actually consumed by the importer

Initiative: q54-optical-lab-p12-4c4c2c-asset-custody
Stage: implementation-note
Author: codex
Version: v1
Date: 2026-10-03
Status: in_review
Related Files: runtime-integration design v3 section3 at674af14; C2c1 approval4cac1be and evidence6b882f6
Owner Summary: Close preflight-to-parser byte drift for canonical Go2 CPU materialization. No worker/GPU/result admission and no new frozen workload digest.

Base:6b882f6. This is the asset-custody portion of approved C2c, not C2c/C4 completion.

## Actual behavior and trust boundary

- `capture_case_asset` performs its own full canonical Git/remote/inventory/content
  preflight. A supplied earlier exact-type evidence object is only a consistency
  constraint; it cannot substitute for that fresh preflight.
- Capture rereads each file into immutable bytes and checks actual byte length
  and SHA-256 against the freshly validated record. It checks manifest bytes too,
  then repeats the existing full preflight and compares evidence before returning.
  The mapping is a read-only view of a private dictionary, with immutable values.
- Each capture read rejects symlinks observed in path components, checks root
  containment, uses no-follow/nonblocking open flags when the OS provides them,
  and checks regular-file status and descriptor identity before/after reading.
  Path identity must still agree. Windows path stat versus fstat ctime semantics
  differ; ctime is compared on the same descriptor, while dev/inode/size/mtime
  are compared across descriptor/path. Content identity always uses SHA-256.
- Canonical Go2 scene construction calls capture itself, then passes these bytes
  to the sole existing importer. Caller-supplied hash labels never bypass capture.
- The importer adds an optional byte-input mode. It copies mapping structure and
  requires normalized relative names and immutable bytes. XML and OBJ reads use
  memory streams; missing/escaping references reject rather than reopening disk.
  Captured Go2 meshes use the frozen OBJ format, process=False and an explicit
  empty resolver, preventing MTL/texture/filesystem/network resolver fallback.
- The general path-based importer remains a compatibility API, not a canonical
  worker admission path. All material, geometry, transform and lighting math
  remains shared. CPU byte records are descriptive input, not unforgeable Python
  capabilities, GPU readiness or execution authority. The approved closed trusted
  worker model remains unchanged; this is not a hostile-filesystem sandbox.

Capture is the read boundary: changed bytes between preflight and capture fail.
Changes after successful capture cannot change the already owned parser input;
the snapshot does not certify future disk state. All bytes, including unused
manifest-listed files, must match before any parse or GPU preparation. No new
cache, network fetch, dependency install or surrogate Go2 is added.

## Validation and coverage obligations

Local Python3.14 tests: custody mechanics17passed,2skipped (real trimesh and
POSIX-only checks require robot_sim); ownership/camera88passed. Ruff passed.
An initial local run exposed fstat/path-stat ctime disagreement on Windows; the
comparison was corrected without dropping descriptor-change or content checks.

The historical C2c1 AST baseline is unchanged. Tests project only exact new IO
statements/signature back to the original form; every other mathematical AST
node still matches the historical hash. They do not regenerate hashes from new
math or blanket-remove imports. New tests exercise actual capture entrypoints,
byte/manifest/inventory drift, forged/stale evidence, immutable ownership,
missing/escaping references, explicit resolver and real-parser no-reopen behavior.
The POSIX FIFO check runs in a bounded subprocess so a regression cannot hang
the test process indefinitely. Tiny Git fixtures test mechanics only, never
canonical Go2 correctness. Remote acceptance must run without fixture patches.

Before scoped approval: run these tests in robot_sim with zero mandatory skips;
recompute both real Go2 scene/fixture digests and 1080p replay; compare actual
legacy and captured importer bounds/geometry; check import closure and all
affected benchmark/compatibility tests. Record failures distinctly from environment
limitations, independently review the frozen commit, and check owned processes
after every round. Primary183 remains unreachable at this check;101 is reachable.
The existing dirty common worktree on101 is preserved; tests use a new detached
worktree, not a reset/checkout of that directory.

## Gap scan and remaining gates

This does not supply the resolved120-camera artifacts, bound native scalars,
v13 row/schema/binding changes or final worker source allowlists. Those remain
the next C2c requirements. C2d/e/f, R2/S/host/depth-two, C4d/B3, product GPU
preparation/submission, a3b, results/baselines and C4 completion remain closed.
No measured transaction or new performance threshold is introduced. The byte
capture/importer is durable CPU preparation; no second importer or temporary
copy of mathematical kernels exists to maintain or later delete.
