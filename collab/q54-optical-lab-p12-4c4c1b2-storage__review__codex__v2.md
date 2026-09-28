Initiative: q54-optical-lab-p12-4c4c1b2-storage
Stage: independent code review
Author: codex
Version: v2
Date: 2026-09-28
Status: accepted
Related Files: optics/warp_native_render.py; tests/gpu/test_warp_render_buffers.py; tests/unit/optics/test_warp_render_buffer_contract.py; tests/unit/optics/test_warp_native_kernel_source.py
Owner Summary: Approve the exact storage-only checkpoint. Both prior P2 counterexamples independently fail closed; no substantive findings remain in this reviewed scope.

Reviewer session: `/root/c4c1b2_storage_review_v2`
Verdict: **Approve**
Overall storage base: `c0046c4`
Prior candidate / repair base: `fe292cf40f0eefc2a6ae999185d360e9f8a6d8c0`
Reviewed HEAD: `aa97f093dd90313f5c2daee710b99e056abf05da`

## Scope and isolation

This fresh independent session reviewed only the caller-owned physical buffer ABI, storage ownership/retention, frozen snapshots, output disjointness, read-only input overlap preflight, and seven migrated kernel source/identity tests. It is not approval of full C4c1b2 or full C4c1.

Read AGENTS.md, collab/README.md, collab/REVIEW_PRINCIPLES.md, accepted design requests v1/v2/v3, the storage implementation note, and the complete prior review and repair follow-up. Inspected the actual overall storage delta and repair delta, implementation call/data flow, focused tests, and the installed Warp descriptor/argument-packing code. Author reports were not used as proof of runtime success.

Local checkout: `C:/Users/AG/新建文件夹/p12_4_c4c1b2_impl`, branch `implementation/c4c1b2`.
Remote isolated checkout: `/data3/ang.gao/robot_sim_c4c1b2_review2`, branch `review/c4c1b2-storage-v2`.

Both checkouts independently reported the exact reviewed HEAD and clean tracked/untracked state before and after verification. No implementation edits, nested reviewers, recorded commands, recorded-bundle execution, performance runs, or activation changes were made. Independent scratch checks ran through stdin, without creating scratch files in either checkout. This report is the only reviewer-created evidence file and is outside the candidate.

## Findings and prior finding closure

No substantive findings remain within the exact reviewed scope.

1. **Prior P2: cached Warp descriptor drift — closed.** The shared layout checker inspects an existing exact installed-runtime `array_t`, comparing data/grad pointers, rank, every shape slot, and every stride slot against the current array metadata. It rejects an instance-level `__ctype__` override. It neither invokes `__ctype__` nor repairs the descriptor. The original mutation of `origins.__ctype__().data` to `directions.ptr` now raises `ValueError: origins: cached launch descriptor disagrees with storage`. The bad descriptor remains unchanged on rejection; restoring it permits validation again.

2. **Prior P2: stale input contiguity flag — closed.** Outputs and supplied read-only inputs now call the same physical-layout checker before dense intervals are derived. The exact original `root[:2]` input with `strides=(40,)`, stale `is_contiguous=True`, and its second element at the intensity output pointer now raises `ValueError: scene: inconsistent contiguous strides/rank`. Restoration permits validation again. Rank/size drift, valid disjoint input views, and the intended non-owning read-only input policy are covered by the executed tests.

The installed Warp 1.12.0 source confirms that `array.__ctype__()` returns its cached `self.ctype` when present. Installed `warp/_src/context.py:6929` packs an array using `value.__ctype__()`. The repair therefore checks execution-relevant state, not an unrelated label or mirrored schema.

## Independent verification

Connection used:

```powershell
ssh -i C:/Users/AG/.ssh/tileops_ed25519 -o BatchMode=yes ang.gao@192.168.80.101
```

The packet designated this fallback because the primary host was unreachable; this reviewer did not independently retest the primary transport.

Observed environment:

```text
Python 3.12.13
Warp 1.12.0
Torch 2.9.1+cu128; torch.version.cuda = 12.8
robot_sim NVCC: release 12.8, V12.8.93
Warp initialization banner: CUDA Toolkit 12.9, Driver 13.3
cuda:0 NVIDIA H200 NVL
cuda:1 NVIDIA A100-SXM4-40GB
cuda:2 NVIDIA A100-SXM4-40GB
```

All Python/test/lint commands used `/data3/ang.gao/anaconda3/bin/conda run -n robot_sim`. The standalone remote shell's ambient `nvcc` instead reports 11.5.119; the reported 12.8.93 above was separately verified with `conda run -n robot_sim nvcc --version`. No NVCC compilation was requested by this storage review.

From the isolated remote checkout:

```sh
/data3/ang.gao/anaconda3/bin/conda run -n robot_sim python -m pytest -q \
  tests/gpu/test_warp_render_buffers.py \
  tests/unit/optics/test_warp_render_buffer_contract.py \
  tests/unit/optics/test_warp_native_kernel_source.py
```

Observed result: **55 passed in 3.08s**, with no skipped tests. This was a new reviewer execution on the reviewed HEAD.

```sh
git diff --check c0046c4 HEAD
/data3/ang.gao/anaconda3/bin/conda run -n robot_sim python -m ruff check \
  optics/warp_native_render.py tests/gpu/test_warp_render_buffers.py \
  tests/unit/optics/test_warp_render_buffer_contract.py \
  tests/unit/optics/test_warp_native_kernel_source.py
/data3/ang.gao/anaconda3/bin/conda run -n robot_sim python -m ruff format --check \
  optics/warp_native_render.py tests/gpu/test_warp_render_buffers.py \
  tests/unit/optics/test_warp_render_buffer_contract.py \
  tests/unit/optics/test_warp_native_kernel_source.py
```

Observed results: diff check clean; Ruff `All checks passed!`; format check `4 files already formatted`. The local diff check also passed.

An additional independent stdin script ran through:

```sh
cd /data3/ang.gao/robot_sim_c4c1b2_review2 &&
/data3/ang.gao/anaconda3/bin/conda run --no-capture-output -n robot_sim python -
```

Its original-counterexample checks used the following sequence (the same test allocation helper supplies valid arrays):

```python
import runpy
import warp as wp
from optics.warp_native_render import WarpDirectLightBuffers

wp.init()
m = runpy.run_path("tests/gpu/test_warp_render_buffers.py")
buffers = WarpDirectLightBuffers(**m["_kwargs"](wp))
descriptor = buffers.origins.__ctype__()
original_data = descriptor.data
try:
    descriptor.data = buffers.directions.ptr
    try:
        buffers.validate_bound()
    except ValueError as exc:
        print("ORIGINAL_CTYPE_REJECTED", str(exc))
    else:
        raise AssertionError("original cached descriptor counterexample accepted")
    assert descriptor.data == buffers.directions.ptr
finally:
    descriptor.data = original_data
buffers.validate_bound()

values = m["_kwargs"](wp)
root = wp.empty((100,), dtype=wp.float32, device=values["origins"].device)
values["intensity"] = root[10:15]
buffers = WarpDirectLightBuffers(**values)
scene_input = root[:2]
original_strides = scene_input.strides
try:
    scene_input.strides = (40,)
    assert scene_input.is_contiguous
    assert scene_input.ptr + scene_input.strides[0] == buffers.intensity.ptr
    try:
        buffers.validate_bound(read_only_inputs=(("scene", scene_input),))
    except ValueError as exc:
        print("ORIGINAL_STRIDE40_REJECTED", str(exc))
    else:
        raise AssertionError("original stale flag counterexample accepted")
finally:
    scene_input.strides = original_strides
buffers.validate_bound(read_only_inputs=(("scene", scene_input),))
```

Observed output:

```text
ORIGINAL_CTYPE_REJECTED origins: cached launch descriptor disagrees with storage
ORIGINAL_STRIDE40_REJECTED scene: inconsistent contiguous strides/rank
```

The additional script also independently verified:

- On both cuda:0 and cuda:1, all combinations of ray counts 0/1/5 and diagnostics false/true construct and validate while class-level `wp.array.__ctype__` is patched to raise. Every array's cache remains `None`.
- Explicit legitimate descriptor materialization afterward is accepted for all those combinations, including zero-sized arrays. Revalidation still succeeds with `__ctype__` patched to raise, proving inspection does not recreate the descriptor.
- Mutating otherwise-unused trailing shape or stride slot 3 is rejected for both an output and a supplied input. The mutation remains intact until explicitly restored.
- Replacing either role's cache with an arbitrary object is rejected; restoration permits validation.

The script finished with `INDEPENDENT_REPRO_AND_EXTRA_CHECKS_PASS`. An initial version of this reviewer-only script had a Python tuple-parenthesis syntax error before execution; it was corrected and rerun successfully. That tooling error is not a candidate failure.

The focused suite separately exercises allocation/copy/launch/init/device-lookup/synchronization entrypoints patched to fail during validation and the data/grad/shape/strides/rank cached-descriptor mutation matrix for both roles. Static inspection confirms validation contains no command recording, descriptor materialization, array allocation, transfer, or enqueue call.

Other inspected invariants: exact 14-primary/zero-or-nine-overlay ordering; strict ray-count/mode types; profile-independent storage; empty/single/multiple rays; pairwise active-interval disjointness; legitimate shared-root slices; retained root/view chains; non-owning output rejection; frozen assignment/deletion and baseline re-entry rejection; object and physical-metadata drift; CPU/wrong-device rejection; and no execution/lifecycle API added by the wrapper.

All seven migrated kernel names were checked for exact shared object identity and absence of definitions in `warp_execution.py`. A production-tree source search found their runtime definitions only in `warp_native_render.py`; the same module also intentionally supplies CPU-only unavailable-runtime stubs. Compatibility call sites import/use those same names. No kernel mathematics changed in the reviewed storage delta.

## Limits and remaining gates

- Approval binds only `aa97f093dd90313f5c2daee710b99e056abf05da` and this storage/source-test scope. The parent full non-slow suite is separate evidence, not a reviewer-run result in this report.
- Original `00f1790`/`f771e61` Git history remains unavailable. This report reviews from recovered `c0046c4` and does not represent it as original ancestry.
- The real GPU checks use fresh allocations and current code. Normal Warp caches were present; no old test output was reused as proof. No recorded render work was executed.
- The checker deliberately depends on the installed Warp `array_t` implementation. This approval does not certify untested Warp versions or arbitrary hostile rewriting of runtime classes and all baselines.
- Read-only input provenance/retention authority remains the future factory's responsibility; allowing a valid non-owning input view here does not certify that authority.
- Deterministic reset, four fixed shade ABIs, camera binding, evidence, factory, recorded bundle, contained execution/lifecycle, numerical oracle parity, and full b2 remain pending and unapproved by this report.
- C4c1c onward, C4c2-C4c4, C4d/B3, product worker GPU preparation, a3b, result activation, baselines, host payload, and shipping paths remain closed. No performance or product-result claim is made.

No blocking finding or required evidence gap remains for the bounded storage checkpoint. No additional non-blocking review comment is required.
