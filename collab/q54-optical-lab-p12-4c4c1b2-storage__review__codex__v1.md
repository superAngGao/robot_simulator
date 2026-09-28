Initiative: q54-optical-lab-p12-4c4c1b2-storage
Stage: independent code review
Author: codex
Version: v1
Date: 2026-09-28
Status: in_review
Reviewer session: `/root/c4c1b2_storage_review_v1`
Verdict: Request changes
Base: `c0046c4`
HEAD: `fe292cf40f0eefc2a6ae999185d360e9f8a6d8c0`
Related Files: `optics/warp_native_render.py`, targeted buffer and source-ownership tests
Owner Summary: Two reproducible P2 storage-preflight gaps remain. The targeted suite independently passed, but neither counterexample is covered.

## Scope and isolation

Reviewed only the b2 storage checkpoint: caller-owned physical buffer ABI, output ownership/retention, storage snapshots, disjoint intervals, preflight input/output overlap checks, and seven-entry migrated-kernel identity coverage.

Read `AGENTS.md`, `collab/README.md`, `collab/REVIEW_PRINCIPLES.md`, accepted design v1/v2/v3, and the implementation note. Inspected the actual seven-file storage delta and relevant installed Warp argument-packing implementation.

The reviewer did not author or edit implementation, spawn nested reviewers, execute any recorded bundle, or change gates.

Local checkout: `C:/Users/AG/新建文件夹/p12_4_c4c1b2_impl`
Remote checkout: `/data3/ang.gao/robot_sim_c4c1b2_review1`
Remote branch: `review/c4c1b2-storage`

Both checkouts reported the reviewed HEAD and clean tracked/untracked state. Remote state was rechecked after verification. Reproductions ran through stdin and created no scratch files.

## Findings

### P2 — Cached Warp launch descriptors can disagree with validated output storage

Location: `optics/warp_native_render.py:68`, particularly snapshot construction at lines 116–125 and comparison at lines 230–237.

Trigger:

```python
buffers = WarpDirectLightBuffers(**valid_arrays)
descriptor = buffers.origins.__ctype__()
descriptor.data = buffers.directions.ptr
buffers.validate_bound()  # Unexpectedly succeeds.
```

This is one mutation of the live array’s cached descriptor; the buffer signature and retention baseline remain untouched.

Observed on Warp 1.12.0:

```text
REPRO_CTYPE_ACCEPTED 22011707392 22011707904 22011707904
```

The values are respectively the validated `origins.ptr`, the actual `origins.__ctype__().data`, and `directions.ptr`.

Consequence: preflight accepts “disjoint” origins/directions while Warp’s actual array argument descriptor aliases them. The installed `warp/_src/types.py` implementation of `array.__ctype__()` returns cached `self.ctype` unchanged when present. Installed `warp/_src/context.py:6929` packs the argument using `value.__ctype__()`. Thus the mismatch is execution-relevant, not merely unused metadata.

No kernel execution was required or performed to establish that fact.

Existing checks miss it because they inspect Python-level pointer, shape, rank, strides, and ownership, but neither inspect nor validate an existing `value.ctype`. Existing drift tests mutate those Python-level attributes, not the cached descriptor.

Suggested direction: validate any existing cached descriptor against the canonical pointer/rank/shape/stride metadata during untimed inspection. Apply the same physical-descriptor consistency rule wherever read-only input intervals are checked. Do not silently repair inconsistent descriptors.

Required repair evidence:

- Reject cached descriptor pointer drift with the baseline untouched.
- Reject cached descriptor shape, rank, and stride drift.
- Accept legitimate descriptor materialization after construction when its contents match.
- Preserve allocation/launch/synchronization-free validation.
- Keep bundle execution closed.

Severity is P2 because this violates the storage/preflight guarantee at the current checkpoint. No currently enabled product bundle path was demonstrated.

### P2 — Read-only input overlap checks trust a stale contiguity flag

Location: `optics/warp_native_render.py:240–250`.

Trigger:

```python
root = wp.empty((100,), dtype=wp.float32, device=device)
values["intensity"] = root[10:15]
buffers = WarpDirectLightBuffers(**values)

scene_input = root[:2]
scene_input.strides = (40,)  # Single-point physical-layout drift.

buffers.validate_bound(read_only_inputs=(("scene", scene_input),))
# Unexpectedly succeeds.
```

Observed:

```text
REPRO_INPUT_STRIDE_ACCEPTED True (40,) True
```

These values establish that:

- `scene_input.is_contiguous` remains `True`;
- its actual stride is 40 bytes;
- input element 1 is exactly at `buffers.intensity.ptr`.

Consequence: preflight declares the input/output intervals disjoint using the assumed dense eight-byte interval, although the input’s actual second element aliases a mutable output. This contradicts the checkpoint’s claim that supplied scene/BVH inputs are checked for overlap.

Existing checks miss it because the read-only branch trusts `is_contiguous` and calculates `prod(shape) * itemsize` without independently checking strides, rank, or size. The output branch already acknowledges and handles this mutable-flag problem.

Suggested direction: share physical-layout validation between outputs and read-only inputs, separating native-output ownership requirements from common layout/descriptor checks. Reject inconsistent contiguous strides before calculating dense active intervals. This does not require introducing future prepared-input provenance or factory authority.

Required repair evidence:

- Reject the exact reproduction above before any recording/enqueue.
- Cover stale contiguity flags and malformed input rank/size/strides.
- Retain acceptance of valid disjoint native inputs and intended input-view ownership policies.
- Preserve validation’s no-runtime-work behavior.

## Verification

Environment observed independently:

```text
Python 3.12.13
Warp 1.12.0
CUDA devices:
  cuda:0 NVIDIA H200 NVL
  cuda:1 NVIDIA A100-SXM4-40GB
  cuda:2 NVIDIA A100-SXM4-40GB
```

Used the designated fallback through:

```powershell
ssh -i C:/Users/AG/.ssh/tileops_ed25519 -o BatchMode=yes ang.gao@192.168.80.101
```

Targeted execution in the clean remote checkout:

```sh
/data3/ang.gao/anaconda3/bin/conda run -n robot_sim python -m pytest -q \
  tests/gpu/test_warp_render_buffers.py \
  tests/unit/optics/test_warp_render_buffer_contract.py \
  tests/unit/optics/test_warp_native_kernel_source.py
```

Result: **41 passed in 3.02s**. This was a new reviewer execution, not the author’s prior result.

Static checks:

```sh
git diff --check c0046c4 HEAD

/data3/ang.gao/anaconda3/bin/conda run -n robot_sim python -m ruff check \
  optics/warp_native_render.py \
  tests/gpu/test_warp_render_buffers.py \
  tests/unit/optics/test_warp_render_buffer_contract.py \
  tests/unit/optics/test_warp_native_kernel_source.py
```

Results: diff check clean; Ruff “All checks passed!”

The complete adversarial script was piped through SSH into:

```sh
cd /data3/ang.gao/robot_sim_c4c1b2_review1 &&
/data3/ang.gao/anaconda3/bin/conda run --no-capture-output -n robot_sim python -
```

Script:

```python
import runpy
import sys
import warp as wp
from optics.warp_native_render import WarpDirectLightBuffers

wp.init()
m = runpy.run_path("tests/gpu/test_warp_render_buffers.py")

values = m["_kwargs"](wp)
buffers = WarpDirectLightBuffers(**values)
descriptor = buffers.origins.__ctype__()
original_data = descriptor.data
try:
    descriptor.data = buffers.directions.ptr
    buffers.validate_bound()
    print(
        "REPRO_CTYPE_ACCEPTED",
        buffers.origins.ptr,
        buffers.origins.__ctype__().data,
        buffers.directions.ptr,
    )
finally:
    descriptor.data = original_data

values = m["_kwargs"](wp)
root = wp.empty(
    (100,), dtype=wp.float32, device=values["origins"].device
)
values["intensity"] = root[10:15]
buffers = WarpDirectLightBuffers(**values)
scene_input = root[:2]
original_strides = scene_input.strides
try:
    scene_input.strides = (40,)
    buffers.validate_bound(read_only_inputs=(("scene", scene_input),))
    print(
        "REPRO_INPUT_STRIDE_ACCEPTED",
        scene_input.is_contiguous,
        scene_input.strides,
        scene_input.ptr + scene_input.strides[0]
        == buffers.intensity.ptr,
    )
finally:
    scene_input.strides = original_strides

buffers.validate_bound()
print("ENV", sys.version, wp.__version__, wp.get_cuda_device_count())
```

Both mutations were restored before teardown; the reproduction process exited successfully.

High-risk areas otherwise inspected include exact ABI ordering, 14 primary arrays and zero/nine diagnostic arrays, zero-ray storage, strict count/mode types, output identity/order, root/view retention chains, disjoint shared-root slices, overlap rejection, frozen assignment/deletion, baseline reopening, CPU/wrong-device rejection, and all seven migrated kernel object identities.

## Limitations and remaining boundaries

- No original `00f1790`/`f771e61` history verification is claimed. The reviewed storage delta starts from the recovered `c0046c4` snapshot.
- The parent’s separate full non-slow suite is not reviewer-run evidence in this report.
- Normal Warp runtime caches were present; targeted tests exercised fresh allocations and current code. No cached prior test report was reused.
- Factory, evidence, command recording, reset/shade implementation, bundle execution, lifecycle containment, and numerical oracle parity are outside this checkpoint.
- No performance claim or benchmark/result/baseline artifact was produced.

Request changes applies only to this storage checkpoint. It does not reopen prior approvals or imply b2 completion. C4c1c, C4c2–C4c4, C4d/B3, product GPU preparation, a3b/results/baselines, host payload, and shipping paths remain closed.
