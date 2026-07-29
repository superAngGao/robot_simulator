# Q54 P12.5a Implementation Note: Device Channel Boundary

Author: Codex
Status: ready for review
Related design:
`collab/q54-optical-lab-p12-5-system-boundary-consolidation__design-request__codex__v2.md`

## Owner Summary

This implements the first narrow P12.5a boundary-foundation slice:
`optics/device_channel.py`.

The goal is to give optical result-channel materialization a single boundary
component before P12.3 CUDA direct-light returns native Torch CUDA tensors.
Existing staging and async readback code now call this boundary instead of each
consumer deciding whether a channel is a Warp array, Torch tensor, or host array.

This is not a new render backend. It does not implement CUDA direct-light,
backend compatibility tables, public backend overrides, or documentation
consolidation.

## What Changed

### `optics/device_channel.py`

New boundary helpers:

- `channel_is_device(value)`
- `channel_to_torch(value)`
- `channel_to_numpy(value)`
- `stage_channels_to_host(result, channels)`

Supported inputs:

- Torch tensors, including CUDA tensors;
- Warp arrays, via optional `warp.to_torch(...)`;
- host arrays and array-like values.

The helper keeps Torch and Warp optional. Importing `optics.device_channel` does
not require either package to be installed.

### `optics/device.py`

The existing device-result staging path now uses `channel_to_numpy(...)` instead
of a local `_channel_to_numpy(...)` helper.

Canonical dtype behavior remains owned by `optics/device.py`:

- `hit_mask` -> `bool`
- optical float channels -> `float32`
- source id/order channels -> `int64`

`optics/device_channel.py` only materializes host/device values; it does not own
the optical channel schema.

### `tools/optical_pipeline_lab/async_readback.py`

`_torch_device_tensors_for_channels(...)` now uses `channel_to_torch(...)`
instead of calling `wp.to_torch(...)` directly.

The current dependency probe remains conservative:

- async readback still requires Torch + Warp through `_require_torch_and_warp()`;
- the materialization step itself is now ready for native Torch CUDA channels.

This preserves current behavior while removing the direct Warp conversion from
the readback consumer.

### `optics/__init__.py`

Exports the new boundary helpers:

- `channel_is_device`
- `channel_to_numpy`
- `channel_to_torch`
- `stage_channels_to_host`

### `MANIFEST.md`

Registers the new `optics/device_channel.py` module.

## Tests Added

`tests/unit/optics/test_device_optical.py`

- host array/list staging through `stage_channels_to_host(...)`;
- Torch tensor conversion path, skipped automatically if Torch is unavailable.

`tests/unit/optics/test_optical_pipeline_lab.py`

- async readback materialization calls the device-channel boundary and preserves
  the existing contiguous-device-tensor requirement.

## Validation

Commands run:

```bash
PYTHONPATH=. ruff check optics/__init__.py optics/device.py optics/device_channel.py \
  tools/optical_pipeline_lab/async_readback.py \
  tests/unit/optics/test_device_optical.py \
  tests/unit/optics/test_optical_pipeline_lab.py

PYTHONPATH=. pytest -q \
  tests/unit/optics/test_device_optical.py \
  tests/unit/optics/test_optical_pipeline_lab.py
```

Result:

- ruff: clean
- pytest: 202 passed

## Review Questions

1. Is `optics/device_channel.py` the right ownership boundary for result-channel
   materialization, or should these helpers remain inside `optics/device.py`
   until CUDA direct-light lands?
2. Should async readback keep requiring Warp for the availability probe in this
   slice, or should P12.5a immediately split "Torch-native async readback" from
   "Warp-backed async readback"?
3. Is `stage_channels_to_host(...)` useful as a schema-neutral helper, or should
   all staging continue to go through the canonicalized `optics/device.py`
   functions?
