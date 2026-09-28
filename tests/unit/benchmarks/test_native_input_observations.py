from types import SimpleNamespace

import numpy as np
import pytest

from optical_worker import native_input as native


def inputs():
    source = native.CanonicalNativeInput.from_arrays(
        case_id="asymmetric_synthetic_7x3_v1",
        materializer_id="cpu_observation_validator_unit_v1",
        arrays={
            "bvh_child_l": np.array([-1], dtype=np.int32),
            "triangle_v0": np.arange(6, dtype=np.float32).reshape(2, 3),
        },
        array_semantics={"bvh_child_l": "left-child topology", "triangle_v0": "world vertex zero"},
        topology_arrays=("bvh_child_l",),
        semantics={"purpose": "CPU observation comparison, not a runtime authority"},
    )
    prepared = SimpleNamespace(manifest=source.manifest, arrays=source.arrays, backend="warp")
    observations = [(name, value.copy()) for name, value in source.arrays.items()]
    return source, prepared, observations


def test_explicit_host_observations_share_the_formal_comparison_without_readback(monkeypatch):
    source, prepared, observations = inputs()
    reads = []

    def readback(owner, array):
        assert owner is prepared
        reads.append(array)
        return array.copy()

    monkeypatch.setattr(native, "_native_array_to_numpy", readback)
    ordinary = native.validate_prepared_native_input(source, prepared)
    assert len(reads) == 2
    monkeypatch.setattr(
        native, "_native_array_to_numpy", lambda *_: pytest.fail("unexpected implicit readback")
    )
    explicit = native._validate_native_input_host_observations(source, prepared, observations)
    assert ordinary == explicit


@pytest.mark.parametrize("mutation", ["missing", "extra", "wrong_name", "values", "dtype", "shape", "stride"])
def test_host_observation_comparison_rejects_incomplete_or_changed_arrays(mutation):
    source, prepared, observations = inputs()
    name, value = observations[1]
    if mutation == "missing":
        observations.clear()
    elif mutation == "extra":
        observations.append((name, value.copy()))
    elif mutation == "wrong_name":
        observations[1] = ("wrong", value)
    elif mutation == "values":
        value[0, 0] += 1
    elif mutation == "dtype":
        observations[1] = (name, value.astype(np.float64))
    elif mutation == "shape":
        observations[1] = (name, value.reshape(3, 2))
    else:
        observations[1] = (name, np.repeat(value, 2, axis=1)[:, ::2])
        assert not observations[1][1].flags.c_contiguous
    with pytest.raises(ValueError):
        native._validate_native_input_host_observations(source, prepared, observations)
