"""Exact production readiness authority shared by Warp and CUDA lanes."""

from __future__ import annotations

from dataclasses import dataclass, replace

import pytest

from optical_worker.native_contract import (
    NATIVE_INPUT_SCHEMA_VERSION,
    CudaPreparedNativeInput,
    NativeReadinessEvidence,
    WarpPreparedNativeInput,
    assert_cuda_prepared_authority,
    assert_warp_prepared_authority,
    register_native_readiness,
    register_prepared_native_input,
)


@dataclass
class _Manifest:
    sha256: str = "a" * 64
    content_sha256: str = "b" * 64
    topology_sha256: str = "c" * 64
    input_contract: str | None = None
    scene_contract_schema_version: str | None = None
    scene_contract_sha256: str | None = None
    scene_id: str | None = None
    transform_replay_sha256: str | None = None
    transform_frame_count: int = 0
    fixed_render_fixture_sha256: str | None = None


@dataclass
class _Source:
    manifest: _Manifest


def _pair(backend: str):
    source = _Source(_Manifest())
    prepared_type = WarpPreparedNativeInput if backend == "warp" else CudaPreparedNativeInput
    prepared = prepared_type(source.manifest, 1, {"scene": object()}, object(), source)
    readiness = NativeReadinessEvidence(
        schema_version=NATIVE_INPUT_SCHEMA_VERSION,
        backend=backend,
        device_ordinal=1,
        manifest_sha256=source.manifest.sha256,
        content_sha256=source.manifest.content_sha256,
        topology_sha256=source.manifest.topology_sha256,
        input_contract=None,
        scene_contract_schema_version=None,
        scene_contract_sha256=None,
        scene_id=None,
        transform_replay_sha256=None,
        transform_frame_count=0,
        fixed_render_fixture_sha256=None,
        producer_ready_event_complete=True,
        direct_validation_complete=True,
        allocation_complete=True,
        compile_complete=True,
        warmup_frames_completed=1,
    )
    register_prepared_native_input(prepared, source)
    register_native_readiness(readiness, prepared)
    return prepared, readiness


@pytest.mark.parametrize(
    ("backend", "check"),
    (("warp", assert_warp_prepared_authority), ("cuda", assert_cuda_prepared_authority)),
)
def test_exact_registered_readiness_and_prepared_identity(backend, check):
    prepared, readiness = _pair(backend)
    check(prepared, readiness)
    assert NativeReadinessEvidence.from_payload(readiness.to_payload()).sha256 == readiness.sha256

    with pytest.raises((TypeError, ValueError)):
        check(prepared, NativeReadinessEvidence.from_payload(readiness.to_payload()))
    with pytest.raises((TypeError, ValueError)):
        check(replace(prepared), readiness)
    with pytest.raises((TypeError, ValueError)):
        check(prepared, replace(readiness, content_sha256="d" * 64))

    class ReadinessSubclass(NativeReadinessEvidence):
        pass

    with pytest.raises(TypeError):
        check(prepared, ReadinessSubclass(**readiness.to_payload()))

    SameName = type(
        "NativeReadinessEvidence",
        (),
        {"__module__": "optical_worker.native_contract", "to_payload": readiness.to_payload},
    )
    with pytest.raises(TypeError):
        check(prepared, SameName())


def test_benchmark_reexports_exact_production_readiness():
    from benchmarks.optical import native

    assert native.NativeReadinessEvidence is NativeReadinessEvidence
    assert native.NATIVE_INPUT_SCHEMA_VERSION is NATIVE_INPUT_SCHEMA_VERSION


def test_registered_readiness_cannot_cross_backends_or_prepared_owners():
    warp_prepared, warp_readiness = _pair("warp")
    cuda_prepared, cuda_readiness = _pair("cuda")

    with pytest.raises(TypeError):
        assert_cuda_prepared_authority(warp_prepared, warp_readiness)
    with pytest.raises(TypeError):
        assert_warp_prepared_authority(cuda_prepared, cuda_readiness)
    with pytest.raises(ValueError):
        assert_cuda_prepared_authority(cuda_prepared, warp_readiness)
    with pytest.raises(ValueError):
        assert_warp_prepared_authority(warp_prepared, cuda_readiness)


@pytest.mark.parametrize(
    ("backend", "check"),
    (("warp", assert_warp_prepared_authority), ("cuda", assert_cuda_prepared_authority)),
)
def test_replacing_registered_storage_or_event_is_rejected(backend, check):
    prepared, readiness = _pair(backend)
    prepared.arrays["scene"] = object()
    with pytest.raises(ValueError, match="storage"):
        check(prepared, readiness)

    prepared, readiness = _pair(backend)
    object.__setattr__(prepared, "ready_event", object())
    with pytest.raises(ValueError, match="storage"):
        check(prepared, readiness)


@pytest.mark.parametrize("backend", ("warp", "cuda"))
def test_registration_cannot_rebaseline_a_mutated_input(backend):
    prepared, readiness = _pair(backend)
    prepared.arrays["scene"] = object()
    with pytest.raises(ValueError, match="registered twice"):
        register_prepared_native_input(prepared, prepared.source)
    with pytest.raises(ValueError, match="registered twice"):
        register_native_readiness(readiness, prepared)


@pytest.mark.parametrize(
    ("backend", "check"),
    (("warp", assert_warp_prepared_authority), ("cuda", assert_cuda_prepared_authority)),
)
def test_readiness_registration_does_not_certify_unregistered_test_input(backend, check):
    prepared, readiness = _pair(backend)
    unregistered_prepared = replace(prepared)
    unregistered_readiness = replace(readiness)
    register_native_readiness(unregistered_readiness, unregistered_prepared)
    with pytest.raises(ValueError, match="canonical a2 preparation"):
        check(unregistered_prepared, unregistered_readiness)
