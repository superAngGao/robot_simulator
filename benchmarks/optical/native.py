"""Compatibility aliases; implementation lives in optical_worker.native_input."""

from optical_worker.native_input import (
    _FIXED_FIXTURE_SEMANTICS as _FIXED_FIXTURE_SEMANTICS,
)
from optical_worker.native_input import (
    _FIXED_FIXTURE_TOPOLOGY_ARRAYS as _FIXED_FIXTURE_TOPOLOGY_ARRAYS,
)
from optical_worker.native_input import (
    _REPLAY_SEMANTICS as _REPLAY_SEMANTICS,
)
from optical_worker.native_input import (
    CANONICAL_NATIVE_REPLAY_INPUT as CANONICAL_NATIVE_REPLAY_INPUT,
)
from optical_worker.native_input import (
    FIXED_NATIVE_SCENE_INPUT as FIXED_NATIVE_SCENE_INPUT,
)
from optical_worker.native_input import (
    FIXED_RENDER_FIXTURE_SCHEMA_VERSION as FIXED_RENDER_FIXTURE_SCHEMA_VERSION,
)
from optical_worker.native_input import (
    GPU_ENGINE_PUBLISHED_FRAME_INPUT as GPU_ENGINE_PUBLISHED_FRAME_INPUT,
)
from optical_worker.native_input import (
    NATIVE_INPUT_SCHEMA_VERSION as NATIVE_INPUT_SCHEMA_VERSION,
)
from optical_worker.native_input import (
    SCENE_CONTRACT_SCHEMA_VERSION as SCENE_CONTRACT_SCHEMA_VERSION,
)
from optical_worker.native_input import (
    TRANSFORM_REPLAY_SCHEMA_VERSION as TRANSFORM_REPLAY_SCHEMA_VERSION,
)
from optical_worker.native_input import (
    CanonicalFixedRenderFixtureHost as CanonicalFixedRenderFixtureHost,
)
from optical_worker.native_input import (
    CanonicalNativeInput as CanonicalNativeInput,
)
from optical_worker.native_input import (
    CanonicalTransformReplayHost as CanonicalTransformReplayHost,
)
from optical_worker.native_input import (
    CudaPreparedNativeInput as CudaPreparedNativeInput,
)
from optical_worker.native_input import (
    NativeArrayManifest as NativeArrayManifest,
)
from optical_worker.native_input import (
    NativeBackend as NativeBackend,
)
from optical_worker.native_input import (
    NativeInputManifest as NativeInputManifest,
)
from optical_worker.native_input import (
    NativeInputValidationReport as NativeInputValidationReport,
)
from optical_worker.native_input import (
    NativeReadinessEvidence as NativeReadinessEvidence,
)
from optical_worker.native_input import (
    NativeStorageSignature as NativeStorageSignature,
)
from optical_worker.native_input import (
    PackedOpticalSceneHost as PackedOpticalSceneHost,
)
from optical_worker.native_input import (
    PreparedFixedRenderInputs as PreparedFixedRenderInputs,
)
from optical_worker.native_input import (
    PreparedNativeInput as PreparedNativeInput,
)
from optical_worker.native_input import (
    PreparedOpticalLocalScene as PreparedOpticalLocalScene,
)
from optical_worker.native_input import (
    WarpPreparedNativeInput as WarpPreparedNativeInput,
)
from optical_worker.native_input import (
    _array_content_sha256 as _array_content_sha256,
)
from optical_worker.native_input import (
    _array_manifest as _array_manifest,
)
from optical_worker.native_input import (
    _array_manifest_payload as _array_manifest_payload,
)
from optical_worker.native_input import (
    _canonical_json_bytes as _canonical_json_bytes,
)
from optical_worker.native_input import (
    _derive_fixed_fixture_arrays as _derive_fixed_fixture_arrays,
)
from optical_worker.native_input import (
    _fixed_instance_transform as _fixed_instance_transform,
)
from optical_worker.native_input import (
    _freeze_array_mapping as _freeze_array_mapping,
)
from optical_worker.native_input import (
    _insert_disjoint_arrays as _insert_disjoint_arrays,
)
from optical_worker.native_input import (
    _native_array_to_numpy as _native_array_to_numpy,
)
from optical_worker.native_input import (
    _scene_topology_source_sha256 as _scene_topology_source_sha256,
)
from optical_worker.native_input import (
    _sha256_payload as _sha256_payload,
)
from optical_worker.native_input import (
    _sha256_text as _sha256_text,
)
from optical_worker.native_input import (
    _validate_native_input_host_observations as _validate_native_input_host_observations,
)
from optical_worker.native_input import (
    _validate_packed_scene_case as _validate_packed_scene_case,
)
from optical_worker.native_input import (
    _warp_dtype as _warp_dtype,
)
from optical_worker.native_input import (
    assert_cuda_arena_a2_authority as assert_cuda_arena_a2_authority,
)
from optical_worker.native_input import (
    assert_cuda_prepared_authority as assert_cuda_prepared_authority,
)
from optical_worker.native_input import (
    assert_independent_native_inputs as assert_independent_native_inputs,
)
from optical_worker.native_input import (
    assert_operation_allowed as assert_operation_allowed,
)
from optical_worker.native_input import (
    assert_outside_timed_boundary as assert_outside_timed_boundary,
)
from optical_worker.native_input import (
    build_host_median_bvh as build_host_median_bvh,
)
from optical_worker.native_input import (
    build_prepared_cuda_lbvh_rebuild_factory as build_prepared_cuda_lbvh_rebuild_factory,
)
from optical_worker.native_input import (
    build_prepared_cuda_local_scene as build_prepared_cuda_local_scene,
)
from optical_worker.native_input import (
    build_prepared_cuda_scene_update_factory as build_prepared_cuda_scene_update_factory,
)
from optical_worker.native_input import (
    build_prepared_cuda_transform_inputs as build_prepared_cuda_transform_inputs,
)
from optical_worker.native_input import (
    build_prepared_warp_bvh_refit_factory as build_prepared_warp_bvh_refit_factory,
)
from optical_worker.native_input import (
    build_prepared_warp_local_scene as build_prepared_warp_local_scene,
)
from optical_worker.native_input import (
    build_prepared_warp_scene_update_factory as build_prepared_warp_scene_update_factory,
)
from optical_worker.native_input import (
    build_prepared_warp_transform_inputs as build_prepared_warp_transform_inputs,
)
from optical_worker.native_input import (
    certify_native_readiness as certify_native_readiness,
)
from optical_worker.native_input import (
    frozen_workload_content as frozen_workload_content,
)
from optical_worker.native_input import (
    get_case as get_case,
)
from optical_worker.native_input import (
    prepare_cuda_native_input as prepare_cuda_native_input,
)
from optical_worker.native_input import (
    prepare_warp_native_input as prepare_warp_native_input,
)
from optical_worker.native_input import (
    prepared_storage_signatures as prepared_storage_signatures,
)
from optical_worker.native_input import (
    prepared_transform_frame_signatures as prepared_transform_frame_signatures,
)
from optical_worker.native_input import (
    register_native_readiness as register_native_readiness,
)
from optical_worker.native_input import (
    register_prepared_native_input as register_prepared_native_input,
)
from optical_worker.native_input import (
    validate_prepared_native_input as validate_prepared_native_input,
)
from optical_worker.native_input import (
    warp_dtype_from_host_dtype as warp_dtype_from_host_dtype,
)
from optical_worker.native_input import (
    wrap_prepared_fixed_render_inputs as wrap_prepared_fixed_render_inputs,
)
from optical_worker.native_input import (
    wrap_prepared_optical_local_scene as wrap_prepared_optical_local_scene,
)
