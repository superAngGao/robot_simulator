"""Compatibility aliases; implementation lives in optical_worker.provenance."""

from optical_worker.provenance import (
    _CUDA_ARENA_C3_BINDING_AUTHORITIES as _CUDA_ARENA_C3_BINDING_AUTHORITIES,
)
from optical_worker.provenance import (
    _SUPPORTED_OUTPUT_PROFILES as _SUPPORTED_OUTPUT_PROFILES,
)
from optical_worker.provenance import (
    ACCELERATION_RECEIPT_SCHEMA_VERSION as ACCELERATION_RECEIPT_SCHEMA_VERSION,
)
from optical_worker.provenance import (
    DYNAMIC_ACCELERATION_INPUT_NAMES as DYNAMIC_ACCELERATION_INPUT_NAMES,
)
from optical_worker.provenance import (
    DYNAMIC_ACCELERATION_OUTPUT_NAMES as DYNAMIC_ACCELERATION_OUTPUT_NAMES,
)
from optical_worker.provenance import (
    DYNAMIC_DATAFLOW_CONTRACT_SCHEMA_VERSION as DYNAMIC_DATAFLOW_CONTRACT_SCHEMA_VERSION,
)
from optical_worker.provenance import (
    DYNAMIC_RENDER_METADATA_INPUT_NAMES as DYNAMIC_RENDER_METADATA_INPUT_NAMES,
)
from optical_worker.provenance import (
    DYNAMIC_RENDER_SHADE_INPUT_NAMES as DYNAMIC_RENDER_SHADE_INPUT_NAMES,
)
from optical_worker.provenance import (
    DYNAMIC_RENDER_WORLD_INPUT_NAMES as DYNAMIC_RENDER_WORLD_INPUT_NAMES,
)
from optical_worker.provenance import (
    DYNAMIC_SCENE_OUTPUT_NAMES as DYNAMIC_SCENE_OUTPUT_NAMES,
)
from optical_worker.provenance import (
    DYNAMIC_SCENE_UPDATE_RUNTIME_INPUT_NAMES as DYNAMIC_SCENE_UPDATE_RUNTIME_INPUT_NAMES,
)
from optical_worker.provenance import (
    DYNAMIC_TRANSFORM_INPUT_NAMES as DYNAMIC_TRANSFORM_INPUT_NAMES,
)
from optical_worker.provenance import (
    EXACT_ACCELERATION_CONTRACTS as EXACT_ACCELERATION_CONTRACTS,
)
from optical_worker.provenance import (
    EXACT_LANE_BINDINGS as EXACT_LANE_BINDINGS,
)
from optical_worker.provenance import (
    GPU_ENGINE_PUBLISHED_FRAME_INPUT as GPU_ENGINE_PUBLISHED_FRAME_INPUT,
)
from optical_worker.provenance import (
    NATIVE_EXECUTION_BINDING_SCHEMA_VERSION as NATIVE_EXECUTION_BINDING_SCHEMA_VERSION,
)
from optical_worker.provenance import (
    NATIVE_STAGE_EVIDENCE_SCHEMA_VERSION as NATIVE_STAGE_EVIDENCE_SCHEMA_VERSION,
)
from optical_worker.provenance import (
    SHIPPING_PUBLICATION_RECEIPT_SCHEMA_VERSION as SHIPPING_PUBLICATION_RECEIPT_SCHEMA_VERSION,
)
from optical_worker.provenance import (
    AccelerationReceipt as AccelerationReceipt,
)
from optical_worker.provenance import (
    C3OneShotNativeExecutionBindingEvidence as C3OneShotNativeExecutionBindingEvidence,
)
from optical_worker.provenance import (
    NativeStageBinding as NativeStageBinding,
)
from optical_worker.provenance import (
    NativeStageEvidence as NativeStageEvidence,
)
from optical_worker.provenance import (
    NativeStorageSignature as NativeStorageSignature,
)
from optical_worker.provenance import (
    ShippingPublicationReceipt as ShippingPublicationReceipt,
)
from optical_worker.provenance import (
    _assert_cuda_arena_c3_binding_authority as _assert_cuda_arena_c3_binding_authority,
)
from optical_worker.provenance import (
    _canonical_sha256 as _canonical_sha256,
)
from optical_worker.provenance import (
    _register_cuda_arena_c3_binding_authority as _register_cuda_arena_c3_binding_authority,
)
from optical_worker.provenance import (
    _require_exact_storage as _require_exact_storage,
)
from optical_worker.provenance import (
    _require_exact_storage_names as _require_exact_storage_names,
)
from optical_worker.provenance import (
    _require_named_storage as _require_named_storage,
)
from optical_worker.provenance import (
    _select_storage as _select_storage,
)
from optical_worker.provenance import (
    _sha256_text as _sha256_text,
)
from optical_worker.provenance import (
    _storage_by_name as _storage_by_name,
)
from optical_worker.provenance import (
    _storage_sequence as _storage_sequence,
)
from optical_worker.provenance import (
    _validate_stage_dataflow as _validate_stage_dataflow,
)
from optical_worker.provenance import (
    build_c3_one_shot_native_execution_binding as build_c3_one_shot_native_execution_binding,
)
from optical_worker.provenance import (
    canonical_dataflow_binding_sha256 as canonical_dataflow_binding_sha256,
)
from optical_worker.provenance import (
    canonical_dataflow_contract_id as canonical_dataflow_contract_id,
)
from optical_worker.provenance import (
    canonical_storage_manifest_sha256 as canonical_storage_manifest_sha256,
)
from optical_worker.provenance import (
    dynamic_scene_update_input_storage as dynamic_scene_update_input_storage,
)
