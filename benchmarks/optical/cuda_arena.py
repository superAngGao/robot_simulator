"""Compatibility aliases for the production CUDA arena C3 binding owner."""

from optical_worker.cuda_arena import (
    _canonical_sha256 as _canonical_sha256,
    _ordered_signatures as _ordered_signatures,
    _plain_payload as _plain_payload,
    _render_output_signatures as _render_output_signatures,
    _require_exact_objects as _require_exact_objects,
    _stage_evidence as _stage_evidence,
    _storage_signature as _storage_signature,
    _tensor_identity as _tensor_identity,
    _validate_target_a2_allocations as _validate_target_a2_allocations,
    build_cuda_c3_binding_from_arena_target as build_cuda_c3_binding_from_arena_target,
)
