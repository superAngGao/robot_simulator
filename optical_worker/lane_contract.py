"""Frozen native input and acceleration identities; no runtime discovery."""

INPUT_CONTRACT_SCHEMA_VERSION = "q54.p12.4.input-contract.v2"

FIXED_NATIVE_SCENE_INPUT = "fixed_native_scene_v1"
CANONICAL_NATIVE_REPLAY_INPUT = "canonical_native_replay_v1"
GPU_ENGINE_PUBLISHED_FRAME_INPUT = "gpu_engine_published_frame_v1"

INPUT_CONTRACT_BY_BOUNDARY = {
    "R0": FIXED_NATIVE_SCENE_INPUT,
    "R1": CANONICAL_NATIVE_REPLAY_INPUT,
    "R2": CANONICAL_NATIVE_REPLAY_INPUT,
    "S": GPU_ENGINE_PUBLISHED_FRAME_INPUT,
}

EXACT_LANE_BINDINGS = {
    "warp": {
        "renderer_id": "warp_bvh_direct_light",
        "acceleration_by_class": {
            "fixed_bvh_fixture": "fixed_bvh_fixture",
            "production_static": "cuda_morton_karras_lbvh_rebuild_v1",
            "production_dynamic": "cpu_median_leaf4_topology_warp_refit_v1",
            "production_physics": "cpu_median_leaf4_topology_warp_refit_v1",
        },
    },
    "cuda": {
        "renderer_id": "cuda_direct_light",
        "acceleration_by_class": {
            "fixed_bvh_fixture": "fixed_bvh_fixture",
            "production_static": "cuda_morton_karras_lbvh_rebuild_v1",
            "production_dynamic": "cuda_morton_karras_lbvh_rebuild_v1",
            "production_physics": "cuda_morton_karras_lbvh_rebuild_v1",
        },
    },
}

EXACT_ACCELERATION_CONTRACTS = {
    "warp": {
        "fixed_bvh_fixture": {
            "topology_origin": "cpu_median_longest_axis_leaf4_source_order_v1",
            "topology_lifetime": "prepared_input_immutable_v1",
            "timed_scene_update_operation": "none",
            "timed_acceleration_operation": "none",
            "operation_classification": "none",
        },
        "cpu_median_leaf4_topology_warp_refit_v1": {
            "topology_origin": "cpu_median_longest_axis_leaf4_source_order_v1",
            "topology_lifetime": "prepared_lane_session_immutable_v1",
            "timed_scene_update_operation": "warp_local_scene_transform_and_aabb_out_v1",
            "timed_acceleration_operation": "warp_bvh_leaf_internal_refit_out_v1",
            "operation_classification": "refit",
        },
        "cuda_morton_karras_lbvh_rebuild_v1": {
            "topology_origin": "cuda_morton30_karras_source_index_tiebreak_v1",
            "topology_lifetime": "transaction_output_arena_lease_v1",
            "timed_scene_update_operation": "warp_local_scene_transform_and_aabb_out_v1",
            "timed_acceleration_operation": ("cuda_lbvh_scene_bounds_morton_sort_topology_bounds_out_v1"),
            "operation_classification": "rebuild",
        },
    },
    "cuda": {
        "fixed_bvh_fixture": {
            "topology_origin": "cpu_median_longest_axis_leaf4_source_order_v1",
            "topology_lifetime": "prepared_input_immutable_v1",
            "timed_scene_update_operation": "none",
            "timed_acceleration_operation": "none",
            "operation_classification": "none",
        },
        "cuda_morton_karras_lbvh_rebuild_v1": {
            "topology_origin": "cuda_morton30_karras_source_index_tiebreak_v1",
            "topology_lifetime": "transaction_output_arena_lease_v1",
            "timed_scene_update_operation": "torch_cuda_local_scene_transform_and_aabb_out_v1",
            "timed_acceleration_operation": ("cuda_lbvh_scene_bounds_morton_sort_topology_bounds_out_v1"),
            "operation_classification": "rebuild",
        },
    },
}
