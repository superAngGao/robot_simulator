"""Fixed CUDA extension recipes shared by legacy and captured preparation.

This module describes builds; importing it neither imports a backend nor builds
or admits an extension. Source strings have exactly one repository owner.
"""

from dataclasses import dataclass

from . import _cuda_direct_light_source, _cuda_lbvh_source, _cuda_scene_source


@dataclass(frozen=True, slots=True)
class CudaExtensionRecipe:
    name: str
    source_module: object
    entries: tuple[str, ...]
    generate_bindings: bool


CUDA_EXTENSION_RECIPES = (
    CudaExtensionRecipe(
        "robot_sim_cuda_scene_v1",
        _cuda_scene_source,
        ("update_triangles_world_aabb_out", "update_planes_world_out"),
        False,
    ),
    CudaExtensionRecipe(
        "robot_sim_cuda_lbvh_native_v3",
        _cuda_lbvh_source,
        ("cuda_lbvh_workspace_requirements", "cuda_lbvh_rebuild_out"),
        False,
    ),
    CudaExtensionRecipe(
        "robot_sim_cuda_direct_light_v8",
        _cuda_direct_light_source,
        (
            "pinhole_camera_rays_out",
            "first_hit_rays_out",
            "shade_direct_light_no_shadow_out",
            "shade_direct_light_no_shadow_diagnostics_out",
            "shade_direct_light_with_shadows_out",
            "shade_direct_light_with_shadows_diagnostics_out",
        ),
        True,
    ),
)


def cuda_extension_recipe(recipe_id: str) -> CudaExtensionRecipe:
    if type(recipe_id) is str:
        for recipe in CUDA_EXTENSION_RECIPES:
            if recipe.name == recipe_id:
                return recipe
    raise ValueError("unknown fixed CUDA extension recipe")


def cuda_extension_kwargs(recipe_id: str) -> dict:
    """Return fresh kwargs for the sole fixed recipe; no caller source or hook."""
    recipe = cuda_extension_recipe(recipe_id)
    result = {
        "name": recipe.name,
        "cpp_sources": [recipe.source_module.CPP_SOURCE],
        "cuda_sources": [recipe.source_module.CUDA_SOURCE],
        "with_cuda": True,
        "is_python_module": True,
        "use_pch": False,
        "extra_cflags": ["-O2"],
        "extra_cuda_cflags": ["-O2"],
        "verbose": False,
    }
    if recipe.generate_bindings:
        result["functions"] = list(recipe.entries)
    return result
