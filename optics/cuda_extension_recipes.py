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
    cflags: tuple[str, ...] = ("-O2",)
    cuda_cflags: tuple[str, ...] = ("-O2",)


CUDA_EXTENSION_RECIPES = (
    CudaExtensionRecipe(
        "robot_sim_cuda_scene_v3",
        _cuda_scene_source,
        ("update_triangles_world_aabb_out", "update_planes_world_out"),
        False,
        cuda_cflags=("-O2", "--fmad=false"),
    ),
    CudaExtensionRecipe(
        "robot_sim_cuda_lbvh_native_v3",
        _cuda_lbvh_source,
        ("cuda_lbvh_workspace_requirements", "cuda_lbvh_rebuild_out"),
        False,
    ),
    CudaExtensionRecipe(
        "robot_sim_cuda_direct_light_v10",
        _cuda_direct_light_source,
        (
            "pinhole_camera_rays_out",
            "first_hit_rays_out",
            "shade_direct_light_no_shadow_out",
            "shade_direct_light_no_shadow_diagnostics_out",
            "shade_direct_light_with_shadows_out",
            "shade_direct_light_with_shadows_diagnostics_out",
        ),
        False,
        cuda_cflags=("-O2", "--fmad=false"),
    ),
)


def cuda_extension_recipe(recipe_id: str) -> CudaExtensionRecipe:
    if type(recipe_id) is str:
        for recipe in CUDA_EXTENSION_RECIPES:
            if recipe.name == recipe_id:
                return recipe
    raise ValueError("unknown fixed CUDA extension recipe")


def cuda_extension_entries(module, recipe_id: str) -> tuple:
    """Read the fixed exported callables; this check grants no live authority."""
    from types import ModuleType

    recipe = cuda_extension_recipe(recipe_id)
    if type(module) is not ModuleType or vars(module).get("__name__") != recipe.name:
        raise ValueError("CUDA extension module differs from the fixed recipe")
    entries = tuple(vars(module).get(name) for name in recipe.entries)
    if any(not callable(entry) or entry.__module__ != recipe.name or entry.__name__ != name
           for name, entry in zip(recipe.entries, entries, strict=True)):
        raise ValueError("CUDA extension entry differs from the fixed ABI")
    if len({id(entry) for entry in entries}) != len(entries):
        raise ValueError("CUDA extension entries alias")
    return entries
