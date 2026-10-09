"""Fixed recipe resolution shared by CUDA factories; no loading/build machinery."""

from .cuda_extension_recipes import cuda_extension_recipe


def load_cuda_extension(recipe_name, *, extensions=None, context=None):
    recipe = cuda_extension_recipe(recipe_name)
    if extensions is not None:
        from optical_worker.native_setup import _require_cuda_extension

        return _require_cuda_extension(extensions, recipe.name, context)
    from .cuda_extension_compatibility import load_cuda_extension as load_deployed

    return load_deployed(recipe.name)
