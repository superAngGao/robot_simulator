"""Private Warp compilation owners for the captured, supervised setup session.

No backend import, process manager, command execution or readiness issuance.
The source session alone supplies live runtime and guard authority.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import stat
import tempfile
from types import MappingProxyType
from typing import NamedTuple


_CACHES = {}
_MODULE_SETS = {}
_DEFAULT_CONFIG = MappingProxyType({
    "version": "1.12.0", "verify_fp": False, "verify_cuda": False,
    "print_launches": False, "mode": "release", "optimization_level": None,
    "verbose": False, "verbose_warnings": False, "quiet": False,
    "verify_autograd_array_access": False, "enable_vector_component_overwrites": False,
    "legacy_scalar_return_types": False, "cache_kernels": True, "kernel_cache_dir": None,
    "cuda_output": None, "ptx_target_arch": None, "cuda_arch_suffix": None,
    "lineinfo": False, "line_directives": True, "compile_time_trace": False,
    "enable_backward": True, "enable_mathdx_gemm": True, "llvm_cuda": False,
    "enable_graph_capture_module_load_by_default": True, "enable_mempools_at_init": True,
    "max_unroll": 16, "enable_tiles_in_stack_memory": True,
    "use_precompiled_headers": True, "load_module_max_workers": 0,
})
_MODULE_OPTIONS = MappingProxyType({
    "max_unroll": 16, "enable_backward": True, "enable_mathdx_gemm": None,
    "fast_math": False, "fuse_fp": True, "lineinfo": False, "cuda_output": None,
    "mode": None, "optimization_level": None, "block_dim": 256,
    "compile_time_trace": False, "strip_hash": False,
})
_DEFINITIONS = (
    ("optics.device_scene", (
        "_update_triangles_derived_aabb_world_kernel", "_update_triangles_derived_world_kernel",
        "_update_planes_world_kernel",
    ), ("_compose_world_transform", "_transform_source_point", "_indexed_world_triangle",
        "_write_triangle_world", "_outward_bound"), ()),
    ("optics.device_bvh", (
        "_refit_bvh_leaf_bounds_kernel", "_refit_bvh_internal_bounds_kernel",
    ), (), ()),
    ("optics.warp_native_render", (
        "_pinhole_camera_raygen_kernel", "_brute_force_first_hit_kernel",
        "_device_scene_first_hit_kernel", "_device_scene_first_hit_aabb_kernel",
        "_init_bvh_hit_mask_kernel", "_device_scene_bvh_first_hit_kernel",
        "warp_direct_light_no_shadow_primary_v1", "warp_direct_light_no_shadow_diagnostics_v1",
        "warp_direct_light_with_shadows_primary_v1", "warp_direct_light_with_shadows_diagnostics_v1",
        "_reset_bvh_primary_counters_kernel", "_reset_shadow_primary_counters_kernel",
        "_reset_shadow_diagnostic_counters_kernel",
    ), (
        "_is_better_hit", "_intersect_aabb_axis", "_intersect_aabb_for_ray",
        "_intersect_aabb_for_ray_with_interval", "_intersect_device_triangle_for_ray",
        "_is_occluded_for_ray", "_direct_light_begin", "_direct_light_sample",
        "_direct_light_accumulate", "_direct_light_write", "_shade_unshadowed_ray", "_shade_shadowed_ray",
    ), ("_ShadowTrace", "_ShadowTotals", "_DirectLightSurface", "_DirectLightSample")),
)
# Installed include inputs are explicit data files, never permission to scan/import.
_HEADERS = (
    "aot.h", "array.h", "builtin.h", "bvh.h", "crt.h", "cuda_crt.h", "cuda_util.h",
    "error.h", "exports.h", "fabric.h", "hashgrid.h", "initializer_array.h",
    "intersect.h", "intersect_adj.h", "intersect_tri.h", "mat.h", "matnn.h", "mesh.h",
    "nanovdb/GridHandle.h", "nanovdb/HostBuffer.h", "nanovdb/NanoVDB.h",
    "nanovdb/NodeManager.h", "nanovdb/PNanoVDB.h", "noise.h", "quat.h", "rand.h",
    "range.h", "scan.h", "solid_angle.h", "sort.h", "spatial.h", "svd.h", "temp_buffer.h",
    "texture.h", "tile.h", "tile_bvh.h", "tile_mesh.h", "tile_radix_sort.h", "tile_reduce.h",
    "tile_scan.h", "tuple.h", "vec.h", "version.h", "volume.h", "volume_builder.h",
    "volume_impl.h", "warp.h",
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()


def _matches(actual, expected):
    return (type(actual) is dict and set(actual) == set(expected)
            and all(type(actual[key]) is type(value) and actual[key] == value for key, value in expected.items()))


class _Cache(NamedTuple):
    path: Path
    directory_identity: tuple
    module: object
    config: object
    context: object
    settings: tuple
    records: list


class _CompiledModule(NamedTuple):
    python_module: object
    module: object
    mapping: dict
    executable: object
    definitions: tuple
    definition_facts: tuple
    module_hash: bytes
    handle: object
    metadata: bytes
    artifacts: tuple
    source_sha256: str
    guard: tuple


class WarpModuleSet:
    __slots__ = ("session", "binding", "modules")

    def __init__(self):
        raise TypeError("Warp module sets are issued by their source session")

    def __setattr__(self, name, value):
        raise TypeError("Warp module set is frozen")

    def __delattr__(self, name):
        raise TypeError("Warp module set is frozen")


def configure_warp_cache(session, wp):
    """Called inside the existing runtime guard, before the sole Warp init."""
    session.bound_controller()._require_setup()
    _require(session not in _CACHES, "Warp cache configuration is single-use")
    _CACHES[session] = None
    context = session.external.records["warp._src.context"]["module"]
    _require(context.runtime is None, "Warp was initialized before cache selection")
    original = {name: getattr(wp.config, name) for name in _DEFAULT_CONFIG}
    _require(_matches(original, _DEFAULT_CONFIG), "unexpected initial Warp configuration")
    raw_path = tempfile.mkdtemp(prefix="robot-sim-warp-", dir="/tmp")
    session.runtime_owners.append(raw_path)
    path = Path(raw_path)
    session._hold_runtime_owner(path, "warp_compilation_cache")
    _require(path.parent == Path("/tmp") and path.resolve(strict=True) == path
             and not path.is_symlink() and not any(path.iterdir()), "Warp cache is not fresh and private")
    observed = path.stat()
    _require(stat.S_IMODE(observed.st_mode) == 0o700, "Warp cache permissions differ")
    wp.config.kernel_cache_dir = str(path)
    wp.config.use_precompiled_headers = False
    wp.config.cuda_output = "cubin"
    wp.config.quiet = True
    # The checked defaults already select cache_kernels=True, llvm_cuda=False,
    # and serial loading. Preserve every other numerical/runtime option.
    settings = dict(_DEFAULT_CONFIG) | {
        "kernel_cache_dir": str(path), "use_precompiled_headers": False, "cuda_output": "cubin", "quiet": True,
    }
    cache = _Cache(path, (observed.st_dev, observed.st_ino), wp, wp.config, context, tuple(settings.items()), [])
    session._hold_runtime_owner(cache, "warp_compilation_configuration")
    _CACHES[session] = cache
    validate_warp_cache(session, wp)


def validate_warp_cache(session, wp):
    cache = _CACHES.get(session)
    _require(cache is not None and cache.module is wp and wp.config is cache.config,
             "Warp cache configuration identity drift")
    actual = {name: getattr(wp.config, name) for name, _ in cache.settings}
    _require(_matches(actual, dict(cache.settings)), "Warp compilation configuration drift")
    path = cache.path
    observed = path.stat()
    _require(path.resolve(strict=True) == path and not path.is_symlink()
             and (observed.st_dev, observed.st_ino) == cache.directory_identity
             and stat.S_IMODE(observed.st_mode) == 0o700, "Warp cache directory changed")
    return cache


def validate_compilation_inputs(session):
    rows = {(row["root"], row["path"]): row for row in session.external.environment.to_payload()["files"]}
    _require(all(rows.get(("site", "warp/native/" + name), {}).get("kind") == "data" for name in _HEADERS),
             "Warp compilation header inventory is incomplete")
    # File bytes were checked by external-environment admission. Recheck before
    # native compilation, using that same trusted environment digest/validator.
    environment = session.external.environment
    environment.verify_files(expected_sha256=session.external.expected_sha256)


def definition_owners(session, index):
    name, kernel_names, function_names, struct_names = _DEFINITIONS[index]
    python_module = session.modules[name][0]
    kernels = tuple(getattr(python_module, key) for key in kernel_names)
    functions = tuple(getattr(python_module, key) for key in function_names)
    structs = tuple(getattr(python_module, key) for key in struct_names)
    module = kernels[0].module
    context = session.external.records["warp._src.context"]["module"]
    codegen = session.external.records["warp._src.codegen"]["module"]
    _require(type(module) is context.Module and module.name == name, "unexpected Warp module definition")
    _require(all(type(value) is context.Kernel for value in kernels)
             and all(type(value) is context.Function for value in functions)
             and all(type(value) is codegen.Struct for value in structs), "unexpected Warp definition type")
    _require(len(module.kernels) == len(kernels) and len(module.functions) == len(functions)
             and len(module.structs) == len(structs), "Warp definition inventory drift")
    _require(all(module.kernels.get(value.key) is value for value in kernels)
             and all(module.functions.get(value.key) is value for value in functions)
             and all(module.structs.get(value.key) is value for value in structs), "Warp registry identity drift")
    _require(all(type(value.options) is dict and not value.options for value in kernels),
             "Warp kernel compilation options drift")
    for key, value in zip(kernel_names + function_names, kernels + functions):
        _require(value.module is module and value.func.__name__ == key and value.func.__module__ == name,
                 "Warp definition source identity drift")
    for key, value in zip(struct_names, structs):
        _require(value.module is module and value.cls.__name__ == key and value.cls.__module__ == name,
                 "Warp struct source identity drift")
    expected_options = dict(_MODULE_OPTIONS)
    if name in ("optics.device_scene", "optics.warp_native_render"):
        expected_options["fuse_fp"] = False
    _require(_matches(module.options, expected_options), "Warp module compilation options drift")
    return python_module, module, kernels + functions + structs


def _type_facts(value, types):
    # Warp's own type code freezes scalar/vector/matrix layout. Array annotations
    # and struct fields are mutable, so retain their values as well as owners.
    code = types.get_type_code(value)
    if types.is_array(value):
        return (value, code, value.ndim, _type_facts(value.dtype, types))
    if types.type_is_struct(value):
        return (value, code, value.ctype, value.instance_type,
                tuple((name, _type_facts(var.type, types)) for name, var in value.vars.items()))
    return (value, code, getattr(value, "_type_", None))


def _definition_facts(definitions, types):
    facts = []
    for value in definitions:
        if hasattr(value, "func"):
            facts.append((value, value.func, value.func.__code__, value.adj,
                          tuple((arg.label, _type_facts(arg.type, types)) for arg in value.adj.args),
                          (_json(value.options), value.key, value.hash, value.get_mangled_name())
                          if hasattr(value, "options") else None))
        else:
            facts.append((value, value.cls, value.module, value.ctype, value.instance_type,
                          tuple((name, _type_facts(var.type, types)) for name, var in value.vars.items()),
                          value.hash, value.native_name))
    return tuple(facts)


def _artifact_bytes(root, relative):
    path = root
    parts = Path(relative).parts
    _require(parts and not Path(relative).is_absolute() and all(part not in {".", ".."} for part in parts),
             "Warp artifact path is not relative")
    for part in parts:
        path = path / part
        _require(not path.is_symlink(), "Warp artifact redirects outside its cache")
    _require(path.resolve(strict=True).is_relative_to(root) and path.is_file(), "Warp artifact outside cache")
    return path.read_bytes()


def _artifact_facts(cache, module, target):
    # These are fixed metadata methods of the pinned Warp Module, not a cache
    # scanner or an alternative compiler. Native calls remain in load_module.
    identifier = module.get_module_identifier()
    name = module._get_compile_output_name(target)
    _require(name == identifier + ".sm90.cubin", "Warp output is not the selected sm_90 CUBIN")
    names = (name, module._get_meta_name(), identifier + ".cu")
    data = tuple(_artifact_bytes(cache.path, identifier + "/" + part) for part in names)
    _require(data[0].startswith(b"\x7fELF\x02\x01\x01"), "Warp compiler output is not an ELF64 CUBIN")
    facts = tuple((identifier + "/" + part, len(value), hashlib.sha256(value).hexdigest())
                  for part, value in zip(names, data))
    return facts, _json(json.loads(data[1]))


def compile_module(session, binding, index):
    """One original armed guard owns checks, compilation and returned roots."""
    controller = session.bound_controller()
    controller._require_setup()
    wp, target, stream = binding.module, binding.target, binding.stream
    cache = validate_warp_cache(session, wp)
    _require(type(index) is int and index == len(cache.records) and 0 <= index < 3,
             "Warp compilation order is single-use")
    python_module, module, definitions = definition_owners(session, index)
    session._hold_runtime_owner(module, "warp_compile_module")
    session._hold_runtime_owner(definitions, "warp_compile_definitions")
    mapping = module.execs
    session._hold_runtime_owner(mapping, "warp_compile_executables")
    _require(type(mapping) is dict and not mapping and not module.failed_builds, "Warp executable already exists")
    _require(target.arch == 90 and target.is_cuda and stream.device is target, "Warp compilation target drift")
    types = session.external.records["warp._src.types"]["module"]
    # Module hashing initializes each kernel's hash/mangled entry. Snapshot that
    # legitimate initialized state, then reject changes through load and reuse.
    expected_hash = module.get_module_hash(256)
    before = _definition_facts(definitions, types)
    identifier = module.get_module_identifier()
    _require(not (cache.path / identifier).exists(), "Warp module cache predates compilation")
    session._runtime_leaf("warp_compile_" + str(index), "runtime_import_or_init")
    with wp.ScopedDevice(target), wp.ScopedStream(stream):
        wp.load_module(module=module, device=target, recursive=False, block_dim=256, max_workers=0)
        executable = mapping[(target.context, 256)]
        session.runtime_owners.append(executable)
        session._hold_runtime_owner(executable, "warp_compiled_module_exec", device_ordinal=int(target.ordinal))
        _require(type(executable) is cache.context.ModuleExec and executable.device is target
                 and executable.handle is not None and executable.handle != 0
                 and executable.module_hash == expected_hash and module.execs is mapping
                 and set(mapping) == {(target.context, 256)},
                 "Warp loaded executable identity drift")
        _require(_definition_facts(definitions, types) == before, "Warp definitions changed during compilation")
        artifacts, metadata = _artifact_facts(cache, module, target)
        _require(_json(executable.meta) == metadata, "Warp executable metadata differs from compiled file")
        guard = controller._guard_reference
        record = _CompiledModule(python_module, module, mapping, executable, definitions, before,
                                 expected_hash, executable.handle, metadata, artifacts,
                                 session.sources[session.modules[python_module.__name__][1].relative][1],
                                 (guard["sequence"], guard["role"]))
        session._hold_runtime_owner(record, "warp_compiled_module_record")
        cache.records.append(record)
        return record


def issue_module_set(session, binding, records):
    controller = session.bound_controller()
    _require(controller._guard_client.state == "IDLE" and not controller._replay.failed
             and not controller._context_active and controller._guard_reference is None
             and not controller._replay.compromised,
             "Warp module issuance requires healthy IDLE target scope")
    _require(len(records) == 3 and _MODULE_SETS.get(session) is None, "Warp module set is incomplete or repeated")
    cache = validate_warp_cache(session, binding.module)
    _require(type(records) is tuple and len(cache.records) == 3
             and all(actual is held for actual, held in zip(records, cache.records))
             and len({record.guard for record in records}) == 3,
             "Warp compilation records are copied, foreign or incomplete")
    controller.require_adopted(binding, cache, *records)
    completed = {(event["guard"]["sequence"], event["guard"]["role"])
                 for event in controller._guard_events if event["operation"] == "normal_disarmed"}
    _require(all(record.guard in completed for record in records), "Warp compile guard did not complete")
    owner = object.__new__(WarpModuleSet)
    object.__setattr__(owner, "session", session)
    object.__setattr__(owner, "binding", binding)
    object.__setattr__(owner, "modules", records)
    session._hold_runtime_owner(owner, "warp_compiled_module_set")
    _MODULE_SETS[session] = (owner, binding, records)
    return owner


def validate_module_set(session, owner):
    session.check_live()
    controller = session.bound_controller()
    controller._require_setup()
    registered = _MODULE_SETS.get(session)
    _require(type(owner) is WarpModuleSet and registered is not None and registered[0] is owner,
             "Warp module set is copied, foreign or unregistered")
    _, binding, records = registered
    _require(owner.session is session and owner.binding is binding and owner.modules is records,
             "Warp module set identity drift")
    cache = validate_warp_cache(session, binding.module)
    _require(len(cache.records) == 3 and all(actual is held for actual, held in zip(records, cache.records)),
             "Warp compilation record registry changed")
    controller.require_adopted(owner, binding, *records)
    types = session.external.records["warp._src.types"]["module"]
    for index, record in enumerate(records):
        python_module, module, definitions = definition_owners(session, index)
        _require(python_module is record.python_module and module is record.module
                 and module.execs is record.mapping
                 and set(module.execs) == {(binding.target.context, 256)}
                 and module.execs[(binding.target.context, 256)] is record.executable,
                 "Warp executable mapping changed or a new variant appeared")
        _require(_definition_facts(definitions, types) == record.definition_facts
                 and module.get_module_hash(256) == record.module_hash
                 and record.executable.module_hash == record.module_hash
                 and record.executable.handle == record.handle and record.executable.device is binding.target
                 and _json(record.executable.meta) == record.metadata,
                 "Warp executable or kernel ABI changed")
        _require(_artifact_facts(cache, module, binding.target) == (record.artifacts, record.metadata),
                 "Warp compiled artifact bytes changed")
        controller.require_adopted(module, record.mapping, record.executable)
    return records
