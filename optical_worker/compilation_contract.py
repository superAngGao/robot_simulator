"""Inert fixed compilation phases and evidence vocabulary; no build authority."""

INPUT_PHASES = (
    "setup_definitions", "external_environment", "canonical_materialization",
    "runtime_initialization", "native_upload", "input_validation",
)
CUDA_COMPILE_PHASES = (
    "cuda_scene_compile", "cuda_lbvh_compile", "cuda_direct_light_compile",
)
# Position is the wire sequence, shared by the driver and the supervisor.
CUDA_FOUNDATION_PHASES = INPUT_PHASES + CUDA_COMPILE_PHASES + ("terminal_disposal",)
CUDA_COMPILE_SEQUENCES = tuple(range(len(INPUT_PHASES), len(INPUT_PHASES) + len(CUDA_COMPILE_PHASES)))
CUDA_TERMINAL_SEQUENCE = len(CUDA_FOUNDATION_PHASES) - 1
CUDA_SOURCE_PHASES = (
    ("input_validated", "compiling_cuda_scene", "cuda_scene_compiled"),
    ("cuda_scene_compiled", "compiling_cuda_lbvh", "cuda_lbvh_compiled"),
    ("cuda_lbvh_compiled", "compiling_cuda_direct_light", "cuda_compiled"),
)
CUDA_COMPILE_STEPS = (
    "build_entered", "generated", "create_entered", "module_returned",
    "exec_completed", "entries_bound", "owner_registered", "hook_restored",
    "guard_completed",
)
CUDA_COMPILATION_VERSION = "q54.p12.4.cuda-compilation.v1"
CUDA_COMPILATION_SELECTION_VERSION = "q54.p12.4.cuda-compilation-selection.v1"
COMPILER_JOB_VERSION = "q54.p12.4.compiler-job.v1"
CUDA_CONTINUATION_SOURCE_MODULES = (
    "optical_worker.cuda_compilation", "optical_worker.cuda_compilation_evidence",
    "optical_worker.compilation_contract", "optics.cuda_extension_recipes",
    "optics._cuda_scene_source", "optics._cuda_lbvh_source", "optics._cuda_direct_light_source",
)
CUDA_NATIVE_SITES = ("cuda_compile", "cuda_extension_create", "cuda_extension_exec")

# Pinned dependency grammar. These inert facts are shared by collection/replay;
# they neither construct an initialization owner nor grant import permission.
SETUPTOOLS_INITIALIZATION_POLICY = "setuptools_83_conda_cuda_g1_v1"
SETUPTOOLS_ALIAS_ACTIONS = ("alias_created", "alias_executed", "alias_published")
_SETUPTOOLS_ALIAS_FIELDS = ("source_module_token", "alias_spec_token", "delegate_token", "wrapper_token")
SETUPTOOLS_ACTION_FIELDS = (
    ("begun", ("vendor_root", "original_paths", "active_paths")),
    ("hooked", ("hack_module_token", "finder_token", "insert_original_token", "remove_original_token",
                "insert_wrapper_token", "remove_wrapper_token", "producer_token")),
    ("shim_entered", ("override_module_token",)),
    ("alias_spec", ("source_module_token", "source_spec_token", "source_loader_token",
                    "alias_spec_token", "delegate_token", "wrapper_token")),
    *((action, _SETUPTOOLS_ALIAS_FIELDS) for action in SETUPTOOLS_ALIAS_ACTIONS),
    ("shim_left", ()),
    ("hooks_restored", ()),
    ("metadata_read", ("read_index", "distribution_token", "bytes_token", "file_root", "file_path",
                       "sha256", "version")),
    ("resource_read", ("read_index", "package_token", "reader_token", "bytes_token", "handle_token",
                       "file_root", "file_path", "sha256")),
    ("completed", ("setuptools_module_token", "build_ext_module_token", "cpp_extension_module_token")),
    ("failed", ("exception_token",)),
)
SETUPTOOLS_HELPER_BINDINGS = (
    ("hooked", "finder_token", "setuptools_initialization.finder"),
    ("hooked", "producer_token", "setuptools_initialization.spec_producer"),
    ("hooked", "insert_original_token", "setuptools_initialization.insert_original"),
    ("hooked", "remove_original_token", "setuptools_initialization.remove_original"),
    ("hooked", "insert_wrapper_token", "setuptools_initialization.insert_wrapper"),
    ("hooked", "remove_wrapper_token", "setuptools_initialization.remove_wrapper"),
    ("alias_spec", "alias_spec_token", "setuptools_initialization.alias_spec"),
    ("alias_spec", "delegate_token", "setuptools_initialization.alias_delegate"),
    ("alias_spec", "wrapper_token", "setuptools_initialization.alias_wrapper"),
    ("metadata_read", "distribution_token", "setuptools_initialization.metadata_distribution"),
    ("metadata_read", "bytes_token", "setuptools_initialization.metadata_bytes"),
    ("resource_read", "reader_token", "setuptools_initialization.resource_reader"),
    ("resource_read", "bytes_token", "setuptools_initialization.resource_bytes"),
)


def setuptools_event_fields(action: str) -> frozenset[str]:
    if type(action) is str:
        for name, fields in SETUPTOOLS_ACTION_FIELDS:
            if action == name:
                return frozenset(("action", "initialization_token", *fields))
    raise ValueError("unknown setuptools initialization action")


def setuptools_role_anchor(action: str, field: str, read_index: int | None = None) -> tuple[str, str]:
    """Closed successful-helper roles; ordinary file owners have separate joins."""
    fields = setuptools_event_fields(action)
    if type(field) is not str or field not in fields:
        raise ValueError("unknown setuptools initialization field")
    reading = action in {"metadata_read", "resource_read"}
    if reading:
        if type(read_index) is not int or read_index < 0:
            raise ValueError("invalid setuptools read occurrence")
    elif read_index is not None:
        raise ValueError("unexpected setuptools read occurrence")
    if field == "initialization_token":
        return "setuptools_initialization", "setuptools_initialization"
    if action == "resource_read" and field == "handle_token":
        return "external_import_owner", f"setuptools_initialization.resource_handle.{read_index}"
    binding_action = "alias_spec" if action in SETUPTOOLS_ALIAS_ACTIONS else action
    for name, slot, operation in SETUPTOOLS_HELPER_BINDINGS:
        if (binding_action, field) == (name, slot):
            return "external_import_owner", operation
    raise ValueError("setuptools field requires an ordinary-source or failed-owner join")


def cuda_native_operation(recipe_id: str, site: str) -> tuple[str, str]:
    for name, suffix, runtime_identity in (
        ("cuda_compile", ".build", "torch_cuda_fixed_jit_v1"),
        ("cuda_extension_create", ".create", "python_extension_create_v1"),
        ("cuda_extension_exec", ".exec", "python_extension_exec_v1"),
    ):
        if site == name:
            return recipe_id + suffix, runtime_identity
    raise ValueError("unknown CUDA compilation native site")


def cuda_foundation_phase(sequence: int, role: str) -> str:
    if type(sequence) is not int or not 0 <= sequence < len(CUDA_FOUNDATION_PHASES):
        raise ValueError("unknown CUDA foundation sequence")
    expected = "emergency" if sequence == CUDA_TERMINAL_SEQUENCE else "setup"
    if type(role) is not str or role != expected:
        raise ValueError("wrong CUDA foundation guard role")
    return CUDA_FOUNDATION_PHASES[sequence]


def cuda_role_anchor(recipe_id: str, field: str, entry_name: str | None = None) -> tuple[str, str]:
    """Literal role spelling; recipe and entry membership are checked by consumers."""
    if type(recipe_id) is not str:
        raise ValueError("invalid CUDA recipe identity")
    if field == "entry_token" and type(entry_name) is str:
        return "cuda_compile_entry", recipe_id + ".entry." + entry_name
    if entry_name is not None:
        raise ValueError("unexpected CUDA entry name")
    for key, role, suffix in (
        ("spec_token", "cuda_compile_spec", ".spec"),
        ("loader_token", "cuda_compile_loader", ".loader"),
        ("module_token", "cuda_compile_module", ".module"),
        ("compiled_owner_token", "cuda_compiled_owner", ".owner"),
    ):
        if field == key:
            return role, recipe_id + suffix
    raise ValueError("unknown CUDA generated role")
