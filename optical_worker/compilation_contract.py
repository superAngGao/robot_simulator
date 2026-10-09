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
