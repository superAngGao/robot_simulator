"""Device-resident optical scene cache primitives."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from physics.publish import GpuPublishedFrame

from .host_bvh import _canonical_sha256 as _canonical_sha256
from .registry import OpticalWorldRegistry
from .scene_contract import PackedOpticalSceneHost, validate_indexed_geometry
from .query_numerics import BUILD_EPS

try:  # pragma: no cover - exercised in GPU environments.
    import warp as wp

    _HAS_WARP = True
except Exception:  # pragma: no cover - keeps CPU-only imports working.
    wp = None
    _HAS_WARP = False

_MAX_INT64_ROLE_BITS = 63
WARP_SCENE_UPDATE_OPERATION_ID = "warp_local_scene_transform_and_aabb_out_v1"
WARP_SCENE_COMMAND_SCHEMA_VERSION = "q54.p12.4.warp-scene-command.v1"
_DEVICE_SCENE_ARRAY_NAMES = (
    "material_albedo_rgb",
    "light_kind",
    "light_position_or_direction_world",
    "light_intensity",
    "light_color_rgb",
    "geometry_vertices_local",
    "geometry_vertex_offsets",
    "triangle_vertex_indices",
    "triangle_instance_index",
    "triangle_primitive_global_id",
    "triangle_primitive_index_within_instance",
    "triangle_geometry_index",
    "triangle_geometry_primitive_index",
    "triangle_source_order_key",
    "triangle_role_mask",
    "triangle_numeric_instance_id",
    "triangle_material_index",
    "plane_normal_local",
    "plane_point_local",
    "plane_instance_index",
    "plane_source_order_key",
    "plane_role_mask",
    "plane_numeric_instance_id",
    "plane_material_index",
    "instance_body_index",
    "instance_X_body_geometry_R",
    "instance_X_body_geometry_r",
)


@dataclass(frozen=True)
class DeviceOpticalRoleTable:
    """Deterministic role-to-bitmask mapping for device visibility checks."""

    role_to_mask: MappingProxyType

    @classmethod
    def from_roles(cls, roles: object) -> "DeviceOpticalRoleTable":
        unique_roles = tuple(sorted({str(role) for role in roles}))
        if len(unique_roles) > _MAX_INT64_ROLE_BITS:
            raise ValueError("L5C role bitmask supports at most 63 roles")
        return cls(MappingProxyType({role: 1 << index for index, role in enumerate(unique_roles)}))

    @classmethod
    def from_registry(cls, registry: OpticalWorldRegistry) -> "DeviceOpticalRoleTable":
        roles: set[str] = set()
        for instance in registry.instances:
            roles.update(instance.roles)
        return cls.from_roles(roles)

    @classmethod
    def from_mapping(cls, role_to_mask: object) -> "DeviceOpticalRoleTable":
        mapping = dict(role_to_mask)
        if any(not isinstance(role, str) or not role for role in mapping):
            raise ValueError("role names must be non-empty strings")
        expected = {role: 1 << index for index, role in enumerate(sorted(mapping))}
        if mapping != expected:
            raise ValueError("role-to-mask mapping is not canonical")
        return cls(MappingProxyType(mapping))

    def mask_for(self, role: str) -> int:
        """Return zero for unknown roles so device queries naturally miss."""
        return int(self.role_to_mask.get(role, 0))

    def mask_for_roles(self, roles: object) -> int:
        mask = 0
        for role in roles:
            mask |= self.mask_for(str(role))
        return mask


@dataclass(frozen=True)
class DeviceOpticalScene:
    """Long-lived device buffers built from an `OpticalWorldRegistry`.

    `instance_X_body_geometry_*` follows the host convention:
    `X_body_geometry` is the pose of the geometry frame expressed in the body
    frame. It maps geometry-local points into the body frame. World-static
    instances use `instance_body_index == -1`, in which case
    `X_body_geometry` is interpreted directly as world-from-geometry.
    """

    device: object
    role_table: DeviceOpticalRoleTable
    num_instances: int
    num_materials: int
    num_lights: int
    num_triangles: int
    num_planes: int
    max_body_index: int
    material_albedo_rgb: object
    light_kind: object
    light_position_or_direction_world: object
    light_intensity: object
    light_color_rgb: object
    geometry_vertices_local: object
    geometry_vertex_offsets: object
    triangle_vertex_indices: object
    triangle_instance_index: object
    triangle_primitive_global_id: object
    triangle_primitive_index_within_instance: object
    triangle_geometry_index: object
    triangle_geometry_primitive_index: object
    triangle_source_order_key: object
    triangle_role_mask: object
    triangle_numeric_instance_id: object
    triangle_material_index: object
    plane_normal_local: object
    plane_point_local: object
    plane_instance_index: object
    plane_source_order_key: object
    plane_role_mask: object
    plane_numeric_instance_id: object
    plane_material_index: object
    instance_body_index: object
    instance_X_body_geometry_R: object
    instance_X_body_geometry_r: object
    scene_contract_sha256: str | None = None
    resources: tuple[object, ...] = ()


@dataclass(frozen=True)
class DeviceOpticalSceneSnapshot:
    """Per-frame device optical scene buffers.

    L5C.0 intentionally allocates fresh world primitive buffers per snapshot.
    This binds buffer lifetime to the snapshot object and avoids introducing a
    device buffer ring before the event dependency model has been validated.
    """

    frame_id: int
    sim_time: float
    env_idx: int
    scene: DeviceOpticalScene
    triangle_v0_world: object
    triangle_e1_world: object
    triangle_e2_world: object
    triangle_normal_world: object
    plane_normal_world: object
    plane_point_world: object
    triangle_aabb_min: object | None = None
    triangle_aabb_max: object | None = None
    ready_event: object | None = None
    resources: tuple[object, ...] = ()


@dataclass(frozen=True)
class WarpSceneTransformInputs:
    """One already-created Warp transform view bound outside native timing."""

    x_world_R: object
    x_world_r: object
    env_idx: int
    input_contract: str
    transform_replay_sha256: str | None
    transform_frame_index: int | None
    resources: tuple[object, ...] = ()


@dataclass(frozen=True)
class WarpWorldSceneBuffers:
    """Caller-owned mutable world geometry and AABB outputs.

    The allocation helper below exists for compatibility callers and C3
    one-shot tests. C4 must supply the final arena-owned instances.
    """

    device: object
    triangle_v0_world: object
    triangle_e1_world: object
    triangle_e2_world: object
    triangle_normal_world: object
    plane_normal_world: object
    plane_point_world: object
    triangle_aabb_min: object
    triangle_aabb_max: object
    resources: tuple[object, ...] = ()

    def as_snapshot(
        self,
        scene: DeviceOpticalScene,
        *,
        frame_id: int,
        sim_time: float,
        env_idx: int,
    ) -> DeviceOpticalSceneSnapshot:
        """Create an untimed compatibility view without recording an event."""

        if scene.device != self.device:
            raise ValueError("Warp world-scene buffers and local scene devices must match")
        return DeviceOpticalSceneSnapshot(
            frame_id=int(frame_id),
            sim_time=float(sim_time),
            env_idx=int(env_idx),
            scene=scene,
            triangle_v0_world=self.triangle_v0_world,
            triangle_e1_world=self.triangle_e1_world,
            triangle_e2_world=self.triangle_e2_world,
            triangle_normal_world=self.triangle_normal_world,
            triangle_aabb_min=self.triangle_aabb_min,
            triangle_aabb_max=self.triangle_aabb_max,
            plane_normal_world=self.plane_normal_world,
            plane_point_world=self.plane_point_world,
            ready_event=None,
            resources=(self,) + tuple(self.resources),
        )


@dataclass(frozen=True)
class WarpSceneUpdateCommandEvidence:
    """Independent facts captured while a Warp scene bundle is sealed."""

    schema_version: str
    operation_id: str
    device_ordinal: int
    stream_handle: int
    native_input_manifest_sha256: str
    scene_contract_sha256: str
    input_contract: str
    transform_replay_sha256: str | None
    transform_frame_index: int | None
    input_storage_sha256: str
    output_storage_sha256: str
    dispatch_count: int
    dispatch_plan_sha256: str

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.__dict__)


class WarpSceneUpdateCommandBundle:
    """Immutable pre-recorded Warp scene commands for one storage graph."""

    __slots__ = (
        "_binding_signature",
        "_commands",
        "_frozen_commands",
        "_frozen_device_ordinal",
        "_frozen_dispatch_count",
        "_frozen_dispatch_plan_sha256",
        "_frozen_evidence",
        "_frozen_evidence_sha256",
        "_frozen_input_contract",
        "_frozen_native_input_manifest_sha256",
        "_frozen_owners",
        "_frozen_stream_handle",
        "_poisoned",
        "_sealed",
        "evidence",
        "local_scene",
        "outputs",
        "stream",
        "transforms",
    )

    def __init__(
        self,
        *,
        commands: tuple[object, ...],
        evidence: WarpSceneUpdateCommandEvidence,
        local_scene: DeviceOpticalScene,
        transforms: WarpSceneTransformInputs,
        outputs: WarpWorldSceneBuffers,
        stream: object,
    ) -> None:
        self._sealed = False
        self._binding_signature: tuple[object, ...] | None = None
        self._commands = commands
        self._poisoned = False
        self.evidence = evidence
        self.local_scene = local_scene
        self.transforms = transforms
        self.outputs = outputs
        self.stream = stream
        self._frozen_commands = commands
        self._frozen_owners = (local_scene, transforms, outputs, stream)
        self._frozen_evidence = evidence
        self._frozen_device_ordinal = evidence.device_ordinal
        self._frozen_stream_handle = evidence.stream_handle
        self._frozen_native_input_manifest_sha256 = evidence.native_input_manifest_sha256
        self._frozen_input_contract = evidence.input_contract
        self._frozen_dispatch_count = evidence.dispatch_count
        self._frozen_dispatch_plan_sha256 = evidence.dispatch_plan_sha256
        self._frozen_evidence_sha256 = evidence.sha256
        self._freeze()

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False) or getattr(self, "_binding_signature", None) is not None:
            raise AttributeError("Warp scene-update command bundle is immutable")
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        raise AttributeError("Warp scene-update command bundle is immutable")

    @property
    def dispatch_count(self) -> int:
        return len(self._commands)

    @property
    def poisoned(self) -> bool:
        return self._poisoned

    def validate_bound(self) -> None:
        """Revalidate the exact sealed command/storage graph outside timing."""

        self._validate_lightweight_bound()
        scene = self.local_scene
        transforms = self.transforms
        outputs = self.outputs
        _validate_warp_stream(self.stream, device=scene.device)
        if int(scene.device.ordinal) != self._frozen_device_ordinal:
            raise ValueError("Warp scene command device changed after sealing")
        if transforms.input_contract != self._frozen_input_contract:
            raise ValueError("Warp scene command input contract changed after sealing")
        _validate_warp_transform_inputs(scene, transforms)
        _validate_warp_world_scene_buffers(scene, outputs)
        input_arrays = _warp_scene_command_inputs(scene, transforms)
        output_arrays = _warp_world_scene_arrays(outputs)
        protected_arrays = _warp_local_scene_arrays(scene) + (
            ("x_world_R", transforms.x_world_R),
            ("x_world_r", transforms.x_world_r),
        )
        _validate_disjoint_warp_storage(protected_arrays, output_arrays)

        expected_evidence = WarpSceneUpdateCommandEvidence(
            schema_version=WARP_SCENE_COMMAND_SCHEMA_VERSION,
            operation_id=WARP_SCENE_UPDATE_OPERATION_ID,
            device_ordinal=int(scene.device.ordinal),
            stream_handle=int(self.stream.cuda_stream),
            native_input_manifest_sha256=self._frozen_native_input_manifest_sha256,
            scene_contract_sha256=scene.scene_contract_sha256,
            input_contract=self._frozen_input_contract,
            transform_replay_sha256=transforms.transform_replay_sha256,
            transform_frame_index=transforms.transform_frame_index,
            input_storage_sha256=_warp_storage_manifest_sha256(input_arrays),
            output_storage_sha256=_warp_storage_manifest_sha256(output_arrays),
            dispatch_count=len(self._commands),
            dispatch_plan_sha256=_canonical_sha256(_warp_scene_dispatch_plan(scene)),
        )
        if self.evidence != expected_evidence:
            raise ValueError("Warp scene command evidence changed after sealing")
        if self._bundle_signature() != self._binding_signature:
            raise ValueError("Warp scene command binding changed after sealing")

    def enqueue(self) -> None:
        """Launch only the frozen commands on the frozen stream."""

        if self._poisoned:
            raise RuntimeError("Warp scene-update command bundle is poisoned")
        try:
            self._validate_lightweight_bound()
            for command in self._commands:
                command.launch(stream=self.stream)
        except Exception:
            object.__setattr__(self, "_poisoned", True)
            raise

    def _freeze(self) -> None:
        if self._sealed or self._binding_signature is not None:
            raise RuntimeError("Warp scene-update command bundle seal is one-way")
        object.__setattr__(self, "_binding_signature", self._bundle_signature())
        object.__setattr__(self, "_sealed", True)

    def _validate_lightweight_bound(self) -> None:
        if not self._sealed or self._binding_signature is None:
            raise ValueError("Warp scene command bundle is not sealed")
        local_scene, transforms, outputs, stream = self._frozen_owners
        if (
            self.local_scene is not local_scene
            or self.transforms is not transforms
            or self.outputs is not outputs
            or self.stream is not stream
        ):
            raise ValueError("Warp scene command owners changed after sealing")
        if self._commands is not self._frozen_commands:
            raise ValueError("Warp scene command tuple changed after sealing")
        if self.evidence is not self._frozen_evidence:
            raise ValueError("Warp scene command evidence object changed after sealing")
        if (
            len(self._commands) != self._frozen_dispatch_count
            or self.evidence.dispatch_count != self._frozen_dispatch_count
        ):
            raise ValueError("Warp scene command dispatch count changed after sealing")
        if int(self.stream.cuda_stream) != self._frozen_stream_handle:
            raise ValueError("Warp scene command stream changed after sealing")

    def _bundle_signature(self) -> tuple[object, ...]:
        return (
            id(self._commands),
            tuple(id(command) for command in self._commands),
            id(self._frozen_commands),
            tuple(id(command) for command in self._frozen_commands),
            id(self.local_scene),
            id(self.transforms),
            id(self.outputs),
            id(self.stream),
            self._frozen_device_ordinal,
            self._frozen_stream_handle,
            self._frozen_native_input_manifest_sha256,
            self._frozen_input_contract,
            self._frozen_dispatch_count,
            self._frozen_dispatch_plan_sha256,
            self._frozen_evidence_sha256,
            self.evidence.sha256,
        )


class WarpSceneUpdateCommandFactory:
    """Seal caller-owned world outputs into allocation-free Warp commands."""

    __slots__ = (
        "_frozen_device_ordinal",
        "_frozen_input_contract",
        "_frozen_local_scene_storage_sha256",
        "_frozen_native_input_manifest_sha256",
        "_frozen_scene_contract_sha256",
        "_frozen_stream_handle",
        "_sealed",
        "input_contract",
        "local_scene",
        "native_input_manifest_sha256",
        "stream",
    )

    def __init__(
        self,
        local_scene: DeviceOpticalScene,
        *,
        stream: object,
        native_input_manifest_sha256: str,
        input_contract: str,
    ) -> None:
        _require_warp()
        if not local_scene.scene_contract_sha256:
            raise ValueError("Warp scene command requires scene-contract provenance")
        if not _sha256_text(native_input_manifest_sha256) or not input_contract:
            raise ValueError("Warp scene command provenance must be non-empty")
        _validate_warp_stream(stream, device=local_scene.device)
        self.local_scene = local_scene
        self.stream = stream
        self.native_input_manifest_sha256 = native_input_manifest_sha256
        self.input_contract = input_contract
        self._frozen_device_ordinal = int(local_scene.device.ordinal)
        self._frozen_stream_handle = int(stream.cuda_stream)
        self._frozen_scene_contract_sha256 = local_scene.scene_contract_sha256
        self._frozen_local_scene_storage_sha256 = _warp_storage_manifest_sha256(
            _warp_local_scene_arrays(local_scene)
        )
        self._frozen_native_input_manifest_sha256 = native_input_manifest_sha256
        self._frozen_input_contract = input_contract
        self._sealed = True

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("Warp scene-update command factory is immutable")
        object.__setattr__(self, name, value)

    def seal(
        self,
        *,
        transforms: WarpSceneTransformInputs,
        outputs: WarpWorldSceneBuffers,
        _context=None,
    ) -> WarpSceneUpdateCommandBundle:
        """Validate and record an immutable command bundle outside timing."""

        self._validate_frozen_state()
        scene = self.local_scene
        if transforms.input_contract != self.input_contract:
            raise ValueError("Warp transform input contract does not match the factory")
        _validate_warp_transform_inputs(scene, transforms)
        _validate_warp_world_scene_buffers(scene, outputs)
        input_arrays = _warp_scene_command_inputs(scene, transforms)
        output_arrays = _warp_world_scene_arrays(outputs)
        protected_arrays = _warp_local_scene_arrays(scene) + (
            ("x_world_R", transforms.x_world_R),
            ("x_world_r", transforms.x_world_r),
        )
        _validate_disjoint_warp_storage(protected_arrays, output_arrays)

        from optical_worker.native_setup import _warp_command_recording

        with _warp_command_recording(wp, scene.device, self.stream, context=_context) as record:
            commands: list[object] = []
            if scene.num_triangles > 0:
                commands.append(
                    record(
                        _update_triangles_derived_aabb_world_kernel,
                        dim=scene.num_triangles,
                        inputs=(
                            scene.geometry_vertices_local,
                            scene.triangle_vertex_indices,
                            scene.triangle_instance_index,
                            scene.instance_body_index,
                            scene.instance_X_body_geometry_R,
                            scene.instance_X_body_geometry_r,
                            transforms.x_world_R,
                            transforms.x_world_r,
                            int(transforms.env_idx),
                            outputs.triangle_v0_world,
                            outputs.triangle_e1_world,
                            outputs.triangle_e2_world,
                            outputs.triangle_normal_world,
                            outputs.triangle_aabb_min,
                            outputs.triangle_aabb_max,
                        ),
                    )
                )
            if scene.num_planes > 0:
                commands.append(
                    record(
                        _update_planes_world_kernel,
                        dim=scene.num_planes,
                        inputs=(
                            scene.plane_normal_local,
                            scene.plane_point_local,
                            scene.plane_instance_index,
                            scene.instance_body_index,
                            scene.instance_X_body_geometry_R,
                            scene.instance_X_body_geometry_r,
                            transforms.x_world_R,
                            transforms.x_world_r,
                            int(transforms.env_idx),
                            outputs.plane_normal_world,
                            outputs.plane_point_world,
                        ),
                    )
                )
        dispatches = _warp_scene_dispatch_plan(scene)

        evidence = WarpSceneUpdateCommandEvidence(
            schema_version=WARP_SCENE_COMMAND_SCHEMA_VERSION,
            operation_id=WARP_SCENE_UPDATE_OPERATION_ID,
            device_ordinal=int(scene.device.ordinal),
            stream_handle=int(self.stream.cuda_stream),
            native_input_manifest_sha256=self.native_input_manifest_sha256,
            scene_contract_sha256=scene.scene_contract_sha256,
            input_contract=self.input_contract,
            transform_replay_sha256=transforms.transform_replay_sha256,
            transform_frame_index=transforms.transform_frame_index,
            input_storage_sha256=_warp_storage_manifest_sha256(input_arrays),
            output_storage_sha256=_warp_storage_manifest_sha256(output_arrays),
            dispatch_count=len(commands),
            dispatch_plan_sha256=_canonical_sha256(dispatches),
        )
        return WarpSceneUpdateCommandBundle(
            commands=tuple(commands),
            evidence=evidence,
            local_scene=scene,
            transforms=transforms,
            outputs=outputs,
            stream=self.stream,
        )

    def _validate_frozen_state(self) -> None:
        scene = self.local_scene
        _validate_warp_stream(self.stream, device=scene.device)
        if int(scene.device.ordinal) != self._frozen_device_ordinal:
            raise ValueError("Warp scene factory device drifted after construction")
        if int(self.stream.cuda_stream) != self._frozen_stream_handle:
            raise ValueError("Warp scene factory stream drifted after construction")
        if scene.scene_contract_sha256 != self._frozen_scene_contract_sha256:
            raise ValueError("Warp scene factory contract drifted after construction")
        if (
            _warp_storage_manifest_sha256(_warp_local_scene_arrays(scene))
            != self._frozen_local_scene_storage_sha256
        ):
            raise ValueError("Warp scene factory storage drifted after construction")
        if (
            self.native_input_manifest_sha256 != self._frozen_native_input_manifest_sha256
            or self.input_contract != self._frozen_input_contract
        ):
            raise ValueError("Warp scene factory provenance drifted after construction")


def _warp_scene_dispatch_plan(scene: DeviceOpticalScene) -> list[dict[str, object]]:
    dispatches: list[dict[str, object]] = []
    if scene.num_triangles > 0:
        dispatches.append({"kernel": "update_triangles_derived_aabb_world", "dim": scene.num_triangles})
    if scene.num_planes > 0:
        dispatches.append({"kernel": "update_planes_world", "dim": scene.num_planes})
    return dispatches


def allocate_warp_world_scene_buffers(
    scene: DeviceOpticalScene,
    *,
    stream: object | None = None,
    _context=None,
) -> WarpWorldSceneBuffers:
    """Allocate one caller-owned C3 test/compatibility output bundle.

    This helper is intentionally not an arena and must run before native
    timing. The sealed command factory itself never allocates outputs.
    """

    _require_warp()
    if stream is not None:
        _validate_warp_stream(stream, device=scene.device)
    from optical_worker.native_setup import _output_array_allocation

    with _output_array_allocation(wp, scene.device, lane="warp", stream=stream, context=_context) as empty:
        arrays = {
            name: empty(name, (count, 3), wp.float64 if name in ('triangle_v0_world', 'triangle_e1_world', 'triangle_e2_world', 'plane_normal_world', 'plane_point_world') else wp.float32)
            for name, count in (
                ("triangle_v0_world", scene.num_triangles), ("triangle_e1_world", scene.num_triangles),
                ("triangle_e2_world", scene.num_triangles), ("triangle_normal_world", scene.num_triangles),
                ("plane_normal_world", scene.num_planes), ("plane_point_world", scene.num_planes),
                ("triangle_aabb_min", scene.num_triangles), ("triangle_aabb_max", scene.num_triangles),
            )
        }
    return WarpWorldSceneBuffers(
        device=scene.device,
        resources=(scene,),
        **arrays,
    )


def warp_replay_transform_inputs(
    x_world_R: object,
    x_world_r: object,
    *,
    frame_index: int,
    input_contract: str,
    transform_replay_sha256: str,
    resources: tuple[object, ...] = (),
) -> WarpSceneTransformInputs:
    """Create leading-env Warp views for one already-uploaded replay frame."""

    _require_warp()
    if not isinstance(frame_index, int) or isinstance(frame_index, bool) or frame_index < 0:
        raise ValueError("transform replay frame index must be a non-negative integer")
    if not input_contract or not _sha256_text(transform_replay_sha256):
        raise ValueError("transform replay provenance must be non-empty")
    _validate_warp_array(x_world_R, name="replay_x_world_R", dtype=wp.float32, ndim=4)
    _validate_warp_array(x_world_r, name="replay_x_world_r", dtype=wp.float32, ndim=3)
    if tuple(x_world_R.shape[:2]) != tuple(x_world_r.shape[:2]):
        raise ValueError("Warp replay transform shapes do not agree")
    if tuple(x_world_R.shape[2:]) != (3, 3) or tuple(x_world_r.shape[2:]) != (3,):
        raise ValueError("Warp replay transform trailing shapes are invalid")
    if frame_index >= int(x_world_R.shape[0]):
        raise IndexError("transform replay frame index is out of range")
    if int(x_world_R.shape[1]) == 0:
        # Warp 1.12 rejects slicing an array that has a later zero-length
        # dimension. Static zero-body replays never dereference transforms, so
        # retain the exact base allocation and select the frame as the env.
        rotation_view = x_world_R
        translation_view = x_world_r
        env_idx = frame_index
    else:
        rotation_view = x_world_R[frame_index : frame_index + 1]
        translation_view = x_world_r[frame_index : frame_index + 1]
        env_idx = 0
    return WarpSceneTransformInputs(
        x_world_R=rotation_view,
        x_world_r=translation_view,
        env_idx=env_idx,
        input_contract=input_contract,
        transform_replay_sha256=transform_replay_sha256,
        transform_frame_index=frame_index,
        resources=tuple(resources) + (x_world_R, x_world_r),
    )


def _validate_warp_transform_inputs(
    scene: DeviceOpticalScene,
    transforms: WarpSceneTransformInputs,
) -> None:
    if not isinstance(transforms, WarpSceneTransformInputs):
        raise TypeError("transforms must be WarpSceneTransformInputs")
    _validate_warp_array(
        transforms.x_world_R,
        name="x_world_R",
        dtype=wp.float32,
        ndim=4,
        device=scene.device,
    )
    _validate_warp_array(
        transforms.x_world_r,
        name="x_world_r",
        dtype=wp.float32,
        ndim=3,
        device=scene.device,
    )
    rotation_shape = tuple(transforms.x_world_R.shape)
    translation_shape = tuple(transforms.x_world_r.shape)
    if rotation_shape[2:] != (3, 3) or translation_shape[2:] != (3,):
        raise ValueError("Warp transform trailing shapes are invalid")
    if rotation_shape[:2] != translation_shape[:2]:
        raise ValueError("Warp transform env/body dimensions do not agree")
    if (
        not isinstance(transforms.env_idx, int)
        or isinstance(transforms.env_idx, bool)
        or transforms.env_idx < 0
        or transforms.env_idx >= rotation_shape[0]
    ):
        raise ValueError("Warp transform env index is out of range")
    if scene.max_body_index >= rotation_shape[1]:
        raise ValueError("Warp transforms do not cover the scene body indices")
    has_replay_hash = transforms.transform_replay_sha256 is not None
    has_replay_index = transforms.transform_frame_index is not None
    if has_replay_hash != has_replay_index:
        raise ValueError("Warp replay hash and frame index must appear together")
    if has_replay_hash:
        if not _sha256_text(transforms.transform_replay_sha256):
            raise ValueError("Warp transform replay hash is invalid")
        if (
            not isinstance(transforms.transform_frame_index, int)
            or isinstance(transforms.transform_frame_index, bool)
            or transforms.transform_frame_index < 0
        ):
            raise ValueError("Warp transform replay frame index is invalid")


def _validate_warp_world_scene_buffers(
    scene: DeviceOpticalScene,
    outputs: WarpWorldSceneBuffers,
) -> None:
    if not isinstance(outputs, WarpWorldSceneBuffers):
        raise TypeError("outputs must be WarpWorldSceneBuffers")
    if wp.get_device(outputs.device) != scene.device:
        raise ValueError("Warp world-scene output device does not match the local scene")
    expected = {
        "triangle_v0_world": scene.num_triangles,
        "triangle_e1_world": scene.num_triangles,
        "triangle_e2_world": scene.num_triangles,
        "triangle_normal_world": scene.num_triangles,
        "triangle_aabb_min": scene.num_triangles,
        "triangle_aabb_max": scene.num_triangles,
        "plane_normal_world": scene.num_planes,
        "plane_point_world": scene.num_planes,
    }
    for name, count in expected.items():
        _validate_warp_array(
            getattr(outputs, name),
            name=name,
            dtype=wp.float64 if name in ('triangle_v0_world', 'triangle_e1_world', 'triangle_e2_world', 'plane_normal_world', 'plane_point_world') else wp.float32,
            ndim=2,
            shape=(count, 3),
            device=scene.device,
        )


def _validate_warp_stream(stream: object, *, device: object) -> None:
    if getattr(stream, "device", None) != wp.get_device(device):
        raise ValueError("Warp prepared stream device does not match")
    handle = getattr(stream, "cuda_stream", None)
    if not isinstance(handle, int) or isinstance(handle, bool) or handle <= 0:
        raise ValueError("Warp prepared stream must expose a nonzero CUDA handle")


def _validate_warp_array(
    value: object,
    *,
    name: str,
    dtype: object,
    ndim: int,
    shape: tuple[int, ...] | None = None,
    device: object | None = None,
) -> None:
    if not isinstance(value, wp.array):
        raise TypeError(f"Warp array {name!r} has the wrong type")
    if value.dtype != dtype or int(value.ndim) != ndim:
        raise ValueError(f"Warp array {name!r} dtype/rank does not match")
    if shape is not None and tuple(value.shape) != shape:
        raise ValueError(f"Warp array {name!r} shape does not match")
    if not value.is_contiguous:
        raise ValueError(f"Warp array {name!r} must be contiguous")
    if device is not None and wp.get_device(value.device) != wp.get_device(device):
        raise ValueError(f"Warp array {name!r} is on the wrong device")
    if int(value.capacity) > 0 and int(value.ptr or 0) <= 0:
        raise ValueError(f"Warp array {name!r} has a null pointer")


def _warp_scene_command_inputs(
    scene: DeviceOpticalScene,
    transforms: WarpSceneTransformInputs,
) -> tuple[tuple[str, object], ...]:
    return (
        ("geometry_vertices_local", scene.geometry_vertices_local),
        ("triangle_vertex_indices", scene.triangle_vertex_indices),
        ("triangle_instance_index", scene.triangle_instance_index),
        ("plane_normal_local", scene.plane_normal_local),
        ("plane_point_local", scene.plane_point_local),
        ("plane_instance_index", scene.plane_instance_index),
        ("instance_body_index", scene.instance_body_index),
        ("instance_X_body_geometry_R", scene.instance_X_body_geometry_R),
        ("instance_X_body_geometry_r", scene.instance_X_body_geometry_r),
        ("x_world_R", transforms.x_world_R),
        ("x_world_r", transforms.x_world_r),
    )


def _warp_local_scene_arrays(
    scene: DeviceOpticalScene,
) -> tuple[tuple[str, object], ...]:
    return tuple((name, getattr(scene, name)) for name in _DEVICE_SCENE_ARRAY_NAMES)


def _warp_world_scene_arrays(
    outputs: WarpWorldSceneBuffers,
) -> tuple[tuple[str, object], ...]:
    return tuple(
        (name, getattr(outputs, name))
        for name in (
            "triangle_v0_world",
            "triangle_e1_world",
            "triangle_e2_world",
            "triangle_normal_world",
            "plane_normal_world",
            "plane_point_world",
            "triangle_aabb_min",
            "triangle_aabb_max",
        )
    )


def _validate_disjoint_warp_storage(
    inputs: tuple[tuple[str, object], ...],
    outputs: tuple[tuple[str, object], ...],
) -> None:
    seen_outputs: list[tuple[str, int, int]] = []
    input_ranges = tuple(
        (name, int(value.ptr or 0), int(value.capacity)) for name, value in inputs if int(value.capacity) > 0
    )
    for name, value in outputs:
        start = int(value.ptr or 0)
        size = int(value.capacity)
        if size <= 0:
            continue
        end = start + size
        for other_name, other_start, other_size in input_ranges:
            if start < other_start + other_size and other_start < end:
                raise ValueError(f"Warp output {name!r} overlaps input {other_name!r}")
        for other_name, other_start, other_end in seen_outputs:
            if start < other_end and other_start < end:
                raise ValueError(f"Warp outputs {name!r} and {other_name!r} overlap")
        seen_outputs.append((name, start, end))


def _warp_storage_manifest_sha256(values: tuple[tuple[str, object], ...]) -> str:
    return _canonical_sha256(
        [
            {
                "name": name,
                "device_ordinal": int(value.device.ordinal),
                "ptr": int(value.ptr or 0),
                "capacity": int(value.capacity),
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "strides": list(value.strides),
            }
            for name, value in values
        ]
    )


def _sha256_text(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


class DeviceOpticalSceneCache:
    """Device scene cache that realizes registry primitives for a GPU frame."""

    def __init__(self, registry: OpticalWorldRegistry, *, device=None, stream=None) -> None:
        self.scene = build_device_optical_scene(registry, device=device, stream=stream)

    def snapshot_from_gpu_frame(
        self,
        frame: GpuPublishedFrame,
        *,
        env_idx: int = 0,
        stream=None,
        include_aabb: bool = False,
    ) -> DeviceOpticalSceneSnapshot:
        return update_device_optical_scene_from_gpu_frame(
            self.scene,
            frame,
            env_idx=env_idx,
            stream=stream,
            include_aabb=include_aabb,
        )


def build_device_optical_scene(
    registry: OpticalWorldRegistry,
    *,
    device=None,
    stream=None,
) -> DeviceOpticalScene:
    """Upload registry geometry and metadata into long-lived device buffers."""
    packed_scene = PackedOpticalSceneHost.from_registry(registry)
    return build_device_optical_scene_from_packed(
        packed_scene,
        device=device,
        stream=stream,
    )


def build_device_optical_scene_from_packed(
    packed_scene: PackedOpticalSceneHost,
    *,
    device=None,
    stream=None,
) -> DeviceOpticalScene:
    """Upload one canonical host contract into Warp-owned CUDA arrays."""
    _require_warp()
    validate_indexed_geometry(packed_scene)
    wp.init()
    resolved_device = wp.get_device("cuda:0" if device is None else device)
    arrays: dict[str, object] = {}
    with _scoped_stream(stream):
        for name, value in packed_scene.arrays.items():
            arrays[name] = _wp_array(
                value,
                dtype=_warp_dtype_for_numpy(value.dtype),
                device=resolved_device,
            )
        return device_optical_scene_from_arrays(
            packed_scene,
            arrays,
            device=resolved_device,
            resources=(),
        )


def device_optical_scene_from_arrays(
    packed_scene: PackedOpticalSceneHost,
    arrays: dict[str, object],
    *,
    device: object,
    resources: tuple[object, ...],
) -> DeviceOpticalScene:
    """Wrap exact caller-owned Warp arrays without allocating or copying."""
    _require_warp()
    resolved_device = wp.get_device(device)
    validate_indexed_geometry(packed_scene)
    if set(arrays) != set(packed_scene.arrays):
        raise ValueError("Warp scene arrays do not match the packed scene contract")
    for name, host in packed_scene.arrays.items():
        value = arrays[name]
        if wp.get_device(value.device).ordinal != resolved_device.ordinal:
            raise ValueError(f"Warp scene array {name!r} is on the wrong device")
        if tuple(value.shape) != tuple(host.shape):
            raise ValueError(f"Warp scene array {name!r} shape does not match")
        if value.dtype != _warp_dtype_for_numpy(host.dtype):
            raise ValueError(f"Warp scene array {name!r} dtype does not match")
    return DeviceOpticalScene(
        device=resolved_device,
        role_table=DeviceOpticalRoleTable.from_mapping(packed_scene.role_to_mask),
        num_instances=packed_scene.num_instances,
        num_materials=packed_scene.num_materials,
        num_lights=packed_scene.num_lights,
        num_triangles=packed_scene.num_triangles,
        num_planes=packed_scene.num_planes,
        max_body_index=packed_scene.max_body_index,
        **{name: arrays[name] for name in packed_scene.arrays},
        scene_contract_sha256=packed_scene.scene_contract_sha256,
        resources=tuple(resources) + (packed_scene,),
    )


def update_device_optical_scene_from_gpu_frame(
    scene: DeviceOpticalScene,
    frame: GpuPublishedFrame,
    *,
    env_idx: int = 0,
    stream=None,
    include_aabb: bool = False,
) -> DeviceOpticalSceneSnapshot:
    """Realize per-frame world primitive buffers from a borrowed GPU frame."""
    _require_warp()
    if env_idx < 0:
        raise ValueError("env_idx must be >= 0")
    _validate_gpu_frame_transform_shape(scene, frame, env_idx=env_idx)
    _wait_on_event(frame.ready_event, stream=stream, device=scene.device)

    with _scoped_stream(stream):
        triangle_v0_world = wp.zeros((scene.num_triangles, 3), dtype=wp.float64, device=scene.device)
        triangle_e1_world = wp.zeros((scene.num_triangles, 3), dtype=wp.float64, device=scene.device)
        triangle_e2_world = wp.zeros((scene.num_triangles, 3), dtype=wp.float64, device=scene.device)
        triangle_normal_world = wp.zeros((scene.num_triangles, 3), dtype=wp.float32, device=scene.device)
        triangle_aabb_min = None
        triangle_aabb_max = None
        if include_aabb:
            triangle_aabb_min = wp.zeros((scene.num_triangles, 3), dtype=wp.float32, device=scene.device)
            triangle_aabb_max = wp.zeros((scene.num_triangles, 3), dtype=wp.float32, device=scene.device)
        plane_normal_world = wp.zeros((scene.num_planes, 3), dtype=wp.float64, device=scene.device)
        plane_point_world = wp.zeros((scene.num_planes, 3), dtype=wp.float64, device=scene.device)

        if scene.num_triangles > 0:
            wp.launch(
                _update_triangles_derived_aabb_world_kernel
                if include_aabb
                else _update_triangles_derived_world_kernel,
                dim=scene.num_triangles,
                inputs=[
                    scene.geometry_vertices_local,
                    scene.triangle_vertex_indices,
                    scene.triangle_instance_index,
                    scene.instance_body_index,
                    scene.instance_X_body_geometry_R,
                    scene.instance_X_body_geometry_r,
                    frame.x_world_R_wp,
                    frame.x_world_r_wp,
                    int(env_idx),
                    triangle_v0_world,
                    triangle_e1_world,
                    triangle_e2_world,
                    triangle_normal_world,
                ]
                + ([triangle_aabb_min, triangle_aabb_max] if include_aabb else []),
                device=scene.device,
                stream=stream,
            )
        if scene.num_planes > 0:
            wp.launch(
                _update_planes_world_kernel,
                dim=scene.num_planes,
                inputs=[
                    scene.plane_normal_local,
                    scene.plane_point_local,
                    scene.plane_instance_index,
                    scene.instance_body_index,
                    scene.instance_X_body_geometry_R,
                    scene.instance_X_body_geometry_r,
                    frame.x_world_R_wp,
                    frame.x_world_r_wp,
                    int(env_idx),
                    plane_normal_world,
                    plane_point_world,
                ],
                device=scene.device,
                stream=stream,
            )

        ready_event = (stream or wp.get_stream(scene.device)).record_event()

    return DeviceOpticalSceneSnapshot(
        frame_id=frame.frame_id,
        sim_time=frame.sim_time,
        env_idx=env_idx,
        scene=scene,
        triangle_v0_world=triangle_v0_world,
        triangle_e1_world=triangle_e1_world,
        triangle_e2_world=triangle_e2_world,
        triangle_normal_world=triangle_normal_world,
        triangle_aabb_min=triangle_aabb_min,
        triangle_aabb_max=triangle_aabb_max,
        plane_normal_world=plane_normal_world,
        plane_point_world=plane_point_world,
        ready_event=ready_event,
        resources=(frame,),
    )


def _validate_gpu_frame_transform_shape(
    scene: DeviceOpticalScene,
    frame: GpuPublishedFrame,
    *,
    env_idx: int,
) -> None:
    rotation_shape = tuple(frame.x_world_R_wp.shape)
    translation_shape = tuple(frame.x_world_r_wp.shape)
    if len(rotation_shape) != 4 or rotation_shape[2:] != (3, 3):
        raise ValueError("GpuPublishedFrame.x_world_R_wp must have shape (num_envs, num_bodies, 3, 3)")
    if len(translation_shape) != 3 or translation_shape[2] != 3:
        raise ValueError("GpuPublishedFrame.x_world_r_wp must have shape (num_envs, num_bodies, 3)")
    if rotation_shape[:2] != translation_shape[:2]:
        raise ValueError("GpuPublishedFrame transform arrays must agree on env/body dimensions")
    if env_idx >= rotation_shape[0]:
        raise IndexError(f"env_idx {env_idx} is out of range for GpuPublishedFrame transforms")
    if scene.max_body_index >= rotation_shape[1]:
        raise IndexError(
            f"body_index {scene.max_body_index} is out of range for GpuPublishedFrame transforms"
        )


def _wp_array(values: np.ndarray, *, dtype, device):
    return wp.array(values, dtype=dtype, device=device)


def _warp_dtype_for_numpy(dtype):
    mapping = {
        np.dtype(np.bool_): wp.bool,
        np.dtype(np.int8): wp.int8,
        np.dtype(np.int16): wp.int16,
        np.dtype(np.int32): wp.int32,
        np.dtype(np.int64): wp.int64,
        np.dtype(np.uint8): wp.uint8,
        np.dtype(np.uint16): wp.uint16,
        np.dtype(np.uint32): wp.uint32,
        np.dtype(np.uint64): wp.uint64,
        np.dtype(np.float16): wp.float16,
        np.dtype(np.float32): wp.float32,
        np.dtype(np.float64): wp.float64,
    }
    try:
        return mapping[np.dtype(dtype)]
    except KeyError as exc:
        raise ValueError(f"unsupported Warp scene dtype {np.dtype(dtype)}") from exc


def _scoped_stream(stream):
    if stream is None:
        return nullcontext()
    return wp.ScopedStream(stream)


def _wait_on_event(event, *, stream, device) -> None:
    if event is None:
        return
    (stream or wp.get_stream(device)).wait_event(event)


def _require_warp() -> None:
    if not _HAS_WARP:
        raise ImportError("DeviceOpticalSceneCache requires the optional warp package")


if _HAS_WARP:
    wp.set_module_options({"fuse_fp": False})
    _QUERY_BUILD_EPS = wp.constant(wp.float64(BUILD_EPS))

    @wp.func
    def _compose_world_transform(
        instance: int, instance_body_index: wp.array(dtype=wp.int32),
        instance_R: wp.array2d(dtype=wp.float64), instance_r: wp.array2d(dtype=wp.float64),
        frame_R: wp.array4d(dtype=wp.float32), frame_r: wp.array3d(dtype=wp.float32), env: int,
    ):
        Rwb = wp.mat33d()
        for axis in range(3):
            Rwb[axis, axis] = wp.float64(1.0)
        rwb = wp.vec3d()
        body = instance_body_index[instance]
        if body >= 0:
            for row in range(3):
                rwb[row] = wp.float64(frame_r[env, body, row])
                for col in range(3):
                    Rwb[row, col] = wp.float64(frame_R[env, body, row, col])
        R = wp.mat33d()
        r = wp.vec3d()
        for row in range(3):
            for col in range(3):
                R[row, col] = ((Rwb[row, 0] * instance_R[instance, col]
                               + Rwb[row, 1] * instance_R[instance, 3 + col])
                               + Rwb[row, 2] * instance_R[instance, 6 + col])
            r[row] = ((Rwb[row, 0] * instance_r[instance, 0]
                       + Rwb[row, 1] * instance_r[instance, 1])
                       + Rwb[row, 2] * instance_r[instance, 2]) + rwb[row]
        return R, r

    @wp.func
    def _transform_source_point(R: wp.mat33d, r: wp.vec3d, p: wp.vec3d):
        result = wp.vec3d()
        for row in range(3):
            result[row] = ((R[row, 0] * p[0] + R[row, 1] * p[1]) + R[row, 2] * p[2]) + r[row]
        return result

    @wp.func
    def _indexed_world_triangle(
        primitive: int, vertices: wp.array2d(dtype=wp.float64), indices: wp.array2d(dtype=wp.int64),
        triangle_instance: wp.array(dtype=wp.int32), instance_body: wp.array(dtype=wp.int32),
        instance_R: wp.array2d(dtype=wp.float64), instance_r: wp.array2d(dtype=wp.float64),
        frame_R: wp.array4d(dtype=wp.float32), frame_r: wp.array3d(dtype=wp.float32), env: int,
    ):
        R, r = _compose_world_transform(triangle_instance[primitive], instance_body,
                                        instance_R, instance_r, frame_R, frame_r, env)
        i0 = indices[primitive, 0]
        i1 = indices[primitive, 1]
        i2 = indices[primitive, 2]
        p0 = _transform_source_point(R, r, wp.vec3d(vertices[i0, 0], vertices[i0, 1], vertices[i0, 2]))
        p1 = _transform_source_point(R, r, wp.vec3d(vertices[i1, 0], vertices[i1, 1], vertices[i1, 2]))
        p2 = _transform_source_point(R, r, wp.vec3d(vertices[i2, 0], vertices[i2, 1], vertices[i2, 2]))
        return p0, p1, p2

    @wp.func
    def _write_triangle_world(
        primitive: int, p0: wp.vec3d, p1: wp.vec3d, p2: wp.vec3d,
        v0_world: wp.array2d(dtype=wp.float64), e1_world: wp.array2d(dtype=wp.float64),
        e2_world: wp.array2d(dtype=wp.float64), normal_world: wp.array2d(dtype=wp.float32),
    ):
        e1 = p1 - p0
        e2 = p2 - p0
        normal = wp.cross(e1, e2)
        norm = wp.sqrt((normal[0]*normal[0] + normal[1]*normal[1]) + normal[2]*normal[2])
        unit = wp.vec3d()
        if norm > _QUERY_BUILD_EPS:
            unit = normal / norm
        for axis in range(3):
            v0_world[primitive, axis] = p0[axis]
            e1_world[primitive, axis] = e1[axis]
            e2_world[primitive, axis] = e2[axis]
            normal_world[primitive, axis] = wp.float32(unit[axis])

    @wp.func
    def _outward_bound(value: wp.float64, upper: int):
        rounded = wp.float32(value)
        if upper == 0 and wp.float64(rounded) > value:
            if rounded == wp.float32(0.0):
                rounded = wp.cast(wp.int32(-2147483647), wp.float32)
            elif rounded > wp.float32(0.0):
                rounded = wp.cast(wp.cast(rounded, wp.int32) - wp.int32(1), wp.float32)
            else:
                rounded = wp.cast(wp.cast(rounded, wp.int32) + wp.int32(1), wp.float32)
        elif upper == 1 and wp.float64(rounded) < value:
            if rounded == wp.float32(0.0):
                rounded = wp.cast(wp.int32(1), wp.float32)
            elif rounded > wp.float32(0.0):
                rounded = wp.cast(wp.cast(rounded, wp.int32) + wp.int32(1), wp.float32)
            else:
                rounded = wp.cast(wp.cast(rounded, wp.int32) - wp.int32(1), wp.float32)
        return rounded

    @wp.kernel
    def _update_triangles_derived_aabb_world_kernel(
        geometry_vertices_local: wp.array2d(dtype=wp.float64),
        triangle_vertex_indices: wp.array2d(dtype=wp.int64),
        triangle_instance_index: wp.array(dtype=wp.int32),
        instance_body_index: wp.array(dtype=wp.int32),
        instance_X_body_geometry_R: wp.array2d(dtype=wp.float64),
        instance_X_body_geometry_r: wp.array2d(dtype=wp.float64),
        frame_x_world_R: wp.array4d(dtype=wp.float32),
        frame_x_world_r: wp.array3d(dtype=wp.float32), env_idx: int,
        triangle_v0_world: wp.array2d(dtype=wp.float64),
        triangle_e1_world: wp.array2d(dtype=wp.float64),
        triangle_e2_world: wp.array2d(dtype=wp.float64),
        triangle_normal_world: wp.array2d(dtype=wp.float32),
        triangle_aabb_min: wp.array2d(dtype=wp.float32),
        triangle_aabb_max: wp.array2d(dtype=wp.float32),
    ):
        primitive = wp.tid()
        p0, p1, p2 = _indexed_world_triangle(primitive, geometry_vertices_local, triangle_vertex_indices,
            triangle_instance_index, instance_body_index, instance_X_body_geometry_R,
            instance_X_body_geometry_r, frame_x_world_R, frame_x_world_r, env_idx)
        _write_triangle_world(primitive, p0, p1, p2, triangle_v0_world, triangle_e1_world,
                              triangle_e2_world, triangle_normal_world)
        for axis in range(3):
            triangle_aabb_min[primitive, axis] = _outward_bound(wp.min(p0[axis], wp.min(p1[axis], p2[axis])), 0)
            triangle_aabb_max[primitive, axis] = _outward_bound(wp.max(p0[axis], wp.max(p1[axis], p2[axis])), 1)

    @wp.kernel
    def _update_triangles_derived_world_kernel(
        geometry_vertices_local: wp.array2d(dtype=wp.float64),
        triangle_vertex_indices: wp.array2d(dtype=wp.int64),
        triangle_instance_index: wp.array(dtype=wp.int32),
        instance_body_index: wp.array(dtype=wp.int32),
        instance_X_body_geometry_R: wp.array2d(dtype=wp.float64),
        instance_X_body_geometry_r: wp.array2d(dtype=wp.float64),
        frame_x_world_R: wp.array4d(dtype=wp.float32),
        frame_x_world_r: wp.array3d(dtype=wp.float32), env_idx: int,
        triangle_v0_world: wp.array2d(dtype=wp.float64),
        triangle_e1_world: wp.array2d(dtype=wp.float64),
        triangle_e2_world: wp.array2d(dtype=wp.float64),
        triangle_normal_world: wp.array2d(dtype=wp.float32),
    ):
        primitive = wp.tid()
        p0, p1, p2 = _indexed_world_triangle(primitive, geometry_vertices_local, triangle_vertex_indices,
            triangle_instance_index, instance_body_index, instance_X_body_geometry_R,
            instance_X_body_geometry_r, frame_x_world_R, frame_x_world_r, env_idx)
        _write_triangle_world(primitive, p0, p1, p2, triangle_v0_world, triangle_e1_world,
                              triangle_e2_world, triangle_normal_world)

    @wp.kernel
    def _update_planes_world_kernel(
        plane_normal_local: wp.array2d(dtype=wp.float64),
        plane_point_local: wp.array2d(dtype=wp.float64),
        plane_instance_index: wp.array(dtype=wp.int32), instance_body_index: wp.array(dtype=wp.int32),
        instance_X_body_geometry_R: wp.array2d(dtype=wp.float64),
        instance_X_body_geometry_r: wp.array2d(dtype=wp.float64),
        frame_x_world_R: wp.array4d(dtype=wp.float32), frame_x_world_r: wp.array3d(dtype=wp.float32),
        env_idx: int, plane_normal_world: wp.array2d(dtype=wp.float64),
        plane_point_world: wp.array2d(dtype=wp.float64),
    ):
        primitive = wp.tid()
        R, r = _compose_world_transform(plane_instance_index[primitive], instance_body_index,
            instance_X_body_geometry_R, instance_X_body_geometry_r, frame_x_world_R, frame_x_world_r, env_idx)
        point = _transform_source_point(R, r, wp.vec3d(plane_point_local[primitive, 0],
            plane_point_local[primitive, 1], plane_point_local[primitive, 2]))
        normal = _transform_source_point(R, wp.vec3d(), wp.vec3d(plane_normal_local[primitive, 0],
            plane_normal_local[primitive, 1], plane_normal_local[primitive, 2]))
        norm = wp.sqrt((normal[0]*normal[0] + normal[1]*normal[1]) + normal[2]*normal[2])
        if norm > _QUERY_BUILD_EPS:
            normal = normal / norm
        else:
            normal = wp.vec3d()
        for axis in range(3):
            plane_point_world[primitive, axis] = point[axis]
            plane_normal_world[primitive, axis] = normal[axis]

else:
    def _update_triangles_derived_world_kernel(*args, **kwargs):  # pragma: no cover
        raise ImportError("DeviceOpticalSceneCache requires the optional warp package")

    def _update_planes_world_kernel(*args, **kwargs):  # pragma: no cover
        raise ImportError("DeviceOpticalSceneCache requires the optional warp package")
