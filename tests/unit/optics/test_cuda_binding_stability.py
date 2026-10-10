"""CPU tensor metadata with production CUDA binding capture and checks."""

from contextlib import nullcontext
from dataclasses import dataclass
from types import SimpleNamespace as NS

import numpy as np
import pytest

from optics import cuda_lbvh as lbvh
from optics import cuda_scene as scene
from optics import cuda_timeline as timeline
from tests.unit.optics.test_guarded_native_readback import _CudaArray


@pytest.fixture
def dynamic_bindings(monkeypatch):
    @dataclass(frozen=True)
    class Device:
        type: str = "cuda"
        index: int = 1

    device = Device()

    class Tensor(_CudaArray):
        is_cuda = True

        def dim(self):
            return len(self.shape)

    def tensor(shape, dtype=np.float32):
        return Tensor(np.zeros(shape, dtype=dtype), device)

    runtime = NS(Tensor=Tensor, device=lambda *args: device,
        float32=np.dtype("float32"), float64=np.dtype("float64"), int32=np.dtype("int32"),
        int64=np.dtype("int64"), uint8=np.dtype("uint8"),
        empty=lambda shape, dtype, **kwargs: tensor(shape, dtype),
        cuda=NS(is_available=lambda: True, device_count=lambda: 2,
                device=lambda value: nullcontext(), stream=lambda value: nullcontext()))
    monkeypatch.setattr(timeline, "torch", runtime)
    monkeypatch.setattr(scene, "_torch_module", lambda: runtime)
    monkeypatch.setattr(lbvh, "torch", runtime)
    monkeypatch.setattr(lbvh, "_HAS_TORCH_CUDA_LBVH_DEPS", True)
    context = timeline.CudaOpticalExecutionContext(device=device, stream=NS(device=device, cuda_stream=17))
    arrays = {
        "geometry_vertices_local": tensor((3, 3), np.float64),
        "triangle_vertex_indices": tensor((1, 3), np.int64),
        "triangle_instance_index": tensor((1,), np.int32),
        "plane_normal_local": tensor((1, 3), np.float64),
        "plane_point_local": tensor((1, 3), np.float64),
        "plane_instance_index": tensor((1,), np.int32),
        "instance_body_index": tensor((1,), np.int32),
        "instance_X_body_geometry_R": tensor((1, 9), np.float64),
        "instance_X_body_geometry_r": tensor((1, 3), np.float64),
    }
    local = scene.CudaOpticalLocalScene(device_ordinal=1, arrays=arrays, role_to_mask={},
        num_instances=1, num_materials=1, num_lights=0, num_triangles=1, num_planes=1,
        max_body_index=0, scene_contract_sha256="a" * 64, ready_event=None, resources=())
    calls = []

    class Event:
        def __init__(self):
            self.device, self.cuda_event = device, id(self)

        def record(self, stream):
            assert stream is context.stream
            calls.append("record")

    runtime.cuda.Event = Event
    module = NS(update_triangles_world_aabb_out=lambda *args: calls.append("triangles"),
                update_planes_world_out=lambda *args: calls.append("planes"),
                cuda_lbvh_rebuild_out=lambda *args: calls.append("lbvh"))
    scene_factory = scene.CudaSceneUpdateStageFactory(context=context, local_scene=local, module=module,
        native_input_manifest_sha256="b" * 64, input_contract="canonical_native_replay_v1",
        _prepared_extension_token=scene._PREPARED_EXTENSION_TOKEN)
    transforms = scene.CudaSceneTransformInputs(x_world_R=tensor((1, 1, 3, 3)), x_world_r=tensor((1, 1, 3)),
        env_idx=0, input_contract="canonical_native_replay_v1", transform_replay_sha256="c" * 64,
        transform_frame_index=0)
    world = scene.allocate_cuda_world_scene_buffers(local)
    scene_stage = scene_factory.bind(transforms=transforms, outputs=world)
    # CPU fixture sizing, not evidence of the real CUB sizing query/alignment.
    requirements = lbvh.CudaLbvhWorkspaceRequirements(lbvh.CUDA_LBVH_WORKSPACE_SCHEMA_VERSION,
        lbvh.CUDA_LBVH_EXTENSION_ABI_VERSION, lbvh._CUDA_LBVH_EXTENSION_SOURCE_SHA256,
        1, (9, 0), 1, 1, 128, 128, 128, 1)
    factory = lbvh.CudaLbvhRebuildStageFactory(context=context, primitive_count=1,
        requirements=requirements, module=module, native_input_manifest_sha256="b" * 64,
        scene_contract_sha256="a" * 64, input_contract="canonical_native_replay_v1",
        _prepared_extension_token=lbvh._PREPARED_EXTENSION_TOKEN)
    outputs = lbvh.allocate_cuda_lbvh_buffers(primitive_count=1, device=device)
    workspace = lbvh.CudaLbvhWorkspace(tensor((128,), np.uint8), requirements)
    lbvh_stage = factory.bind(world=world, outputs=outputs, workspace=workspace)
    return NS(context=context, scene=scene_stage, lbvh=lbvh_stage, arrays=arrays,
              tensor=tensor, calls=calls, Device=Device, event=Event())


def test_dynamic_stability_does_not_rebuild_hashes_or_manifests(dynamic_bindings, monkeypatch):
    x = dynamic_bindings

    def late_rebuild(*args, **kwargs):
        pytest.fail("execution rebuilt a binding or hashed a storage manifest")

    monkeypatch.setattr(scene, "_canonical_sha256", late_rebuild)
    monkeypatch.setattr(lbvh, "_canonical_sha256", late_rebuild)
    for stage in (x.scene, x.lbvh):
        stage._assert_cuda_transaction_stage_stable(x.context)
    assert not x.calls


def test_dynamic_transaction_uses_only_prepared_bindings(dynamic_bindings, monkeypatch):
    x = dynamic_bindings
    transaction = x.context.begin_transaction(stages=(x.scene, x.lbvh), completion_event=x.event)

    def late_rebuild(*args, **kwargs):
        pytest.fail("dynamic execution rebuilt prepared binding metadata")

    monkeypatch.setattr(scene, "_canonical_sha256", late_rebuild)
    monkeypatch.setattr(lbvh, "_canonical_sha256", late_rebuild)
    monkeypatch.setattr(timeline._CudaBindingSnapshot, "capture", late_rebuild)
    x.scene.enqueue(transaction)
    x.lbvh.enqueue(transaction)
    assert transaction.finish().event is x.event
    assert x.calls == ["triangles", "planes", "lbvh", "record"]


@pytest.mark.parametrize("stage_name", ["scene", "lbvh"])
@pytest.mark.parametrize(
    "drift", ["copy", "pointer", "shape", "stride", "dtype", "device", "field", "scalar"],
)
def test_dynamic_stability_rejects_changed_real_binding(dynamic_bindings, stage_name, drift):
    x = dynamic_bindings
    stage = getattr(x, stage_name)
    tensor = stage.outputs.triangle_v0_world if stage_name == "scene" else stage.outputs.node_bounds_min
    if drift == "copy":
        copied = object.__new__(type(stage))
        for name in stage.__slots__:
            object.__setattr__(copied, name, getattr(stage, name))
        stage = copied
    elif drift == "pointer":
        tensor.storage.data = tensor.storage.data.copy()
    elif drift == "shape":
        tensor.shape = (3, 1)
    elif drift == "stride":
        tensor.data = tensor.data[:, ::-1]
    elif drift == "dtype":
        tensor.dtype = np.dtype("int32")
    elif drift == "device":
        tensor.device = x.Device(index=0)
    elif drift == "field":
        field = "triangle_v0_world" if stage_name == "scene" else "node_bounds_min"
        object.__setattr__(stage.outputs, field, x.tensor((1, 3)))
    else:
        if stage_name == "scene":
            object.__setattr__(stage.transforms, "env_idx", 1)
        else:
            object.__setattr__(stage.workspace.requirements, "required_bytes", 64)
    with pytest.raises(ValueError, match="changed"):
        stage._assert_cuda_transaction_stage_stable(x.context)
    assert not x.calls


def test_dynamic_stability_checks_scene_mapping_and_workspace(dynamic_bindings):
    x = dynamic_bindings
    x.arrays["geometry_vertices_local"] = x.tensor((3, 3), np.float64)
    with pytest.raises(ValueError, match="changed"):
        x.scene._assert_cuda_transaction_stage_stable(x.context)
    x.lbvh.workspace.storage.storage.data = x.lbvh.workspace.storage.storage.data.copy()
    with pytest.raises(ValueError, match="changed"):
        x.lbvh._assert_cuda_transaction_stage_stable(x.context)


def test_dynamic_stability_accepts_equal_device_value_and_mutable_tensor_contents(dynamic_bindings):
    x = dynamic_bindings
    x.context.device = x.Device()
    x.scene.outputs.triangle_v0_world.data.fill(7)
    x.lbvh.outputs.node_bounds_min.data.fill(3)
    x.scene._assert_cuda_transaction_stage_stable(x.context)
    x.lbvh._assert_cuda_transaction_stage_stable(x.context)


@pytest.mark.parametrize("owner_name", ["transforms", "scene_outputs", "world", "lbvh_outputs", "workspace"])
def test_dynamic_preparation_rejects_mutable_resource_owner_sequences(dynamic_bindings, owner_name):
    from dataclasses import replace

    x = dynamic_bindings
    if owner_name in {"transforms", "scene_outputs"}:
        transforms, outputs = x.scene.transforms, x.scene.outputs
        if owner_name == "transforms":
            transforms = replace(transforms, resources=[])
        else:
            outputs = replace(outputs, resources=[])
        with pytest.raises(TypeError, match="immutable tuples"):
            x.scene.factory.bind(transforms=transforms, outputs=outputs)
    else:
        world, outputs, workspace = x.lbvh.world, x.lbvh.outputs, x.lbvh.workspace
        if owner_name == "world":
            world = replace(world, resources=[])
        elif owner_name == "lbvh_outputs":
            outputs = replace(outputs, resources=[])
        else:
            workspace = replace(workspace, resources=[])
        with pytest.raises(TypeError, match="immutable tuples"):
            x.lbvh.factory.bind(world=world, outputs=outputs, workspace=workspace)
    assert not x.calls
