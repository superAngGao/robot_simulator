"""Explicit physics runtime owners for Optical Pipeline Lab scenarios."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import numpy as np

from . import dynamic_frames

PhysicsFrameStepFn = Callable[[int], object]
HeightForFrame = Callable[[int], float]


@dataclass
class PhysicsLabScenarioRuntime:
    """Own the lifecycle boundary for one explicit physics-backed lab scenario."""

    engine: object
    registry: object
    base_frame: object
    step_frame_fn: PhysicsFrameStepFn
    bounds_min: object | None = None
    bounds_max: object | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)
    close_fn: Callable[[], None] | None = None
    _closed: bool = field(default=False, init=False)

    @property
    def closed(self) -> bool:
        """Return whether this runtime owner has been closed."""

        return self._closed

    def step_frame(self, frame_index: int) -> object:
        """Advance/select physics time for one frame and return its published frame."""

        if self._closed:
            raise RuntimeError("PhysicsLabScenarioRuntime is closed")
        return self.step_frame_fn(frame_index)

    def close(self) -> None:
        """Release runtime-owner resources if the underlying engine exposes cleanup."""

        if self._closed:
            return
        if self.close_fn is not None:
            self.close_fn()
        else:
            _close_engine_if_available(self.engine)
        self._closed = True

    def __enter__(self) -> "PhysicsLabScenarioRuntime":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def create_physics_body_triangle_lab_runtime(
    *,
    device: str = "cuda:0",
    initial_height: float = 0.5,
    height_for_frame: HeightForFrame | None = None,
    dt: float = 1.0e-4,
    bounds_min: object | None = None,
    bounds_max: object | None = None,
    metadata: Mapping[str, object] | None = None,
    synchronize_event: Callable[[object], None] | None = None,
) -> PhysicsLabScenarioRuntime:
    """Create the narrow synthetic body-triangle physics runtime owner.

    Scripted stepping teleports the root height (`q[6]`) for each frame instead
    of integrating from the previous frame. This keeps the smoke deterministic
    and focused on runtime/render ownership rather than dynamics quality.
    """

    merged = _merge_single_ball_model(_build_ball_model())
    engine = _create_gpu_engine(merged, device=device)
    base_frame = _step_engine_to_body_height(
        engine,
        merged,
        body_height=float(initial_height),
        dt=float(dt),
        synchronize_event=synchronize_event,
    )
    registry = dynamic_frames.make_body_bound_triangle_registry(geometry_z_offset=0.25)

    def step_frame(frame_index: int) -> object:
        body_height = (
            float(height_for_frame(frame_index)) if height_for_frame is not None else float(initial_height)
        )
        return _step_engine_to_body_height(
            engine,
            merged,
            body_height=body_height,
            dt=float(dt),
            synchronize_event=synchronize_event,
        )

    return PhysicsLabScenarioRuntime(
        engine=engine,
        registry=registry,
        base_frame=base_frame,
        step_frame_fn=step_frame,
        bounds_min=(-0.2, -0.2, 0.0) if bounds_min is None else bounds_min,
        bounds_max=(0.4, 0.4, 1.2) if bounds_max is None else bounds_max,
        metadata={
            "producer": "gpu_engine",
            "runtime_owner": "physics_body_triangle_lab",
            **dict(metadata or {}),
        },
    )


def _step_engine_to_body_height(
    engine: object,
    merged: object,
    *,
    body_height: float,
    dt: float,
    synchronize_event: Callable[[object], None] | None,
) -> object:
    q, _ = merged.tree.default_state()
    q = np.asarray(q).copy()
    q[6] = float(body_height)
    qdot = np.zeros(merged.nv, dtype=q.dtype)
    engine.step(q=q, qdot=qdot, dt=float(dt))
    frame = engine.latest_published_frame()
    if frame is None:
        raise RuntimeError("physics engine did not publish a frame")
    ready_event = getattr(frame, "ready_event", None)
    if synchronize_event is not None and ready_event is not None:
        synchronize_event(ready_event)
    return frame


def _close_engine_if_available(engine: object) -> None:
    for method_name in ("close", "destroy"):
        method = getattr(engine, method_name, None)
        if callable(method):
            method()
            return


def _build_ball_model():
    from physics.geometry import BodyCollisionGeometry, ShapeInstance, SphereShape
    from physics.joint import FreeJoint
    from physics.robot_tree import Body, RobotTreeNumpy
    from physics.spatial import SpatialInertia, SpatialTransform
    from robot.model import RobotModel

    tree = RobotTreeNumpy(gravity=9.81)
    tree.add_body(
        Body(
            name="ball",
            index=0,
            joint=FreeJoint("root"),
            inertia=SpatialInertia(1.0, np.eye(3) * 0.001, np.zeros(3)),
            X_tree=SpatialTransform.identity(),
            parent=-1,
        )
    )
    tree.finalize()
    return RobotModel(
        tree=tree,
        geometries=[BodyCollisionGeometry(0, [ShapeInstance(SphereShape(0.05))])],
        contact_body_names=["ball"],
    )


def _merge_single_ball_model(model: object) -> object:
    from physics.merged_model import merge_models

    return merge_models({"ball": model})


def _create_gpu_engine(merged: object, *, device: str) -> object:
    from physics.gpu_engine import GpuEngine

    return GpuEngine(merged, num_envs=1, device=device)
