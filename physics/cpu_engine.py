"""
CpuEngine — CPU physics engine using StepPipeline + collision detection.

Operates on the MergedModel's unified tree. Collision detection runs on
all body pairs (intra-robot + cross-robot) uniformly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

import numpy as np
from numpy.typing import NDArray

from .constraint_solvers import wrap_solver
from .contact_tolerances import CONTACT_CONVEX_MARGIN
from .dynamics_cache import DynamicsCache
from .engine import ContactInfo, PhysicsEngine, StepOutput
from .force_source import PassiveForceSource
from .gjk_epa import gjk_epa_query, ground_contact_query, halfspace_convex_query
from .publish import (
    AckPolicy,
    BorrowedFrameLease,
    ConsumerState,
    CpuPublishedFrame,
    HostSnapshotSpec,
    PublishPlan,
    PublishPolicy,
    SnapshotHandle,
)
from .solvers.pgs_solver import ContactConstraint
from .solvers.pgs_split_impulse import PGSSplitImpulseSolver
from .step_pipeline import StepPipeline
from .terrain import HalfSpaceTerrain

if TYPE_CHECKING:
    from .merged_model import MergedModel


class CpuEngine(PhysicsEngine):
    """CPU physics engine with full collision detection.

    Uses StepPipeline for the two-stage dynamics pipeline and
    GJK/EPA-based collision detection on the merged body list.

    Args:
        merged : MergedModel (multi-root tree + collision data).
        solver : Contact solver (default: PGSSplitImpulseSolver).
        dt     : Default time step [s] (can be overridden in step()).
    """

    def __init__(
        self,
        merged: "MergedModel",
        solver=None,
        dt: float = 2e-4,
    ) -> None:
        super().__init__(merged)
        solver = solver or PGSSplitImpulseSolver(max_iter=60, erp=0.8, slop=0.005)
        wrapped = wrap_solver(solver)
        self._pipeline = StepPipeline(
            dt=dt,
            force_sources=[PassiveForceSource()],
            constraint_solver=wrapped,
        )
        self._dt = dt
        self._last_contacts: List[ContactConstraint] = []
        self._publish_policy = PublishPolicy()
        self._publish_frame_id = -1
        self._publish_step_index = -1
        self._publish_sim_time = 0.0
        self._publish_consumers: list[ConsumerState] = []
        self._latest_published_frame: CpuPublishedFrame | None = None

    def step(
        self,
        q: NDArray,
        qdot: NDArray,
        tau: NDArray,
        dt: float | None = None,
    ) -> StepOutput:
        dt = dt or self._dt
        tree = self.merged.tree

        # Build DynamicsCache (FK + body_v)
        cache = DynamicsCache.from_tree(tree, q, qdot, dt)

        # Collision detection on merged body list
        contacts = self._detect_contacts(cache)
        self._last_contacts = contacts

        # Run pipeline (smooth forces → constraint → integrate)
        self._pipeline.dt = dt
        q_new, qdot_new = self._pipeline.step(tree, q, qdot, tau, contacts, cache=cache)

        # Build output
        X_world = tree.forward_kinematics(q_new)
        v_bodies = tree.body_velocities(q_new, qdot_new)
        contact_active = np.array([True] * len(contacts) if contacts else [])

        output = StepOutput(
            q_new=q_new,
            qdot_new=qdot_new,
            X_world=X_world,
            v_bodies=v_bodies,
            contact_active=contact_active,
            force_state=self._pipeline.last_force_state,
        )
        self._publish_after_step(output, dt)
        return output

    @property
    def publish_policy(self) -> PublishPolicy:
        return self._publish_policy

    def set_publish_policy(self, policy: PublishPolicy) -> None:
        self._publish_policy = policy

    def register_consumer(self, consumer: ConsumerState) -> None:
        for idx, existing in enumerate(self._publish_consumers):
            if existing.consumer_id == consumer.consumer_id:
                self._publish_consumers[idx] = consumer
                return
        self._publish_consumers.append(consumer)

    def unregister_consumer(self, consumer_id: str) -> None:
        self._publish_consumers[:] = [
            consumer for consumer in self._publish_consumers if consumer.consumer_id != consumer_id
        ]

    def step_and_publish(
        self,
        q: NDArray,
        qdot: NDArray,
        tau: NDArray,
        dt: float | None = None,
    ) -> CpuPublishedFrame:
        self.step(q=q, qdot=qdot, tau=tau, dt=dt)
        if self._latest_published_frame is None:
            raise RuntimeError("CPU publish failed to produce a latest frame")
        return self._latest_published_frame

    def _publish_after_step(self, output: StepOutput, dt: float) -> None:
        # frame_id follows the physics-step timeline, not just the materialized
        # publish timeline. Skipped publishes therefore create gaps visible to
        # consumers (e.g. 0, 2, 4, ...) rather than renumbering published
        # frames densely.
        next_frame_id = self._publish_frame_id + 1
        plan = PublishPlan.from_policy(next_frame_id, self._publish_policy)
        self._publish_frame_id = next_frame_id
        self._publish_step_index += 1
        self._publish_sim_time += dt
        if not plan.do_publish_core:
            return
        self._latest_published_frame = CpuPublishedFrame(
            frame_id=self._publish_frame_id,
            sim_time=self._publish_sim_time,
            step_index=self._publish_step_index,
            env_mask=None,
            q=np.asarray(output.q_new).copy(),
            qdot=np.asarray(output.qdot_new).copy(),
            X_world=output.X_world,
            v_bodies=output.v_bodies,
            contact_count=len(self._last_contacts),
            contacts=self.query_contacts(),
            telemetry=output.force_state,
        )

    def latest_published_frame(self) -> CpuPublishedFrame | None:
        return self._latest_published_frame

    def borrow_latest_frame(self, consumer_id: str) -> BorrowedFrameLease[CpuPublishedFrame]:
        consumer = self._require_consumer(consumer_id)
        frame = self._latest_published_frame
        if frame is None:
            return BorrowedFrameLease(None)

        consumer.latest_seen_frame_id = frame.frame_id
        ack_policy = AckPolicy.default_for(consumer)

        def _release_callback(released_frame: CpuPublishedFrame) -> None:
            if ack_policy.ack_point == "on_borrow_complete":
                consumer.acked_frame_id = max(consumer.acked_frame_id, released_frame.frame_id)

        return BorrowedFrameLease(frame, on_release=_release_callback)

    def snapshot_frame_to_host(
        self, consumer_id: str, frame_id: int, spec: HostSnapshotSpec
    ) -> SnapshotHandle[dict[str, object]]:
        consumer = self._require_consumer(consumer_id)
        frame = self._require_published_frame(frame_id)
        consumer.latest_seen_frame_id = frame.frame_id

        fields = set(spec.fields)
        snapshot: dict[str, object] = {
            "frame_id": frame.frame_id,
            "step_index": frame.step_index,
            "sim_time": frame.sim_time,
        }
        if "q" in fields:
            snapshot["q"] = np.asarray(frame.q).copy()
        if "qdot" in fields:
            snapshot["qdot"] = np.asarray(frame.qdot).copy()
        if "X_world" in fields:
            snapshot["X_world"] = frame.X_world
        if "v_bodies" in fields:
            snapshot["v_bodies"] = frame.v_bodies
        if "contact_count" in fields:
            snapshot["contact_count"] = int(frame.contact_count)
        if "contacts" in fields:
            snapshot["contacts"] = frame.contacts
        if "telemetry" in fields:
            snapshot["telemetry"] = frame.telemetry

        ack_policy = AckPolicy.default_for(consumer)
        if ack_policy.ack_point == "on_snapshot_staged":
            consumer.acked_frame_id = max(consumer.acked_frame_id, frame.frame_id)
        return SnapshotHandle(snapshot, frame_id=frame.frame_id)

    def _require_consumer(self, consumer_id: str) -> ConsumerState:
        for consumer in self._publish_consumers:
            if consumer.consumer_id == consumer_id:
                return consumer
        raise KeyError(f"Unknown publish consumer {consumer_id!r}. Register it first.")

    def _require_published_frame(self, frame_id: int) -> CpuPublishedFrame:
        """Return the current published frame.

        CPU reference path keeps only the latest published frame; therefore
        `frame_id` must match `latest_published_frame().frame_id`.
        Historical frame lookup is intentionally unsupported here.
        """
        if self._latest_published_frame is None or self._latest_published_frame.frame_id != frame_id:
            raise KeyError(f"Published frame {frame_id} is not currently available.")
        return self._latest_published_frame

    def _detect_contacts(self, cache: DynamicsCache) -> List[ContactConstraint]:
        """Detect all contacts using GJK/EPA: body-ground + body-body."""
        contacts: List[ContactConstraint] = []
        merged = self.merged
        X_world = cache.X_world
        terrain = merged.terrain

        # 1. Ground contacts (GJK/EPA per body, all shapes)
        for body_idx, _local_pos in merged.contact_points:
            geom = merged.collision_shapes[body_idx] if body_idx < len(merged.collision_shapes) else None
            if geom is None or not geom.shapes:
                continue
            X_body = X_world[body_idx]
            for si in geom.shapes:
                X_shape = si.world_pose(X_body)
                if isinstance(terrain, HalfSpaceTerrain):
                    manifold = halfspace_convex_query(
                        si.shape,
                        X_shape,
                        hs_normal_world=terrain.normal_world,
                        hs_point_world=terrain.point_on_plane,
                    )
                else:
                    gz = terrain.height_at(X_shape.r[0], X_shape.r[1])
                    manifold = ground_contact_query(si.shape, X_shape, ground_z=gz)
                if manifold is not None and manifold.depth > 1e-10:
                    for pi, pt in enumerate(manifold.points):
                        contacts.append(
                            ContactConstraint(
                                body_i=body_idx,
                                body_j=-1,
                                point=pt,
                                normal=manifold.normal.copy(),
                                tangent1=np.zeros(3),
                                tangent2=np.zeros(3),
                                depth=manifold.depth_at(pi),
                                mu=getattr(terrain, "mu", 0.8),
                                condim=3,
                            )
                        )

        # 2. Body-body contacts (GJK/EPA per shape)
        for bi, bj in merged.collision_pairs:
            geom_i = merged.collision_shapes[bi] if bi < len(merged.collision_shapes) else None
            geom_j = merged.collision_shapes[bj] if bj < len(merged.collision_shapes) else None
            if geom_i is None or geom_j is None or not geom_i.shapes or not geom_j.shapes:
                continue
            for si_i in geom_i.shapes:
                X_i = si_i.world_pose(X_world[bi])
                for si_j in geom_j.shapes:
                    X_j = si_j.world_pose(X_world[bj])
                    manifold = gjk_epa_query(
                        si_i.shape,
                        X_i,
                        si_j.shape,
                        X_j,
                        margin=CONTACT_CONVEX_MARGIN,
                    )
                    if manifold is not None and manifold.depth > 1e-10:
                        for pi, pt in enumerate(manifold.points):
                            contacts.append(
                                ContactConstraint(
                                    body_i=bi,
                                    body_j=bj,
                                    point=pt,
                                    normal=manifold.normal.copy(),
                                    tangent1=np.zeros(3),
                                    tangent2=np.zeros(3),
                                    depth=manifold.depth_at(pi),
                                    mu=0.8,
                                    condim=3,
                                )
                            )

        return contacts

    def query_contacts(self, env_idx: int = 0) -> List[ContactInfo]:
        """Return contacts from the most recent step() as ContactInfo list.

        Args:
            env_idx: Ignored (CpuEngine is single-env).
        """
        return [
            ContactInfo(
                body_i=c.body_i,
                body_j=c.body_j,
                depth=float(c.depth),
                normal=np.asarray(c.normal, dtype=np.float64).copy(),
                point=np.asarray(c.point, dtype=np.float64).copy(),
            )
            for c in self._last_contacts
        ]
