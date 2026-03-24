"""
Tests for CollisionFilter — bitmask, explicit exclude, auto-exclude, and
integration with AABBSelfCollision and load_urdf.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np

from physics.collision import AABBSelfCollision, BodyAABB
from physics.collision_filter import CollisionFilter

# ---------------------------------------------------------------------------
# Unit tests: CollisionFilter standalone
# ---------------------------------------------------------------------------


class TestCollisionFilterBasic:
    def test_same_body_never_collides(self):
        f = CollisionFilter(5)
        assert not f.should_collide(2, 2)

    def test_default_all_collide(self):
        f = CollisionFilter(5)
        assert f.should_collide(0, 1)
        assert f.should_collide(0, 4)
        assert f.should_collide(3, 4)

    def test_num_bodies_and_repr(self):
        f = CollisionFilter(10)
        assert f.num_bodies == 10
        assert "10" in repr(f)


class TestAutoExclude:
    def test_parent_child_excluded(self):
        """Parent-child pairs are auto-excluded."""
        # Tree: 0 → 1 → 2, 0 → 3
        parent_list = [-1, 0, 1, 0]
        f = CollisionFilter(4)
        f.auto_exclude_adjacent(parent_list)

        assert not f.should_collide(0, 1)
        assert not f.should_collide(1, 2)
        assert not f.should_collide(0, 3)
        # Non-adjacent should still collide
        assert f.should_collide(0, 2)
        assert f.should_collide(1, 3)
        assert f.should_collide(2, 3)

    def test_excluded_pairs_set(self):
        parent_list = [-1, 0, 1]
        f = CollisionFilter(3)
        f.auto_exclude_adjacent(parent_list)

        excluded = f.excluded_pairs()
        assert (0, 1) in excluded
        assert (1, 2) in excluded
        assert len(excluded) == 2

    def test_num_excluded(self):
        parent_list = [-1, 0, 0, 1]
        f = CollisionFilter(4)
        f.auto_exclude_adjacent(parent_list)
        assert f.num_excluded == 3  # (0,1), (0,2), (1,3)


class TestBitmask:
    def test_group_mask_filtering(self):
        """Bitmask filtering: bodies only collide if group & mask != 0 both ways."""
        f = CollisionFilter(4)
        # Body 0,1 in group A (bit 0); body 2,3 in group B (bit 1)
        f.set_group_mask(0, group=0b01, mask=0b01)  # A only collides with A
        f.set_group_mask(1, group=0b01, mask=0b01)
        f.set_group_mask(2, group=0b10, mask=0b10)  # B only collides with B
        f.set_group_mask(3, group=0b10, mask=0b10)

        assert f.should_collide(0, 1)  # A-A
        assert f.should_collide(2, 3)  # B-B
        assert not f.should_collide(0, 2)  # A-B
        assert not f.should_collide(1, 3)  # A-B

    def test_asymmetric_mask(self):
        """One-way mask: A sees B but B doesn't see A → no collision."""
        f = CollisionFilter(2)
        f.set_group_mask(0, group=0b01, mask=0b11)  # body 0 can collide with both
        f.set_group_mask(1, group=0b10, mask=0b10)  # body 1 only with group B

        # body 0's group (01) & body 1's mask (10) = 0 → filtered out
        assert not f.should_collide(0, 1)

    def test_default_mask_collides_with_everything(self):
        """Default 0xFFFFFFFF group/mask collides with any custom group."""
        f = CollisionFilter(2)
        f.set_group(1, 0b0001)  # body 1 has a specific group
        # body 0 still has default 0xFFFFFFFF for both
        assert f.should_collide(0, 1)


class TestExplicitExclude:
    def test_exclude_pair(self):
        f = CollisionFilter(5)
        f.exclude_pair(1, 3)
        assert not f.should_collide(1, 3)
        assert not f.should_collide(3, 1)  # order doesn't matter
        assert f.should_collide(1, 2)

    def test_exclude_pairs_bulk(self):
        f = CollisionFilter(5)
        f.exclude_pairs([(0, 2), (1, 4)])
        assert not f.should_collide(0, 2)
        assert not f.should_collide(1, 4)
        assert f.should_collide(0, 1)

    def test_exclude_overrides_bitmask(self):
        """Explicit exclude trumps bitmask (even if bitmask says collide)."""
        f = CollisionFilter(3)
        # All default masks → should collide
        f.exclude_pair(0, 2)
        assert not f.should_collide(0, 2)


class TestFilterPairs:
    def test_filter_pairs_list(self):
        f = CollisionFilter(4)
        f.auto_exclude_adjacent([-1, 0, 1, 0])
        candidates = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        result = f.filter_pairs(candidates)
        # 0-1 excluded (parent-child), 1-2 excluded, 0-3 excluded
        assert (0, 1) not in result
        assert (1, 2) not in result
        assert (0, 3) not in result
        assert (0, 2) in result
        assert (1, 3) in result
        assert (2, 3) in result


# ---------------------------------------------------------------------------
# Integration: CollisionFilter + AABBSelfCollision
# ---------------------------------------------------------------------------


class TestFilterWithAABBSelfCollision:
    def test_filter_reduces_pairs(self):
        """Explicit exclude should reduce the number of collision pairs."""
        parent_list = [-1, 0, 0, 1, 1]  # 0→1→3, 0→2, 1→4
        geom_half = np.array([0.1, 0.1, 0.1])

        # Without filter (legacy): only parent-child excluded
        sc_no_filter = AABBSelfCollision()
        for i in range(5):
            sc_no_filter.add_body(BodyAABB(i, geom_half))
        sc_no_filter.build_pairs(parent_list)
        pairs_no_filter = sc_no_filter.num_pairs

        # With filter: add extra exclusion (2, 3)
        f = CollisionFilter(5)
        f.auto_exclude_adjacent(parent_list)
        f.exclude_pair(2, 3)

        sc_filter = AABBSelfCollision()
        for i in range(5):
            sc_filter.add_body(BodyAABB(i, geom_half))
        sc_filter.build_pairs(parent_list, collision_filter=f)
        pairs_filter = sc_filter.num_pairs

        assert pairs_filter == pairs_no_filter - 1

    def test_bitmask_filter_in_aabb(self):
        """Bitmask should be respected during pair building."""
        parent_list = [-1, 0, 0]
        f = CollisionFilter(3)
        f.auto_exclude_adjacent(parent_list)
        # Bodies 1 and 2 are non-adjacent, so normally they'd collide
        # But set bitmask to prevent it
        f.set_group_mask(1, group=0b01, mask=0b01)
        f.set_group_mask(2, group=0b10, mask=0b10)

        sc = AABBSelfCollision()
        for i in range(3):
            sc.add_body(BodyAABB(i, np.array([0.1, 0.1, 0.1])))
        sc.build_pairs(parent_list, collision_filter=f)

        assert sc.num_pairs == 0  # 0-1, 0-2 adj-excluded; 1-2 bitmask-excluded

    def test_from_geometries_with_filter(self):
        """from_geometries accepts collision_filter kwarg."""
        from physics.geometry import BodyCollisionGeometry, BoxShape, ShapeInstance

        parent_list = [-1, 0, 0]
        geoms = [
            BodyCollisionGeometry(0, [ShapeInstance(BoxShape((0.2, 0.2, 0.2)))]),
            BodyCollisionGeometry(1, [ShapeInstance(BoxShape((0.1, 0.1, 0.1)))]),
            BodyCollisionGeometry(2, [ShapeInstance(BoxShape((0.1, 0.1, 0.1)))]),
        ]
        f = CollisionFilter(3)
        f.auto_exclude_adjacent(parent_list)
        f.exclude_pair(1, 2)

        sc = AABBSelfCollision.from_geometries(geoms, parent_list, collision_filter=f)
        assert sc.num_pairs == 0  # all pairs excluded


# ---------------------------------------------------------------------------
# Integration: load_urdf with collision_exclude_pairs
# ---------------------------------------------------------------------------


_URDF_THREE_LINKS = """\
<robot name="test">
  <link name="base">
    <inertial><mass value="5.0"/><origin xyz="0 0 0"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>
    <collision><geometry><box size="0.4 0.3 0.2"/></geometry></collision>
  </link>
  <link name="arm_L">
    <inertial><mass value="1.0"/><origin xyz="0 0 -0.1"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
    <collision><geometry><sphere radius="0.05"/></geometry></collision>
  </link>
  <link name="arm_R">
    <inertial><mass value="1.0"/><origin xyz="0 0 -0.1"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
    <collision><geometry><sphere radius="0.05"/></geometry></collision>
  </link>
  <joint name="shoulder_L" type="revolute">
    <parent link="base"/><child link="arm_L"/>
    <origin xyz="-0.2 0 0"/><axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57"/>
  </joint>
  <joint name="shoulder_R" type="revolute">
    <parent link="base"/><child link="arm_R"/>
    <origin xyz="0.2 0 0"/><axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57"/>
  </joint>
</robot>
"""


class TestLoadURDFCollisionFilter:
    def _write_urdf(self) -> str:
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".urdf", delete=False)
        f.write(_URDF_THREE_LINKS)
        f.close()
        return f.name

    def test_collision_filter_created(self):
        """load_urdf should always create a CollisionFilter on the model."""
        from robot import load_urdf

        path = self._write_urdf()
        try:
            model = load_urdf(path, floating_base=True, contact_links=[])
        finally:
            os.unlink(path)

        assert model.collision_filter is not None
        assert isinstance(model.collision_filter, CollisionFilter)

    def test_parent_child_auto_excluded(self):
        """Parent-child pairs should be auto-excluded in the filter."""
        from robot import load_urdf

        path = self._write_urdf()
        try:
            model = load_urdf(path, floating_base=True, contact_links=[])
        finally:
            os.unlink(path)

        f = model.collision_filter
        tree = model.tree
        # base→arm_L and base→arm_R are parent-child
        base_idx = next(b.index for b in tree.bodies if b.name == "base")
        arm_l_idx = next(b.index for b in tree.bodies if b.name == "arm_L")
        arm_r_idx = next(b.index for b in tree.bodies if b.name == "arm_R")

        assert not f.should_collide(base_idx, arm_l_idx)
        assert not f.should_collide(base_idx, arm_r_idx)
        # arm_L vs arm_R are non-adjacent → should collide by default
        assert f.should_collide(arm_l_idx, arm_r_idx)

    def test_explicit_exclude_pairs(self):
        """collision_exclude_pairs should add extra exclusions."""
        from robot import load_urdf

        path = self._write_urdf()
        try:
            model = load_urdf(
                path,
                floating_base=True,
                contact_links=[],
                collision_exclude_pairs=[("arm_L", "arm_R")],
            )
        finally:
            os.unlink(path)

        f = model.collision_filter
        tree = model.tree
        arm_l_idx = next(b.index for b in tree.bodies if b.name == "arm_L")
        arm_r_idx = next(b.index for b in tree.bodies if b.name == "arm_R")

        assert not f.should_collide(arm_l_idx, arm_r_idx)

    def test_exclude_reduces_self_collision_pairs(self):
        """Excluding arm_L-arm_R should reduce self-collision pairs."""
        from robot import load_urdf

        path = self._write_urdf()
        try:
            model_no_excl = load_urdf(path, floating_base=True, contact_links=[])
            model_excl = load_urdf(
                path,
                floating_base=True,
                contact_links=[],
                collision_exclude_pairs=[("arm_L", "arm_R")],
            )
        finally:
            os.unlink(path)

        assert model_excl.self_collision.num_pairs < model_no_excl.self_collision.num_pairs

    def test_invalid_link_in_exclude_warns(self):
        """Non-existent link name in collision_exclude_pairs should warn, not crash."""
        from robot import load_urdf

        path = self._write_urdf()
        try:
            model = load_urdf(
                path,
                floating_base=True,
                contact_links=[],
                collision_exclude_pairs=[("arm_L", "nonexistent")],
            )
        finally:
            os.unlink(path)

        # Should still build successfully
        assert model.collision_filter is not None
