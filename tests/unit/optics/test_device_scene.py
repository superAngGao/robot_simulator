from __future__ import annotations

import pytest

from optics import DeviceOpticalRoleTable, OpticalInstanceSpec, OpticalMaterialSpec, OpticalWorldRegistry


def test_device_role_table_assigns_deterministic_int64_masks():
    table = DeviceOpticalRoleTable.from_roles({"segmentation", "depth", "rgb"})

    assert table.mask_for("depth") == 1
    assert table.mask_for("rgb") == 2
    assert table.mask_for("segmentation") == 4
    assert table.mask_for("missing") == 0
    assert table.mask_for_roles({"depth", "segmentation"}) == 5


def test_device_role_table_rejects_more_than_63_roles():
    with pytest.raises(ValueError, match="63 roles"):
        DeviceOpticalRoleTable.from_roles({f"role_{index}" for index in range(64)})


def test_device_role_table_from_registry_uses_instance_roles():
    registry = OpticalWorldRegistry()
    registry.add_material(OpticalMaterialSpec("mat"))
    registry.add_plane_geometry("plane", normal_local=[0.0, 0.0, 1.0], point_local=[0.0, 0.0, 0.0])
    registry.add_instance(
        OpticalInstanceSpec(
            "depth_plane",
            "plane",
            "mat",
            roles=frozenset({"depth", "segmentation"}),
        )
    )

    table = DeviceOpticalRoleTable.from_registry(registry)

    assert table.mask_for("depth") != 0
    assert table.mask_for("segmentation") != 0
    assert table.mask_for("rgb") == 0
