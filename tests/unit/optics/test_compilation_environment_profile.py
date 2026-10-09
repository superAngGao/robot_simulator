"""Frozen deployment declarations; these tests do not execute dependencies."""

import json
import os
from pathlib import Path

import pytest

from optical_worker.runtime_environment import RuntimeEnvironmentContract


DATA = Path(__file__).resolve().parents[2] / "data" / "runtime_environments"


def _profiles():
    return tuple(json.loads((DATA / name).read_bytes()) for name in (
        "183_cuda_go2_author_v1.json", "183_cuda_compilation_author_v1.json"))


def test_compilation_profile_preserves_every_predecessor_rule():
    old, new = _profiles()
    assert set(new) == set(old) | {"setuptools_initialization"}
    for key, value in old.items():
        if isinstance(value, list):
            assert all(row in new[key] for row in value), key
        elif key != "schema_version":
            assert new[key] == value, key
    # This extension cannot expand the original input materialization cut.
    for key in ("modules", "namespaces", "absent_probes"):
        assert all(row["phase"] == "runtime" for row in new[key] if row not in old[key])
    for key in ("publications", "native_globals", "module_replacements", "disabled_optional_modules"):
        assert new[key] == old[key]


@pytest.mark.skipif(os.name != "posix", reason="actual POSIX deployment path semantics required")
def test_compilation_profile_parses_without_loading_packages():
    _, payload = _profiles()
    contract = RuntimeEnvironmentContract.from_payload(payload)
    roots = {r["id"]: r["path"] for r in payload["roots"]}
    assert contract.external_search_roots() == (roots["site"],)
    assert contract.external_search_roots(setuptools_active=True) == (roots["site"], roots["setuptools_vendor"])
    # Children execute as their requested names; only the root is an alias.
    assert contract.expected_file_module_name("distutils.core") == "distutils.core"


def test_compilation_profile_covers_lazy_build_imports_without_other_platforms():
    _, payload = _profiles()
    modules = {r["name"]: r for r in payload["modules"]}
    # Both are reached from function bodies, missed by top-level-only scanning:
    # monkey.patch_all and cpp_extension._jit_compile's cold-build branch.
    for name in ("setuptools._core_metadata", "torch.utils.hipify.hipify_python",
                 "torch.utils.hipify.cuda_to_hip_mappings", "torch.utils.hipify.constants"):
        assert modules[name]["phase"] == "runtime"
    assert not any(name.split(".")[0] in {"backports", "win32com", "com", "Cython"} for name in modules)
    assert "distutils" not in modules  # Only the owned alias transaction publishes this root.
    assert modules["packaging"]["root"] == "site"
    assert modules["jaraco.text"]["root"] == "setuptools_vendor"
