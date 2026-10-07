"""CPU contract tests, not cold backend admission or GPU lifecycle acceptance."""

from __future__ import annotations

import copy
import hashlib
import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from optical_worker.runtime_environment import RUNTIME_ENVIRONMENT_VERSION, RuntimeEnvironmentContract


@pytest.fixture
def deployment(tmp_path):
    root = tmp_path / "site-packages"
    root.mkdir()
    records = [
        ("fake/__init__.py", "python", b"raise AssertionError('MUST NOT EXECUTE')\n"),
        ("fake/_native.so", "extension", b"not a real extension; no loading is allowed"),
        ("fake/owner.py", "python", b"raise AssertionError('MUST NOT EXECUTE')\n"),
        ("fake-1.dist-info/METADATA", "metadata", b"Name: fake\nVersion: 1\n"),
        ("fake/lib/native.so", "native", b"not a native library"),
    ]
    files = []
    for relative, kind, data in records:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        files.append(
            {
                "root": "site",
                "path": relative,
                "kind": kind,
                "size": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    (root / "vendor").mkdir()
    executable = Path(sys.executable).resolve()
    payload = {
        "schema_version": RUNTIME_ENVIRONMENT_VERSION,
        "interpreter": {
            "path": executable.as_posix(),
            "sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
            "implementation": sys.implementation.name,
            "version": list(sys.version_info[:3]),
        },
        "roots": [{"id": "site", "path": root.resolve().as_posix(), "kind": "site"}],
        "files": sorted(files, key=lambda r: (r["root"], r["path"])),
        "distributions": [
            {"name": "fake", "version": "1", "root": "site", "metadata": "fake-1.dist-info/METADATA"}
        ],
        "modules": [
            {
                "name": "fake",
                "root": "site",
                "path": "fake/__init__.py",
                "is_package": True,
                "phase": "materialization",
            },
            {
                "name": "fake._native",
                "root": "site",
                "path": "fake/_native.so",
                "is_package": False,
                "phase": "runtime",
            },
            {
                "name": "fake.owner",
                "root": "site",
                "path": "fake/owner.py",
                "is_package": False,
                "phase": "materialization",
            },
        ],
        "absent_probes": [{"name": "optional_absent", "requester": "fake.owner", "phase": "materialization"}],
        "disabled_optional_modules": [],
        "publications": [
            {
                "name": "fake._native.child",
                "kind": "created_module",
                "creator": "fake._native",
                "publisher": "fake",
                "attributes": ["child"],
                "phase": "runtime",
            },
            {
                "name": "fake.alias",
                "kind": "module_alias",
                "creator": "fake.owner",
                "publisher": "fake",
                "attributes": ["value"],
                "phase": "materialization",
            },
        ],
        "namespaces": [
            {
                "name": "vendor",
                "phase": "materialization",
                "locations": [
                    {"root": "site", "path": "vendor"},
                ],
            }
        ],
        "settings": {"TORCH_DEVICE_BACKEND_AUTOLOAD": "0", "warp_quiet": True},
    }
    return root, payload


def test_environment_is_canonical_detached_and_never_executes_inventory(deployment):
    _, payload = deployment
    before = dict(sys.modules)
    contract = RuntimeEnvironmentContract.from_payload(payload)
    expected = contract.sha256
    assert RuntimeEnvironmentContract(contract.encoded).to_payload() == payload
    payload["modules"][0]["name"] = "different"
    returned = contract.to_payload()
    returned["modules"].clear()
    assert contract.to_payload()["modules"][0]["name"] == "fake"
    contract.verify_files(expected_sha256=expected)
    assert not any(name == "fake" or name.startswith("fake.") for name in sys.modules)
    assert all(sys.modules.get(name) is value for name, value in before.items())
    with pytest.raises(FrozenInstanceError):
        contract.encoded = b"{}"


@pytest.mark.parametrize(
    "table", ["roots", "files", "distributions", "modules", "absent_probes", "publications", "namespaces"]
)
def test_duplicate_rules_reject(deployment, table):
    _, payload = deployment
    payload[table].append(copy.deepcopy(payload[table][0]))
    with pytest.raises(ValueError, match="order or duplicate"):
        RuntimeEnvironmentContract.from_payload(payload)


@pytest.mark.parametrize("path", [".", "..", "fake/../owner.py", "/absolute", "fake\\owner.py", "a//b"])
def test_inventory_path_escape_or_noncanonical_spelling_rejects(deployment, path):
    _, payload = deployment
    payload["files"][0]["path"] = path
    with pytest.raises(ValueError):
        RuntimeEnvironmentContract.from_payload(payload)


@pytest.mark.parametrize(
    "mutation",
    [
        "unknown_file",
        "wrong_file_kind",
        "publication_collision",
        "creator_cycle",
        "missing_publisher",
        "optional_present",
        "optional_wrong_requester",
        "invented_phase",
        "plugin_enabled",
        "quiet_integer",
        "missing_parent",
        "nonpackage_parent",
        "namespace_root",
        "duplicate_root",
        "native_module_root",
        "wildcard",
        "repository",
    ],
)
def test_semantically_invalid_environment_rejects(deployment, mutation):
    _, p = deployment
    if mutation == "unknown_file":
        p["modules"][0]["path"] = "not-in-inventory.py"
    elif mutation == "wrong_file_kind":
        p["modules"][0]["path"] = "fake/lib/native.so"
    elif mutation == "publication_collision":
        p["publications"][1]["name"] = "fake.owner"
    elif mutation == "creator_cycle":
        p["publications"][0]["creator"] = "fake.alias"
    elif mutation == "missing_publisher":
        p["publications"][0]["publisher"] = "unknown"
    elif mutation == "optional_present":
        p["absent_probes"][0]["name"] = "fake.owner"
    elif mutation == "optional_wrong_requester":
        p["absent_probes"][0]["requester"] = "unknown"
    elif mutation == "invented_phase":
        p["modules"][0]["phase"] = "anywhere"
    elif mutation == "plugin_enabled":
        p["settings"]["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "1"
    elif mutation == "quiet_integer":
        p["settings"]["warp_quiet"] = 1
    elif mutation == "missing_parent":
        p["publications"][0]["name"] = "absent.child"
    elif mutation == "nonpackage_parent":
        p["modules"][0]["is_package"] = False
    elif mutation == "namespace_root":
        p["namespaces"][0]["locations"][0]["root"] = "elsewhere"
    elif mutation == "duplicate_root":
        p["roots"].append({"id": "sub", "path": p["roots"][0]["path"], "kind": "site"})
    elif mutation == "native_module_root":
        p["roots"][0]["kind"] = "native"
    elif mutation == "wildcard":
        p["publications"][0]["name"] = "fake.*"
    elif mutation == "repository":
        p["modules"][0]["name"] = "benchmarks"
    with pytest.raises((ValueError, TypeError)):
        RuntimeEnvironmentContract.from_payload(p)


def test_filesystem_and_trusted_digest_checks(deployment):
    root, p = deployment
    contract = RuntimeEnvironmentContract.from_payload(p)
    with pytest.raises(ValueError, match="trusted launch digest"):
        contract.verify_files(expected_sha256="0" * 64)
    target = root / "fake/owner.py"
    target.write_bytes(b"replacement with same manifest labels")
    with pytest.raises(ValueError, match="file bytes changed"):
        contract.verify_files(expected_sha256=contract.sha256)


def test_frozen_version_label_must_match_real_metadata(deployment):
    _, p = deployment
    p["distributions"][0]["version"] = "wrong"
    contract = RuntimeEnvironmentContract.from_payload(p)
    with pytest.raises(ValueError, match="METADATA"):
        contract.verify_files(expected_sha256=contract.sha256)


def test_public_constructor_and_rebound_bytes_cannot_skip_validation(deployment):
    _, p = deployment
    contract = RuntimeEnvironmentContract.from_payload(p)
    old_hash = contract.sha256
    with pytest.raises(ValueError):
        RuntimeEnvironmentContract(b"{}")
    p["distributions"][0]["version"] = "different"
    changed = RuntimeEnvironmentContract.from_payload(p)
    object.__setattr__(contract, "encoded", changed.encoded)
    with pytest.raises(ValueError, match="trusted launch digest"):
        contract.verify_files(expected_sha256=old_hash)


def test_import_is_stdlib_only_in_a_fresh_process():
    code = """
import sys
import optical_worker.runtime_environment
assert not {'numpy', 'torch', 'warp', 'trimesh', 'benchmarks'}.intersection(sys.modules)
"""
    subprocess.run([sys.executable, "-S", "-B", "-c", code], check=True, timeout=15)
