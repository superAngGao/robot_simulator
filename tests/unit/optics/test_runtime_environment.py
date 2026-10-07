"""CPU contract tests, not cold backend admission or GPU lifecycle acceptance."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from optical_worker.protocol import canonical_sha256
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


def _self_alias_payload(deployment):
    _, payload = deployment
    payload["publications"] = [
        {
            "name": "fake.native_alias",
            "kind": "module_alias",
            "creator": "fake._native",
            "publisher": "fake._native",
            "attributes": [],
            "phase": "runtime",
        }
    ]
    return payload


def test_native_self_selector_is_metadata_only_and_preserves_qualified_creator(deployment):
    payload = _self_alias_payload(deployment)
    contract = RuntimeEnvironmentContract.from_payload(payload)
    contract.verify_files(expected_sha256=contract.sha256)
    assert contract.to_payload() == payload
    assert not any(n == "fake" or n.startswith("fake.") for n in sys.modules)


def test_intrinsic_name_query_is_cpu_only_and_only_for_declared_file_rows(deployment):
    _, payload = deployment
    original = RuntimeEnvironmentContract.from_payload(payload)
    assert original.expected_file_module_name("fake") == "fake"
    assert original.expected_file_module_name("fake._native") == "fake._native"
    for name in ("scipy._lib._uarray._uarray", "vendor", "fake.alias", "fake._native.child"):
        with pytest.raises(ValueError, match="declared file module"):
            original.expected_file_module_name(name)
    row = next(r for r in payload["modules"] if r["name"] == "fake._native")
    row["name"] = "scipy._lib._uarray._uarray"
    payload["modules"].sort(key=lambda r: r["name"])
    payload["publications"] = []
    # Parent names are explicit even in this CPU-only metadata fixture. The
    # disposable namespace directory is not a claim about real SciPy layout.
    payload["namespaces"].extend(
        {"name": parent, "phase": "materialization", "locations": [{"root": "site", "path": "vendor"}]}
        for parent in ("scipy", "scipy._lib", "scipy._lib._uarray")
    )
    payload["namespaces"].sort(key=lambda r: r["name"])
    before = dict(sys.modules)
    contract = RuntimeEnvironmentContract.from_payload(payload)
    assert contract.expected_file_module_name(row["name"]) == "uarray._uarray"
    assert dict(sys.modules) == before
    assert contract.to_payload() == payload  # no new serialized caller field


@pytest.mark.parametrize("drift", ["python", "package"])
def test_intrinsic_native_row_wrong_kind_rejected_before_execution(deployment, drift):
    _, payload = deployment
    row = next(r for r in payload["modules"] if r["name"] == "fake._native")
    row["name"] = "scipy._lib._uarray._uarray"
    payload["modules"].sort(key=lambda r: r["name"])
    payload["publications"] = []
    if drift == "package":
        row["is_package"] = True
    else:
        row["path"] = "fake/owner.py"
    with pytest.raises(ValueError, match="nonpackage extension"):
        RuntimeEnvironmentContract.from_payload(payload)


def test_intrinsic_metadata_cache_does_not_cache_identity_or_file_verification(deployment):
    root, payload = deployment
    original = RuntimeEnvironmentContract.from_payload(payload)
    assert original.expected_file_module_name("fake._native") == "fake._native"
    changed = copy.deepcopy(payload)
    changed["modules"] = [r for r in changed["modules"] if r["name"] != "fake._native"]
    changed["publications"] = []
    other = RuntimeEnvironmentContract.from_payload(changed)
    with pytest.raises(ValueError, match="declared file module"):
        other.expected_file_module_name("fake._native")
    # Returning to the original exact bytes is valid; no observed module/file
    # fact was cached. Same contract identity with changed bytes revalidates too.
    assert original.expected_file_module_name("fake._native") == "fake._native"
    object.__setattr__(original, "encoded", other.encoded)
    with pytest.raises(ValueError, match="declared file module"):
        original.expected_file_module_name("fake._native")
    original = RuntimeEnvironmentContract.from_payload(payload)
    assert original.expected_file_module_name("fake._native") == "fake._native"
    (root / "fake/_native.so").write_bytes(b"changed after metadata caching")
    with pytest.raises(ValueError, match="file bytes changed"):
        original.verify_files(expected_sha256=original.sha256)


def test_active_remote_fixture_version_and_cold_test_digest_move_together():
    # CPU-only packaging check: do not import the GPU test or initialize a
    # backend, and do not pretend Windows validates POSIX deployment paths.
    root = Path(__file__).resolve().parents[3]
    payload = json.loads(
        (root / "tests/data/runtime_environments/183_warp_synthetic_author_v3.json").read_bytes()
    )
    tree = ast.parse((root / "tests/gpu/test_native_runtime_cold_warp.py").read_bytes())
    pins = [
        node.value.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "ENVIRONMENT_SHA256" for t in node.targets)
        and isinstance(node.value, ast.Constant)
    ]
    assert payload["schema_version"] == RUNTIME_ENVIRONMENT_VERSION
    assert pins == [canonical_sha256(payload)]


@pytest.mark.parametrize(
    "drift",
    [
        "python",
        "package",
        "publisher",
        "phase",
        "created",
        "tuple",
        "collision",
        "cycle",
        "v1",
        "v2",
    ],
)
def test_native_self_selector_rejects_non_native_or_unfrozen_relation(deployment, drift):
    payload = _self_alias_payload(deployment)
    row = payload["publications"][0]
    if drift == "python":
        row.update(creator="fake.owner", publisher="fake.owner", phase="materialization")
    elif drift == "package":
        payload["modules"][1]["is_package"] = True
    elif drift == "publisher":
        row["publisher"] = "fake"
    elif drift == "phase":
        row["phase"] = "materialization"
    elif drift == "created":
        row["kind"] = "created_module"
    elif drift == "tuple":
        row["attributes"] = ()
    elif drift == "collision":
        row["name"] = "fake._native"
    elif drift == "cycle":
        row["creator"] = row["publisher"] = row["name"]
    else:
        payload["schema_version"] = "q54.p12.4.runtime-environment." + drift
    with pytest.raises(ValueError):
        RuntimeEnvironmentContract.from_payload(payload)


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
