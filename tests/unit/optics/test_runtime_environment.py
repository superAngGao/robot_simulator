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
from optical_worker.historical.compilation_contract import setuptools_initialization_descriptor
from optical_worker.runtime_environment import (
    RUNTIME_ENVIRONMENT_VERSION, RUNTIME_ENVIRONMENT_COMPILATION_VERSION, RuntimeEnvironmentContract,
)


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
        "native_globals": [],
        "module_replacements": [],
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


@pytest.fixture
def setuptools_profile(deployment):
    """Synthetic file facts only, deliberately not a complete dependency closure."""
    root, payload = deployment
    payload["schema_version"] = RUNTIME_ENVIRONMENT_COMPILATION_VERSION
    payload["setuptools_initialization"] = setuptools_initialization_descriptor()
    payload["interpreter"]["version"] = [3, 12, 13]
    vendor = root / "setuptools" / "_vendor"
    vendor.mkdir(parents=True)
    payload["roots"].append({"id": "setuptools_vendor", "path": vendor.resolve().as_posix(), "kind": "site"})

    def file(root_id, relative, kind, data):
        path = (root if root_id == "site" else vendor) / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        payload["files"].append({"root": root_id, "path": relative, "kind": kind,
                                 "size": len(data), "sha256": hashlib.sha256(data).hexdigest()})

    for name, package in (("setuptools", True), ("setuptools._distutils", True),
                          ("setuptools.version", False), ("setuptools.command", True),
                          ("setuptools.command.build_ext", False), ("_distutils_hack", True),
                          ("_distutils_hack.override", False), ("distutils.core", False),
                          ("jaraco.text", True), ("torch", True), ("torch.utils", True),
                          ("torch.utils.cpp_extension", False)):
        root_id = "setuptools_vendor" if name.startswith("jaraco.") else "site"
        relative = name.replace("distutils.", "setuptools._distutils.", 1) if name.startswith("distutils.") else name
        relative = relative.replace(".", "/") + ("/__init__.py" if package else ".py")
        file(root_id, relative, "python", b"raise AssertionError('inventory must never execute')\n")
        payload["modules"].append({"name": name, "root": root_id, "path": relative,
                                   "is_package": package, "phase": "runtime"})
    metadata = "setuptools-83.0.0-py3.12.egg-info/PKG-INFO"
    file("site", metadata, "metadata", b"Name: setuptools\nVersion: 83.0.0\n")
    file("setuptools_vendor", "jaraco/text/Lorem ipsum.txt", "data", b"fixture resource, not the installed data")
    payload["distributions"].append({"name": "setuptools", "version": "83.0.0", "root": "site", "metadata": metadata})
    for name, root_id, path in (("jaraco", "setuptools_vendor", "jaraco"),
                               ("distutils.compilers", "site", "setuptools/_distutils/compilers"),
                               ("distutils.compilers.C", "site", "setuptools/_distutils/compilers/C")):
        ((root if root_id == "site" else vendor) / path).mkdir(parents=True, exist_ok=True)
        payload["namespaces"].append({"name": name, "phase": "runtime", "locations": [{"root": root_id, "path": path}]})
    for field, key in (("files", lambda r: (r["root"], r["path"])), ("roots", lambda r: r["id"]),
                        ("modules", lambda r: r["name"]), ("namespaces", lambda r: r["name"]),
                        ("distributions", lambda r: r["name"])):
        payload[field].sort(key=key)
    return payload


def test_v6_null_has_v5_file_semantics_and_does_not_rewrite_legacy_payload(deployment):
    _, payload = deployment
    original = RuntimeEnvironmentContract.from_payload(payload)
    original_bytes = original.encoded
    payload["schema_version"] = RUNTIME_ENVIRONMENT_COMPILATION_VERSION
    payload["setuptools_initialization"] = None
    current = RuntimeEnvironmentContract.from_payload(payload)
    current.verify_files(expected_sha256=current.sha256)
    assert original.encoded == original_bytes and original.setuptools_initialization is None
    assert current.external_search_roots() == original.external_search_roots()
    with pytest.raises(ValueError, match="not declared"):
        current.external_search_roots(setuptools_active=True)


def test_setuptools_alias_namespaces_and_site_priority_are_explicit_facts(setuptools_profile):
    payload = setuptools_profile
    before_path, before_modules = tuple(sys.path), dict(sys.modules)
    contract = RuntimeEnvironmentContract.from_payload(payload)
    roots = {r["id"]: str(Path(r["path"])) for r in payload["roots"]}
    assert contract.external_search_roots() == (roots["site"],)
    assert contract.external_search_roots(setuptools_active=True) == (roots["site"], roots["setuptools_vendor"])
    exposed = contract.setuptools_initialization
    exposed["resource"]["path"] = "elsewhere"
    assert contract.setuptools_initialization == setuptools_initialization_descriptor()
    assert not any(r["name"] == "distutils" for r in contract.to_payload()["modules"])
    assert contract.expected_file_module_name("distutils.core") == "distutils.core"
    with pytest.raises(ValueError, match="declared file"):
        contract.expected_file_module_name("distutils")
    assert tuple(sys.path) == before_path and dict(sys.modules) == before_modules


@pytest.mark.parametrize("mutation", [
    lambda p: p["setuptools_initialization"].update(alias="another_alias"),
    lambda p: p["setuptools_initialization"]["resource"].update(path="../outside"),
    lambda p: p["setuptools_initialization"].update(extra_permission=True),
    lambda p: p["interpreter"].update(version=[3, 14, 0]),
    lambda p: next(r for r in p["roots"] if r["id"] == "setuptools_vendor").update(path="/wrong/vendor"),
    lambda p: next(r for r in p["distributions"] if r["name"] == "setuptools").update(version="84.0.0"),
    lambda p: next(r for r in p["files"] if r["path"].endswith("Lorem ipsum.txt")).update(kind="python"),
    lambda p: next(r for r in p["modules"] if r["name"] == "distutils.core").update(phase="materialization"),
    lambda p: next(r for r in p["modules"] if r["name"] == "distutils.core").update(path="fake/owner.py"),
    lambda p: next(r for r in p["namespaces"] if r["name"] == "distutils.compilers")["locations"].append(
        {"root": "site", "path": "setuptools/_distutils/compilers/extra"}),
    lambda p: p["modules"].remove(next(r for r in p["modules"] if r["name"] == "_distutils_hack.override")),
])
def test_setuptools_profile_cannot_expand_or_redirect_fixed_rules(setuptools_profile, mutation):
    mutation(setuptools_profile)
    with pytest.raises(ValueError):
        RuntimeEnvironmentContract.from_payload(setuptools_profile)


def test_setuptools_root_cannot_be_a_namespace_or_a_second_source(setuptools_profile):
    payload = setuptools_profile
    payload["namespaces"].append({"name": "distutils", "phase": "runtime",
                                   "locations": [{"root": "site", "path": "setuptools/_distutils"}]})
    payload["namespaces"].sort(key=lambda r: r["name"])
    with pytest.raises(ValueError, match="alias collides"):
        RuntimeEnvironmentContract.from_payload(payload)


def test_v5_cannot_smuggle_a_v6_descriptor_or_alias_parent(setuptools_profile):
    payload = setuptools_profile
    payload["schema_version"] = RUNTIME_ENVIRONMENT_VERSION
    with pytest.raises(ValueError, match="environment fields"):
        RuntimeEnvironmentContract.from_payload(payload)
    del payload["setuptools_initialization"]
    with pytest.raises(ValueError, match="unlisted module parent: distutils"):
        RuntimeEnvironmentContract.from_payload(payload)


def test_alias_child_cannot_reuse_the_legacy_publication_exception(setuptools_profile):
    payload = setuptools_profile
    payload["publications"].append({"name": "distutils.unlisted", "kind": "module_alias",
                                     "creator": "setuptools", "publisher": "setuptools",
                                     "attributes": ["injected"], "phase": "runtime"})
    payload["publications"].sort(key=lambda r: r["name"])
    with pytest.raises(ValueError, match="ordinary file or namespace"):
        RuntimeEnvironmentContract.from_payload(payload)


def test_cached_configuration_copies_and_rebound_bytes_cannot_share_permission(deployment):
    _, payload = deployment
    contract = RuntimeEnvironmentContract.from_payload(payload)
    other = RuntimeEnvironmentContract(contract.encoded)
    first, second = contract.to_payload(), other.to_payload()
    first["roots"][0]["path"] = "invented"
    first["interpreter"]["version"][0] = 999
    first["files"].clear()
    assert second == payload and contract.to_payload() == payload
    original = contract.encoded
    # A previously validated object's mutated encoding must not inherit its
    # former metadata cache entry, even when the JSON itself is canonical.
    from optical_worker.protocol import canonical_json_bytes

    changed = copy.deepcopy(payload)
    changed["modules"][0]["phase"] = "invented"
    object.__setattr__(contract, "encoded", canonical_json_bytes(changed))
    with pytest.raises(ValueError):
        contract.to_payload()
    object.__setattr__(contract, "encoded", original)
    assert contract.to_payload() == other.to_payload() == payload


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


def _assert_active_terminal_profile_pins(profile_source, consumer_source):
    root = Path(__file__).resolve().parents[3]
    profiles = [
        ast.literal_eval(node.value)
        for node in ast.parse(profile_source).body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "PROFILES" for t in node.targets)
    ]
    assert len(profiles) == 1 and set(profiles[0]) == {"warp", "cuda"}
    for filename, digest, _ in profiles[0].values():
        payload = json.loads((root / "tests/data/runtime_environments" / filename).read_bytes())
        assert payload["schema_version"] == RUNTIME_ENVIRONMENT_VERSION
        assert digest == canonical_sha256(payload)
    consumer = ast.parse(consumer_source)
    assert any(isinstance(node, ast.ImportFrom)
               and node.module == "tests.gpu.test_native_runtime_fault_matrix"
               and any(alias.name == "PROFILES" and alias.asname is None for alias in node.names)
               for node in consumer.body)
    selection = ast.parse("filename, environment_sha256, deadline = PROFILES[lane]").body[0]
    assert any(ast.dump(node) == ast.dump(selection) for node in ast.walk(consumer))


def test_active_remote_fixture_version_and_cold_test_digest_move_together():
    # Read the sole shared table and its actual consumer without importing a
    # GPU test or pretending Windows can verify POSIX deployment files.
    root = Path(__file__).resolve().parents[3]
    _assert_active_terminal_profile_pins(
        (root / "tests/gpu/test_native_runtime_fault_matrix.py").read_bytes(),
        (root / "tests/gpu/test_native_runtime_cold_warp.py").read_bytes(),
    )


@pytest.mark.parametrize("drift", ["filename", "digest", "consumer"])
def test_active_terminal_profile_pin_drift_is_not_hidden_by_shared_table(drift):
    root = Path(__file__).resolve().parents[3]
    profiles = (root / "tests/gpu/test_native_runtime_fault_matrix.py").read_text(encoding="utf-8")
    consumer = (root / "tests/gpu/test_native_runtime_cold_warp.py").read_text(encoding="utf-8")
    if drift == "filename":
        profiles = profiles.replace("183_warp_synthetic_author_v5.json", "183_warp_go2_author_v3.json")
    elif drift == "digest":
        payload = json.loads((root / "tests/data/runtime_environments/183_warp_synthetic_author_v5.json").read_bytes())
        profiles = profiles.replace(canonical_sha256(payload), "0" * 64)
    else:
        consumer = consumer.replace("PROFILES[lane]", "OTHER_PROFILES[lane]")
    with pytest.raises(AssertionError):
        _assert_active_terminal_profile_pins(profiles, consumer)


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
