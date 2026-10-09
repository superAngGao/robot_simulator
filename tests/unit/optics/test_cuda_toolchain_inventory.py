"""Fixed-root scanner and production completeness comparison, using CPU files."""

import copy
import hashlib
from pathlib import Path

import pytest

from optical_worker import cuda_compilation as compilation


@pytest.fixture
def layout(tmp_path, monkeypatch):
    headers = tmp_path / "headers"
    (headers / "nested").mkdir(parents=True)
    (headers / "first.h").write_bytes(b"first")
    (headers / "nested/second.h").write_bytes(b"second")
    binary = tmp_path / "fixed-tool"
    binary.write_bytes(b"not executable; never run")
    empty = tmp_path / "empty"
    empty.mkdir()
    absent = tmp_path / "optional-library-root"
    roots = tuple(sorted(((str(headers), "tree", True), (str(binary), "file", True),
                          (str(empty), "files", True), (str(absent), "files", False))))
    monkeypatch.setattr(compilation, "_toolchain_roots", lambda: roots)
    monkeypatch.setattr(compilation, "_toolchain_tools", lambda: {"cxx": str(binary)})
    return headers, binary, absent


def audit(payload):
    value = object.__new__(compilation._Toolchain)
    value.payload = payload
    value.audit = lambda: None  # Environment/private-directory proof is separate.
    value.inventory_checked = False
    value.audit_inventory()
    assert value.inventory_checked
    return value


def test_complete_inventory_hashes_nested_inputs_and_records_absence(layout, monkeypatch):
    headers, binary, absent = layout
    monkeypatch.setattr(Path, "read_bytes", lambda self: pytest.fail("inventory must use the shared streaming hasher"))
    payload = compilation._toolchain_inventory()
    assert len(payload["files"]) == 3
    assert payload["files"] == sorted(payload["files"], key=lambda row: row["path"])
    first = next(row for row in payload["files"] if row["path"] == str(headers / "first.h"))
    assert first == {"path": str(headers / "first.h"), "size": 5, "sha256": hashlib.sha256(b"first").hexdigest()}
    assert next(row for row in payload["roots"] if row["path"] == str(absent)) == {
        "path": str(absent), "kind": "files", "present": False, "resolved": None}
    assert payload["tools"] == {"cxx": str(binary)}
    audit(payload)


@pytest.mark.parametrize("change", ["omit_root", "omit_header", "change_digest", "extra_file", "tool_substitution"])
def test_caller_cannot_define_its_own_inventory_completeness(layout, change):
    payload = compilation._toolchain_inventory()
    if change == "omit_root":
        payload["roots"].pop()
    elif change == "omit_header":
        payload["files"] = [row for row in payload["files"] if not row["path"].endswith("second.h")]
    elif change == "change_digest":
        payload["files"][0]["sha256"] = "0" * 64
    elif change == "extra_file":
        payload["files"].append(copy.deepcopy(payload["files"][0]))
    else:
        payload["tools"]["cxx"] = str(layout[0] / "first.h")
    with pytest.raises(RuntimeError, match="coverage or bytes changed"):
        audit(payload)


@pytest.mark.parametrize("change", ["added", "modified", "removed", "optional_root_appeared"])
def test_deployment_drift_invalidates_an_earlier_complete_inventory(layout, change):
    headers, _, absent = layout
    payload = compilation._toolchain_inventory()
    if change == "added":
        (headers / "new.h").write_bytes(b"unfrozen")
    elif change == "modified":
        (headers / "first.h").write_bytes(b"other")
    elif change == "removed":
        (headers / "first.h").unlink()
    else:
        absent.mkdir()
    with pytest.raises(RuntimeError, match="coverage or bytes changed"):
        audit(payload)


def symlink(path, target, *, directory=False):
    try:
        path.symlink_to(target, target_is_directory=directory)
    except OSError as error:
        pytest.skip("OS did not permit the fixture symlink: " + str(error))


def test_symlink_chain_preserves_parent_resolution_before_dotdot(layout):
    headers, _, _ = layout
    outside = headers.parent / "outside"
    (outside / "nested").mkdir(parents=True)
    (outside / "value.h").write_bytes(b"resolved after directory symlink")
    symlink(headers / "shortcut", outside / "nested", directory=True)
    symlink(headers / "linked.h", "shortcut/../value.h")
    payload = compilation._toolchain_inventory()
    # Windows readlink may expose the extended-length path spelling. The Linux
    # deployment uses POSIX names; here compare the actual resolved file.
    assert any(Path(row["path"]).samefile(outside / "value.h") for row in payload["files"])
    assert {row["path"] for row in payload["symlinks"]} == {str(headers / "shortcut"), str(headers / "linked.h")}
    audit(payload)
    payload["symlinks"].pop()
    with pytest.raises(RuntimeError, match="coverage or bytes changed"):
        audit(payload)


def test_symlink_cycle_fails_before_any_build(layout):
    headers, _, _ = layout
    symlink(headers / "cycle.h", "cycle.h")
    with pytest.raises(RuntimeError, match="symlink cycle"):
        compilation._toolchain_inventory()


def test_file_change_during_hash_is_rejected(layout, monkeypatch):
    original = compilation._hash_file

    def changed(path):
        result = original(path)
        path.write_bytes(b"changed during inventory")
        return result

    monkeypatch.setattr(compilation, "_hash_file", changed)
    with pytest.raises(RuntimeError, match="changed while hashing"):
        compilation._toolchain_inventory()
