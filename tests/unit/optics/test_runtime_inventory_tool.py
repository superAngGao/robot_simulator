"""Offline wheel facts must not execute installed code or grant admission."""

import base64
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from tools.optical_runtime_inventory import inventory


@pytest.fixture
def installed(tmp_path):
    root = tmp_path / "site"
    root.mkdir()
    metadata = "example_pkg-1.2.dist-info/METADATA"
    record = "example_pkg-1.2.dist-info/RECORD"
    files = {
        metadata: b"Name: Example-Pkg\nVersion: 1.2\nRequires-Dist: other-package\n\n",
        "example_pkg/__init__.py": b"raise RuntimeError('must never execute package')\n",
        "example_pkg/native.so": b"not a real binary; must never load",
        "example_pkg/hook.pth": b"import forbidden_startup_hook\n",
    }
    rows = []
    for relative, data in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip("=")
        rows.append([relative, "sha256=" + digest, str(len(data))])
    rows.extend(
        [[record, "", ""], ["../../../bin/example", "", ""], ["example_pkg/__pycache__/x.pyc", "", ""]]
    )
    with (root / record).open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows(rows)
    return root, record, rows


def test_inventory_reads_bytes_without_importing_packages_or_dependencies(installed):
    root, _, _ = installed
    before = dict(sys.modules)
    result = inventory(root, ["example_pkg"])
    assert result["purpose"] == "offline_facts_only_not_runtime_admission"
    assert result["distributions"][0]["requires_dist"] == ["other-package"]
    assert len(result["files"]) == 5
    assert {row["reason"] for row in result["omitted"]} == {"external_entry_point", "bytecode_disabled"}
    assert "example_pkg" not in sys.modules and "other_package" not in sys.modules
    assert all(sys.modules.get(name) is value for name, value in before.items())
    assert not {"modules", "publications", "absent_probes", "namespaces"} & result.keys()
    assert result == inventory(root, ["Example.Pkg"])


@pytest.mark.parametrize("mutation", ["bytes", "size", "duplicate", "escape", "missing_identity"])
def test_changed_or_malformed_installation_fails_without_a_partial_inventory(installed, mutation):
    root, record, rows = installed
    if mutation == "bytes":
        (root / "example_pkg/native.so").write_bytes(b"changed")
    elif mutation == "size":
        rows[0][2] = "9999"
    elif mutation == "duplicate":
        rows.append(rows[0])
    elif mutation == "escape":
        rows.append(["../outside.py", "", ""])
    else:
        rows = rows[1:]
    with (root / record).open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows(rows)
    with pytest.raises(ValueError):
        inventory(root, ["example-pkg"])


@pytest.mark.parametrize("names", [[], ["missing"], ["Example-Pkg", "example_pkg"]])
def test_explicit_distribution_selection_is_not_automatic_dependency_discovery(installed, names):
    with pytest.raises(ValueError):
        inventory(installed[0], names)


def test_manual_outside_site_is_recorded_as_an_exclusion_not_read(installed):
    root, record, rows = installed
    rows.append(["../../../share/man/man1/example.1", "sha256=not-checked-outside-root", "12"])
    with (root / record).open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows(rows)
    result = inventory(root, ["example-pkg"])
    assert {
        "distribution": "example-pkg",
        "path": "../../../share/man/man1/example.1",
        "reason": "external_manual",
    } in result["omitted"]
    assert len(result["files"]) == 5


def test_file_changed_during_hash_is_rejected(installed, monkeypatch):
    from tools import optical_runtime_inventory as tool

    root, _, _ = installed
    original = tool.os.fstat
    calls = []

    def changed(fd):
        value = original(fd)
        calls.append(True)
        if len(calls) == 1:
            (root / "example_pkg-1.2.dist-info/METADATA").write_bytes(b"changed after open")
        return value

    monkeypatch.setattr(tool.os, "fstat", changed)
    with pytest.raises(ValueError, match="changed while reading"):
        inventory(root, ["example-pkg"])


def test_cli_is_isolated_and_refuses_to_overwrite_evidence(installed, tmp_path):
    script = Path(__file__).resolve().parents[3] / "tools" / "optical_runtime_inventory.py"
    output = tmp_path / "inventory.json"
    command = [
        sys.executable,
        "-I",
        "-S",
        "-B",
        str(script),
        "--site-root",
        str(installed[0]),
        "--distribution",
        "example-pkg",
        "--output",
        str(output),
    ]
    first = subprocess.run(command, check=True, capture_output=True, text=True, timeout=10)
    data = output.read_bytes()
    assert json.loads(first.stdout)["sha256"] == hashlib.sha256(data).hexdigest()
    second = subprocess.run(command, capture_output=True, text=True, timeout=10)
    assert second.returncode != 0 and "overwrite" in second.stderr
    assert output.read_bytes() == data
    unsafe = subprocess.run(
        [sys.executable, str(script), *command[5:]], capture_output=True, text=True, timeout=10
    )
    assert unsafe.returncode != 0 and "requires -I -S -B" in unsafe.stderr
