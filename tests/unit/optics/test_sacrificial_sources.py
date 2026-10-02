from __future__ import annotations

import ast
import copy
import json
import os
import py_compile
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tests.support.sacrificial_sources import (
    REPOSITORY_SOURCES,
    RepositoryImportLedger,
    VerifiedRepositorySourceLoader,
    frozen_source_manifest,
)

ROOT = Path(__file__).resolve().parents[3]


def test_changed_digest_and_incomplete_transitive_source_closure_are_rejected():
    original = frozen_source_manifest(ROOT)
    for mutation in ("digest", "remove", "extra"):
        manifest = copy.deepcopy(original)
        if mutation == "digest":
            manifest["sources"][0]["sha256"] = "0" * 64
        elif mutation == "remove":
            manifest["sources"].pop()
        else:
            manifest["sources"].append(dict(path="not-approved.py", sha256="0" * 64))
        with pytest.raises(RuntimeError, match="frozen repository closure"):
            RepositoryImportLedger(manifest)


def test_import_finder_rejects_real_unlisted_module_before_execution():
    code = """
import sys
sys.path.insert(0, sys.argv[1])
from pathlib import Path
from tests.support.sacrificial_sources import RepositoryImportLedger, frozen_source_manifest
ledger = RepositoryImportLedger(frozen_source_manifest(Path(sys.argv[1])))
ledger.install()
try:
    import tests.support.sacrificial_protocol_child
except RuntimeError as exc:
    assert 'outside frozen closure' in str(exc)
else:
    raise AssertionError('unlisted repository source executed')
assert 'tests.support.sacrificial_protocol_child' not in sys.modules
assert not any(name.split('.')[0] in ('warp', 'torch', 'optics') for name in sys.modules)
ledger.verify()
"""
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-c", code, str(ROOT)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_cold_schema_import_admits_complete_render_contract_closure():
    code = """
import json, sys
from pathlib import Path
sys.path.extend(json.loads(sys.argv[2]))
sys.path.insert(0, sys.argv[1])
from tests.support.sacrificial_sources import RepositoryImportLedger, frozen_source_manifest
ledger = RepositoryImportLedger(frozen_source_manifest(Path(sys.argv[1])))
ledger.install()
import benchmarks.optical.schema
assert 'optical_worker.render_parameters' in sys.modules
assert 'optical_worker.camera_contract' in sys.modules
ledger.verify()
try:
    import tests.support.sacrificial_protocol_child
except RuntimeError as exc:
    assert 'outside frozen closure' in str(exc)
else:
    raise AssertionError('unlisted source admitted after legitimate imports')
"""
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            "-c",
            code,
            str(ROOT),
            json.dumps([p for p in sys.path if p and not Path(p).resolve().is_relative_to(ROOT)]),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_guardian_is_constructed_before_site_packages_or_gpu_driver_import():
    tree = ast.parse((ROOT / "tests/support/sacrificial_warp_child.py").read_text())
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    guardian = next(
        node.lineno
        for node in ast.walk(main)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "SacrificialRenderGuardian"
    )
    imports = [
        node
        for node in ast.walk(main)
        if isinstance(node, ast.ImportFrom) and node.module == "tests.support.sacrificial_warp_case"
    ]
    site = next(
        node.lineno
        for node in ast.walk(main)
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "sys.path.append"
    )
    assert guardian < site < imports[0].lineno
    assert not any(
        isinstance(node, (ast.Import, ast.ImportFrom))
        and (getattr(node, "module", "") or "").split(".")[0] in {"warp", "torch", "optics"}
        for node in tree.body
    )


def test_test_only_supervisor_does_not_change_product_ak_protocol():
    # Test helpers are not part of the product worker's source policy.
    contract = json.loads(
        (ROOT / "benchmarks/results/q54_p12_4/closed_worker_runtime_contract.json").read_text()
    )
    assert not any("sacrificial" in path for path in contract["trusted_source_paths"])


@pytest.mark.parametrize(
    "module,relative",
    [
        ("optical_worker", "optical_worker/__init__.py"),
        ("tests.support.sacrificial_supervisor", "tests/support/sacrificial_supervisor.py"),
    ],
)
@pytest.mark.parametrize(
    "cache_mode", [py_compile.PycInvalidationMode.TIMESTAMP, py_compile.PycInvalidationMode.UNCHECKED_HASH]
)
def test_ledger_executes_verified_source_not_stale_cache(tmp_path, module, relative, cache_mode):
    # Harmless old/new constant fixture, outside the real candidate checkout.
    for name in REPOSITORY_SOURCES:
        destination = tmp_path / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
    target = tmp_path / relative
    target.write_text("VALUE = 'old'\n", encoding="utf-8")
    os.utime(target, (1700000000, 1700000000))
    py_compile.compile(str(target), doraise=True, invalidation_mode=cache_mode)
    target.write_text("VALUE = 'new'\n", encoding="utf-8")
    os.utime(target, (1700000000, 1700000000))
    code = """
import importlib, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
sys.dont_write_bytecode = True
from tests.support.sacrificial_sources import RepositoryImportLedger, frozen_source_manifest
ledger = RepositoryImportLedger(frozen_source_manifest(Path(sys.argv[1])))
ledger.install()
module = importlib.import_module(sys.argv[2])
assert module.VALUE == 'new', module.VALUE
ledger.verify()
"""
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-c", code, str(tmp_path), module], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_loader_rejects_source_changed_after_manifest_check(tmp_path):
    import hashlib

    target = tmp_path / "constant.py"
    source = b"VALUE = 'new'\n"
    target.write_bytes(source)
    loader = VerifiedRepositorySourceLoader("constant", str(target), hashlib.sha256(source).hexdigest())
    target.write_bytes(b"VALUE = 'old'\n")
    with pytest.raises(RuntimeError, match="changed before compilation"):
        loader.get_code("constant")
    with pytest.raises(RuntimeError, match="identity changed"):
        loader.get_code("other")


@pytest.mark.parametrize(
    "relative", ["tests/support/sacrificial_warp_child.py", "tools/optical_sacrificial_warp_supervisor.py"]
)
def test_bootstrap_uses_fresh_cache_namespace_before_any_repository_import(relative):
    tree = ast.parse((ROOT / relative).read_text())
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    first_repository_import = min(
        node.lineno
        for node in ast.walk(main)
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("tests.")
    )
    prefix = next(
        node.lineno
        for node in ast.walk(main)
        if isinstance(node, ast.Assign)
        and any(ast.unparse(target) == "sys.pycache_prefix" for target in node.targets)
    )
    create = next(
        node
        for node in ast.walk(main)
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "bootstrap_cache.mkdir"
    )
    assert create.lineno < prefix < first_repository_import
    assert any(
        keyword.arg == "exist_ok" and isinstance(keyword.value, ast.Constant) and keyword.value.value is False
        for keyword in create.keywords
    )
