from __future__ import annotations

import ast
import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from tests.support.sacrificial_sources import RepositoryImportLedger, frozen_source_manifest

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
