"""Read-only C2 source policy; no worker/GPU admission is exercised here."""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from optical_worker.manifest import AST_POLICY, AST_POLICY_SHA256, validate_trusted_source_ast
from optical_worker.native_source_policy import (
    NATIVE_AST_POLICY_SHA256,
    NATIVE_AST_POLICY_VERSION,
    validate_native_source_bytes,
    validate_native_source_file,
)
from optical_worker.protocol import ClosedWorkerProtocolError, canonical_sha256

ROOT = Path(__file__).resolve().parents[3]
RENDERER = "optics/warp_native_render.py"


def _check(source: bytes, path="optical_worker/future_module.py"):
    return validate_native_source_bytes(path, source, expected_sha256=hashlib.sha256(source).hexdigest())


@pytest.fixture(scope="module")
def renderer_release():
    # The worker release is LF Git bytes. Never normalize input inside the
    # verifier to make this Windows CRLF worktree appear to be that release.
    return subprocess.check_output(["git", "show", f"HEAD:{RENDERER}"], cwd=ROOT, timeout=20)


def test_real_reviewed_renderer_passes_without_importing_it(renderer_release):
    before = set(sys.modules)
    assert _check(renderer_release, RENDERER) is None
    assert set(sys.modules) == before


@pytest.mark.parametrize("path", [
    "optical_worker/workloads.py", "optical_worker/native_input.py",
    "optical_worker/native_setup.py", "optical_worker/native_contract.py",
    "optical_worker/render_parameters.py", "optical_worker/provenance.py",
    "optical_worker/wait_protocol.py", "optical_worker/wait_supervision.py",
    "optics/cuda_arena.py", "optics/cuda_timeline.py", "optics/cuda_scene.py",
    "optics/cuda_lbvh.py", "optics/_cuda_lbvh_source.py", "optics/cuda_direct_light.py",
    "optics/device_scene.py", "optics/device_bvh.py", "optics/mjcf_visual.py",
    "optics/camera_presets.py", "optical_worker/native_source_policy.py",
])
def test_current_production_source_checks_do_not_execute_code(path):
    # Representative actual sources, deliberately NOT a transitive allowlist.
    source = (ROOT / path).read_bytes()
    validate_native_source_file(ROOT, path, expected_sha256=hashlib.sha256(source).hexdigest())


@pytest.mark.parametrize("source", [
    b"import ctypes", b"import ctypes as c", b"from ctypes import addressof",
    b"from ctypes import *", b"import pickle", b"import importlib.util",
    b"import benchmarks.optical.native", b"from examples import anything",
    b"from tools.optical_pipeline_lab import camera_presets", b"import tests.support",
    b"run = eval", b"x = builtins.exec", b"__import__('warp')",
    b"runtime.core.wp_cuda_launch_kernel(1)", b"runtime.core.wp_cuda_event_record(1)",
    b"x = callback.__closure__", b"x = callback.__wrapped__",
    b"getattr(callback, '__closure__')", b"vars(callback)['__wrapped__']",
    b"ctypes.c_float(1)", b"ctypes = replacement",
])
def test_forbidden_imports_aliases_and_native_ingress_reject(source):
    with pytest.raises(ClosedWorkerProtocolError):
        _check(source)


@pytest.mark.parametrize("old,new", [
    (b"import ctypes", b"import ctypes as ctypes"),
    (b"import ctypes", b"import ctypes\nimport ctypes"),
    (b"import ctypes", b"from ctypes import addressof"),
    (b"import ctypes", b"import ctypes\nescaped = ctypes"),
    (b"import ctypes", b"import ctypes\nescaped = ctypes.addressof"),
    (b"import ctypes", b"import ctypes\nctypes.CDLL('anything')"),
    (b"ctypes.addressof(parameter)", b"ctypes.cast(parameter, ctypes.c_int32)"),
    (b"bytes(parameter)", b"bytearray(parameter)"),
    (b"\n    snapshots = []", b"\n    snapshots = []\n    command.params[0] = None"),
    (b"\n    snapshots = []", b"\n    snapshots = []\n    escaped = ctypes"),
])
def test_rehashed_renderer_changes_cannot_expand_exception(renderer_release, old, new):
    assert old in renderer_release
    changed = renderer_release.replace(old, new)
    ast.parse(changed)  # A syntax error is not evidence of semantic rejection.
    with pytest.raises(ClosedWorkerProtocolError):
        _check(changed, RENDERER)  # Deliberately recompute the caller's hash.


def test_exact_source_pin_rejects_even_comment_and_newline_changes(renderer_release):
    with pytest.raises(ClosedWorkerProtocolError, match="reviewed release"):
        _check(renderer_release + b"\n# changed release\n", RENDERER)
    with pytest.raises(ClosedWorkerProtocolError, match="snapshot function has drifted"):
        _check(renderer_release.replace(b"\n", b"\r\n"), RENDERER)
    with pytest.raises(ClosedWorkerProtocolError, match="forbidden module"):
        _check(renderer_release, "optics/another_renderer.py")


@pytest.mark.parametrize("name,alias", [
    (name, alias) for name in ("eval", "exec", "compile", "__import__", "*")
    for alias in ("", " as run") if not (name == "*" and alias)
])
def test_forbidden_builtin_imports_with_or_without_alias_reject(name, alias):
    with pytest.raises(ClosedWorkerProtocolError, match="forbidden dynamic builtin"):
        _check(f"from builtins import {name}{alias}\n".encode())


@pytest.mark.parametrize("reflector", ["getattr", "setattr", "delattr", "hasattr"])
@pytest.mark.parametrize("attribute", [
    "__closure__", "__wrapped__", "wp_cuda_launch_kernel", "wp_cuda_event_record",
])
@pytest.mark.parametrize("spelling", ["direct", "qualified", "import_alias", "module_alias"])
def test_literal_reflection_rejects_ordinary_builtin_spellings(reflector, attribute, spelling):
    prefix, name = {
        "direct": ("", reflector),
        "qualified": ("import builtins\n", f"builtins.{reflector}"),
        "import_alias": (f"from builtins import {reflector} as reflect\n", "reflect"),
        "module_alias": ("import builtins as b\n", f"b.{reflector}"),
    }[spelling]
    with pytest.raises(ClosedWorkerProtocolError, match="reflects a forbidden"):
        _check(f"{prefix}{name}(runtime.core, {attribute!r})\n".encode())


@pytest.mark.parametrize("source", [
    b"import builtins; builtins.getattr(runtime.core, 'wp_cuda_launch_kernel')(1)",
    b"from builtins import eval as run; run('1 + 1')",
    b"from builtins import compile as build; build('x=1', 'probe', 'exec')",
    b"__builtins__['eval']('1+1')",
    b"import builtins; vars(builtins).get('eval')('1+1')",
    b"getattr(object=runtime.core, name='wp_cuda_event_record')",
])
def test_review_repros_and_literal_keyword_or_subscript_variants_reject(source):
    with pytest.raises(ClosedWorkerProtocolError):
        _check(source)


@pytest.mark.parametrize("source", [
    b"from builtins import bytes as copy_bytes; x = copy_bytes()",
    b"import builtins; x = builtins.getattr(value, 'ordinary_attribute')",
    b"from builtins import getattr as reflect; x = reflect(value, 'ordinary_attribute')",
    b"policy = {'forbidden_names': ['eval', 'exec'], 'attributes': ['__closure__']}",
])
def test_permitted_builtins_reflection_and_inert_policy_data_still_pass(source):
    assert _check(source) is None


@pytest.mark.parametrize("path", ["optical_worker/protocol.py", "optical_worker/lifecycle.py"])
def test_existing_cpu_regex_source_is_admitted_only_as_exact_release(path):
    source = subprocess.check_output(["git", "show", f"HEAD:{path}"], cwd=ROOT, timeout=20)
    assert b"re.compile(" in source
    assert _check(source, path) is None
    for changed in (
        source + b"\n# new release\n",
        source.replace(b"import re", b"import re\nre = fake_runtime"),
        source.replace(b"re.compile(", b"builtins.compile("),
    ):
        ast.parse(changed)
        with pytest.raises(ClosedWorkerProtocolError):
            _check(changed, path)
    with pytest.raises(ClosedWorkerProtocolError):
        _check(source, "optical_worker/unreviewed_regex.py")


def test_posix_actual_release_file_validates(renderer_release, tmp_path):
    digest = hashlib.sha256(renderer_release).hexdigest()
    if (ROOT / RENDERER).read_bytes() == renderer_release:
        # Actual deployed checkout, not reconstructed source on the server.
        validate_native_source_file(ROOT, RENDERER, expected_sha256=digest)
    else:
        # Explicit LF release fixture on Windows; not a claim that the original
        # CRLF working file matches the frozen POSIX source bytes.
        release_file = tmp_path / RENDERER
        release_file.parent.mkdir()
        release_file.write_bytes(renderer_release)
        validate_native_source_file(tmp_path, RENDERER, expected_sha256=digest)


@pytest.mark.parametrize("path", [
    "", "../optics/x.py", "/optics/x.py", "optics/../x.py", "optics//x.py",
    "./optics/x.py", "optics\\x.py", "C:/optics/x.py", "optics/x.pyc",
    "benchmarks/optical/x.py", "examples/x.py", "tests/x.py", "tools/unreviewed.py",
])
def test_paths_are_canonical_production_python_only(path):
    with pytest.raises(ClosedWorkerProtocolError):
        _check(b"x = 1", path)


@pytest.mark.parametrize("source,digest", [
    (b"x=1", "0" * 64), (b"x=1", "A" * 64), (b"x=1", "abcd"),
    ("x=1", "0" * 64), (bytearray(b"x=1"), "0" * 64), (b"x=1", None),
])
def test_byte_and_digest_mismatch_reject(source, digest):
    with pytest.raises(ClosedWorkerProtocolError):
        validate_native_source_bytes("optics/x.py", source, expected_sha256=digest)


@pytest.mark.parametrize("source", [b"\xff", b"x = (", b"x = '\x00'"])
def test_unparseable_sources_fail_closed(source):
    with pytest.raises(ClosedWorkerProtocolError, match="cannot be parsed"):
        _check(source)


def test_file_read_is_nonexecuting_and_requires_exact_digest(tmp_path):
    path = tmp_path / "optics" / "x.py"
    path.parent.mkdir()
    path.write_bytes(b"raise RuntimeError('must not execute')\n")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    validate_native_source_file(tmp_path, "optics/x.py", expected_sha256=digest)
    path.write_bytes(b"raise RuntimeError('changed')\n")
    with pytest.raises(ClosedWorkerProtocolError, match="manifest"):
        validate_native_source_file(tmp_path, "optics/x.py", expected_sha256=digest)
    with pytest.raises(ClosedWorkerProtocolError, match="unavailable"):
        validate_native_source_file(tmp_path, "optics/missing.py", expected_sha256=digest)


@pytest.mark.parametrize("directory", [False, True])
def test_symlink_file_or_parent_rejected(tmp_path, directory):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "x.py").write_bytes(b"x=1")
    root = tmp_path / "repo"
    root.mkdir()
    try:
        if directory:
            (root / "optics").symlink_to(outside, target_is_directory=True)
        else:
            (root / "optics").mkdir()
            (root / "optics" / "x.py").symlink_to(outside / "x.py")
    except OSError as exc:
        pytest.skip(f"symlinks unavailable on this test host: {exc}")
    with pytest.raises(ClosedWorkerProtocolError, match="symlinked"):
        validate_native_source_file(root, "optics/x.py", expected_sha256=hashlib.sha256(b"x=1").hexdigest())


def test_cold_import_remains_cpu_only_and_does_not_mutate_c0_policy():
    script = f"""
import sys
sys.path.insert(0, {str(ROOT)!r})
from optical_worker.manifest import AST_POLICY, AST_POLICY_SHA256
before = repr(AST_POLICY), AST_POLICY_SHA256
import optical_worker.native_source_policy
assert (repr(AST_POLICY), AST_POLICY_SHA256) == before
assert not set(sys.modules) & {{'numpy', 'torch', 'warp', 'optics', 'benchmarks'}}
"""
    subprocess.run([sys.executable, "-I", "-S", "-c", script], check=True, timeout=20)
    assert AST_POLICY["schema_version"].endswith("v1")
    assert AST_POLICY_SHA256 == canonical_sha256(AST_POLICY)
    assert NATIVE_AST_POLICY_VERSION.endswith("v4")
    assert NATIVE_AST_POLICY_SHA256 != AST_POLICY_SHA256
    with pytest.raises(ClosedWorkerProtocolError, match="forbidden module"):
        validate_trusted_source_ast(ROOT / RENDERER)


def test_cleanup_obligation_and_old_worker_gate_remain_explicit():
    ledger = json.loads((ROOT / "benchmarks/results/q54_p12_4/cleanup_ledger.json").read_text())
    entry = next(x for x in ledger["entries"] if x["entry_id"] == "native_worker_source_policy")
    assert "optical_worker/native_source_policy.py" in entry["paths"]
    assert entry["must_resolve_by"] == "P12.4e"
    for branch in ("warp_selected", "cuda_selected"):
        assert entry[branch]["action"] == "collapse"
    contract_path = ROOT / "benchmarks/results/q54_p12_4/closed_worker_runtime_contract.json"
    contract = json.loads(contract_path.read_text())
    assert contract["gpu_preparation_enabled"] is False
    assert contract["protocol_version"].endswith("v1")
    assert "optical_worker/native_source_policy.py" not in contract["trusted_source_paths"]
