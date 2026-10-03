"""Actual isolated source entry and test-only fault injection; no GPU imports."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import py_compile
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from optical_worker.native_source_policy import validate_native_source_bytes
from optical_worker.native_sources import CATALOG_PATH
from optical_worker.protocol import ClosedWorkerProtocolError
from tests.unit.optics.test_native_sources import _git
from tests.unit.optics.test_native_sources import release as _release_fixture

release = _release_fixture

ENTRY = "optical_worker/native_bootstrap.py"


def _run(command, *, timeout=30):
    child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        stdout, stderr = child.communicate(timeout=timeout)
    except BaseException as error:
        child.kill()
        stdout, stderr = child.communicate(timeout=10)
        assert child.poll() is not None
        if isinstance(error, subprocess.TimeoutExpired):
            # Windows communicate reader threads publish bytes only after EOF.
            error.output, error.stderr = stdout, stderr
        raise
    assert child.poll() is not None
    return child.returncode, stdout, stderr


def _args(release, lane="warp"):
    root, revision = release
    return [str(root), revision, lane, hashlib.sha256((root / CATALOG_PATH).read_bytes()).hexdigest(), "1"]


def _direct(release, lane="warp", args=None):
    return _run(
        [
            sys.executable,
            "-I",
            "-S",
            "-B",
            str(release[0] / ENTRY),
            *(_args(release, lane) if args is None else args),
        ]
    )


def _report(result):
    code, stdout, stderr = result
    assert code == 0, stderr.decode()
    assert len(stdout) >= 4 and int.from_bytes(stdout[:4], "big") == len(stdout) - 4
    value = json.loads(stdout[4:])
    assert value["status"] == "cpu_sources_verified"
    assert len(value["execution_trace"]) == 12
    assert len(value["repository_modules"]) == 7
    for name in {item["module"] for item in value["execution_trace"]}:
        assert [item["operation"] for item in value["execution_trace"] if item["module"] == name] == [
            "started",
            "completed",
        ]
    return value


def _copy(release, tmp_path):
    root = tmp_path / "release"
    shutil.copytree(release[0], root)
    return root, release[1]


def _internal(release, before="", after=""):
    # Fixed isolated test driver calls the REAL production classes, not a loader
    # duplicate. It is intentionally not an accepted product entry/callback.
    root, revision, lane, digest, _ = _args(release)
    script = f"""
import importlib.util, importlib, sys, types
from pathlib import Path
root = Path({root!r})
spec = importlib.util.spec_from_file_location('source_test', root / {ENTRY!r})
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)
entry.__name__ = '__main__'
sys.modules['__main__'] = entry
{before}
session = entry._install(root, {revision!r}, {lane!r}, {digest!r})
{after}
"""
    return _run([sys.executable, "-I", "-S", "-B", "-c", script])


@pytest.mark.parametrize("lane", ["warp", "cuda"])
def test_actual_source_entry(release, lane):
    value = _report(_direct(release, lane))
    assert value["lane"] == lane
    assert value["repository_revision"] == release[1]


@pytest.mark.parametrize("lane", ["warp", "cuda"])
def test_actual_posix_source_entry_without_packaging(lane):
    if os.name != "posix":
        pytest.skip("actual POSIX release test is required remotely")
    root = Path(__file__).resolve().parents[3]
    _report(_direct((root, _git(root, "rev-parse", "HEAD")), lane))


@pytest.mark.parametrize("relative", ["optical_worker/__init__.py", "optical_worker/native_sources.py"])
@pytest.mark.parametrize(
    "mode", [py_compile.PycInvalidationMode.TIMESTAMP, py_compile.PycInvalidationMode.UNCHECKED_HASH]
)
def test_actual_core_ignores_valid_stale_bytecode(release, tmp_path, relative, mode):
    copied = _copy(release, tmp_path)
    path = copied[0] / relative
    original, stat = path.read_bytes(), path.stat()
    poison = b"raise RuntimeError('CACHED_SENTINEL_EXECUTED')\n"
    poison += b"#" * (len(original) - len(poison))
    assert len(poison) == len(original)
    path.write_bytes(poison)
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    cached = py_compile.compile(str(path), doraise=True, invalidation_mode=mode)
    cache_bytes = Path(cached).read_bytes()
    path.write_bytes(original)
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    # Demonstrate that ordinary isolated -B imports really consume this cache.
    code, _, stderr = _run(
        [
            sys.executable,
            "-I",
            "-S",
            "-B",
            "-c",
            f"import sys; sys.path.insert(0, {str(copied[0])!r}); import optical_worker.native_sources",
        ]
    )
    assert code != 0 and b"CACHED_SENTINEL_EXECUTED" in stderr
    _report(_direct(copied))
    assert Path(cached).read_bytes() == cache_bytes


@pytest.mark.parametrize(
    "mutation", ["revision", "catalog", "lane", "extra", "missing", "source", "sourceless", "symlink"]
)
def test_cold_failures_have_no_success_report(release, tmp_path, mutation):
    copied = _copy(release, tmp_path)
    args = _args(copied)
    path = copied[0] / "optical_worker/protocol.py"
    if mutation == "revision":
        args[1] = "0" * 40
    elif mutation == "catalog":
        args[3] = "0" * 64
    elif mutation == "lane":
        args[2] = "other"
    elif mutation == "extra":
        args.append("callback")
    elif mutation == "source":
        path.write_bytes(path.read_bytes() + b"\n# drift\n")
    elif mutation == "sourceless":
        py_compile.compile(str(path), cfile=str(path.with_suffix(".pyc")), doraise=True)
        path.unlink()
    elif mutation == "symlink":
        destination = tmp_path / "foreign.py"
        destination.write_bytes(path.read_bytes())
        path.unlink()
        try:
            path.symlink_to(destination)
        except OSError:
            pytest.skip("host cannot create symlinks; POSIX run is mandatory")
    else:
        path.unlink()
    code, stdout, stderr = _direct(copied, args=args)
    assert code != 0 and stdout == b"" and b"source_bootstrap_failed" in stderr


@pytest.mark.parametrize(
    "action",
    [
        "importlib.reload(module)",
        "module.__loader__.exec_module(module)",
        "session.finder.find_spec(module.__name__)",
        "sys.modules[module.__name__] = types.ModuleType(module.__name__); session.audit()",
        "module.__spec__.loader = object(); session.audit()",
        "module.__file__ = '/foreign/core.py'; session.audit()",
        "sys.modules['optical_worker'].__path__.append('/foreign'); session.audit()",
    ],
)
def test_exact_execution_identity_and_no_reexecution(release, action):
    result = _internal(
        release,
        after=f"""
entry._verify(session)
module = sys.modules['optical_worker.native_sources']
try:
    {action}
except entry.SourceBootstrapError:
    pass
else:
    raise AssertionError('drift accepted')
assert session.failed
try:
    entry._verify(session)
except entry.SourceBootstrapError:
    pass
else:
    raise AssertionError('failed session revived')
print('rejected')
""",
    )
    assert result[0] == 0, result[2].decode()
    assert result[1].splitlines() == [b"rejected"]


@pytest.mark.parametrize(
    "name", ["optics.cuda_scene", "optical_worker.native_driver", "benchmarks", "optical_worker.unknown"]
)
def test_unlisted_or_gpu_import_fails_before_other_finders(release, name):
    result = _internal(
        release,
        after=f"""
entry._verify(session)
class SentinelFinder:
    def find_spec(self, fullname, path=None, target=None):
        raise AssertionError('fallback/body must not execute')
sys.meta_path.insert(1, SentinelFinder())
try:
    importlib.import_module({name!r})
except entry.SourceBootstrapError:
    pass
else:
    raise AssertionError('denied import accepted')
assert session.failed
assert not {{'numpy', 'torch', 'warp'}}.intersection(sys.modules)
print('rejected')
""",
    )
    assert result[0] == 0, result[2].decode()
    assert result[1].splitlines() == [b"rejected"]


def test_preloaded_module_rejected(release):
    result = _internal(release, before="sys.modules['optical_worker'] = types.ModuleType('optical_worker')")
    assert result[0] != 0 and b"repository module preloaded" in result[2]


@pytest.mark.parametrize("name", ["optical_worker", "optical_worker.native_sources"])
@pytest.mark.parametrize("field", ["spec_name", "package"])
def test_review_core_spec_and_package_drift_is_terminal(release, name, field):
    result = _internal(
        release,
        after=f"""
entry._verify(session)
module = sys.modules[{name!r}]
if {field!r} == 'spec_name':
    module.__spec__.name = 'foreign.alias'
else:
    module.__package__ = 'foreign.package'
for attempt in range(2):
    try:
        entry._verify(session)
    except entry.SourceBootstrapError:
        assert session.failed
    else:
        raise AssertionError('identity drift accepted or session revived')
print('rejected')
""",
    )
    assert result[0] == 0, result[2].decode()
    assert result[1].splitlines() == [b"rejected"]


@pytest.mark.parametrize(
    "mutation",
    [
        "sys.modules.pop('__main__')",
        "replacement = types.ModuleType('__main__'); replacement.__file__ = entry.__file__; "
        "sys.modules['__main__'] = replacement",
        "replacement = types.ModuleType('__main__'); replacement.__dict__.update(entry.__dict__); "
        "sys.modules['__main__'] = replacement",
        "entry.__file__ = '/foreign/bootstrap.py'",
        "entry.__name__ = 'foreign.alias'",
        "entry.__package__ = 'foreign.package'",
        "entry.__loader__ = object()",
        "entry.__spec__ = object()",
    ],
)
def test_review_bootstrap_owner_is_held_and_replacement_is_terminal(release, mutation):
    result = _internal(
        release,
        after=f"""
entry._verify(session)
assert session.bootstrap_owner is entry
{mutation}
for attempt in range(2):
    try:
        entry._verify(session)
    except entry.SourceBootstrapError:
        assert session.failed
    else:
        raise AssertionError('bootstrap drift accepted or session revived')
assert session.bootstrap_owner is entry
print('rejected')
""",
    )
    assert result[0] == 0, result[2].decode()
    assert result[1].splitlines() == [b"rejected"]


def test_loader_exception_retains_failed_record_and_prevents_retry(release):
    result = _internal(
        release,
        after="""
def broken_compiler(*args, **kwargs):
    raise RuntimeError('compile_fault')
entry._CapturedSourceLoader.source_to_code = broken_compiler
try:
    entry._verify(session)
except RuntimeError as error:
    assert str(error) == 'compile_fault'
else:
    raise AssertionError('compile failure accepted')
assert session.failed
assert session.trace == [('started', 'optical_worker'), ('failed', 'optical_worker')]
assert session.modules['optical_worker'][2] == 'started'
assert 'optical_worker' not in sys.modules
try:
    entry._verify(session)
except entry.SourceBootstrapError:
    pass
else:
    raise AssertionError('failed session revived')
print('rejected')
""",
    )
    assert result[0] == 0, result[2].decode()
    assert result[1].splitlines() == [b"rejected"]


def test_post_capture_drift_never_executes_changed_source(release, tmp_path):
    copied = _copy(release, tmp_path)
    result = _internal(
        copied,
        after="""
path = root / 'optical_worker/__init__.py'
path.write_text("raise RuntimeError('DRIFT_BODY_EXECUTED')\\n")
try:
    entry._verify(session)
except Exception as error:
    assert 'DRIFT_BODY_EXECUTED' not in str(error)
else:
    raise AssertionError('drift accepted')
assert session.failed
assert ('completed', 'optical_worker') in session.trace
assert 'optical_worker' in session.modules
print('rejected')
""",
    )
    assert result[0] == 0, result[2].decode()
    assert result[1].splitlines() == [b"rejected"]


@pytest.mark.parametrize(
    "suffix",
    [
        "\nimport importlib.machinery\n",
        "\nimport importlib.util as alias\n",
        "\nimportlib.util.spec_from_file_location('escape', '/tmp/escape.py')\n",
        "\nexec('pass')\n",
        "\nCaptured = _CapturedSourceLoader\n",
    ],
)
def test_release_pinned_policy_rejects_parseable_rehashed_mutants(release, suffix):
    source = (release[0] / ENTRY).read_bytes()
    validate_native_source_bytes(ENTRY, source, expected_sha256=hashlib.sha256(source).hexdigest())
    mutated = source + suffix.encode()
    ast.parse(mutated)
    with pytest.raises(ClosedWorkerProtocolError):
        validate_native_source_bytes(ENTRY, mutated, expected_sha256=hashlib.sha256(mutated).hexdigest())


@pytest.mark.parametrize("point", ["verification", "response"])
def test_parent_alive_deadline_kills_and_reaps_real_bootstrap(release, point):
    # Fixed test-only fault injection into the actual entry. The production
    # command has no sleep/callback/fault selector. Parent remains alive.
    args = _args(release)
    script = f"""
import importlib.util, sys, time
spec = importlib.util.spec_from_file_location('source_test', {str(release[0] / ENTRY)!r})
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)
entry.__name__ = '__main__'
sys.modules['__main__'] = entry
sys.argv = [{str(release[0] / ENTRY)!r}, *{args!r}]
def stall(*args):
    print('at_stall', file=sys.stderr, flush=True)
    time.sleep(60)
if {point!r} == 'verification':
    entry._verify = stall
else:
    entry.os.write = stall
entry.main()
"""
    with pytest.raises(subprocess.TimeoutExpired) as caught:
        _run([sys.executable, "-I", "-S", "-B", "-c", script], timeout=5)
    assert b"at_stall" in caught.value.stderr
    assert not caught.value.stdout


@pytest.mark.parametrize(
    "payload",
    [
        b'{"lanes":{},"lanes":{}}\n',
        b'{"lanes": NaN}\n',
        b"{}",
        b" {}\n",
        b"[]\n",
    ],
)
def test_malformed_released_catalog_before_core_execution(release, tmp_path, payload):
    copied = _copy(release, tmp_path)
    (copied[0] / CATALOG_PATH).write_bytes(payload)
    _git(copied[0], "add", CATALOG_PATH)
    _git(copied[0], "commit", "-qm", "malformed catalog test")
    copied = copied[0], _git(copied[0], "rev-parse", "HEAD")
    result = _direct(copied)
    assert result[0] != 0 and not result[1]
    assert b"source_bootstrap_failed" in result[2]


def test_capture_scope_rejects_escape_before_read(release, tmp_path):
    copied = _copy(release, tmp_path)
    catalog_path = copied[0] / CATALOG_PATH
    catalog = json.loads(catalog_path.read_bytes())
    catalog["lanes"]["warp"][0] = "../outside.py"
    catalog["source_digests"].append({"relative_path": "../outside.py", "sha256": "0" * 64})
    catalog_path.write_text(json.dumps(catalog, sort_keys=True, separators=(",", ":")) + "\n", newline="\n")
    _git(copied[0], "add", CATALOG_PATH)
    _git(copied[0], "commit", "-qm", "escape catalog test")
    result = _direct((copied[0], _git(copied[0], "rev-parse", "HEAD")))
    assert result[0] != 0 and not result[1]
    assert b"source path scope" in result[2]


def test_loader_uses_held_source_and_rejects_alternate_entry(release):
    result = _internal(
        release,
        after="""
entry._verify(session)
loader = session.loaders['optical_worker.protocol']
assert loader.get_data(loader.origin) is session.sources[loader.relative][0]
assert loader.get_source(loader.name).encode() == session.sources[loader.relative][0]
for action in (lambda: loader.get_data(loader.origin + 'c'),
               lambda: loader.get_filename('alias'),
               lambda: loader.get_code(loader.name),
               lambda: entry._install(root, session.revision, session.lane, session.catalog_sha256)):
    try:
        action()
    except entry.SourceBootstrapError:
        pass
    else:
        raise AssertionError('alternate entry accepted')
session.audit()
print('rejected')
""",
    )
    assert result[0] == 0, result[2].decode()
    assert result[1].splitlines() == [b"rejected"]
