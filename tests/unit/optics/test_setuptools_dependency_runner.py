"""CPU runner restrictions and source binding, never installed dependency entry."""

import hashlib
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from tests.support.setuptools_dependency_process import CATALOG, PROFILE, PROFILE_SHA256, _cpu_only, _reap_cpu_group
from tests.unit.optics.test_native_bootstrap import _internal
from tests.unit.optics.test_native_sources import ROOT, release


def test_profile_pin_is_exact_git_release_bytes():
    data = subprocess.check_output(["git", "show", "HEAD:" + PROFILE], cwd=ROOT, timeout=20)
    assert hashlib.sha256(data).hexdigest() == PROFILE_SHA256
    if os.name == "posix":
        assert (ROOT / PROFILE).read_bytes() == data


def test_cpu_hook_allows_only_exact_source_git_commands(tmp_path):
    revision = "a" * 40
    hook = _cpu_only(tmp_path, revision)
    for argv in (["git", "rev-parse", "HEAD"], ["git", "show", revision + ":" + CATALOG]):
        hook("subprocess.Popen", ("git", argv, tmp_path, None))
    for args in (
        ("nvcc", ["nvcc", "--version"], tmp_path, None),
        ("git", ["git", "status"], tmp_path, None),
        ("git", ["git", "show", "b" * 40 + ":" + CATALOG], tmp_path, None),
        ("git", ["git", "rev-parse", "HEAD"], tmp_path.parent, None),
        ("git", ["git", "rev-parse", "HEAD"], tmp_path, {}),
        ("git", ["git", "rev-parse", "HEAD"], str(tmp_path), None),
        ("sh", ["git", "rev-parse", "HEAD"], tmp_path, None),
    ):
        with pytest.raises(RuntimeError, match="forbids this process"):
            hook("subprocess.Popen", args)
    for name in ("torch", "torch.cuda", "warp", "numpy", "scipy", "trimesh"):
        with pytest.raises(RuntimeError, match="backend import"):
            hook("import", (name,))
    for event in ("os.system", "os.fork", "os.posix_spawn", "ctypes.dlopen"):
        with pytest.raises(RuntimeError, match="process/library"):
            hook(event, ())


@pytest.mark.skipif(sys.platform != "linux", reason="runner's exact Popen audit ABI is Linux-only")
def test_actual_cold_binding_and_reaudits_under_runner_hook(release):
    # Load the outer test helper without installing it as a repository module.
    # The source session/finder/guard binding and audits are the real code.
    before = f"""
import socket
driver_spec = importlib.util.spec_from_file_location('cpu_test_driver', {str(ROOT / 'tests/support/setuptools_dependency_process.py')!r})
driver = importlib.util.module_from_spec(driver_spec)
driver_spec.loader.exec_module(driver)
"""
    after = """
entry._verify(session)
sys.addaudithook(driver._cpu_only(root, session.revision))
a, b = socket.socketpair()
try:
    session.bind_guarded_custody(a, {
        'protocol_version': 'q54.closed-worker.guarded-wait.v1',
        'session': 'a' * 32, 'device_ordinal': 1,
        'ack_timeout_ms': 1000, 'operation_timeout_ms': 120000})
    session.audit()
    session.snapshot.verify_unchanged()
    session.audit()
    assert not session.failed
    assert not {n.split('.')[0] for n in sys.modules} & {'torch', 'warp', 'numpy', 'scipy', 'setuptools', '_distutils_hack', 'distutils'}
    print('real binding and repeated source audits passed')
finally:
    a.close()
    b.close()
"""
    result = _internal(release, before=before, after=after, lane="cuda")
    assert result[0] == 0, result[2].decode(errors="replace")
    assert result[1].splitlines() == [b"real binding and repeated source audits passed"]


@pytest.mark.skipif(sys.platform != "linux", reason="actual Linux private process group required")
@pytest.mark.parametrize("parent_exits", [False, True])
def test_cpu_cleanup_stops_owned_descendant_group(tmp_path, parent_exits):
    identity = tmp_path / "descendant.pid"
    child_code = f"""
import subprocess, sys, time
from pathlib import Path
descendant = subprocess.Popen([sys.executable, '-I', '-S', '-B', '-c', 'import time; time.sleep(30)'])
Path({str(identity)!r}).write_text(str(descendant.pid))
if not {parent_exits!r}:
    descendant.wait()
"""
    process = subprocess.Popen([sys.executable, "-I", "-S", "-B", "-c", child_code], start_new_session=True)
    cleaned = False
    try:
        deadline = time.monotonic() + 5
        published = ""
        while time.monotonic() < deadline:
            published = identity.read_text() if identity.exists() else ""
            if published.isdecimal():
                break
            time.sleep(0.01)
        assert published.isdecimal()
        descendant = int(published)
        assert os.getpgid(descendant) == process.pid
        if parent_exits:
            assert process.wait(timeout=5) == 0
        _reap_cpu_group(process)
        cleaned = True
        assert process.poll() is not None
        # A grandchild belongs to the OS reaper after its parent exits; this
        # CPU helper claims termination, not compiler subreaper/wait ownership.
        status = Path(f"/proc/{descendant}/stat")
        deadline = time.monotonic() + 5
        while status.exists() and time.monotonic() < deadline:
            try:
                state = status.read_text().rsplit(")", 1)[1].split()[0]
            except FileNotFoundError:
                break
            if state == "Z":
                break
            time.sleep(0.01)
        assert not status.exists() or status.read_text().rsplit(")", 1)[1].split()[0] == "Z"
    finally:
        if not cleaned:
            _reap_cpu_group(process)
