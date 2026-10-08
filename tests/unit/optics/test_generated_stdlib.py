"""Real bootstrap entrypoints with disposable CPU interpreter-file faults.

The scratch directory simulates trusted startup intrinsics for fault injection;
it is not proof of the actual deployment. Real lazy sysconfig is tested separately.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from tests.unit.optics.test_native_bootstrap import _internal
from tests.unit.optics.test_native_sources import release as _release_fixture

release = _release_fixture
pytestmark = pytest.mark.skipif(
    os.name != "posix" or sys.version_info[:2] != (3, 12),
    reason="generated-module fault contract requires CPython3.12 POSIX",
)


def _fixture_source(tmp_path):
    base = tmp_path / "interpreter_stdlib"
    base.mkdir()
    # The independent alias descriptor also uses the simulated stdlib root.
    # Preserve its real source unchanged; only the generated-data file is faulted.
    package = base / "multiprocessing"
    package.mkdir()
    (package / "__init__.py").write_bytes((Path(sys._stdlib_dir) / "multiprocessing/__init__.py").read_bytes())
    name = f"_sysconfigdata_{sys.abiflags}_{sys.platform}_{sys.implementation._multiarch}"
    origin = base / (name + ".py")
    marker = tmp_path / "body_ran"
    origin.write_text(f"open({str(marker)!r}, 'w').write('executed')\nbuild_time_vars = {{}}\n")
    before = f"sys._stdlib_dir = {str(base)!r}\nsys.path.append(sys._stdlib_dir)"
    return name, origin, marker, before


@pytest.mark.parametrize("mutation", ["bytes", "missing", "symlink", "cached_bytes", "cached_missing"])
def test_generated_source_drift_fails_in_real_preload_or_boundary(release, tmp_path, mutation):
    name, origin, marker, before = _fixture_source(tmp_path)
    other = tmp_path / "other.py"
    other.write_bytes(origin.read_bytes())
    result = _internal(release, before=before, after=f"""
entry._verify(session)
name = {name!r}
origin = Path({str(origin)!r})
mutation = {mutation!r}
if mutation.startswith('cached_'):
    importlib.import_module(name)
    session.audit()
if mutation in ('bytes', 'cached_bytes'):
    origin.write_bytes(origin.read_bytes() + b'\\n')
else:
    origin.unlink()
    if mutation == 'symlink':
        origin.symlink_to({str(other)!r})
try:
    if mutation.startswith('cached_'):
        session.audit()
    else:
        importlib.import_module(name)
except entry.SourceBootstrapError:
    assert session.failed and session.failures
else:
    raise AssertionError('source drift accepted')
assert not {{'numpy','warp','torch','benchmarks'}}.intersection(sys.modules)
print('source_drift_rejected')
""")
    assert result[0] == 0, result[2].decode()
    assert result[1].strip() == b"source_drift_rejected"
    assert marker.exists() is mutation.startswith("cached_")


@pytest.mark.parametrize("mutation", ["origin", "loader", "loader_path", "package", "name"])
def test_generated_wrong_resolved_spec_never_executes_body(release, tmp_path, mutation):
    name, origin, marker, before = _fixture_source(tmp_path)
    before += f"""
original = importlib.machinery.PathFinder.find_spec
def injected(cls, fullname, path=None, target=None):
    spec = original(fullname, path, target)
    if fullname == {name!r}:
        mutation = {mutation!r}
        if mutation == 'origin': spec.origin += '.wrong'
        elif mutation == 'loader': spec.loader = object()
        elif mutation == 'loader_path': spec.loader.path += '.wrong'
        elif mutation == 'package': spec.submodule_search_locations = []
        elif mutation == 'name': spec.name += '_wrong'
    return spec
importlib.machinery.PathFinder.find_spec = classmethod(injected)
"""
    result = _internal(release, before=before, after=f"""
entry._verify(session)
try:
    importlib.import_module({name!r})
except entry.SourceBootstrapError as error:
    assert 'spec or loader drift' in str(error)
    assert session.failed
else:
    raise AssertionError('bad generated spec accepted')
assert {name!r} not in sys.modules
print('generated_spec_rejected')
""")
    assert result[0] == 0, result[2].decode()
    assert result[1].strip() == b"generated_spec_rejected" and not marker.exists()


@pytest.mark.parametrize("suffix", [".child", "_other", "_foreign_abi"])
def test_generated_name_is_not_a_prefix_permission(release, tmp_path, suffix):
    name, origin, marker, before = _fixture_source(tmp_path)
    denied = name + suffix
    result = _internal(release, before=before, after=f"""
entry._verify(session)
assert session.is_interpreter_name({name!r})
assert not session.is_interpreter_name({denied!r})
try:
    session.finder.find_spec({denied!r})
except entry.SourceBootstrapError:
    assert session.failed
else:
    raise AssertionError('generated prefix accepted')
assert {name!r} not in sys.modules
print('prefix_rejected')
""")
    assert result[0] == 0, result[2].decode()
    assert result[1].strip() == b"prefix_rejected" and not marker.exists()
