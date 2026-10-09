"""Exercise the actual child exception block without a native worker or exit."""
import ast
from pathlib import Path
from types import SimpleNamespace as NS

import pytest


@pytest.mark.parametrize("diagnostic_failure", [None, "inspect", "write"])
def test_roots_diagnostic_never_replaces_original_failure(diagnostic_failure):
    path = Path(__file__).resolve().parents[3] / "tests/support/native_runtime_process.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    blocks = [node for node in ast.walk(tree) if isinstance(node, ast.Try)
              and len(node.body) == 1 and isinstance(node.body[0], ast.Assign)
              and isinstance(node.body[0].value, ast.Call)
              and isinstance(node.body[0].value.func, ast.Attribute)
              and node.body[0].value.func.attr == "prepare_warp_modules"]
    assert len(blocks) == 1
    original = RuntimeError("fixed compile error")
    diagnostic = OSError("fixed diagnostic error")

    class Session:
        def prepare_warp_modules(self):
            raise original

    class BrokenModules(dict):
        def __getitem__(self, key):
            raise diagnostic

    session = Session()
    session.runtime_owners = []
    module = NS(execs={})
    python_module = NS(first_kernel=NS(module=module))
    session.modules = (BrokenModules() if diagnostic_failure == "inspect" else
                       {"optics.device_scene": (python_module, None, "completed")})
    writes = []

    def write(directory, name, record):
        if diagnostic_failure == "write":
            raise diagnostic
        writes.append((name, record))

    fault = "runtime_warp_modules_load_before_raise"
    namespace = dict(session=session, fault=fault, RUNTIME_WARP_COMPILE_FAULTS=(fault,),
                     warp_setup=NS(_DEFINITIONS=(("optics.device_scene", ("first_kernel",)),),
                                   _CACHES={session: NS(records=[])}, _MODULE_SETS={}),
                     binding=NS(target=NS(context=42)),
                     controller=NS(guardian=NS(contains_exact_owner=lambda owner: True,
                                               state="TRANSFERRED_TO_QUARANTINE")),
                     load_attempts=[], calls=[], directory=None, _write=write)
    with pytest.raises(RuntimeError) as caught:
        exec(compile(ast.Module(body=blocks, type_ignores=[]), str(path), "exec"), namespace)
    assert caught.value is original
    if diagnostic_failure:
        assert session.runtime_owners == [diagnostic] and not writes
    else:
        assert session.runtime_owners == [] and writes[0][0] == "warp_failure_roots.json"
