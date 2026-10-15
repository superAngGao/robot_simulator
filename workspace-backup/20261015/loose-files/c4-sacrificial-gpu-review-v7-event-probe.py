"""Independent bounded probes; no optical execution and no candidate writes."""
import inspect
from pathlib import Path
import runpy
import subprocess
import sys

if "--child" not in sys.argv:
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--child"],
        capture_output=True, text=True, timeout=90,
    )
    print(result.stdout, end="")
    print(result.stderr, end="", file=sys.stderr)
    raise SystemExit(result.returncode)

root, mode = sys.argv[1:3]
sys.path.insert(0, root)
import warp as wp
import torch
from tests.support.sacrificial_leaf_audit import RecordedLeafAudit, RecordedLeafContamination

wp.init()
print("CANDIDATE_AUDIT", sys.modules[RecordedLeafAudit.__module__].__file__, flush=True)
print("RUNTIME", sys.version, torch.__version__, wp.__version__, flush=True)

if mode == "original_wait_closure":
    original_path = sys.argv[3]
    original_wait = torch.cuda.Event.wait
    original_launch = wp.Launch.launch
    calls = {"original_public_wait": 0, "original_launch": 0}
    state = {"active": False, "audit": None, "globals": None}

    def trace(frame, kind, value):
        if frame.f_code is RecordedLeafAudit.run.__code__:
            if kind == "call":
                state["active"] = True
                state["audit"] = frame.f_locals["self"]
            elif kind == "return":
                state["active"] = False
        if state["active"] and kind == "call":
            if frame.f_code is original_wait.__code__:
                calls["original_public_wait"] += 1
            if frame.f_code is original_launch.__code__:
                calls["original_launch"] += 1
            if frame.f_code.co_name == "contaminated":
                state["globals"] = frame.f_globals
        return trace

    sys.argv = [original_path, root, "torch_wait", "--child"]
    sys.settrace(trace)
    try:
        runpy.run_path(original_path, run_name="__original_v6_probe__")
    except RecordedLeafContamination as exc:
        assert str(exc) == "torch.cuda.Event.wait", str(exc)
    else:
        raise AssertionError("unchanged original wait operation did not fail closed")
    finally:
        sys.settrace(None)
        sys.setprofile(None)
    audit = state["audit"]
    assert audit is not None and state["globals"] is not None
    observed = state["globals"]["observed"]
    assert calls == {"original_public_wait": 0, "original_launch": 0}, calls
    assert observed["native_wait_enter"] == observed["native_wait_return"] == 0, observed
    assert audit.entry_calls == audit.restoration_checks == 1
    assert audit.native_launches == audit.native_events == audit.native_queries == 0
    assert audit.rejected == ["torch.cuda.Event.wait"]
    for _, owner, name, original in audit._resolved:
        assert inspect.getattr_static(owner, name) is original
    assert torch.cuda.Event.wait is original_wait and wp.Launch.launch is original_launch
    wp.synchronize_device(audit.stream.device)
    print("UNCHANGED_V6_OPERATION_CLOSED", calls, observed)
    print("EXACT_STATIC_DESCRIPTOR_RESTORATION", audit.evidence())
    raise SystemExit(0)

if mode == "original_query_closure":
    original_v6, original_v5 = sys.argv[3:5]
    source = Path(original_v6).read_text()
    diagnostic = "assert getattr(owner, name) is original"
    assert source.count(diagnostic) == 1
    source = source.replace(diagnostic, "assert __import__('inspect').getattr_static(owner, name) is original")
    sys.argv = [original_v6, root, "original_closure", original_v5, "--child"]
    # Only the restoration diagnostic adapts to raw descriptors. Original v5
    # operation remains runpy-executed unchanged by the v6 closure observer.
    exec(compile(source, original_v6, "exec"), {"__name__": "__diagnostic_v6__", "__file__": original_v6})
    raise AssertionError("original closure must exit")

assert mode in ("generic_query", "generic_wait", "generic_stream_query")

@wp.kernel
def independent_empty_kernel():
    pass

target = wp.get_device("cuda:0")
stream = wp.Stream(device=target)
sample = wp.empty(1, dtype=wp.int32, device=target)
event = wp.Event(device=target)
stream.record_event(event)
command = wp.launch(independent_empty_kernel, dim=1, device=target, stream=stream, record_cmd=True)
wp.synchronize_device(target)
generic_stream = torch.Stream(device="cuda:0")
generic_event = torch.Event(device="cuda:0")
generic_event.record(generic_stream)
generic_event.synchronize()
assert type(generic_event) is torch.Event and type(generic_stream) is torch.Stream
assert torch.Event is not torch.cuda.Event and torch.Stream is not torch.cuda.Stream
original_launch = wp.Launch.launch
observed = {"native_enter": 0, "native_return": 0}
operation_owner = generic_stream if mode == "generic_stream_query" else generic_event
operation_name = "wait" if mode == "generic_wait" else "query"
original_operation = inspect.getattr_static(type(operation_owner), operation_name)
mutability = None
try:
    setattr(type(operation_owner), operation_name, original_operation)
    mutability = "mutable"
except (TypeError, AttributeError) as exc:
    mutability = str(exc)

def profile(frame, kind, value):
    if kind in ("c_call", "c_return") and getattr(value, "__self__", None) is operation_owner:
        if getattr(value, "__name__", None) == operation_name:
            observed["native_enter" if kind == "c_call" else "native_return"] += 1

def contaminated(instance, stream=None):
    if mode == "generic_query":
        assert generic_event.query() is True
    elif mode == "generic_wait":
        generic_event.wait(generic_stream)
    else:
        assert generic_stream.query() is True
    return original_launch(instance, stream=stream)

wp.Launch.launch = contaminated
try:
    audit = RecordedLeafAudit(wp, commands=(command,), stream=stream,
                              sample_array=sample, event=event, fault="normal")
    sys.setprofile(profile)
    try:
        audit.run(command)
    finally:
        sys.setprofile(None)
    assert observed == {"native_enter": 1, "native_return": 1}, observed
    evidence = audit.evidence()
    assert evidence["native_launches"] == evidence["entry_calls"] == evidence["restoration_checks"] == 1
    assert evidence["native_events"] == evidence["native_queries"] == 0 and evidence["rejected"] == []
    for _, owner, name, original in audit._resolved:
        assert inspect.getattr_static(owner, name) is original
    assert inspect.getattr_static(type(operation_owner), operation_name) is original_operation
    print("PUBLIC_GENERIC_OPERATION_ESCAPED", mode, observed)
    print("EXACT_PUBLIC_TYPES", type(generic_event), type(generic_stream), "MUTABILITY", mutability)
    print("AUDIT_REPORTED_CLEAN", evidence)
finally:
    wp.Launch.launch = original_launch
    generic_stream.synchronize()
    wp.synchronize_device(target)
