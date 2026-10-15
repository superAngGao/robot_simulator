"""Independent review probes, outside the frozen candidate."""
import inspect
import json
from pathlib import Path
import runpy
import subprocess
import sys

if "--child" not in sys.argv:
    result = subprocess.run([sys.executable, __file__, *sys.argv[1:], "--child"],
                            capture_output=True, text=True, timeout=90)
    print(result.stdout, end="")
    print(result.stderr, end="", file=sys.stderr)
    raise SystemExit(result.returncode)

root, mode = sys.argv[1:3]
sys.path.insert(0, root)
import warp as wp
import torch
from tests.support.sacrificial_leaf_audit import RecordedLeafAudit, RecordedLeafContamination

def monitor_state():
    return tuple((sys.monitoring.get_tool(i), sys.monitoring.get_events(i)) for i in range(6))

before = monitor_state()
print("RUNTIME", sys.version.split()[0], torch.__version__, wp.__version__)
print("MODULE", inspect.getfile(RecordedLeafAudit))

if mode.startswith("original_"):
    original_path = sys.argv[3]
    original_mode = mode.removeprefix("original_")
    captured = {}
    original_launch = wp.Launch.launch
    entered = []
    def trace(frame, event, arg):
        if frame.f_code is RecordedLeafAudit.run.__code__ and event == "call":
            captured["audit"] = frame.f_locals["self"]
        if frame.f_code.co_name == "contaminated" and event == "call":
            captured["globals"] = frame.f_globals
        if frame.f_code is original_launch.__code__ and event == "call":
            entered.append("original_launch")
        return trace
    sys.argv = [original_path, root, original_mode, "--child"]
    sys.settrace(trace)
    try:
        runpy.run_path(original_path, run_name="__original_v7__")
    except RecordedLeafContamination as exc:
        expected = "torch.Stream.query" if original_mode == "generic_stream_query" else (
            "torch.Event.wait" if original_mode == "generic_wait" else "torch.Event.query")
        assert str(exc) == expected, str(exc)
    else:
        raise AssertionError("original generic method escaped")
    finally:
        sys.settrace(None)
        sys.setprofile(None)
    audit = captured["audit"]
    scope = captured["globals"]
    # CPython's legacy profiler receives c_call before another monitoring tool
    # can refuse CALL. This is dispatch notification, not C-body entry proof.
    assert scope["observed"] == {"native_enter": 1, "native_return": 0}, scope["observed"]
    assert entered == [], entered
    assert audit.rejected == [expected]
    assert audit.native_launches == audit.native_queries == audit.native_events == 0
    assert audit.entry_calls == audit.restoration_checks == 1
    assert audit.native_event_guard.calls == audit.native_event_guard.restoration_checks == 1
    for _, owner, name, descriptor in audit._resolved:
        assert inspect.getattr_static(owner, name) is descriptor
    assert inspect.getattr_static(type(scope["operation_owner"]), scope["operation_name"]) is scope["original_operation"]
    assert wp.Launch.launch is original_launch
    assert monitor_state() == before
    # Independently calibrate the retained original profiler after restoration.
    sys.setprofile(scope["profile"])
    try:
        if original_mode == "generic_wait":
            scope["generic_event"].wait(scope["generic_stream"])
        else:
            assert scope["operation_owner"].query() is True
    finally:
        sys.setprofile(None)
    assert scope["observed"] == {"native_enter": 2, "native_return": 1}
    scope["generic_stream"].synchronize()
    print("UNCHANGED_ORIGINAL_REFUSED", original_mode, "C dispatch 1/return 0; restored calibration 1/1; Warp body 0; native body verified separately by GDB")
    raise SystemExit(0)

assert mode == "aliases"
wp.init()

@wp.kernel
def empty_kernel():
    pass

target = wp.get_device("cuda:0")
stream = wp.Stream(device=target)
sample = wp.empty(1, dtype=wp.int32, device=target)
event = wp.Event(device=target)
command = wp.launch(empty_kernel, dim=1, device=target, stream=stream, record_cmd=True)
native_stream = torch.Stream(device="cuda:0")
native_event = torch.Event(device="cuda:0")
native_event.record(native_stream)
native_event.synchronize()
wp.synchronize_device(target)
cached_query = native_event.query
cached_wait = native_event.wait
cached_stream_query = native_stream.query
event_ctor = torch.Event
stream_ctor = torch.Stream
unbound_query = torch.Event.query
unbound_wait = torch.Event.wait
unbound_stream_query = torch.Stream.query
operations = (
    ("cached_query", "torch.Event.query", lambda: cached_query()),
    ("unbound_query", "torch.Event.query", lambda: unbound_query(native_event)),
    ("cached_wait", "torch.Event.wait", lambda: cached_wait(native_stream)),
    ("unbound_wait", "torch.Event.wait", lambda: unbound_wait(native_event, native_stream)),
    ("cached_stream_query", "torch.Stream.query", lambda: cached_stream_query()),
    ("unbound_stream_query", "torch.Stream.query", lambda: unbound_stream_query(native_stream)),
    ("cached_event_ctor", "torch.Event", lambda: event_ctor(device="cuda:0")),
    ("cached_stream_ctor", "torch.Stream", lambda: stream_ctor(device="cuda:0")),
)
original_launch = wp.Launch.launch
for label, expected, operation in operations:
    calls = []
    def profile(frame, kind, value):
        if kind in ("c_call", "c_return"):
            receiver = getattr(value, "__self__", None)
            if receiver is native_event or receiver is native_stream or value in (unbound_query, unbound_wait, unbound_stream_query):
                calls.append((kind, getattr(value, "__name__", "?")))
    def contaminated(instance, stream=None):
        operation()
        raise AssertionError("forbidden call returned")
    wp.Launch.launch = contaminated
    try:
        audit = RecordedLeafAudit(wp, commands=(command,), stream=stream, sample_array=sample, event=event, fault="normal")
        sys.setprofile(profile)
        try:
            audit.run(command)
        except RecordedLeafContamination as exc:
            assert str(exc) == expected, (label, str(exc))
        else:
            raise AssertionError(label)
        finally:
            sys.setprofile(None)
        assert not any(kind == "c_return" for kind, _ in calls), (label, calls)
        assert audit.native_launches == 0 and audit.rejected == [expected]
        assert audit.native_event_guard.calls == audit.native_event_guard.restoration_checks == 1
        assert audit.restoration_checks == audit.entry_calls == 1
        for _, owner, name, descriptor in audit._resolved:
            assert inspect.getattr_static(owner, name) is descriptor
        assert monitor_state() == before
        print("ALIAS_REJECTED_AND_RESTORED", label)
    finally:
        wp.Launch.launch = original_launch
native_stream.synchronize()
wp.synchronize_device(target)
# Positive exact native command and reusable monitor state after every refusal.
audit = RecordedLeafAudit(wp, commands=(command,), stream=stream, sample_array=sample, event=event, fault="normal")
audit.run(command)
assert audit.native_launches == audit.entry_calls == audit.restoration_checks == 1
assert audit.native_event_guard.calls == 0
assert monitor_state() == before
wp.synchronize_device(target)
print("RESTORED_NORMAL_RECORDED_LAUNCH_PASS")
