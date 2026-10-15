"""Independent scratch audit probes; no optical bundle or candidate edits."""
import os
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
from tests.support.sacrificial_leaf_audit import RecordedLeafAudit, RecordedLeafContamination

wp.init()
core = wp._src.context.runtime.core
print("CANDIDATE_AUDIT", sys.modules[RecordedLeafAudit.__module__].__file__, flush=True)

if mode == "original_closure":
    original_path = sys.argv[3]
    originals = (wp.Event.is_complete, wp.Stream.is_complete, wp.Launch.launch,
                 core.wp_cuda_event_query, core.wp_cuda_stream_query, core.wp_cuda_event_elapsed_time)
    calls = {"property_body": 0, "launch_body": 0, "query_observer": 0}
    active = False
    audit = None

    def trace(frame, event, arg):
        global active, audit
        if frame.f_code is RecordedLeafAudit.run.__code__:
            if event == "call":
                active = True
                audit = frame.f_locals["self"]
            elif event == "return":
                active = False
        if active and event == "call":
            if frame.f_code is originals[0].fget.__code__:
                calls["property_body"] += 1
            if frame.f_code is originals[2].__code__:
                calls["launch_body"] += 1
            if frame.f_code.co_name == "observed_query":
                calls["query_observer"] += 1
        return trace

    sys.argv = [original_path, root, "--child"]
    sys.settrace(trace)
    try:
        runpy.run_path(original_path, run_name="__original_probe__")
    except RecordedLeafContamination as exc:
        assert str(exc) == "warp.Event.is_complete", str(exc)
    else:
        raise AssertionError("original probe did not fail closed")
    finally:
        sys.settrace(None)
    assert audit is not None
    assert calls == {"property_body": 0, "launch_body": 0, "query_observer": 0}, calls
    assert audit.entry_calls == audit.restoration_checks == 1
    assert audit.native_launches == audit.native_queries == audit.native_events == 0
    assert audit.rejected == ["warp.Event.is_complete"]
    restored = (wp.Event.is_complete, wp.Stream.is_complete, wp.Launch.launch,
                core.wp_cuda_event_query, core.wp_cuda_stream_query, core.wp_cuda_event_elapsed_time)
    assert all(a is b for a, b in zip(originals, restored, strict=True))
    for _, owner, name, original in audit._resolved:
        assert getattr(owner, name) is original
    wp.synchronize_device(audit.stream.device)
    print("ORIGINAL_UNCHANGED_BODY_FAILS_CLOSED", calls)
    print("DESCRIPTORS_CALLABLES_AND_NATIVE_ENTRIES_RESTORED", audit.evidence())
    raise SystemExit(0)


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
original_launch = wp.Launch.launch
originals = tuple((name, getattr(core, name)) for name in (
    "wp_cuda_event_query", "wp_cuda_stream_query", "wp_cuda_event_elapsed_time"))
native_bodies = []

if mode == "torch_wait":
    import torch
    torch_stream = torch.cuda.ExternalStream(stream.cuda_stream, device=0)
    torch_event = torch.cuda.Event()
    torch_event.record(torch_stream)
    torch_event.synchronize()
    observed = {"public_wait": 0, "native_wait_enter": 0, "native_wait_return": 0}

    def profile(frame, kind, value):
        if kind == "call" and frame.f_code is torch.cuda.Event.wait.__code__:
            observed["public_wait"] += 1
        if kind in ("c_call", "c_return") and getattr(value, "__self__", None) is torch_event:
            if getattr(value, "__name__", None) == "wait":
                observed["native_wait_enter" if kind == "c_call" else "native_wait_return"] += 1

    def contaminated(instance, stream=None):
        torch_event.wait(torch_stream)
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
        wp.synchronize_device(target)
        assert observed == {"public_wait": 1, "native_wait_enter": 1, "native_wait_return": 1}, observed
        evidence = audit.evidence()
        assert evidence["native_launches"] == evidence["entry_calls"] == evidence["restoration_checks"] == 1
        assert evidence["native_events"] == evidence["native_queries"] == 0 and evidence["rejected"] == []
        print("TORCH_PUBLIC_AND_NATIVE_EVENT_WAIT_ESCAPED", observed)
        print("AUDIT_REPORTED_CLEAN", evidence)
    finally:
        wp.Launch.launch = original_launch
        wp.synchronize_device(target)
else:
    assert mode == "native_queries"
    for name, original in originals:
        def forbidden_original(*args, _name=name, **kwargs):
            native_bodies.append(_name)
            raise AssertionError("native query body should be denied before entry")

        setattr(core, name, forbidden_original)
        def contaminated(instance, stream=None, _name=name):
            getattr(core, _name)(event.cuda_event)
            return original_launch(instance, stream=stream)

        wp.Launch.launch = contaminated
        try:
            audit = RecordedLeafAudit(wp, commands=(command,), stream=stream,
                                      sample_array=sample, event=event, fault="normal")
            try:
                audit.run(command)
            except RecordedLeafContamination as exc:
                assert str(exc) == f"warp.runtime.core.{name}"
            else:
                raise AssertionError("native query not rejected")
            assert audit.native_queries == audit.entry_calls == audit.restoration_checks == 1
            assert audit.native_events == audit.native_launches == 0
            assert not native_bodies
            assert getattr(core, name) is forbidden_original
            assert wp.Launch.launch is contaminated
            print("NATIVE_QUERY_REJECTED_BEFORE_ORIGINAL_AND_RESTORED", name)
        finally:
            wp.Launch.launch = original_launch
            setattr(core, name, original)
    wp.synchronize_device(target)
