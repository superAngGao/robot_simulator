"""Real-device audit probe; no optical render bundle or candidate file edits."""
import os
from pathlib import Path
import subprocess
import sys

if "--child" not in sys.argv:
    completed = subprocess.run([sys.executable, str(Path(__file__).resolve()), sys.argv[1], "--child"],
                               capture_output=True, text=True, timeout=90)
    print(completed.stdout, end="")
    print(completed.stderr, end="", file=sys.stderr)
    raise SystemExit(completed.returncode)

sys.path.insert(0, sys.argv[1])
import warp as wp
from tests.support.sacrificial_leaf_audit import RecordedLeafAudit


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
core = wp._src.context.runtime.core
original_query = core.wp_cuda_event_query
queries = []


def observed_query(handle):
    queries.append(handle)
    return original_query(handle)


def launch_with_forbidden_query(instance, stream=None):
    # The supported public property really enters the installed native query.
    assert event.is_complete
    return original_launch(instance, stream=stream)


wp.Launch.launch = launch_with_forbidden_query
core.wp_cuda_event_query = observed_query
try:
    audit = RecordedLeafAudit(wp, commands=(command,), stream=stream,
                              sample_array=sample, event=event, fault="normal")
    audit.run(command)
    wp.synchronize_device(target)
    evidence = audit.evidence()
    assert queries == [event.cuda_event]
    assert evidence["native_launches"] == evidence["entry_calls"] == evidence["restoration_checks"] == 1
    assert evidence["native_events"] == 0 and evidence["rejected"] == []
    print("CANDIDATE_AUDIT", sys.modules[RecordedLeafAudit.__module__].__file__)
    print("PUBLIC_EVENT_QUERY_ESCAPED", len(queries))
    print("AUDIT_REPORTED_CLEAN", evidence)
finally:
    wp.Launch.launch = original_launch
    core.wp_cuda_event_query = original_query
