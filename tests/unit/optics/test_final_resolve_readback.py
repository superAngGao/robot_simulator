"""Original worker RESOLVE and shared D2H bodies; only native leaves are modeled."""
# ruff: noqa: F401, F811

import os
from copy import copy, deepcopy
from types import SimpleNamespace as NS

import numpy as np
import pytest

from optical_worker.guarded_evidence import GuardedLifecycleEvidence
from optical_worker.native_readback import _copy_prepared_native_readback
from optical_worker.wait_protocol import GuardProtocolError
from tests.support.native_runtime_process import _prepared_worker_readback
from tests.unit.optics.test_final_resolve_reader import (
    bound_worker,
    executable_graph,
    final_sample_count,
    final_session,
    fixed_inputs,
    fixed_storage_session,
    graph_session,
    request,
    resolving_reader,
    worker_graph,
)

pytestmark = [pytest.mark.parametrize("fail_at", [None]), pytest.mark.parametrize("diagnostic", [False])]


@pytest.fixture
def readback_directory(request, tmp_path):
    return tmp_path if getattr(request, "param", False) else None


@pytest.fixture
def staged_readback(resolving_reader, monkeypatch, readback_directory):
    x = resolving_reader
    state = NS(calls=[], pending=[], fault=None, readback_event=None,
               failure=RuntimeError("injected final readback failure"))
    event_type = x.p.runtime.Event
    original_init = event_type.__init__
    original_record, original_wait = event_type.record, event_type.synchronize

    def final_leaf(kind):
        x.controller._require_resolution_reader(x.reader, after_wait=True)
        assert x.worker._active_sample.state == "COMPLETION_RECORDED"
        state.calls.append((kind, x.controller._guard_ticket))
        if state.fault == kind:
            raise state.failure

    class View:
        def __init__(self, host):
            self.host = host

        def copy_(self, source, *, non_blocking):
            assert non_blocking is True and x.f.state.device is x.p.device
            if x.p.lane == "cuda":
                assert x.f.state.stream is x.f.stream
            x.controller.require_adopted(self, self.host, source)
            state.pending.append((self.host, source.data))
            final_leaf("copy")

    def view(host, **kwargs):
        x.controller._require_setup()
        return View(host)

    def create(event, *, enable_timing=False, device=None):
        x.controller._require_setup()
        assert state.readback_event is None
        original_init(event, enable_timing=enable_timing, device=device)
        state.readback_event = event

    def record(event, stream):
        if event is not state.readback_event:
            return original_record(event, stream)
        assert stream is x.f.stream
        event.device = stream.device
        if x.controller.guardian.state == "SEALED":
            final_leaf("record")
            if state.fault == "foreign_record":
                event.device = NS(type="cuda", is_cuda=True, index=0, ordinal=0)
            elif state.fault == "unknown_record":
                event.device = None
        else:
            x.controller._require_setup()

    def wait(event):
        if event is not state.readback_event:
            return original_wait(event)
        if x.controller.guardian.state == "SEALED":
            final_leaf("wait")
            for host, source in state.pending:
                np.copyto(host, source)
            state.pending.clear()
        else:
            x.controller._require_setup()

    class Scope:
        def __init__(self, owner, field):
            self.owner, self.field = owner, field

        def __enter__(self):
            self.previous = getattr(x.f.state, self.field)
            setattr(x.f.state, self.field, self.owner)

        def __exit__(self, *_):
            setattr(x.f.state, self.field, self.previous)
            if state.fault == "exit" and x.controller._resolution_copy_started:
                raise state.failure

    monkeypatch.setattr(x.p.runtime.cuda, "device", lambda owner: Scope(owner, "device"))
    monkeypatch.setattr(x.p.runtime.cuda, "stream", lambda owner: Scope(owner, "stream"))
    monkeypatch.setattr(x.p.runtime, "from_numpy", view, raising=False)
    monkeypatch.setattr(x.p.runtime, "array", view)
    monkeypatch.setattr(x.p.runtime, "copy",
                        lambda dst, src, *, stream: dst.copy_(src, non_blocking=True), raising=False)
    monkeypatch.setattr(event_type, "__init__", create)
    monkeypatch.setattr(event_type, "record", record)
    monkeypatch.setattr(event_type, "synchronize", wait)
    if readback_directory is not None:
        # The shared fixture bypasses HELLO while exercising real native worker
        # control. These two descriptive wire identities are modeled here.
        x.controller._worker_session._worker_evidence = NS(
            process_id=os.getpid(), process_start_nonce="d" * 32,
        )
    controller_wait = x.controller._wait_resolving_terminal
    worker_resolve = x.controller._worker_session._resolve_native
    reader_fail = x.reader._fail
    with _prepared_worker_readback(x.session, directory=readback_directory) as held:
        x.readback_holder, x.readback_state = held, state
        x.readback_directory = readback_directory
        yield x
    if readback_directory is not None:
        assert x.controller._wait_resolving_terminal == controller_wait
        assert x.controller._worker_session._resolve_native == worker_resolve
        assert x.reader._fail == reader_fail


@pytest.fixture
def copying_worker(staged_readback, worker_graph):
    assert staged_readback is worker_graph
    return worker_graph


def _install_copy(x, monkeypatch, *, before_wait=False, duplicate=False):
    wait = x.controller._wait_resolving_terminal
    prepared = x.readback_holder["readback"]

    def observed(event, runtime):
        if not before_wait:
            wait(event, runtime)
        result = _copy_prepared_native_readback(prepared, x.controller, final_reader=x.reader)
        assert result is prepared.observations
        if duplicate:
            _copy_prepared_native_readback(prepared, x.controller, final_reader=x.reader)

    monkeypatch.setattr(x.controller, "_wait_resolving_terminal", observed)


@pytest.mark.parametrize("final_sample_count", [4], indirect=True)
def test_copy_wait_precedes_disarm_and_reuses_original_staging(copying_worker, monkeypatch):
    x = copying_worker
    prepared = x.readback_holder["readback"]
    _install_copy(x, monkeypatch)
    expected_hosts = tuple(id(host) for _, host in prepared.observations)
    complete = x.controller._guard_client.complete
    completed = []

    def disarm(ticket):
        assert x.readback_state.calls[-1] == ("wait", ticket)
        assert not x.readback_state.pending
        assert x.worker._active_sample.state == "COMPLETION_RECORDED"
        completed.append(ticket)
        return complete(ticket)

    monkeypatch.setattr(x.controller._guard_client, "complete", disarm)
    for generation, record in enumerate(x.worker._samples, 1):
        pending = request(
            x.worker, "EXECUTE_PREPARED_SAMPLE", prepared_token=record.prepared_token,
        )["payload"]
        # Model distinct device generations; this is not renderer or sentinel evidence.
        for _, _, source, _ in prepared.copies:
            source.data.fill(generation)
        resolved = request(x.worker, "RESOLVE_PREPARED_SAMPLE", **pending)["payload"]
        assert record.state == "RESOLVED_HELD"
        assert all(np.all(host == generation) for _, host in prepared.observations)
        assert tuple(id(host) for _, host in prepared.observations) == expected_hosts
        calls = x.readback_state.calls[-len(prepared.copies) - 2:]
        assert [kind for kind, _ in calls] == ["copy"] * len(prepared.copies) + ["record", "wait"]
        assert all(ticket is completed[-1] for _, ticket in calls)
        request(x.worker, "RELEASE_RESOLVED_SAMPLE", **resolved)
    assert len(completed) == 4 and len({id(ticket) for ticket in completed}) == 4
    assert sum(kind == "wait" for kind, _ in x.sample_calls) == 4
    assert x.final_graph.arena.lifecycle_evidence().slot_reclaim_count == 4
    GuardedLifecycleEvidence.from_payload(x.controller.evidence().to_payload())


@pytest.mark.parametrize("failure", [
    "before_wait", "copy", "record", "wait", "foreign_record", "unknown_record",
    "foreign_before", "duplicate", "unknown_before", "copied_prepared", "exit",
])
def test_readback_failure_keeps_generation_unresolved(copying_worker, monkeypatch, failure):
    x = copying_worker
    prepared = x.readback_holder["readback"]
    if failure == "copied_prepared":
        x.readback_holder["readback"] = copy(prepared)
    _install_copy(x, monkeypatch,
                  before_wait=failure == "before_wait", duplicate=failure == "duplicate")
    pending = request(x.worker, "EXECUTE_PREPARED_SAMPLE",
                      prepared_token=x.worker._samples[0].prepared_token)["payload"]
    x.readback_state.fault = failure
    if failure == "foreign_before":
        prepared.event.device = NS(type="cuda", is_cuda=True, index=0, ordinal=0)
    elif failure == "unknown_before":
        prepared.event.device = None
    elif failure == "exit" and x.p.lane == "warp":
        scope_type = type(prepared.device_scope)
        original_exit = scope_type.__exit__

        def failed_exit(scope, *args):
            result = original_exit(scope, *args)
            if scope is prepared.device_scope:
                raise x.readback_state.failure
            return result

        monkeypatch.setattr(scope_type, "__exit__", failed_exit)
    with pytest.raises((RuntimeError, ValueError, AttributeError)) as caught:
        request(x.worker, "RESOLVE_PREPARED_SAMPLE", **pending)
    if failure in {"copy", "record", "wait", "exit"}:
        assert caught.value is x.readback_state.failure
    assert x.controller.unknown_device_scope
    assert x.controller.guardian.state == "TRANSFERRED_TO_QUARANTINE"
    arena = x.final_graph.arena.lifecycle_evidence()
    assert arena.completion_resolution_count == arena.slot_reclaim_count == 0
    assert x.worker._samples[0].state == "COMPLETION_RECORDED"
    assert x.controller._resolution_ticket is None and x.controller._worker_resolution is None
    payload = x.controller.evidence().to_payload()
    GuardedLifecycleEvidence.from_payload(payload)
    assert payload["summary"]["emergency_sync_attempt_count"] == 0
    if failure.startswith("foreign"):
        assert x.controller.possible_dirty_device_ordinals == (0, 1)
        assert not any(kind == "wait" for kind, _ in x.readback_state.calls)
        forged = deepcopy(payload)
        fact = next(row for row in forged["transition_trace"] if row["operation"] == "final_scope_observed")
        fact["details"]["device_ordinals"] = [1]
        with pytest.raises(GuardProtocolError, match="contains the target"):
            GuardedLifecycleEvidence.from_payload(forged)
    if failure in {"before_wait", "foreign_before", "unknown_before", "copied_prepared"}:
        assert not x.readback_state.calls
    if failure == "before_wait":
        assert not any(kind == "wait" for kind, _ in x.sample_calls)
    before = tuple(x.readback_state.calls)
    with pytest.raises((RuntimeError, ValueError)):
        _copy_prepared_native_readback(prepared, x.controller, final_reader=x.reader)
    assert tuple(x.readback_state.calls) == before
