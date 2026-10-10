"""Original RESOLVE admission with real worker/guard control and modeled getters."""
# Imported fixtures are registered by pytest.
# ruff: noqa: F401, F811

import ctypes
import sys
from copy import copy
from types import SimpleNamespace as NS

import pytest

from optical_worker.native_scope import NativeStateReader
from tests.unit.optics.test_integrated_worker_lifecycle import bound_worker, request, worker_graph
from tests.unit.optics.test_native_final_graph import final_sample_count, final_session
from tests.unit.optics.test_native_render_preparation import graph_session
from tests.unit.optics.test_native_render_warmup import executable_graph
from tests.unit.optics.test_prepared_fixed_render_inputs import fixed_inputs, fixed_storage_session

pytestmark = [pytest.mark.parametrize("fail_at", [None]), pytest.mark.parametrize("diagnostic", [False])]


@pytest.fixture
def resolving_reader(final_session, monkeypatch):
    x = final_session
    session, controller = x.session, x.controller
    original_phase = session.phase
    with controller.segment("setup"):
        session.phase, session._native_scope_record = "initializing_runtime", None
        class Runtime:
            @property
            def default_device(self):
                return x.p.device if x.f.state.device is x.p.device else NS(ordinal=0)

        runtime = x.p.runtime if x.p.lane == "cuda" else Runtime()
        if x.p.lane == "warp":
            controller.adopt_host_owner(
                runtime, role="modeled_query_runtime", operation_id="modeled_query_runtime",
            )
        reader = NativeStateReader(session, x.p.runtime, runtime, x.p.device)
        reader.stream = x.f.stream
        reader.binding = NS(session=session, controller=controller, module=x.p.runtime,
                            runtime=runtime, target=x.p.device, stream=x.f.stream)
        controller.adopt_host_owner(reader.binding, role="runtime_setup_binding", operation_id="binding")
        session._native_scope_record = (
            reader, controller, x.p.runtime, runtime, x.p.device, x.f.stream, reader.binding,
        )
        reader.ctypes, reader.count = ctypes, 8
        reader.cache_established, reader.target_context = True, 9000
        reader.current_stream_baseline = 0 if x.p.lane == "cuda" else x.f.stream.cuda_stream
        if x.p.lane == "cuda":
            monkeypatch.setitem(sys.modules, "torch.cuda", NS(_initialized=True))
            def raw_device(out):
                out._obj.value = 1
                return 0
            reader.raw_device = raw_device
            reader.logical = lambda: int(x.f.state.device is x.p.device)
            reader.primary = lambda ordinal: ordinal == 1
            reader.stream_query = lambda ordinal: (
                x.f.stream.cuda_stream if x.f.state.stream is x.f.stream else 0)
        else:
            reader.devices = tuple(range(8))
            def context_get(out):
                out._obj.value = 9000 if x.f.state.device is x.p.device else None
                return 0
            def context_device(out):
                out._obj.value = 1
                return 0
            def primary(ordinal, flags, active):
                flags._obj.value, active._obj.value = 0, int(ordinal == 1)
                return 0
            reader.context_get, reader.context_device, reader.primary = context_get, context_device, primary
            reader.stream_query = lambda context: x.f.stream.cuda_stream
        session.phase = original_phase
    x.reader = reader
    return x


@pytest.fixture
def observed_worker(resolving_reader, worker_graph):
    assert resolving_reader is worker_graph
    return worker_graph


def _pending(x):
    return request(x.worker, "EXECUTE_PREPARED_SAMPLE",
                   prepared_token=x.worker._samples[0].prepared_token)["payload"]["pending_token"]


def test_reader_observes_before_and_after_the_original_resolve_wait(observed_worker, monkeypatch):
    x = observed_worker
    wait = x.controller._wait_resolving_terminal
    observations = []

    def observed(event, runtime):
        assert not x.controller._resolution_wait_returned
        before = x.reader._state(False)
        ticket = x.controller._resolution_ticket
        wait(event, runtime)
        assert x.controller._require_resolution_reader(x.reader, after_wait=True) is x.worker._active_sample
        observations.append((ticket, before, x.reader._state(False)))
        assert x.worker._active_sample.state == "COMPLETION_RECORDED"
        assert x.worker._active_sample.active.state == "COMPLETION_RECORDED"

    monkeypatch.setattr(x.controller, "_wait_resolving_terminal", observed)
    pending = _pending(x)
    response = request(x.worker, "RESOLVE_PREPARED_SAMPLE", pending_token=pending)
    assert response["status"] == "ok" and len(observations) == 1
    ticket, before, after = observations[0]
    assert before == after and len(x.reader.raw) > 0
    x.controller._guard_client.require_completed_ticket(ticket)
    assert x.worker._samples[0].state == "RESOLVED_HELD"
    assert sum(kind == "wait" for kind, _ in x.sample_calls) == 1
    assert x.controller._resolution_ticket is x.controller._resolving_owner is None
    assert x.controller._resolving_event is None and not x.controller._resolution_wait_returned
    before = tuple(x.reader.raw)
    with pytest.raises(RuntimeError, match="armed segment"):
        x.reader._state(False)
    assert tuple(x.reader.raw) == before


@pytest.mark.parametrize("attack", ["reader", "ticket", "record", "graph", "phase", "event", "before_wait"])
def test_final_reader_rejects_changed_binding_before_any_getter(observed_worker, monkeypatch, attack):
    x = observed_worker
    pending = _pending(x)

    def observed(event, runtime):
        reader = x.reader
        if attack == "reader":
            reader = copy(reader)
        elif attack == "ticket":
            x.controller._guard_ticket = copy(x.controller._guard_ticket)
        elif attack == "record":
            x.worker._active_sample = copy(x.worker._active_sample)
        elif attack == "graph":
            x.worker._graph = copy(x.worker._graph)
        elif attack == "phase":
            x.session.phase = "fixed_final_graph_ready"
        elif attack == "event":
            x.controller._resolving_event = copy(event)
        else:
            x.controller._require_resolution_reader(reader, after_wait=True)
        reader._state(False)
        pytest.fail("changed binding reached a native getter")

    monkeypatch.setattr(x.controller, "_wait_resolving_terminal", observed)
    with pytest.raises((RuntimeError, ValueError)):
        request(x.worker, "RESOLVE_PREPARED_SAMPLE", pending_token=pending)
    assert x.reader.raw == []
    assert not any(kind == "wait" for kind, _ in x.sample_calls)
    assert x.controller.guardian.state == "TRANSFERRED_TO_QUARANTINE"
    assert x.controller._resolution_ticket is x.controller._resolving_owner is None


@pytest.mark.parametrize("failure", ["getter", "input_snapshot"])
def test_resolve_reader_failure_preserves_prefix_and_never_waits(observed_worker, monkeypatch, failure):
    x = observed_worker
    pending = _pending(x)
    injected = RuntimeError("final getter failed")

    def broken(*args):
        raise injected

    def observed(event, runtime):
        if failure == "input_snapshot":
            x.reader.snapshot("copy", "before", 0)
        else:
            monkeypatch.setattr(x.reader, "primary", broken)
            x.reader._state(False)

    monkeypatch.setattr(x.controller, "_wait_resolving_terminal", observed)
    with pytest.raises((RuntimeError, ValueError)) as caught:
        request(x.worker, "RESOLVE_PREPARED_SAMPLE", pending_token=pending)
    if failure == "getter":
        assert caught.value is injected and x.reader.raw
        assert any(error is injected for error in x.reader.held)
    else:
        assert not x.reader.raw
    assert not any(kind == "wait" for kind, _ in x.sample_calls)
    assert x.controller.unknown_device_scope
    assert x.controller.guardian.state == "TRANSFERRED_TO_QUARANTINE"
    prefix = tuple(x.reader.raw)
    with pytest.raises((RuntimeError, ValueError)):
        x.reader._state(False)
    assert tuple(x.reader.raw) == prefix
