"""Final correctness staging through original graph/seal entries, with CPU leaves."""
# Imported fixtures are registered by pytest.
# ruff: noqa: F401, F811

from types import SimpleNamespace as NS

import pytest

from tests.unit.optics.test_native_final_graph import final_sample_count, final_session
from tests.unit.optics.test_native_render_preparation import graph_session
from tests.unit.optics.test_native_render_warmup import executable_graph
from tests.unit.optics.test_prepared_fixed_render_inputs import fixed_inputs, fixed_storage_session

pytestmark = [pytest.mark.parametrize("fail_at", [None]),
              pytest.mark.parametrize("diagnostic", [False, True])]


@pytest.mark.parametrize("failure", [None, "record", "wait"])
def test_final_staging_is_initialized_and_retained_before_original_seal(
    final_session, final_sample_count, monkeypatch, failure,
):
    from optical_worker import native_setup
    from optical_worker.native_readback import _copy_prepared_native_readback
    from tests.support.native_runtime_process import _prepared_worker_readback

    x = final_session
    calls, barriers = [], []
    original_prepare = native_setup._prepare_fixed_final_graph
    injected = RuntimeError("final staging initialization failed")

    def view(host, **kwargs):
        x.controller.require_adopted(host)
        return NS(host=host)

    event_type = x.p.runtime.Event
    original_init = event_type.__init__

    def create(event, *, enable_timing=False, device=None):
        x.controller._require_setup()
        assert x.controller.guardian.state == "CONSTRUCTING"
        original_init(event, enable_timing=enable_timing, device=device)
        calls.append(("create", event))

    def record(event, stream):
        x.controller.require_adopted(event, stream)
        assert stream is x.f.stream and x.controller.guardian.state == "CONSTRUCTING"
        event.device = stream.device
        calls.append(("record", event))
        if failure == "record":
            raise injected

    def synchronize(event):
        x.controller.require_adopted(event)
        calls.append(("wait", event))
        if failure == "wait":
            raise injected

    monkeypatch.setattr(x.p.runtime, "from_numpy", view, raising=False)
    monkeypatch.setattr(x.p.runtime, "array", view)
    monkeypatch.setattr(event_type, "__init__", create)
    monkeypatch.setattr(event_type, "record", record)
    monkeypatch.setattr(event_type, "synchronize", synchronize)

    def barrier(target):
        assert target is x.p.device and x.controller.guardian.state == "SEALED"
        assert [kind for kind, _ in calls] == ["create", "record", "wait"]
        prepared = held["readback"]
        x.controller.require_adopted(prepared, prepared.event)
        assert prepared.event is calls[-1][1]
        barriers.append(True)

    x.p.runtime.cuda.synchronize = x.p.runtime.synchronize_device = barrier
    with _prepared_worker_readback(x.session) as held:
        if failure is not None:
            with pytest.raises(RuntimeError) as caught:
                x.session.prepare_fixed_final_graph(sample_count=final_sample_count)
            assert caught.value is injected
        else:
            graph = x.session.prepare_fixed_final_graph(sample_count=final_sample_count)
            assert held["graph"] is graph and x.controller._final_admission is graph
    assert native_setup._prepare_fixed_final_graph is original_prepare
    assert any(owner is held for owner in x.session.runtime_owners)
    assert any(owner is calls[0][1] for owner in held["retained"])
    if failure is not None:
        assert not barriers and held["readback"] is None
        assert x.controller.guardian.state == "TRANSFERRED_TO_QUARANTINE"
        assert x.controller._final_admission is None
    else:
        assert barriers == [True]
        prepared = held["readback"]
        buffers = x.session.fixed_render_storage[3]
        physical = dict(buffers.arrays if x.p.lane == "warp" else buffers.tensor_items())
        assert len(prepared.observations) == len(physical)
        assert {name: id(array) for name, _, array, _ in prepared.copies} == {
            name: id(array) for name, array in physical.items()
        }
        for name, host in prepared.observations:
            assert host.shape == physical[name].shape and host.dtype == physical[name].dtype
        # Staging is not final resolve authority. An unguarded post-seal call
        # cannot exploit the preparation helper to copy or synchronize.
        before = tuple(calls)
        with pytest.raises(RuntimeError):
            _copy_prepared_native_readback(prepared, x.controller)
        assert tuple(calls) == before
