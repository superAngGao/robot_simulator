"""Native-driver observation and replay use the production graph entry."""
import json
from types import SimpleNamespace as NS
import pytest

from tests.unit.optics.test_native_render_preparation import graph_session
from tests.unit.optics.test_prepared_fixed_render_inputs import fixed_inputs, fixed_storage_session


@pytest.mark.parametrize('fail_at', [None])
@pytest.mark.parametrize('diagnostic', [False, True])
def test_native_graph_observer_and_raw_replay(graph_session, diagnostic, monkeypatch, tmp_path):
    from optical_worker import native_setup, warp_setup
    from optics import warp_native_render
    from tests.support.native_runtime_process import _prepare_observed_render_graph
    from tests.support.cuda_artifact_runtime_acceptance import _validate_render_graph

    g = graph_session
    f, p, session, c = g.f, g.f.p, g.f.session, g.f.controller
    if p.lane == 'warp':
        compiled = warp_setup.validate_module_set(session, session.warp_modules)
        kernel = NS(module=compiled[2].module)
        def launch(actual, *, dim, inputs, device, stream, record_cmd, record_tape):
            assert actual is kernel and device is p.device and stream is f.stream
            assert record_cmd is True and record_tape is False
            return NS(device=device, kernel=kernel, module_exec=compiled[2].executable)
        p.runtime.launch = launch
        def seal(self, *, _context, **kwargs):
            with native_setup._warp_command_recording(p.runtime, p.device, f.stream, context=_context) as record:
                commands = tuple(record(kernel, dim=1, inputs=()) for _ in range(5))
            return NS(commands=commands, validate_resolved_parameters=lambda parameters: None)
        monkeypatch.setattr(warp_native_render.WarpDirectLightRenderCommandFactory, 'seal_camera', seal)
    # One storage guard was already completed by the real fixture entry.
    for _ in range(11 if p.lane == 'warp' else 7):
        with c.segment('setup'):
            pass
    def ticket():
        c._guard_client.require_live_ticket(c._guard_ticket, role='setup')
    def state(pristine):
        ticket()
        interior = f.state.device is p.device
        value = {'primary_contexts': [{'ordinal': 0, 'active': False}, {'ordinal': 1, 'active': True}],
                 'target_current_stream': f.stream.cuda_stream if f.state.stream is f.stream else 0,
                 'held_stream': {'ordinal': 1, 'handle': f.stream.cuda_stream}}
        if p.lane == 'cuda':
            value.update(logical_device=int(interior), raw_device=1)
        else:
            value.update(default_device=int(interior), current_context=123 if interior else None,
                         current_context_device=1 if interior else None)
        return value
    original_prepare = native_setup._prepare_fixed_render_graph
    original_adopt = c.adopt_returned_device_owner
    fault = 'runtime_render_graph_' + ('diagnostics' if diagnostic else 'success')
    binding = NS(module=p.runtime, target=p.device, stream=f.stream)
    result = _prepare_observed_render_graph(session, c, binding, NS(_ticket=ticket, _state=state), tmp_path, fault)
    assert result is session.fixed_render_graph and len(result.frames) == 10
    assert native_setup._prepare_fixed_render_graph is original_prepare
    assert c.adopt_returned_device_owner.__func__ is original_adopt.__func__
    assert c.adopt_returned_device_owner.__self__ is original_adopt.__self__
    _validate_render_graph(tmp_path, p.lane, fault, trace=c._guard_events)
    path = tmp_path / 'render_graph_binding.json'
    original = path.read_bytes()
    for attack in ('warmup', 'entry', 'event', 'scope', 'guard', 'frames'):
        data = json.loads(original)
        if attack == 'warmup':
            data['warmup_frames_completed'] = 10
        elif attack == 'entry':
            entry = data['frames'][0]['entries'][0]
            entry['expected_executable' if p.lane == 'warp' else 'expected_entry'] += 1
        elif attack == 'event':
            data['event_ids'][0] += 1
        elif attack == 'scope':
            data['adoptions'][0]['state']['primary_contexts'][0]['active'] = True
        elif attack == 'guard':
            data['guard']['sequence'] += 1
        else:
            data['frames'].pop()
        path.write_text(json.dumps(data), encoding='utf-8')
        with pytest.raises(RuntimeError):
            _validate_render_graph(tmp_path, p.lane, fault, trace=c._guard_events)
    path.write_bytes(original)
    trace = [row for row in c._guard_events if not (row['operation'] == 'descriptor_observed'
             and row['details']['descriptor']['leaf_kind'] == 'event_wait')]
    with pytest.raises(RuntimeError, match='initialized-event path'):
        _validate_render_graph(tmp_path, p.lane, fault, trace=trace)
