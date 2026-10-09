"""Readback and raw replay through the production warmup entry, with native doubles."""
import json
from types import SimpleNamespace as NS
import pytest

from tests.unit.optics.test_native_render_warmup import executable_graph, graph_session
from tests.unit.optics.test_prepared_fixed_render_inputs import fixed_inputs, fixed_storage_session


@pytest.mark.parametrize('fail_at', [None])
@pytest.mark.parametrize('diagnostic', [False, True])
def test_warmup_observer_readback_and_tamper_replay(executable_graph, diagnostic, monkeypatch, tmp_path):
    from optical_worker import native_setup
    from tests.support.native_runtime_process import _run_observed_render_warmups
    from tests.support.cuda_artifact_runtime_acceptance import _validate_render_warmups
    x = executable_graph
    for _ in range((13 if x.p.lane == 'warp' else 9) - x.initial_completes):
        with x.controller.segment('setup'):
            pass
    def state(pristine):
        x.controller._require_setup()
        inside = x.f.state.device is x.p.device
        value = {'primary_contexts': [{'ordinal': 0, 'active': False}, {'ordinal': 1, 'active': True}],
            'target_current_stream': x.f.stream.cuda_stream if x.f.state.stream is x.f.stream else 0,
            'held_stream': {'ordinal': 1, 'handle': x.f.stream.cuda_stream}}
        if x.p.lane == 'warp':
            value.update(default_device=int(inside), current_context=123 if inside else None,
                         current_context_device=1 if inside else None)
        else:
            value.update(logical_device=int(inside), raw_device=1)
        return value
    rgb = x.session.fixed_render_storage[3].rgb
    rgb.data[:] = [0.1, 0.2, 0.3]
    monkeypatch.setattr(type(rgb), 'numpy', lambda value: value.data.copy(), raising=False)
    monkeypatch.setattr(type(rgb), 'detach', lambda value: value, raising=False)
    monkeypatch.setattr(type(rgb), 'cpu', lambda value: NS(numpy=lambda: value.data.copy()), raising=False)
    request = {'catalog_sha256': x.session.catalog_sha256,
               'guard_configuration': {'session': x.controller._guard_client.configuration.session}}
    (tmp_path / 'render_graph_binding.json').write_text(json.dumps({
        'readiness_sha256': x.graph.readiness.sha256, 'compiled_modules_identity': id(x.graph.compiled_modules),
        'profile': x.graph.profile, 'shadows': x.graph.shadows, 'pass_mode': x.graph.pass_mode,
        'event_ids': [id(x.graph.completion_event)], 'frames': [{'bundle_id': id(bundle)} for _, bundle in x.graph.frames]}))
    (tmp_path / 'input_validation_report.json').write_text(json.dumps({'manifest_sha256': x.session.prepared_input.manifest.sha256}))
    original = native_setup._native_resource_scope
    result = _run_observed_render_warmups(x.session, x.controller,
        NS(module=x.p.runtime, target=x.p.device, stream=x.f.stream), NS(_state=state), tmp_path)
    assert result is x.session.fixed_render_warmups and native_setup._native_resource_scope is original
    assert (tmp_path / 'warmup_rgb.f32').read_bytes() == rgb.data.tobytes()
    fault = 'runtime_render_warmup_' + ('diagnostics' if diagnostic else 'success')
    def validate(trace=None):
        _validate_render_warmups(tmp_path, x.p.lane, 'asymmetric_synthetic_7x3_v1', fault,
            trace=x.controller._guard_events if trace is None else trace, request=request)
    validate()
    path = tmp_path / 'warmup_completion.json'
    original_data = path.read_bytes()
    for attack in ('count', 'wait', 'scope', 'guard', 'transaction', 'bundle'):
        data = json.loads(original_data)
        if attack == 'count': data['evidence']['completion_count'] = 9
        elif attack == 'wait': data['frames'][0]['leaves'][-1]['returned'] = False
        elif attack == 'scope': data['frames'][0]['leaves'][0]['state']['target_current_stream'] += 1
        elif attack == 'guard': data['evidence']['completed_guard_sequences'][0] += 1
        elif attack == 'transaction': data['evidence']['transaction_identities'][1] = data['evidence']['transaction_identities'][0]
        else: data['bundle_ids'][0] += 1
        path.write_text(json.dumps(data))
        with pytest.raises(RuntimeError): validate()
    path.write_bytes(original_data)
    trace = [row for row in x.controller._guard_events if row['operation'] != 'normal_disarmed']
    with pytest.raises(RuntimeError): validate(trace)
    raw = tmp_path / 'warmup_rgb.f32'
    raw.write_bytes(raw.read_bytes()[:-4])
    with pytest.raises(RuntimeError): validate()


@pytest.mark.parametrize('fail_at', [None])
@pytest.mark.parametrize('diagnostic', [False])
def test_failed_observation_preserves_original_error_and_restores_hook(executable_graph, monkeypatch, tmp_path):
    from optical_worker import native_setup
    from tests.support import native_runtime_process as driver
    x = executable_graph
    x.state.fail = 'launch'
    original = native_setup._native_resource_scope
    def broken_write(*args, **kwargs):
        raise OSError('optional diagnostic failed')
    monkeypatch.setattr(driver, '_write', broken_write)
    with pytest.raises(RuntimeError) as caught:
        driver._run_observed_render_warmups(x.session, x.controller,
            NS(module=x.p.runtime, target=x.p.device, stream=x.f.stream), NS(_state=lambda _: {}), tmp_path)
    assert caught.value is x.failure
    assert native_setup._native_resource_scope is original
    assert x.controller.guardian.state == 'TRANSFERRED_TO_QUARANTINE'
