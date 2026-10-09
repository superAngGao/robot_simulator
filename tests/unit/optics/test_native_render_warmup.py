"""Production setup execution/guards with explicit native-call doubles.

These do not prove optical numerical output or native packed command execution.
"""
from dataclasses import replace
from types import SimpleNamespace as NS
import pytest

from tests.unit.optics.test_native_render_preparation import graph_session
from tests.unit.optics.test_prepared_fixed_render_inputs import fixed_inputs, fixed_storage_session


@pytest.fixture
def executable_graph(graph_session, monkeypatch, diagnostic, request):
    from optical_worker import warmup, warp_setup
    from optics import warp_native_render as render

    g = graph_session
    f, p, session, controller = g.f, g.f.p, g.f.session, g.f.controller
    session.catalog_sha256 = 'a' * 64
    calls = []
    state = NS(fail=None, armed=None)
    failure = RuntimeError('injected warmup leaf failure')

    def action(kind, value):
        controller._require_setup()
        assert f.state.device is p.device and f.state.stream is f.stream
        descriptor = next(row['details']['descriptor'] for row in reversed(controller._guard_events)
                          if row['operation'] == 'descriptor_observed')
        assert descriptor['leaf_kind'] == {'launch': 'warmup_launch', 'record': 'event_record',
                                            'wait': 'event_wait'}[kind]
        calls.append((kind, value, controller._guard_ticket))
        if state.fail == kind:
            raise failure

    if p.lane == 'warp':
        compiled = warp_setup.validate_module_set(session, session.warp_modules)
        class Bundle:
            def __init__(self, factory, camera, **kwargs):
                self.camera = camera
                self.binding = NS(factory=factory, traversal_diagnostics=kwargs['traversal_diagnostics'])
                ids = render._render_command_ids(ray_source='camera', num_rays=camera.query.num_rays,
                    shadows=kwargs['shadows'], diagnostics=kwargs['traversal_diagnostics'])
                self.descriptors = tuple(NS(entry_id=name) for name in ids)
                self.commands = tuple(NS(module_exec=compiled[2].executable,
                    kernel=NS(module=compiled[2].module), launch=lambda *, stream, name=name:
                    action('launch', name)) for name in ids)

            def validate_resolved_parameters(self, parameters):
                assert self.camera.query.frame_id == parameters.to_payload()['frame_id']

        monkeypatch.setattr(render, 'WarpDirectLightRenderCommandBundle', Bundle)
        factory_type = render.WarpDirectLightRenderCommandFactory
        monkeypatch.setattr(factory_type, 'device', p.device, raising=False)
        monkeypatch.setattr(factory_type, 'stream', f.stream, raising=False)
        def seal(factory, *, _context, **kwargs):
            return Bundle(factory, **kwargs)
        monkeypatch.setattr(factory_type, 'seal_camera', seal)
    else:
        module = session.cuda_factories[-1].module
        for name in vars(module):
            setattr(module, name, lambda *args, name=name: action('launch', name))

    profile, shadows = getattr(request, 'param', ('direct_light_full', True))
    graph = session.prepare_fixed_render_graph(profile=profile, shadows=shadows,
        pass_mode='diagnostic_profiled' if diagnostic else 'primary_unprofiled')
    event_type = p.runtime.Event
    monkeypatch.setattr(event_type, 'record', lambda event, stream: action('record', event))
    monkeypatch.setattr(event_type, 'synchronize', lambda event: action('wait', event))
    initial_completes = sum(row['operation'] == 'normal_disarmed' for row in controller._guard_events)
    return NS(g=g, f=f, p=p, session=session, controller=controller, graph=graph, calls=calls,
              state=state, failure=failure, initial_completes=initial_completes)


@pytest.mark.parametrize('fail_at', [None])
@pytest.mark.parametrize('diagnostic', [False, True])
@pytest.mark.parametrize('executable_graph', [(profile, shadows) for profile in
    ('direct_light_full', 'rgb_preview', 'render_only') for shadows in (False, True)], indirect=True)
def test_ten_actual_transactions_wait_and_disarm_before_reuse(executable_graph, diagnostic):
    from optical_worker import warmup
    x = executable_graph
    evidence = x.session.run_fixed_render_warmups()
    assert warmup.require_observed_warmups(x.session, evidence) is evidence
    assert evidence.prepared is x.session.prepared_input and evidence.readiness is x.graph.readiness
    assert evidence.readiness.warmup_frames_completed == 0
    assert len({id(tx) for tx in evidence.transactions}) == len({id(c) for c in evidence.completions}) == 10
    assert len(set(evidence.guards)) == 10
    assert all(tx.completion is c and c.event is x.graph.completion_event
               for tx, c in zip(evidence.transactions, evidence.completions, strict=True))
    assert evidence.to_payload()['completion_count'] == 10
    launches = 5 if x.p.lane == 'warp' else 3
    for index, ticket in enumerate(evidence.guards):
        assert any(row['operation'] == 'normal_disarmed' and row['guard']['sequence'] == ticket.sequence
                   for row in x.controller._guard_events)
        rows = [(kind, value) for kind, value, actual in x.calls if actual is ticket]
        assert sum(kind == 'launch' for kind, _ in rows) == launches
        assert sum(kind == 'record' for kind, _ in rows) == (7 if diagnostic else 1)
        assert rows[-2:] == [('record', x.graph.completion_event), ('wait', x.graph.completion_event)]
        if diagnostic and x.p.lane == 'warp':
            assert [kind for kind, _ in rows] == ['record','launch','record',
                'record','launch','launch','record','record','launch','launch','record','record','wait']
    assert sum(row['operation'] == 'normal_disarmed' for row in x.controller._guard_events) == x.initial_completes + 11
    assert x.f.state.device == 0 and x.f.state.stream is None
    before = len(x.calls)
    for transaction in evidence.transactions:
        with pytest.raises(ValueError, match='active frame'):
            warmup._require_warmup_transaction(transaction)
    with pytest.raises(ValueError, match='observed'):
        warmup.require_observed_warmups(x.session, replace(evidence))
    with pytest.raises(Exception, match='once'):
        x.session.run_fixed_render_warmups()
    assert len(x.calls) == before


@pytest.mark.parametrize('fail_at', [None])
@pytest.mark.parametrize('diagnostic', [False])
@pytest.mark.parametrize('attack', ['terminal', 'copied_transaction', 'copied_authority'])
def test_rebinding_or_copying_prepared_warmup_cannot_execute(executable_graph, monkeypatch, attack):
    from optical_worker import warmup
    x = executable_graph
    original = warmup._run_fixed_render_warmups
    def run(authority):
        transaction = warmup._SETUP_WARMUPS[x.session][1][0]
        if attack == 'copied_authority':
            return original(replace(authority))
        if attack == 'copied_transaction':
            copied = object.__new__(type(transaction))
            copied.__dict__.update(vars(transaction))
            with pytest.raises(ValueError, match='registered'):
                warmup._require_warmup_transaction(copied)
            raise x.failure
        if x.p.lane == 'warp':
            transaction._terminal = replace(transaction._terminal, event=object())
        else:
            transaction._completion_event = object()
        return original(authority)
    monkeypatch.setattr(warmup, '_run_fixed_render_warmups', run)
    with pytest.raises((ValueError, RuntimeError)):
        x.session.run_fixed_render_warmups()
    assert x.calls == []
    assert x.session not in warmup._COMPLETED_WARMUPS


@pytest.mark.parametrize('fail_at', [None])
@pytest.mark.parametrize('diagnostic', [False, True])
@pytest.mark.parametrize('failure_kind', ['launch', 'record', 'wait'])
def test_failed_leaf_never_credits_completion_or_advances(executable_graph, failure_kind):
    from optical_worker import warmup
    x = executable_graph
    x.state.fail = failure_kind
    with pytest.raises(RuntimeError) as caught:
        x.session.run_fixed_render_warmups()
    assert caught.value is x.failure
    assert x.session not in warmup._COMPLETED_WARMUPS
    assert warmup._SETUP_WARMUPS[x.session] is None
    assert len({ticket for _, _, ticket in x.calls}) == 1
    summary = x.controller.termination_evidence.lifecycle.to_payload()['summary']
    assert summary['normal_complete_count'] == x.initial_completes + 1
    assert summary['emergency_sync_attempt_count'] == 0
    assert x.controller.guardian.state == 'TRANSFERRED_TO_QUARANTINE'
    assert x.f.state.device == 0 and x.f.state.stream is None


@pytest.mark.parametrize('fail_at', [None])
@pytest.mark.parametrize('diagnostic', [False])
def test_disarmed_ack_failure_after_physical_wait_never_credits_frame(executable_graph, monkeypatch):
    from optical_worker import warmup
    x = executable_graph
    original = x.controller._guard_client.complete
    def complete(ticket):
        if any(kind == 'wait' for kind, _, _ in x.calls):
            raise x.failure
        return original(ticket)
    monkeypatch.setattr(x.controller._guard_client, 'complete', complete)
    with pytest.raises(RuntimeError) as caught:
        x.session.run_fixed_render_warmups()
    assert caught.value is x.failure
    assert x.calls[-1][0] == 'wait'
    assert x.session not in warmup._COMPLETED_WARMUPS
    assert len({ticket for _, _, ticket in x.calls}) == 1
    assert x.controller.guardian.state == 'TRANSFERRED_TO_QUARANTINE'


@pytest.mark.parametrize('lane', ['warp', 'cuda'])
@pytest.mark.parametrize('phase', ['fixed_render_graph_ready', 'warming_fixed_render_graph',
                                  'fixed_render_warmups_ready'])
def test_warmup_sources_are_admitted_only_in_their_phase_and_lane(lane, phase):
    from optical_worker.native_bootstrap import _SourceSession
    session = object.__new__(_SourceSession)
    session.lane, session.phase = lane, phase
    admitted = session.admitted_sources()
    assert ('optical_worker.warmup' in admitted) == (phase != 'fixed_render_graph_ready')
    assert ('optics.warp_timeline' in admitted) == (lane == 'warp' and phase != 'fixed_render_graph_ready')
    assert ('optics.cuda_timeline' in admitted) == (lane == 'cuda')
