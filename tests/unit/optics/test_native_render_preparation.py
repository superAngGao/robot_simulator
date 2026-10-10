"""Real custody/guard/recording helpers with CPU native-return doubles."""
from types import SimpleNamespace as NS
import pytest
from optical_worker import native_setup as setup
from optical_worker.lifecycle import WorkerSetupProtocolError
from tests.unit.optics.test_native_output_allocation import allocation


@pytest.mark.parametrize('allocation', ['warp'], indirect=True)
@pytest.mark.parametrize('fault', [None, 'allocation', 'metadata', 'adoption'])
def test_recording_retains_each_return_and_caught_failure_cannot_continue(allocation, fault):
    p = allocation
    p.state.fault = fault
    p.runtime.launch = lambda kernel, *, dim, inputs, device, stream, record_tape, record_cmd: (
        p.runtime.empty((dim,), dtype='command', device=device))
    retained_record = []

    def run():
        with p.controller.segment():
            context = setup._create_upload_context(p.controller, p.runtime, p.target, p.stream)
            with setup._warp_command_recording(p.runtime, p.target, p.stream, context=context) as record:
                retained_record.append(record)
                record(object(), dim=1, inputs=())
                if fault:
                    with pytest.raises(RuntimeError) as caught:
                        record(object(), dim=1, inputs=())
                    assert caught.value is p.failure
                    with pytest.raises(WorkerSetupProtocolError):
                        record(object(), dim=1, inputs=())
                else:
                    record(object(), dim=1, inputs=())

    if fault:
        with pytest.raises(WorkerSetupProtocolError):
            run()
        summary = p.controller.termination_evidence.lifecycle.to_payload()['summary']
        assert summary['emergency_sync_attempt_count'] == summary['normal_complete_count'] == 0
    else:
        run()
        p.controller.require_adopted(*p.state.arrays)
    assert len(p.state.calls) == 2
    assert p.state.device == 0 and p.state.stream is None
    assert all(any(type(root) is list and any(value is array for value in root)
                   for root in p.controller.guardian._owners) for array in p.state.arrays)
    with pytest.raises(ValueError, match='original scope'):
        retained_record[0](object(), dim=1, inputs=())


@pytest.mark.parametrize('diagnostic', [False, True])
@pytest.mark.parametrize('fault', [None, 'create', 'adoption', 'record', 'wait'])
def test_events_are_retained_and_initialization_wait_covers_every_record(allocation, diagnostic, fault):
    p = allocation
    p.target.type, p.target.is_cuda = 'cuda', True
    p.state.fault = 'adoption' if fault == 'adoption' else None
    actions = []

    class Event:
        def __init__(self, *, enable_timing, device=None):
            assert p.state.device is p.target and p.state.stream is p.stream
            if fault == 'create' and len(p.state.arrays) == 0:
                raise p.failure
            self.device = None if p.lane == 'cuda' else device
            self.timing = enable_timing
            p.state.arrays.append(self)
            actions.append(('create', self))

        def record(self, stream):
            assert stream is p.stream and p.state.device is p.target
            p.controller.require_adopted(self)
            self.device = p.target
            actions.append(('record', self))
            if fault == 'record':
                raise p.failure

        def synchronize(self):
            assert actions[-1] == ('record', self)
            assert self.timing is False
            actions.append(('wait', self))
            if fault == 'wait':
                raise p.failure

    p.runtime.Event = p.runtime.cuda.Event = Event
    p.stream.record_event = lambda event: event.record(p.stream)
    p.runtime.synchronize_event = lambda event: event.synchronize()

    def run():
        with p.controller.segment():
            context = setup._create_upload_context(p.controller, p.runtime, p.target, p.stream)
            return setup._prepare_render_events(p.runtime, p.target, p.stream,
                                               lane=p.lane, diagnostic=diagnostic, context=context)

    triggered = fault is not None and (fault != 'adoption' or diagnostic)
    if triggered:
        with pytest.raises(RuntimeError) as caught:
            run()
        assert caught.value is p.failure
        summary = p.controller.termination_evidence.lifecycle.to_payload()['summary']
        assert summary['normal_complete_count'] == summary['emergency_sync_attempt_count'] == 0
        if fault != 'wait':
            assert not any(action == 'wait' for action, _ in actions)
    else:
        completion, pairs = run()
        assert len(pairs) == (3 if diagnostic else 0)
        assert len(p.state.arrays) == (7 if diagnostic else 1)
        assert [name for name, _, _ in pairs] == (
            ['raygen_kernel', 'first_hit_kernel', 'shade_kernel'] if diagnostic else [])
        assert actions[-2:] == [('record', completion), ('wait', completion)]
        assert sum(action == 'record' for action, _ in actions) == len(p.state.arrays)
    assert p.state.device == 0 and p.state.stream is None
    assert all(any(type(root) is list and any(value is event for value in root)
                   for root in p.controller.guardian._owners) for event in p.state.arrays)


from tests.unit.optics.test_prepared_fixed_render_inputs import fixed_inputs, fixed_storage_session


@pytest.mark.parametrize('phase', ['preparing_warp_modules', 'fixed_render_storage_ready',
                                  'preparing_fixed_render_graph', 'fixed_render_graph_ready'])
@pytest.mark.parametrize('name', ['optics.execution', 'optics.scene', 'optics.cuda_direct_light'])
def test_warp_graph_dependencies_resolve_through_captured_finder(phase, name):
    from pathlib import Path
    from optical_worker.native_bootstrap import _SourceSession, _CapturedSourceFinder, SourceBootstrapError
    session = object.__new__(_SourceSession)
    session.phase, session.lane = phase, 'warp'
    session.root = Path(__file__).resolve().parents[3]
    relative = name.replace('.', '/') + '.py'
    source = (session.root / relative).read_bytes()
    session.sources, session.loaders = {relative: (source, 'unused-digest')}, {}
    session.check_import_topology = lambda: None
    session.is_interpreter_name = lambda fullname: False
    session.check_live = lambda: None
    session.bound_controller = lambda: NS(_require_setup=lambda: None)
    failures = []
    session.fail = failures.append
    finder = _CapturedSourceFinder(session)
    allowed = (name == 'optics.scene' or
               phase in {'preparing_fixed_render_graph', 'fixed_render_graph_ready'} and name == 'optics.execution')
    if allowed:
        spec = finder._find_spec(name)
        assert spec.loader.source is source and spec.loader.relative == relative
        assert spec.loader is session.loaders[name] and not failures
    else:
        with pytest.raises(SourceBootstrapError, match='repository import outside') as caught:
            finder._find_spec(name)
        assert failures == [caught.value] and not session.loaders


@pytest.fixture
def graph_session(fixed_storage_session, monkeypatch, diagnostic):
    from optical_worker.native_input import NativeInputValidationReport
    from optics import cuda_direct_light as cuda, warp_native_render as warp
    from tests.unit.optics.test_guarded_native_readback import _CudaArray

    f = fixed_storage_session
    p, session, c = f.p, f.session, f.controller
    session.prepare_fixed_render_storage(traversal_diagnostics=diagnostic)
    manifest = p.prepared.manifest
    values = {name: getattr(manifest, name) for name in NativeInputValidationReport.__dataclass_fields__
              if name not in {'backend', 'manifest_sha256', 'validated_arrays'}}
    session.validation_report = NativeInputValidationReport(backend=p.lane, manifest_sha256=manifest.sha256,
        validated_arrays=tuple(row.name for row in manifest.arrays), **values)
    extension_set = object()
    session.cuda_extensions = lambda: extension_set
    events, actions = [], []

    class Event:
        def __init__(self, *, enable_timing, device=None):
            self.device = None if p.lane == 'cuda' else device
            self.cuda_event = id(self)
            events.append(self)

        def record(self, stream):
            c.require_adopted(self)
            assert stream is f.stream and f.state.device is p.device
            self.device = p.device
            actions.append(('record', self))

        def synchronize(self):
            actions.append(('wait', self))

    p.runtime.Event = p.runtime.cuda.Event = Event
    f.stream.record_event = lambda event: event.record(f.stream)
    p.runtime.synchronize_event = lambda event: event.synchronize()
    bundles = []
    if p.lane == 'cuda':
        monkeypatch.setattr(_CudaArray, 'is_cuda', True, raising=False)
        monkeypatch.setattr(_CudaArray, 'dim', lambda self: len(self.shape), raising=False)
        def forbidden(*args):
            pytest.fail('preparation executed an optical kernel')
        module = NS(**{name: forbidden for name in ('pinhole_camera_rays_out', 'first_hit_rays_out',
            'shade_direct_light_no_shadow_out', 'shade_direct_light_no_shadow_diagnostics_out',
            'shade_direct_light_with_shadows_out', 'shade_direct_light_with_shadows_diagnostics_out')})
        from optics import cuda_timeline
        monkeypatch.setattr(cuda_timeline, 'torch', p.runtime)
        context = object.__new__(cuda_timeline.CudaOpticalExecutionContext)
        context.device, context.stream, context.stream_handle = p.device, f.stream, f.stream.cuda_stream
        extension_set = object.__new__(setup.CudaExtensionSet)
        object.__setattr__(extension_set, 'session', session)
        binding = NS(session=session, target=p.device, stream=f.stream,
                     stream_identity=(1, f.stream.cuda_stream))
        monkeypatch.setitem(setup._CUDA_EXTENSION_SETS, session, (extension_set, (binding,)))
        session.cuda_extensions = lambda: extension_set
        def require_extension(owner, recipe):
            assert owner is extension_set and recipe == 'robot_sim_cuda_direct_light_v10'
            return module
        session.require_cuda_extension = require_extension
        factory = cuda.CudaDirectLightNativeRenderer(context=context, module=module,
                                                     _prepared_extension_token=cuda._PREPARED_EXTENSION_TOKEN)
        session.cuda_factories = (context, object(), object(), factory)
        for owner in (context, factory, session.cuda_factories):
            session._hold_runtime_owner(owner, 'fixture_cuda_factory')
    else:
        # The Warp native ABI is covered by its existing command tests and the
        # later native round. This double checks helper wiring and frame views.
        from optical_worker import warp_setup
        compiled = (object(), object(), NS(module=object(), executable=object()))
        monkeypatch.setattr(warp_setup, 'validate_module_set', lambda actual, owner: compiled)
        class Factory:
            @classmethod
            def prepare(cls, *, _context, **kwargs):
                with setup._native_resource_scope(p.runtime, p.device, lane='warp', stream=f.stream,
                        context=_context, owner_role='test_factory', scope_id='test_factory'):
                    return cls()

            def seal_camera(self, *, _context, **kwargs):
                from optics.execution import OpticalOutputProfile
                assert type(kwargs['output_profile']) is OpticalOutputProfile
                camera = kwargs['camera']
                assert kwargs['bvh'].frame_id == camera.query.frame_id
                assert kwargs['bvh'].bounds_min is p.prepared.arrays['bvh_bounds_min']
                bundle = NS(camera=camera, values=kwargs, commands=(NS(
                    module_exec=compiled[2].executable, kernel=NS(module=compiled[2].module)),))
                bundle.validate_resolved_parameters = lambda record: camera.query.frame_id == record.to_payload()['frame_id']
                with setup._native_resource_scope(p.runtime, p.device, lane='warp', stream=f.stream,
                        context=_context, owner_role='test_record', scope_id='test_record'):
                    bundles.append(bundle)
                return bundle
        monkeypatch.setattr(warp, 'WarpDirectLightRenderCommandFactory', Factory)
    return NS(f=f, events=events, actions=actions, bundles=bundles)


@pytest.mark.parametrize('fail_at', [None])
@pytest.mark.parametrize('diagnostic', [False, True])
@pytest.mark.parametrize('profile', ['direct_light_full', 'rgb_preview', 'render_only'])
@pytest.mark.parametrize('shadows', [False, True])
def test_provisional_graph_binds_ten_frames_without_render_or_final_readiness(graph_session, profile, shadows, diagnostic):
    from optical_worker.native_bootstrap import SourceBootstrapError
    g = graph_session
    f, session = g.f, g.f.session
    graph = session.prepare_fixed_render_graph(profile=profile, shadows=shadows,
        pass_mode='diagnostic_profiled' if diagnostic else 'primary_unprofiled')
    assert session.phase == 'fixed_render_graph_ready' and session.fixed_render_graph is graph
    assert graph.readiness.warmup_frames_completed == 0
    assert len(graph.frames) == 10 and len(graph.diagnostic_events) == (3 if diagnostic else 0)
    assert len(g.events) == (7 if diagnostic else 1)
    assert g.actions[-2:] == [('record', graph.completion_event), ('wait', graph.completion_event)]
    for index, (record, bundle) in enumerate(graph.frames):
        payload = record.to_payload()
        assert payload['frame_id'] == index + 1
        assert payload['profile'] == profile and payload['shadows'] is shadows
        if f.p.lane == 'cuda':
            bundle._cuda_transaction_stage_evidence().validate_render_parameters(payload)
        else:
            assert bundle.camera.query.frame_id == index + 1
        f.controller.require_adopted(bundle)
    f.controller.require_adopted(graph, graph.readiness)
    assert f.state.device == 0 and f.state.stream is None
    before = len(g.actions)
    with pytest.raises(SourceBootstrapError, match='once'):
        session.prepare_fixed_render_graph(profile=profile, shadows=shadows,
                                          pass_mode='primary_unprofiled')
    assert len(g.actions) == before


@pytest.mark.parametrize('fixed_inputs', ['cuda'], indirect=True)
@pytest.mark.parametrize('fail_at', [None])
@pytest.mark.parametrize('diagnostic', [False])
@pytest.mark.parametrize('attack', ['module', 'context', 'factory', 'factory_tuple', 'stream', 'handle', 'device'])
def test_cuda_graph_rejects_rebound_factory_before_binding(graph_session, monkeypatch, attack):
    from optics import cuda_direct_light as cuda
    g = graph_session
    session, controller = g.f.session, g.f.controller
    context, scene, lbvh, factory = session.cuda_factories
    if attack == 'module':
        factory.module = NS(**vars(factory.module))
    elif attack == 'context':
        factory.context = NS(device=context.device, stream=context.stream)
    elif attack == 'factory':
        replacement = cuda.CudaDirectLightNativeRenderer(context=context, module=factory.module,
            _prepared_extension_token=cuda._PREPARED_EXTENSION_TOKEN)
        session.cuda_factories = (context, scene, lbvh, replacement)
    elif attack == 'factory_tuple':
        session.cuda_factories = tuple(list(session.cuda_factories))
    elif attack == 'stream':
        context.stream = NS(device=context.device, cuda_stream=context.stream_handle)
    elif attack == 'handle':
        context.stream_handle += 1
    else:
        context.device = object()
    def forbidden(*args, **kwargs):
        pytest.fail('foreign factory reached frame binding')
    monkeypatch.setattr(cuda.CudaDirectLightNativeRenderer, 'bind_direct_light', forbidden)
    with pytest.raises((ValueError, RuntimeError)) as caught:
        session.prepare_fixed_render_graph(profile='direct_light_full', shadows=True,
                                          pass_mode='primary_unprofiled')
    assert session.failures[-1] is caught.value
    assert not g.events and not g.actions
    assert controller.guardian.state == 'TRANSFERRED_TO_QUARANTINE'
    assert controller.termination_evidence.lifecycle.to_payload()['summary']['emergency_sync_attempt_count'] == 0
    assert controller.guardian.contains_exact_owner(factory)
    assert g.f.state.device == 0 and g.f.state.stream is None


@pytest.mark.parametrize('fixed_inputs', ['cuda'], indirect=True)
@pytest.mark.parametrize('fail_at', [None])
@pytest.mark.parametrize('diagnostic', [False])
@pytest.mark.parametrize('attack', ['entry', 'renderer', 'context'])
def test_cuda_graph_checks_returned_stages_before_events(graph_session, monkeypatch, attack):
    from optics import cuda_direct_light as cuda
    g = graph_session
    session = g.f.session
    original = cuda.CudaDirectLightNativeRenderer.bind_direct_light
    returned = []
    def bind(self, **kwargs):
        if not returned and attack == 'entry':
            module = self.module
            self.module = NS(**{name: (lambda *args: None) for name in vars(module)})
            try:
                stage = original(self, **kwargs)
            finally:
                self.module = module
        elif not returned and attack == 'renderer':
            other = cuda.CudaDirectLightNativeRenderer(context=self.context, module=self.module,
                _prepared_extension_token=cuda._PREPARED_EXTENSION_TOKEN)
            stage = original(other, **kwargs)
        else:
            stage = original(self, **kwargs)
        returned.append(stage)
        if len(returned) == 10 and attack == 'context':
            self.context.stream = NS(device=self.context.device, cuda_stream=self.context.stream_handle)
        return stage
    monkeypatch.setattr(cuda.CudaDirectLightNativeRenderer, 'bind_direct_light', bind)
    with pytest.raises(ValueError) as caught:
        session.prepare_fixed_render_graph(profile='direct_light_full', shadows=True,
                                          pass_mode='primary_unprofiled')
    assert session.failures[-1] is caught.value
    assert len(returned) == 10 and not g.events
    assert all(g.f.controller.guardian.contains_exact_owner(stage) for stage in returned)
    assert g.f.controller.guardian.state == 'TRANSFERRED_TO_QUARANTINE'


@pytest.mark.parametrize('fixed_inputs', ['cuda'], indirect=True)
@pytest.mark.parametrize('fail_at', [None])
@pytest.mark.parametrize('diagnostic', [False, True])
@pytest.mark.parametrize('drift', [None, 'pointer', 'shape', 'field', 'camera', 'abi', 'copy'])
def test_render_stability_uses_prepared_bindings_and_detects_storage_drift(
    graph_session, monkeypatch, diagnostic, drift,
):
    from copy import copy

    from optics import cuda_direct_light as cuda

    graph = graph_session.f.session.prepare_fixed_render_graph(profile='direct_light_full', shadows=True,
        pass_mode='diagnostic_profiled' if diagnostic else 'primary_unprofiled')
    stage = graph.frames[0][1]

    def late_signature(*args):
        pytest.fail('render execution rebuilt its binding signature')

    monkeypatch.setattr(cuda, '_render_binding_signature', late_signature)
    if drift == 'pointer':
        stage.buffers.origins.storage.data = stage.buffers.origins.storage.data.copy()
    elif drift == 'shape':
        stage.buffers.origins.shape = (1, 3)
    elif drift == 'field':
        object.__setattr__(stage.buffers, 'origins', stage.buffers.directions)
    elif drift == 'camera':
        object.__setattr__(stage.camera, 'width', stage.camera.width + 1)
    elif drift == 'abi':
        object.__setattr__(stage, '_shade_args', tuple(list(stage._shade_args)))
    elif drift == 'copy':
        stage = copy(stage)
    if drift is None:
        stage._assert_cuda_transaction_stage_stable(stage.renderer.context)
    else:
        with pytest.raises(ValueError, match='changed'):
            stage._assert_cuda_transaction_stage_stable(stage.renderer.context)


@pytest.mark.parametrize('fixed_inputs', ['cuda'], indirect=True)
@pytest.mark.parametrize('fail_at', [None])
@pytest.mark.parametrize('diagnostic', [False])
def test_first_hit_executes_the_prepared_entry_without_late_lookup_or_argument_build(
    graph_session, monkeypatch,
):
    g = graph_session
    renderer = g.f.session.cuda_factories[-1]
    calls = []
    for name in vars(renderer.module):
        monkeypatch.setattr(renderer.module, name, lambda *args: None)
    monkeypatch.setattr(renderer.module, 'first_hit_rays_out', lambda *args: calls.append(args))
    graph = g.f.session.prepare_fixed_render_graph(profile='direct_light_full', shadows=True,
                                                  pass_mode='primary_unprofiled')
    stage = graph.frames[0][1]
    transaction = renderer.context.begin_transaction(stages=(stage,), completion_event=graph.completion_event)

    def late_lookup(*args):
        pytest.fail('first-hit execution looked up the module entry or rebuilt its arguments')

    monkeypatch.setattr(renderer.module, 'first_hit_rays_out', late_lookup)
    monkeypatch.setattr(type(stage), '_bind_first_hit_arguments', late_lookup, raising=False)
    stage.enqueue(transaction)
    assert len(calls) == 1 and len(calls[0]) == 35
    assert all(actual is expected for actual, expected in zip(calls[0], stage._first_hit_args, strict=True))


@pytest.mark.parametrize('fixed_inputs', ['cuda'], indirect=True)
@pytest.mark.parametrize('fail_at', [None])
@pytest.mark.parametrize('diagnostic', [False])
@pytest.mark.parametrize('attack', ['entry', 'tensor', 'scalar', 'scalar_type', 'length', 'mutable'])
def test_first_hit_preflight_rejects_changed_fixed_abi(graph_session, attack):
    g = graph_session
    graph = g.f.session.prepare_fixed_render_graph(profile='direct_light_full', shadows=True,
                                                  pass_mode='primary_unprofiled')
    stage = graph.frames[0][1]
    if attack == 'entry':
        object.__setattr__(stage, '_first_hit_entry', lambda *args: None)
    else:
        args = list(stage._first_hit_args)
        if attack == 'tensor':
            args[0] = args[1]
        elif attack == 'scalar':
            args[2] += 1.0
        elif attack == 'scalar_type':
            args[9] = float(args[9])
        elif attack == 'length':
            args.pop()
        object.__setattr__(stage, '_first_hit_args', args if attack == 'mutable' else tuple(args))
    with pytest.raises(ValueError, match='first_hit_rays_out|first-hit fixed ABI'):
        stage._validate_cuda_transaction_stage(stage.renderer.context)
    with pytest.raises(ValueError, match='binding changed'):
        stage._assert_cuda_transaction_stage_stable(stage.renderer.context)


@pytest.mark.parametrize('allocation', ['warp'], indirect=True)
@pytest.mark.parametrize('failed', [False, True])
def test_original_render_factory_is_retained_before_public_load(allocation, monkeypatch, failed):
    from optics import warp_native_render as render
    p = allocation
    monkeypatch.setattr(render, 'wp', p.runtime)
    # Isolate the original prepare method's native-return custody; factory ABI
    # validation is independently covered by recording tests on the GPU host.
    monkeypatch.setattr(render.WarpDirectLightRenderCommandFactory, '__post_init__', lambda self: None)
    factories = []

    def load(**kwargs):
        assert kwargs == dict(module=render.__name__, device=p.target, recursive=False, block_dim=256, max_workers=0)
        assert p.state.device is p.target and p.state.stream is p.stream
        held = [owner for owner in p.controller.guardian._owners
                if type(owner) is render.WarpDirectLightRenderCommandFactory]
        assert len(held) == 1
        factories.extend(held)
        if failed:
            raise p.failure

    p.runtime.load_module = load
    def run():
        with p.controller.segment():
            context = setup._create_upload_context(p.controller, p.runtime, p.target, p.stream)
            return render.WarpDirectLightRenderCommandFactory.prepare(prepared_input=object(),
                readiness=object(), local_scene=object(), device=p.target, stream=p.stream, _context=context)

    if failed:
        with pytest.raises(RuntimeError) as caught:
            run()
        assert caught.value is p.failure
        assert p.controller.guardian.state == 'TRANSFERRED_TO_QUARANTINE'
    else:
        assert run() is factories[0]
    assert len(factories) == 1 and p.state.device == 0 and p.state.stream is None


@pytest.mark.parametrize('fail_at', [None])
@pytest.mark.parametrize('diagnostic', [False])
@pytest.mark.parametrize('attack', ['diagnostics', 'source', 'second_frame', 'event_wait'])
def test_graph_preparation_failures_retain_partial_graph(graph_session, monkeypatch, attack):
    from optical_worker import render_parameters
    from optics import warp_native_render
    g, error = graph_session, RuntimeError('fixed provisional preparation failure')
    f, session = g.f, g.f.session
    kwargs = dict(profile='direct_light_full', shadows=True, pass_mode='primary_unprofiled')
    if attack == 'diagnostics':
        kwargs['pass_mode'] = 'diagnostic_profiled'
    elif attack == 'source':
        session.render_source = NS(packed_scene=object(), sequence=session.render_source.sequence)
    elif attack == 'second_frame':
        original = render_parameters.bind_render_parameters
        def bind(*args, **kwargs):
            if kwargs['sample_index'] == 1:
                raise error
            return original(*args, **kwargs)
        monkeypatch.setattr(render_parameters, 'bind_render_parameters', bind)
    else:
        def fail(*args):
            raise error
        if f.p.lane == 'warp':
            f.p.runtime.synchronize_event = fail
        else:
            monkeypatch.setattr(f.p.runtime.Event, 'synchronize', fail)
    with pytest.raises((ValueError, RuntimeError)) as caught:
        session.prepare_fixed_render_graph(**kwargs)
    if attack in {'second_frame', 'event_wait'}:
        assert caught.value is error
        frames = [owner for owner in session.runtime_owners
                  if type(owner) is list and owner and type(owner[0]) is tuple
                  and len(owner[0]) == 2 and hasattr(owner[0][0], 'canonical_bytes')]
        assert len(frames) == 1 and len(frames[0]) == (1 if attack == 'second_frame' else 10)
        for _, bundle in frames[0]:
            assert f.controller.guardian.contains_exact_owner(bundle)
    else:
        assert g.events == [] and g.bundles == []
    assert not hasattr(session, 'fixed_render_graph')
    assert f.controller.guardian.state == 'TRANSFERRED_TO_QUARANTINE'
    summary = f.controller.termination_evidence.lifecycle.to_payload()['summary']
    assert summary['normal_complete_count'] == 1 and summary['emergency_sync_attempt_count'] == 0
    assert f.state.device == 0 and f.state.stream is None


@pytest.mark.parametrize('allocation', ['warp'], indirect=True)
@pytest.mark.parametrize('fault', [None, 'allocation'])
def test_original_factory_seal_uses_guarded_recording_and_retains_partial_commands(allocation, monkeypatch, fault):
    from optics import warp_native_render as render
    p = allocation
    p.state.fault = fault
    monkeypatch.setattr(render, 'wp', p.runtime)
    monkeypatch.setattr(render.WarpDirectLightRenderCommandFactory, 'validate_bound', lambda self: None)
    descriptors = tuple(NS(kernel=object(), dim=1, arguments=()) for _ in range(3))
    monkeypatch.setattr(render, '_WarpRenderBinding', lambda **kwargs: NS(command_descriptors=lambda: descriptors))
    monkeypatch.setattr(render, 'WarpDirectLightRenderCommandBundle', lambda **kwargs: NS(**kwargs))
    p.runtime.launch = lambda kernel, *, dim, inputs, device, stream, record_tape, record_cmd: (
        p.runtime.empty((dim,), dtype='command', device=device))
    factory = object.__new__(render.WarpDirectLightRenderCommandFactory)
    object.__setattr__(factory, 'device', p.target)
    object.__setattr__(factory, 'stream', p.stream)

    def run():
        with p.controller.segment():
            context = setup._create_upload_context(p.controller, p.runtime, p.target, p.stream)
            return factory._seal(_context=context)

    if fault:
        with pytest.raises(RuntimeError) as caught:
            run()
        assert caught.value is p.failure and len(p.state.calls) == 2
        assert p.controller.guardian.state == 'TRANSFERRED_TO_QUARANTINE'
    else:
        bundle = run()
        assert len(bundle.commands) == 3 and bundle.descriptors is descriptors
    assert all(any(type(root) is list and any(value is command for value in root)
                   for root in p.controller.guardian._owners) for command in p.state.arrays)


def test_foreign_event_is_quarantined_before_record_or_wait(allocation):
    p = allocation
    p.target.type, p.target.is_cuda = 'cuda', True
    foreign = NS(type='cuda', index=0, ordinal=0, is_cuda=True)
    operations = []
    p.runtime.Event = p.runtime.cuda.Event = lambda **kwargs: NS(device=foreign)
    p.stream.record_event = lambda event: operations.append('record')
    p.runtime.synchronize_event = lambda event: operations.append('wait')
    with pytest.raises(WorkerSetupProtocolError):
        with p.controller.segment():
            context = setup._create_upload_context(p.controller, p.runtime, p.target, p.stream)
            setup._prepare_render_events(p.runtime, p.target, p.stream, lane=p.lane,
                                         diagnostic=False, context=context)
    assert operations == []
    assert p.controller.guardian.state == 'TRANSFERRED_TO_QUARANTINE'
    summary = p.controller.termination_evidence.lifecycle.to_payload()['summary']
    assert summary['emergency_sync_attempt_count'] == 0


@pytest.mark.parametrize('allocation', ['cuda'], indirect=True)
@pytest.mark.parametrize('entry', ['arena_r0', 'arena_r1', 'compatibility'])
@pytest.mark.parametrize('diagnostic', [False, True])
def test_standalone_event_builders_wait_after_all_initialization(allocation, monkeypatch, entry, diagnostic):
    import sys
    from optics import cuda_arena, cuda_direct_light

    p = allocation
    actions = []

    class Event:
        def __init__(self, *, enable_timing):
            assert p.state.device is p.target and p.state.stream is p.stream
            self.device, self.timing = None, enable_timing
            actions.append(('create', self))

        def record(self, stream):
            assert stream is p.stream and p.state.device is p.target
            self.device = p.target
            actions.append(('record', self))

        def synchronize(self):
            assert not self.timing
            assert actions[-1] == ('record', self)
            created = [event for kind, event in actions if kind == 'create']
            recorded = [event for kind, event in actions if kind == 'record']
            assert len(created) == len(recorded) and set(created) == set(recorded)
            actions.append(('wait', self))

    p.runtime.cuda.Event = Event
    monkeypatch.setitem(sys.modules, 'torch', p.runtime)
    monkeypatch.setattr(cuda_direct_light, 'torch', p.runtime)
    context = NS(device=p.target, stream=p.stream, _validate_identity=lambda: None)
    if entry == 'compatibility':
        names = ('raygen_kernel',) if diagnostic else ()
        completion, pairs = cuda_direct_light._prepare_compatibility_events(context, names)
    else:
        boundary = 'R0' if entry == 'arena_r0' else 'R1'
        names = setup._render_phase_names(boundary=boundary, diagnostic=diagnostic)
        completion, pairs = cuda_arena._prepare_events(context, boundary=boundary, diagnostic=diagnostic)
    assert tuple(pairs) == names
    assert actions[-2:] == [('record', completion), ('wait', completion)]
    assert p.state.device == 0 and p.state.stream is None
