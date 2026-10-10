"""Ten observed setup completions, using the existing guard and native kernels.

No caller count, parsed record or returned object grants execution authority.
The source session alone owns the live registration and its one active guard.
"""
from contextlib import contextmanager
from dataclasses import dataclass
from .native_setup import SETUP_WARMUP_FRAME_COUNT


_SETUP_WARMUPS = {}
_TRANSACTION_WARMUPS = {}
_COMPLETED_WARMUPS = {}


@dataclass(slots=True)
class _SetupWarmupRecord:
    authority: object
    transactions: tuple
    completions: list
    tickets: list
    active_frame_index: int | None = None
    active_ticket: object = None
    active_operation: object = None


@dataclass(frozen=True, eq=False)
class _SetupWarmupExecutionAuthority:
    session: object
    controller: object
    guardian: object
    graph: object
    runtime: object
    target: object
    stream: object
    stream_handle: int


@dataclass(frozen=True, eq=False)
class WarmupCompletionEvidence:
    """Local observations plus their actual owners; construction is not issuance."""

    session: object
    prepared: object
    readiness: object
    compiled_modules: object
    graph: object
    transactions: tuple
    completions: tuple
    guards: tuple
    source_catalog_sha256: str

    def to_payload(self):
        return {
            "schema_version": "q54.p12.4.setup-warmup-completion.v1",
            "session": self.guards[0].session,
            "target_device_ordinal": self.guards[0].device_ordinal,
            "prepared_manifest_sha256": self.prepared.manifest.sha256,
            "provisional_readiness_sha256": self.readiness.sha256,
            "source_catalog_sha256": self.source_catalog_sha256,
            "compiled_modules_identity": id(self.compiled_modules),
            "profile": self.graph.profile,
            "shadows": self.graph.shadows,
            "pass_mode": self.graph.pass_mode,
            "frame_indices": list(range(len(self.transactions))),
            "transaction_identities": [id(value) for value in self.transactions],
            "terminal_identities": [id(value) for value in self.completions],
            "terminal_event_identities": [id(value.event) for value in self.completions],
            "completed_guard_sequences": [ticket.sequence for ticket in self.guards],
            "completion_count": len(self.completions),
        }


def _require_warmup_transaction(transaction):
    entry = _TRANSACTION_WARMUPS.get(transaction)
    if entry is None:
        raise ValueError("setup transaction has no registered warmup authority")
    authority, index = entry
    record = _SETUP_WARMUPS.get(authority.session)
    if (record is None or record.authority is not authority or record.active_frame_index is None
            or record.transactions[index] is not transaction or record.active_frame_index != index):
        raise ValueError("setup transaction is outside its exact active frame")
    session, controller = authority.session, authority.controller
    session.check_live()
    controller._require_setup()
    if (session.phase != "warming_fixed_render_graph" or session.bound_controller() is not controller
            or controller.guardian is not authority.guardian or authority.guardian.state != "CONSTRUCTING"
            or session.fixed_render_graph is not authority.graph
            or controller._guard_ticket is not record.active_ticket
            or controller.unknown_device_scope
            or controller.possible_dirty_device_ordinals != (controller.target_device_ordinal,)
            or authority.stream.cuda_stream != authority.stream_handle
            or authority.stream.device != authority.target):
        raise ValueError("setup warmup lost its original session, guard or exact target scope")
    graph, bundle = authority.graph, authority.graph.frames[index][1]
    if session.lane == "warp":
        if (transaction.bundle is not bundle or transaction.stream is not authority.stream
                or transaction.target is not authority.target or transaction.runtime is not authority.runtime
                or transaction._terminal.event is not graph.completion_event
                or transaction._diagnostics is not graph.diagnostic_events):
            raise ValueError("Warp setup transaction binding changed")
    elif (transaction._context is not graph.factory.context or transaction._stages != (bundle,)
            or transaction._completion_event is not graph.completion_event
            or transaction._diagnostic_events != graph.diagnostic_events):
        raise ValueError("CUDA setup transaction binding changed")
    controller.require_adopted(authority, transaction, authority.graph)
    return authority


@contextmanager
def _warmup_leaf(transaction, name, kind):
    authority = _require_warmup_transaction(transaction)
    operation = _SETUP_WARMUPS[authority.session].active_operation
    with operation(name, kind):
        yield


def _prepare_cuda_transaction(bundle, context, graph):
    from optics.cuda_timeline import _CudaSetupTransaction

    template = context.prepare_transaction_template(stages=(bundle,), completion_event=graph.completion_event,
        diagnostic_events={name: (start, end) for name, start, end in graph.diagnostic_events})
    return _CudaSetupTransaction(_template=template)


def _prepare_fixed_render_warmups(session, runtime, target, stream):
    from .native_setup import (_FixedRenderPreparation, _create_upload_context,
                               _native_resource_scope, _require_cuda_render_factory)

    controller = session.bound_controller()
    graph = session.fixed_render_graph
    if (type(graph) is not _FixedRenderPreparation or len(graph.frames) != SETUP_WARMUP_FRAME_COUNT
            or graph.readiness.warmup_frames_completed != 0 or session in _SETUP_WARMUPS):
        raise ValueError("warmup requires the original ten-frame provisional graph, once")
    transactions, completions, tickets = [], [], []
    session._hold_runtime_owner(transactions, "setup_warmup_transactions")
    session._hold_runtime_owner(completions, "setup_warmup_completions")
    session._hold_runtime_owner(tickets, "setup_warmup_guards")
    # Reserve one-shot issuance before any preparation can fail.
    _SETUP_WARMUPS[session] = None
    prepared = session.prepared_input
    controller.require_adopted(graph, graph.readiness, prepared)
    session.audit()
    session.external.audit()
    context = _create_upload_context(controller, runtime, target, stream)
    with _native_resource_scope(runtime, target, lane=session.lane, stream=stream, context=context,
            owner_role="setup_warmup_custody", scope_id="setup_warmup_prepare_scope"):
        if session.lane == "warp":
            from . import warp_setup
            from optics.warp_timeline import _WarpSetupTransaction

            warp_setup.validate_module_set(session, graph.compiled_modules)
        else:
            factory, cuda_context = _require_cuda_render_factory(session, graph.compiled_modules)
            if graph.factory is not factory:
                raise ValueError("warmup CUDA factory differs from its provisional graph")
        authority = _SetupWarmupExecutionAuthority(session, controller, controller.guardian, graph,
                                                   runtime, target, stream, stream.cuda_stream)
        session._hold_runtime_owner(authority, "setup_warmup_authority")
        for parameters, bundle in graph.frames:
            if session.lane == "warp":
                transaction = _WarpSetupTransaction(bundle=bundle, parameters=parameters, runtime=runtime,
                    target=target, stream=stream, completion_event=graph.completion_event,
                    diagnostic_events=graph.diagnostic_events)
            else:
                if bundle.renderer is not factory:
                    raise ValueError("warmup CUDA stage differs from its issued renderer")
                bundle._cuda_transaction_stage_evidence().validate_render_parameters(parameters.to_payload())
                transaction = _prepare_cuda_transaction(bundle, cuda_context, graph)
            transactions.append(transaction)
            session._hold_runtime_owner(transaction, "setup_warmup_transaction")
        _SETUP_WARMUPS[session] = _SetupWarmupRecord(authority, tuple(transactions), completions, tickets)
        for index, transaction in enumerate(transactions):
            _TRANSACTION_WARMUPS[transaction] = (authority, index)
    return authority


def _run_fixed_render_warmups(authority):
    from .native_setup import _create_upload_context, _native_resource_scope, _require_cuda_render_factory

    session, controller, graph = authority.session, authority.controller, authority.graph
    record = _SETUP_WARMUPS.get(session)
    if record is None or record.authority is not authority:
        raise ValueError("warmup driver requires its issued preparation")
    transactions, completions, tickets = record.transactions, record.completions, record.tickets
    runtime, target, stream = authority.runtime, authority.target, authority.stream
    prepared = session.prepared_input

    try:
        for index, transaction in enumerate(transactions):
            with controller.segment("setup"):
                ticket = controller._guard_ticket
                context = _create_upload_context(controller, runtime, target, stream)
                with _native_resource_scope(runtime, target, lane=session.lane, stream=stream, context=context,
                        owner_role="setup_warmup_frame", scope_id="setup_warmup_frame_scope") as (_, _, operation):
                    record.active_frame_index, record.active_ticket, record.active_operation = index, ticket, operation
                    try:
                        if session.lane == "warp":
                            from . import warp_setup
                            warp_setup.validate_module_set(session, graph.compiled_modules)
                            transaction.enqueue()
                        else:
                            _require_cuda_render_factory(session, graph.compiled_modules)
                            graph.frames[index][1].enqueue(transaction)
                        completion = transaction.finish()
                        if transaction.completion is not completion or completion.event is not graph.completion_event:
                            raise ValueError("warmup terminal differs from its actual transaction")
                        # Retain the terminal object before the potentially
                        # blocking host wait, including the failure path.
                        session._hold_runtime_owner(completion, "setup_warmup_terminal")
                        with _warmup_leaf(transaction, "warmup_terminal_wait", "event_wait"):
                            if session.lane == "warp":
                                runtime.synchronize_event(completion.event)
                            else:
                                completion.event.synchronize()
                        if index == SETUP_WARMUP_FRAME_COUNT - 1:
                            session.audit()
                            session.external.audit()
                    finally:
                        record.active_frame_index = record.active_ticket = record.active_operation = None
            # The wait returning is insufficient: COMPLETE/DISARMED must also
            # have succeeded on this exact ticket before the next event reuse.
            controller._guard_client.require_completed_ticket(ticket)
            completions.append(completion)
            tickets.append(ticket)
        evidence = WarmupCompletionEvidence(session, prepared, graph.readiness, graph.compiled_modules,
            graph, tuple(transactions), tuple(completions), tuple(tickets), session.catalog_sha256)
        session._hold_runtime_owner(evidence, "observed_setup_warmups")
        _COMPLETED_WARMUPS[session] = evidence
        return evidence
    finally:
        # A completed, failed or escaped transaction never retains live setup
        # authority. Owners stay in the existing guardian until process exit.
        _SETUP_WARMUPS[session] = None


def require_observed_warmups(session, evidence):
    """Validate local issuance before future final-readiness construction."""
    if (type(evidence) is not WarmupCompletionEvidence or _COMPLETED_WARMUPS.get(session) is not evidence
            or evidence.session is not session or evidence.prepared is not session.prepared_input
            or evidence.graph is not session.fixed_render_graph or session.phase != "fixed_render_warmups_ready"
            or len(evidence.completions) != SETUP_WARMUP_FRAME_COUNT):
        raise ValueError("warmup completion was not observed by this exact session")
    session.check_live()
    return evidence
