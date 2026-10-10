"""Recorded Warp phase execution shared by setup and arena transactions.

Authority adapters are separate; command/reset/event ordering has one owner.
Product dispatch remains closed until full worker integration is accepted.
"""

from dataclasses import dataclass


@dataclass(frozen=True, eq=False)
class WarpOpticalCompletion:
    event: object
    transaction: object
    bundle: object


@dataclass(frozen=True, eq=False, init=False, slots=True, repr=False)
class _WarpRecordedTransaction:
    """One ordered render and one terminal record; no waits or kernel copies."""

    bundle: object
    parameters: object
    runtime: object
    target: object
    stream: object
    _stream_handle: int
    _groups: tuple
    _next_phase: int
    _diagnostics: tuple
    _terminal: WarpOpticalCompletion
    _state: str

    def __init__(self, *, bundle, parameters, runtime, target, stream, completion_event, diagnostic_events):
        from .warp_native_render import WarpDirectLightRenderCommandBundle

        if hasattr(self, "_state"):
            raise RuntimeError("Warp setup transaction cannot be initialized again")
        if type(bundle) is not WarpDirectLightRenderCommandBundle:
            raise TypeError("Warp setup requires an exact recorded render bundle")
        bundle.validate_resolved_parameters(parameters)
        factory = bundle.binding.factory
        if factory.device is not target or factory.stream is not stream:
            raise ValueError("Warp setup transaction differs from its recorded scope")
        phases = ("raygen_kernel", "first_hit_kernel", "shade_kernel")
        expected = phases if bundle.binding.traversal_diagnostics else ()
        if tuple(name for name, _, _ in diagnostic_events) != expected:
            raise ValueError("Warp setup diagnostics differ from the recorded variant")
        events = (completion_event, *(event for _, start, end in diagnostic_events for event in (start, end)))
        if len({id(event) for event in events}) != len(events):
            raise ValueError("Warp setup events must be distinct")
        if any(type(event) is not runtime.Event or event.device is not target for event in events):
            raise ValueError("Warp setup events differ from the target device")
        # Derive phase boundaries once from the already validated fixed ABI.
        # Counter resets belong inside first-hit and shade, including zero rays.
        groups = [[], [], []]
        phase = 0
        for descriptor, command in zip(bundle.descriptors, bundle.commands, strict=True):
            if descriptor.entry_id == "_reset_bvh_primary_counters_kernel":
                phase = 1
            elif descriptor.entry_id in {
                "_reset_shadow_primary_counters_kernel",
                "_reset_shadow_diagnostic_counters_kernel",
            }:
                phase = 2
            groups[phase].append(command)
        for name, value in (
            ("bundle", bundle),
            ("parameters", parameters),
            ("runtime", runtime),
            ("target", target),
            ("stream", stream),
            ("_stream_handle", stream.cuda_stream),
            ("_groups", tuple(tuple(group) for group in groups)),
            ("_next_phase", 0),
            ("_diagnostics", diagnostic_events),
            ("_terminal", WarpOpticalCompletion(completion_event, self, bundle)),
            ("_state", "validated_open"),
        ):
            object.__setattr__(self, name, value)

    @property
    def state(self):
        return self._state

    @property
    def completion(self):
        if self._state != "finished":
            raise RuntimeError("Warp terminal record is unavailable before finish")
        return self._terminal

    def enqueue(self):
        self._require_execution()
        if self._state != "validated_open":
            if self._state != "finished":
                object.__setattr__(self, "_state", "poisoned")
            raise RuntimeError("Warp setup transaction cannot enqueue again")
        try:
            if self.stream.device is not self.target or self.stream.cuda_stream != self._stream_handle:
                raise ValueError("Warp setup stream identity changed")
            # Setup validates packed parameters at entry; the final adapter
            # uses the already sealed C3 graph and its exact live lease.
            self._validate_before_enqueue()
            object.__setattr__(self, "_state", "enqueuing")
            for index, commands in enumerate(self._groups):
                pair = self._diagnostics[index] if self._diagnostics else None
                if pair is not None:
                    self._record_event(pair[1], "warmup_phase_start")
                for command in commands:
                    self._launch(command)
                if pair is not None:
                    self._record_event(pair[2], "warmup_phase_end")
                object.__setattr__(self, "_next_phase", self._next_phase + 1)
        except BaseException:
            object.__setattr__(self, "_state", "poisoned")
            raise

    def finish(self):
        self._require_execution()
        if self._state != "enqueuing" or self._next_phase != len(self._groups):
            if self._state != "finished":
                object.__setattr__(self, "_state", "poisoned")
            raise RuntimeError("Warp setup finish requires one complete enqueue")
        try:
            self._record_event(self._terminal.event, "warmup_terminal_record")
        except BaseException:
            object.__setattr__(self, "_state", "poisoned")
            raise
        object.__setattr__(self, "_state", "finished")
        return self._terminal

    def _require_execution(self):
        raise RuntimeError("recorded commands require a registered execution authority")

    def _validate_before_enqueue(self):
        self.bundle.validate_resolved_parameters(self.parameters)

    def _launch(self, command):
        raise RuntimeError("recorded commands have no execution authority")

    def _record_event(self, event, operation):
        raise RuntimeError("recorded events have no execution authority")


@dataclass(frozen=True, eq=False, init=False, slots=True, repr=False)
class _WarpSetupTransaction(_WarpRecordedTransaction):
    """The existing guarded, setup-only authority over shared phase execution."""

    def _require_execution(self):
        from optical_worker.warmup import _require_warmup_transaction

        _require_warmup_transaction(self)

    def _launch(self, command):
        from optical_worker.warmup import _warmup_leaf

        with _warmup_leaf(self, "warmup_recorded_launch", "warmup_launch"):
            command.launch(stream=self.stream)

    def _record_event(self, event, operation):
        from optical_worker.warmup import _warmup_leaf

        with _warmup_leaf(self, operation, "event_record"):
            self.stream.record_event(event)


@dataclass(frozen=True, eq=False, init=False, slots=True, repr=False)
class _WarpArenaTransaction(_WarpRecordedTransaction):
    """Dormant recorded execution; only an exact live arena sample can enter."""

    _arena: object

    def __init__(self, *, arena, **kwargs):
        from .warp_arena import WarpNativeOpticalArena

        if type(arena) is not WarpNativeOpticalArena:
            raise TypeError("Warp transaction requires its exact arena")
        _WarpRecordedTransaction.__init__(self, **kwargs)
        object.__setattr__(self, "_arena", arena)

    def _require_execution(self):
        self._arena._require_active_transaction(self)

    def _validate_before_enqueue(self):
        # Full packed-command, storage and scalar validation is performed during
        # untimed preparation. Live identity/state checks stay fixed-cost.
        self._require_execution()

    def _launch(self, command):
        self._require_execution()
        command.launch(stream=self.stream)

    def _record_event(self, event, operation):
        self._require_execution()
        if type(event) is not self.runtime.Event or event.device is not self.target:
            raise ValueError("Warp terminal/diagnostic event changed target")
        self.stream.record_event(event)
