"""Fixed C1 recorded-leaf observation; neither product authority nor a sandbox."""

from __future__ import annotations

from contextlib import ExitStack
from unittest.mock import patch

from benchmarks.optical.instrumentation import FORBIDDEN_RUNTIME_HOOKS, _resolve_owner


class RecordedLeafContamination(RuntimeError):
    pass


# These additions are specific to a render-only recorded leaf. Later C4 may
# legitimately record terminal/diagnostic events outside this exact interval.
_EXTRA_TARGETS = (
    "warp.copy",
    "warp.clone",
    "warp.array.numpy",
    "warp.array.to",
    "warp.load_module",
    "warp._src.context.Module.load",
    "warp.Event.__init__",
    "warp.Event.is_complete",
    "warp.Event.ipc_handle",
    "warp.Stream.is_complete",
    "warp.get_event_elapsed_time",
    "warp.record_event",
    "warp.wait_event",
    "warp.wait_stream",
    "warp.Stream.record_event",
    "warp.Stream.wait_event",
    "warp.Stream.wait_stream",
    "torch.cuda.Event.query",
    "torch.cuda.Event.elapsed_time",
    "torch.cuda.Event.record",
    "torch.cuda.Event.ipc_handle",
    "torch.cuda.Stream.query",
    "torch.cuda.Stream.record_event",
    "torch.cuda.Stream.wait_event",
    "torch.cuda.Stream.wait_stream",
    "warp.launch",
)

_NATIVE_QUERY_ENTRIES = (
    "wp_cuda_event_query",
    "wp_cuda_stream_query",
    "wp_cuda_event_elapsed_time",
)


class RecordedLeafAudit:
    def __init__(self, wp, *, commands, stream, sample_array, event, fault):
        self.wp = wp
        self.commands = commands
        self.stream = stream
        self.sample_array = sample_array
        self.event = event
        self.fault = fault
        self._active = None
        self.entry_calls = 0
        self.native_launches = 0
        self.native_events = 0
        self.native_queries = 0
        self.rejected = []
        self.restoration_checks = 0
        self._original_launch = wp.Launch.launch
        self._core = wp._src.context.runtime.core
        self._original_native = self._core.wp_cuda_launch_kernel
        self._original_event = self._core.wp_cuda_event_record
        self._query_originals = tuple((name, getattr(self._core, name)) for name in _NATIVE_QUERY_ENTRIES)
        targets = tuple(spec.target for spec in FORBIDDEN_RUNTIME_HOOKS if not spec.entry_point_prefixes)
        targets += _EXTRA_TARGETS
        resolved = []
        seen = set()
        for target in targets:
            owner, attribute = _resolve_owner(target)
            key = (id(owner), attribute)
            if key not in seen:
                seen.add(key)
                original = getattr(owner, attribute)
                if not callable(original) and not (type(original) is property and callable(original.fget)):
                    raise TypeError(
                        f"required audit entry is neither callable nor a readable property: {target}"
                    )
                resolved.append((target, owner, attribute, original))
        self._resolved = tuple(resolved)
        self._native_arguments = tuple(
            (
                command.device.context,
                command.hooks.forward,
                command.bounds.size,
                command.max_blocks,
                command.block_dim,
                command.hooks.forward_smem_bytes,
                command.params_addr,
                stream.cuda_stream,
            )
            for command in commands
        )
        if any(command.adjoint for command in commands):
            raise ValueError("fixed optical validation has no adjoint launch")

    def _deny(self, name):
        self.rejected.append(name)
        raise RecordedLeafContamination(name)

    def _forbidden(self, target, original):
        def wrapped(*args, **kwargs):
            if self._active is not None:
                self._deny(target)
            return original(*args, **kwargs)

        return wrapped

    def _native(self, *args):
        if self._active is None:
            raise RecordedLeafContamination("native launch outside exact recorded leaf")
        expected = self._native_arguments[self._active]
        if len(args) != len(expected) or any(
            (actual is not wanted if index == 6 else actual != wanted)
            for index, (actual, wanted) in enumerate(zip(args, expected, strict=True))
        ):
            self._deny("native launch identity")
        self.native_launches += 1
        return self._original_native(*args)

    def _native_query(self, name):
        def query(*args, **kwargs):
            self.native_queries += 1
            self._deny(f"warp.runtime.core.{name}")

        return query

    def _replacement(self, target, original):
        if type(original) is property:
            # Patch the descriptor without reading it on an instance. Property
            # polling must fail before its installed native query is entered.
            return property(
                self._forbidden(target, original.fget), original.fset, original.fdel, original.__doc__
            )
        return self._forbidden(target, original)

    def _event_record(self, *args):
        self.native_events += 1
        self._deny("warp.runtime.core.wp_cuda_event_record")

    def _launch(self, command, stream=None):
        if self._active is None or command is not self.commands[self._active] or stream is not self.stream:
            self._deny("recorded launch identity")
        self.entry_calls += 1
        # Fixed negative controls really enter the monitored runtime API. No
        # caller callback or synthetic self-reported violation substitutes for it.
        wp = self.wp
        if self.fault == "audit_allocation":
            wp.empty(1, device=stream.device)
        elif self.fault == "audit_copy":
            wp.copy(self.sample_array, self.sample_array, stream=stream)
        elif self.fault == "audit_materialization":
            self.sample_array.numpy()
        elif self.fault == "audit_numpy":
            import numpy as np

            np.asarray([1])
        elif self.fault == "audit_bridge":
            wp.to_torch(self.sample_array)
        elif self.fault == "audit_dlpack":
            import torch.utils.dlpack

            torch.utils.dlpack.to_dlpack(None)
        elif self.fault == "audit_module":
            wp.load_module(module="optics.warp_native_render", device=stream.device)
        elif self.fault == "audit_event":
            stream.record_event(self.event)
        elif self.fault == "audit_event_query":
            _ = self.event.is_complete
        elif self.fault == "audit_stream_query":
            _ = stream.is_complete
        elif self.fault == "audit_event_elapsed":
            wp.get_event_elapsed_time(self.event, self.event, synchronize=False)
        elif self.fault == "audit_sync":
            wp.synchronize_device(stream.device)
        return self._original_launch(command, stream=stream)

    def run(self, command):
        index = next((index for index, item in enumerate(self.commands) if item is command), None)
        if index is None or self._active is not None:
            raise RecordedLeafContamination("unregistered or nested recorded leaf")
        original_count = self.native_launches
        try:
            with ExitStack() as stack:
                for target, owner, attribute, original in self._resolved:
                    if getattr(owner, attribute) is not original:
                        raise RecordedLeafContamination("hook entry changed before installation")
                    stack.enter_context(patch.object(owner, attribute, self._replacement(target, original)))
                # A function is needed here: Python must still bind the Launch
                # instance, not the audit object's bound method.
                audit = self

                def launch(instance, stream=None):
                    return audit._launch(instance, stream)

                stack.enter_context(patch.object(self.wp.Launch, "launch", launch))
                stack.enter_context(patch.object(self._core, "wp_cuda_launch_kernel", self._native))
                stack.enter_context(patch.object(self._core, "wp_cuda_event_record", self._event_record))
                for name, original in self._query_originals:
                    if getattr(self._core, name) is not original:
                        raise RecordedLeafContamination("native query changed before installation")
                    stack.enter_context(patch.object(self._core, name, self._native_query(name)))
                self._active = index
                try:
                    command.launch(stream=self.stream)
                finally:
                    self._active = None
            if self.native_launches != original_count + 1:
                raise RecordedLeafContamination("recorded leaf did not perform exactly one native launch")
        finally:
            assert self.wp.Launch.launch is self._original_launch
            assert self._core.wp_cuda_launch_kernel is self._original_native
            assert self._core.wp_cuda_event_record is self._original_event
            for name, original in self._query_originals:
                assert getattr(self._core, name) is original
            for _, owner, attribute, original in self._resolved:
                assert getattr(owner, attribute) is original
            self.restoration_checks += 1

    def evidence(self):
        return dict(
            entry_calls=self.entry_calls,
            native_launches=self.native_launches,
            native_events=self.native_events,
            native_queries=self.native_queries,
            rejected=list(self.rejected),
            restoration_checks=self.restoration_checks,
            registered_targets=[target for target, *_ in self._resolved],
        )
