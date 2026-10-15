import inspect
import signal
import socket
import sys

from optical_worker.wait_protocol import GuardConfiguration, GuardFrameDecoder, GuardProtocolError
from optical_worker.wait_supervision import GuardSupervisorCore, GuardSupervisorTransport


def line_of(function, marker):
    source, start = inspect.getsourcelines(function)
    return start + next(i for i, line in enumerate(source) if marker in line)


for operation in ('ARMED', 'DISARMED'):
    cfg = GuardConfiguration('a' * 32, 1, 5, 20)
    now = [100]
    core = GuardSupervisorCore(cfg, clock=lambda: now[0])
    left, right = socket.socketpair()
    transport = GuardSupervisorTransport(right, core)
    response = core.accept(cfg.message('ARM', 0, 'setup'))
    if operation == 'DISARMED':
        core.acknowledge_written(response)
        response = core.accept(cfg.message('COMPLETE', 0, 'setup'))
    deadline = core.effective_deadline_ns
    send_line = line_of(GuardSupervisorTransport._service, 'sent = self.channel.send(')
    fence_line = line_of(GuardSupervisorCore._append, 'if updated.state == "TERMINATING":')
    seen = []

    def second(*unused):
        seen.append(('second', core.state, core.effective_deadline_ns))
        core.tick()  # Its documented reentry error propagates to first handler.

    def first(*unused):
        now[0] = deadline
        try:
            core.tick()
        except GuardProtocolError as exc:
            seen.append(('first_caught', core.state, str(exc)))

    sig1, sig2 = getattr(signal, 'SIGUSR1', None), getattr(signal, 'SIGUSR2', None)
    old1 = signal.signal(sig1, first) if sig1 is not None else None
    old2 = signal.signal(sig2, second) if sig2 is not None else None
    injected = [False, False]

    def schedule(frame, event, arg):
        if event != 'line':
            return schedule
        if frame.f_code is GuardSupervisorTransport._service.__code__ and frame.f_lineno == send_line and not injected[0]:
            injected[0] = True
            if sig1 is not None:
                sys.call_tracing(signal.raise_signal, (sig1,))
            else:
                sys.call_tracing(first, ())
        elif frame.f_code is GuardSupervisorCore._append.__code__ and frame.f_lineno == fence_line and frame.f_locals.get('operation') == 'deadline_expired' and not injected[1]:
            injected[1] = True
            sys.settrace(None)
            signal.raise_signal(sig2) if sig2 is not None else second()
        return schedule

    try:
        sys.settrace(schedule)
        try:
            transport.service(readable=False, writable=True)
        except GuardProtocolError as exc:
            seen.append(('outer_caught', str(exc)))
        finally:
            sys.settrace(None)
        left.settimeout(1)
        wire = GuardFrameDecoder().feed(left.recv(1028))
        payload = core.evidence().to_payload()
        assert injected == [True, True], injected
        assert wire == response, wire
        assert core.state == 'TERMINATING' and core.kill_required
        assert payload['summary']['normal_complete_count'] == 0
        assert payload['summary']['effective_deadline_ns'] == deadline
        print({'python': sys.version.split()[0], 'signals': sig1 is not None,
               'operation': operation, 'send_line': send_line, 'fence_line': fence_line,
               'seen': seen, 'wire_after_terminal': wire,
               'events': [event['operation'] for event in payload['trace']],
               'summary': payload['summary']})
    finally:
        sys.settrace(None)
        if sig1 is not None:
            signal.signal(sig1, old1)
            signal.signal(sig2, old2)
        left.close()
        right.close()
