import inspect
import select
import signal
import socket
import sys
import threading

from optical_worker.wait_protocol import GuardConfiguration, GuardProtocolError, GuardWaitClient
from optical_worker.wait_supervision import GuardSupervisorCore, GuardSupervisorTransport


def run(phase):
    cfg = GuardConfiguration('a' * 32, 1, 1000, 3000)
    left, right = socket.socketpair()
    stop = threading.Event()
    reports, errors, nested = [], [], []

    def serve():
        core = GuardSupervisorCore(cfg)
        transport = GuardSupervisorTransport(right, core)
        try:
            while not stop.is_set():
                read, write, _ = select.select([right], [right] if transport.wants_write else [], [], .01)
                transport.service(readable=bool(read), writable=bool(write))
        except BaseException as exc:
            errors.append(repr(exc))
        finally:
            reports.append(core.evidence().to_payload()['summary'])

    thread = threading.Thread(target=serve)
    thread.start()
    client = GuardWaitClient(left, cfg)
    ticket = client.arm('setup') if phase == 'complete' else None
    method = getattr(GuardWaitClient, phase)
    lines, start = inspect.getsourcelines(method)
    marker = 'self._active = ticket' if phase == 'arm' else 'self._active = None'
    target = start + next(i for i, line in enumerate(lines) if line.strip() == marker)

    def on_signal(*unused):
        try:
            client.arm('resolve')
        except GuardProtocolError as exc:
            nested.append((client.state, str(exc)))

    sig = getattr(signal, 'SIGUSR1', None)
    old = signal.signal(sig, on_signal) if sig is not None else None

    def schedule(frame, event, arg):
        if event == 'line' and frame.f_code is method.__code__ and frame.f_lineno == target:
            sys.settrace(None)
            signal.raise_signal(sig) if sig is not None else on_signal()
        return schedule

    try:
        sys.settrace(schedule)
        returned = client.arm('setup') if phase == 'arm' else client.complete(ticket)
        sys.settrace(None)
        after = client.state
        assert len(nested) == 1 and nested[0][0] == 'FAILED', nested
        assert after == ('ARMED' if phase == 'arm' else 'IDLE'), after
        if phase == 'arm':
            client.complete(returned)
        next_ticket = client.arm('resolve')
        client.complete(next_ticket)
        print({'phase': phase, 'signal': sig is not None, 'line': target,
               'nested': nested, 'outer_state': after, 'subsequent_wait': 'succeeded'})
    finally:
        sys.settrace(None)
        if sig is not None:
            signal.signal(sig, old)
        stop.set()
        thread.join(2)
        left.close()
        right.close()
    assert not thread.is_alive() and not errors, errors
    assert reports[0]['normal_complete_count'] == 2, reports


print('python', sys.version.split()[0])
run('arm')
run('complete')
