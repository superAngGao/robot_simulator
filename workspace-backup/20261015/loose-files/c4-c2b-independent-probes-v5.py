"""Reviewer-only probes; run on stdin from frozen candidate cwd."""
import inspect
import select
import signal
import socket
import sys
import threading

from optical_worker.wait_protocol import GuardConfiguration, GuardProtocolError, GuardWaitClient
from optical_worker.wait_supervision import GuardSupervisorCore, GuardSupervisorTransport
from optical_worker.protocol import encode_framed_message


def locate(function, marker):
    source, start = inspect.getsourcelines(function)
    return start + next(i for i, value in enumerate(source) if marker in value)


def injected_call(function, marker, action, outer, predicate=lambda frame: True):
    line = locate(function, marker)
    seen = []
    def handler(*unused):
        seen.append(True)
        action()
    sig = getattr(signal, 'SIGUSR1', None)
    old = signal.signal(sig, handler) if sig is not None else None
    def trace(frame, event, arg):
        if event == 'line' and frame.f_code is function.__code__ and frame.f_lineno == line and predicate(frame):
            sys.settrace(None)
            signal.raise_signal(sig) if sig is not None else handler()
        return trace
    error = None
    try:
        sys.settrace(trace)
        try:
            outer()
        except (GuardProtocolError, OSError) as exc:
            error = type(exc).__name__
    finally:
        sys.settrace(None)
        if sig is not None:
            signal.signal(sig, old)
    assert seen == [True], (marker, seen)
    return error


for phase in ('arm', 'complete'):
    settings = GuardConfiguration('a' * 32, 1, 1000, 3000)
    left, right = socket.socketpair()
    stop = threading.Event()
    failures = []
    def serve():
        core = GuardSupervisorCore(settings)
        transport = GuardSupervisorTransport(right, core)
        try:
            while not stop.is_set():
                read, write, _ = select.select([right], [right] if transport.wants_write else [], [], .01)
                transport.service(readable=bool(read), writable=bool(write))
        except BaseException as exc:
            failures.append(repr(exc))
    server = threading.Thread(target=serve)
    server.start()
    client = GuardWaitClient(left, settings)
    ticket = client.arm('setup') if phase == 'complete' else None
    def nested_client():
        try:
            client.arm('resolve')
        except GuardProtocolError:
            pass
        assert client.state == 'FAILED'
    method = getattr(GuardWaitClient, phase)
    marker = 'self._commit(pending, ("ARMED"' if phase == 'arm' else 'self._commit(pending, ("IDLE"'
    try:
        error = injected_call(method, marker, nested_client,
                              lambda: client.arm('setup') if phase == 'arm' else client.complete(ticket))
        assert error == 'GuardProtocolError' and client.state == 'FAILED'
        try:
            client.arm('resolve')
        except GuardProtocolError:
            pass
        else:
            raise AssertionError('failed client reopened')
        print('client pre-success publication', phase, error, client.state)
    finally:
        stop.set()
        server.join(2)
        left.close()
        right.close()
    assert not server.is_alive() and not failures, failures


for operation in ('ARMED', 'DISARMED'):
    for boundary in ('before_send', 'after_full_send'):
        settings = GuardConfiguration('a' * 32, 1, 5, 20)
        core = GuardSupervisorCore(settings, clock=lambda: 100)
        left, right = socket.socketpair()
        transport = GuardSupervisorTransport(right, core)
        response = core.accept(settings.message('ARM', 0, 'setup'))
        if operation == 'DISARMED':
            core.acknowledge_written(response)
            response = core.accept(settings.message('COMPLETE', 0, 'setup'))
        deadline = core.effective_deadline_ns
        def fault():
            if boundary == 'before_send':
                try:
                    transport.service(readable=False, writable=True)
                except GuardProtocolError:
                    pass
            else:
                core.fault('parent_lost')
        marker = 'sent = self.channel.send(' if boundary == 'before_send' else 'self._offset += sent'
        try:
            error = injected_call(GuardSupervisorTransport._service, marker, fault,
                                  lambda: transport.service(readable=False, writable=True))
            left.settimeout(1)
            received = left.recv(1028)
            assert received == (b'' if boundary == 'before_send' else encode_framed_message(response, max_bytes=1024))
            if received:
                assert left.recv(1028) == b''
            payload = core.evidence().to_payload()
            assert core.state == 'TERMINATING' and core.effective_deadline_ns == deadline
            assert payload['summary']['normal_complete_count'] == 0
            assert [e['details']['operation'] for e in payload['trace'] if e['operation'] == 'ack_written'] == ([] if operation == 'ARMED' else ['ARMED'])
            print('wire ordering', operation, boundary, 'bytes', len(received), 'error', error, 'deadline', deadline)
        finally:
            left.close()
            right.close()

settings = GuardConfiguration('a' * 32, 1, 5, 20)
core = GuardSupervisorCore(settings, clock=lambda: 100)
core.acknowledge_written(core.accept(settings.message('ARM', 0, 'setup')))
response = core.accept(settings.message('COMPLETE', 0, 'setup'))
def nested_fault():
    try:
        core.fault('parent_lost')
    except GuardProtocolError:
        pass
error = injected_call(GuardSupervisorCore._append, 'self._records.setdefault', nested_fault,
                      lambda: core.acknowledge_written(response),
                      lambda frame: frame.f_locals.get('operation') == 'ack_written')
payload = core.evidence().to_payload()
assert error == 'GuardProtocolError'
assert core.state == 'TERMINATING' and core.effective_deadline_ns == 5000100
assert payload['summary']['normal_complete_count'] == 0
print('old supervisor precommit', core.state, core.effective_deadline_ns, 'replay valid')

core = None
def nested_clock():
    if core is not None:
        try:
            core.accept(settings.message('ARM', 0, 'resolve'))
        except GuardProtocolError:
            pass
    return 100
core = GuardSupervisorCore(settings, clock=nested_clock)
try:
    core.accept(settings.message('ARM', 0, 'setup'))
except GuardProtocolError:
    pass
else:
    raise AssertionError('swallowed clock reentry returned ARM')
assert core.state == 'TERMINATING' and core.kill_required
assert core.evidence().to_payload()['summary']['arm_count'] == 0
print('old swallowed clock', core.state, 'kill_required', core.kill_required, 'arms', 0)

print('independent probes passed', sys.version.split()[0], 'actual_signals', hasattr(signal, 'SIGUSR1'))
