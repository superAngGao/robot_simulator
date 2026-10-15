import inspect
import signal
import socket
import sys
from optical_worker.wait_protocol import GuardConfiguration, GuardFrameDecoder, GuardProtocolError
from optical_worker.wait_supervision import GuardSupervisorCore, GuardSupervisorTransport

cfg = GuardConfiguration('a' * 32, 1, 5, 20)
core = GuardSupervisorCore(cfg, clock=lambda: 100)
left, right = socket.socketpair()
transport = GuardSupervisorTransport(right, core)
response = core.accept(cfg.message('ARM', 0, 'setup'))
source, start = inspect.getsourcelines(GuardSupervisorTransport._service)
target = start + next(i for i, line in enumerate(source) if 'sent = self.channel.send(' in line)
nested = []

def on_signal(*unused):
    try:
        transport.service(readable=False, writable=True)
    except GuardProtocolError as exc:
        nested.append((core.state, core.effective_deadline_ns, str(exc)))

sig = getattr(signal, 'SIGUSR1', None)
old = signal.signal(sig, on_signal) if sig is not None else None

def schedule(frame, event, arg):
    if event == 'line' and frame.f_code is GuardSupervisorTransport._service.__code__ and frame.f_lineno == target:
        sys.settrace(None)
        signal.raise_signal(sig) if sig is not None else on_signal()
    return schedule

try:
    sys.settrace(schedule)
    try:
        transport.service(readable=False, writable=True)
    except GuardProtocolError as exc:
        print('outer_error', str(exc))
    finally:
        sys.settrace(None)
    left.settimeout(1)
    received = left.recv(1028)
    decoded = GuardFrameDecoder().feed(received)
    assert nested and nested[0][0] == 'TERMINATING', nested
    assert decoded == response, decoded
    payload = core.evidence().to_payload()
    assert core.state == 'TERMINATING' and core.pending_response is None
    assert not any(e['operation'] == 'ack_written' for e in payload['trace'])
    print('python', sys.version.split()[0], 'signal', sig is not None, 'line', target)
    print('nested', nested)
    print('wire_after_terminal', decoded)
    print('final', payload['summary'])
finally:
    sys.settrace(None)
    if sig is not None:
        signal.signal(sig, old)
    left.close()
    right.close()
