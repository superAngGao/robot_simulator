"""CPU-only real guard peer shared by tests and cold import probes."""

import socket
import threading
import uuid

from optical_worker.wait_protocol import GuardConfiguration
from optical_worker.wait_supervision import GuardSupervisorCore, GuardSupervisorTransport


class Peer:
    def __init__(self, *, configuration=None, channel_pair=None):
        self.config = configuration or GuardConfiguration(uuid.uuid4().hex, 1, 1000, 5000)
        self.channel, self.remote = socket.socketpair() if channel_pair is None else channel_pair
        self.stop = threading.Event()
        self.ready = threading.Event()
        self.errors = []
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()
        assert self.ready.wait(2)

    def run(self):
        self.core = GuardSupervisorCore(self.config)
        transport = GuardSupervisorTransport(self.remote, self.core)
        self.ready.set()
        try:
            while not self.stop.wait(0.001):
                import select

                self.before_service()
                read, write, _ = select.select(
                    [self.remote], [self.remote] if transport.wants_write else [], [], 0.01
                )
                transport.service(readable=bool(read), writable=bool(write))
                self.core.tick()
        except BaseException as error:
            self.errors.append(error)

    def before_service(self):
        pass

    def close(self):
        self.stop.set()
        self.thread.join(2)
        assert not self.thread.is_alive()
        self.channel.close()
        self.remote.close()
        assert not self.errors, self.errors
