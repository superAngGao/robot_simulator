"""Fixed CPU fault probe for the real test supervisor; no GPU imports."""

from __future__ import annotations

import json
import os
import socket
import sys
import time
from pathlib import Path


def main():
    if not sys.flags.isolated or not sys.flags.no_site or len(sys.argv) != 6:
        return 64
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    from optical_worker.lifecycle import DeviceScopeDescriptor
    from tests.support.sacrificial_lifecycle import SacrificialRenderGuardian
    from tests.support.sacrificial_watchdog import GuardedOperationChannel

    role, fault, session = sys.argv[1:4]
    channel = socket.socket(fileno=int(sys.argv[4]))
    journal_fd = int(sys.argv[5])

    def journal(operation, **facts):
        data = (json.dumps(dict(operation=operation, **facts), sort_keys=True) + "\n").encode()
        if os.write(journal_fd, data) != len(data):
            os._exit(74)

    guardian = SacrificialRenderGuardian(session_id=session, target_ordinal=1)
    journal("guardian_created")

    class Owner:
        def __del__(self):
            journal("owner_finalized")

    handshake = GuardedOperationChannel(channel, session=session, ack_timeout=0.15)
    permit = guardian.begin_setup_leaf(
        DeviceScopeDescriptor.exact_target(
            operation_id="cpu_fixture",
            leaf_kind="allocation",
            device_ordinal=1,
            scope_source="cpu_fixture",
            runtime_identity="cpu_fixture",
            unreturned_resource_possible=True,
        )
    )
    guardian.finish_setup_leaf(permit, owners=(Owner(),))

    def wait():
        journal("blocking_callable_entered", role=role)
        if fault == "hang":
            while True:
                time.sleep(60)
        if fault == "raise":
            raise RuntimeError("fixed test wait failure")

    try:
        guardian.setup_barrier(handshake, wait if role == "S" else lambda: None)
        guardian.seal(command_count=2)
        if role == "S":
            guardian.release()
        elif role == "V":
            for position in range(2):
                guardian.before_launch(position)
                journal("launch_latched", position=position)
                guardian.launch_returned(position)
            guardian.validation_drain(handshake, wait)
            guardian.inspection_completed()
            journal("inspection_completed")
            guardian.release()
        else:
            guardian.before_launch(0)
            guardian.poison("fixed_submit_then_raise")
            guardian.emergency_drain(handshake, wait)
            guardian.release()
        journal("released", evidence=guardian.snapshot())
        return 0
    except BaseException:
        guardian.quarantine("fixed_child_failure")
        journal("quarantined", evidence=guardian.snapshot())
        os._exit(74)


if __name__ == "__main__":
    raise SystemExit(main())
