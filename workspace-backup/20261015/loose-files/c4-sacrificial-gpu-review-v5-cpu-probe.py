"""Independent read-only-candidate state/admission probe."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

root = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root))
from optical_worker.lifecycle import DeviceScopeDescriptor
from tests.support.sacrificial_lifecycle import LifetimeError, SacrificialRenderGuardian
from tests.support.sacrificial_sources import canonical_bytes, frozen_source_manifest


class Handshake:
    def arm(self, role):
        return role

    def complete(self, ticket):
        assert ticket == "S"


def rejects(function):
    try:
        function()
    except LifetimeError:
        return
    raise AssertionError("illegal schedule transition accepted")


for count in (2, 4, 5):
    for repetitions in (1, 2):
        guardian = SacrificialRenderGuardian(session_id="a" * 32, target_ordinal=0)
        permit = guardian.begin_setup_leaf(DeviceScopeDescriptor.exact_target(
            operation_id="independent_probe", leaf_kind="allocation", device_ordinal=0,
            scope_source="independent_probe", runtime_identity="probe",
            unreturned_resource_possible=True))
        guardian.finish_setup_leaf(permit, owners=(object(),))
        guardian.setup_barrier(Handshake(), lambda: None)
        plan = ("snapshot:rgb", "snapshot:counter", "sentinel:rgb", "sentinel:counter") if repetitions == 2 else ()
        guardian.seal(command_count=count, repetitions=repetitions, auxiliary_plan=plan)
        rejects(lambda: guardian.before_auxiliary("snapshot:rgb"))
        for index in range(count * repetitions):
            if index == count:
                rejects(lambda: guardian.before_launch(index))
                for name in plan:
                    guardian.before_auxiliary(name)
                    rejects(lambda: guardian.launch_returned(index))
                    rejects(lambda: guardian.validation_drain(Handshake(), lambda: None))
                    guardian.auxiliary_returned(name)
            rejects(lambda: guardian.before_launch(index + 1))
            guardian.before_launch(index)
            rejects(lambda: guardian.before_launch(index))
            rejects(lambda: guardian.auxiliary_returned("snapshot:rgb"))
            rejects(guardian.release)
            guardian.launch_returned(index)
        rejects(lambda: guardian.before_launch(count * repetitions))
        guardian.validation_drain(Handshake(), lambda: None)
        guardian.inspection_completed()
        guardian.release()
        rejects(guardian.release)
        rejects(lambda: guardian.seal(command_count=count))
        print("SCHEDULE_PASS", count, repetitions)

payload = dict(sources=frozen_source_manifest(root), session="b" * 32, target=0,
               profile="direct_light_full", shadows=True, diagnostics=True,
               resident_rays=None, fault="normal", position=0, ack_role=None,
               ack_fault="normal", repetitions=1)
mutations = [dict(repetitions=value) for value in (0, 3, True, "2")]
mutations += [dict(repetitions=2, resident_rays=value) for value in (1, 21)]
mutations += [dict(fault=value) for value in ("auxiliary_submit_then_raise", "second_pass_submit_then_raise")]
mutations += [dict(repetitions=2, resident_rays=False), dict(callback="arbitrary.py")]
for mutation in mutations:
    with tempfile.TemporaryDirectory(prefix="c4-v5-review-admission-") as temporary:
        directory = Path(temporary)
        manifest = directory / "manifest.json"
        manifest.write_bytes(canonical_bytes(payload | mutation))
        read_fd, write_fd = os.pipe()
        try:
            result = subprocess.run([sys.executable, "-I", "-S",
                str(root / "tools/optical_sacrificial_warp_supervisor.py"),
                str(manifest), str(directory), str(read_fd)],
                pass_fds=(read_fd,), capture_output=True, text=True, timeout=15)
        finally:
            os.close(read_fd)
            os.close(write_fd)
        assert result.returncode == 65, (mutation, result.returncode, result.stderr)
        assert not (directory / "journal.jsonl").exists(), mutation
        print("ADMISSION_REJECTED_BEFORE_CHILD", json.dumps(mutation, sort_keys=True))
print("INDEPENDENT_CPU_PROBE_PASS")
