"""Failure/publication regressions through the original source-session entry."""

# Imported fixture uses the actual guarded controller and retained source binding.
# ruff: noqa: F401, F811
import pytest

from optical_worker import native_bootstrap as bootstrap
from optical_worker import warp_setup
from optical_worker.guarded_lifecycle import NativeGuardedSetupLifecycle
from tests.unit.optics.test_guarded_lifecycle import peer
from tests.unit.optics.test_warp_module_preparation import prepared_session


@pytest.mark.parametrize("lane", ["warp", "cuda"])
def test_out_of_order_stage_before_gpu_work_cancels_without_runtime(peer, lane):
    controller = NativeGuardedSetupLifecycle(lane_id=lane, channel=peer.channel, configuration=peer.config)
    session = object.__new__(bootstrap._SourceSession)
    session.lane, session.phase = lane, "runtime_configured"
    session.failed, session.failures, session.runtime_owners = False, [], []
    session.check_live = lambda: None
    session.bound_controller = lambda: controller

    def retain(error):
        session.failed = True
        session.failures.append(error)

    session._retain_failure = retain
    controller.adopt_host_owner(session, role="unit_source", operation_id="unit_source")
    with pytest.raises(bootstrap.SourceBootstrapError, match="completed compilation"):
        session.prepare_fixed_render_storage()
    summary = controller.termination_evidence.lifecycle.to_payload()["summary"]
    assert summary["disposition"] == "cancelled_before_submission_exit"
    assert summary["emergency_sync_attempt_count"] == 0
    assert summary["possible_dirty_device_ordinals"] == []
    assert summary["release_status"] == "completed"


@pytest.mark.parametrize("prepared_session", ["warp", "cuda"], indirect=True)
def test_out_of_order_stage_disposes_healthy_idle_target_work(prepared_session):
    p = prepared_session
    # Actual input upload has completed, but compilation/storage has not.
    with pytest.raises(bootstrap.SourceBootstrapError, match="completed compilation"):
        p.session.prepare_fixed_render_storage()
    summary = p.controller.termination_evidence.lifecycle.to_payload()["summary"]
    assert summary["normal_complete_count"] == 1
    assert summary["emergency_sync_attempt_count"] == 1
    assert summary["disposition"] == "aborted_synced_exit"
    assert p.syncs == [(p.binding.module, p.binding.target)]
    assert not any(e["operation"] == "guard_failed" for e in p.controller._guard_events)


@pytest.mark.parametrize("fault", ["raise", "poison", "phase"])
def test_checkpoint_failure_cannot_publish_ready_phase(prepared_session, monkeypatch, fault):
    p = prepared_session
    error = RuntimeError("checkpoint did not commit")

    def checkpoint():
        assert p.controller._guard_client.state == "IDLE"
        assert p.session.phase == "preparing_warp_modules"
        if fault == "raise":
            raise error
        if fault == "poison":
            p.session.failed = True
        else:
            p.session.phase = "input_validated"

    monkeypatch.setattr(p.session.runtime_observation, "checkpoint", checkpoint)
    with pytest.raises((RuntimeError, bootstrap.SourceBootstrapError)) as caught:
        p.session.prepare_warp_modules()
    if fault == "raise":
        assert caught.value is error
    assert len(p.records) == 3
    assert p.session.phase != "warp_modules_ready"
    assert p.session.failed
    assert p.controller.termination_evidence is not None


def test_source_integrity_failure_never_uses_idle_target_sync(prepared_session, monkeypatch):
    p = prepared_session
    error = bootstrap.SourceBootstrapError("source integrity changed")

    def reject_source():
        raise error

    monkeypatch.setattr(p.session, "check_live", reject_source)
    with pytest.raises(bootstrap.SourceBootstrapError) as caught:
        p.session.prepare_warp_modules()
    assert caught.value is error
    assert not p.records and not p.syncs
    summary = p.controller.termination_evidence.lifecycle.to_payload()["summary"]
    assert summary["normal_complete_count"] == 1
    assert summary["emergency_sync_attempt_count"] == 0
    assert "quarantine" in summary["disposition"]


def test_secondary_retention_failure_preserves_original_and_quarantines(prepared_session, monkeypatch):
    p = prepared_session
    original = RuntimeError("aggregate issuance failed")
    secondary = RuntimeError("failure intake failed")

    def issue(*args):
        raise original

    def retain(error):
        assert error is original
        raise secondary

    monkeypatch.setattr(warp_setup, "issue_module_set", issue)
    monkeypatch.setattr(p.session, "_retain_failure", retain)
    with pytest.raises(RuntimeError) as caught:
        p.session.prepare_warp_modules()
    assert caught.value is original
    assert any(e is original for e in p.controller._failure_holds)
    assert any(e is secondary for e in p.controller._failure_holds)
    assert not p.syncs
    assert p.session.phase == "preparing_warp_modules"
    summary = p.controller.termination_evidence.lifecycle.to_payload()["summary"]
    assert summary["normal_complete_count"] == 5
    assert summary["emergency_sync_attempt_count"] == 0
    assert "quarantine" in summary["disposition"]
