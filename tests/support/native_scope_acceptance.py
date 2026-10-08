"""CPU checks of required scope coverage, never a live runtime authority."""
from optical_worker.runtime_custody import NATIVE_SCOPE_HELPERS, RUNTIME_CUSTODY_VERSION


def assert_native_scope_success(record, artifact, identity):
    """Caller first validates the record, durable selection and attachment.

    Identity is the separately captured same-run OS worker/supervisor/parent
    observation. This is a fixed deployment acceptance check, not attestation.
    """
    payload = record.to_payload()
    assert payload["schema_version"] == RUNTIME_CUSTODY_VERSION
    assert payload["summary"]["input_validation_observed"] is True
    assert payload["summary"]["final_execution_readiness_issued"] is False
    assert payload["configuration"]["device_ordinal"] == 1
    assert artifact["worker"]["record_kind"] == "terminal"
    assert artifact["worker"]["transition_trace"] == payload["lifecycle"]["transition_trace"]
    worker, supervisor = artifact["worker"]["summary"], artifact["supervisor"]["summary"]
    assert worker["release_status"] == "completed" and worker["emergency_sync_attempt_count"] == 1
    assert worker["unknown_device_scope"] is False and worker["possible_dirty_device_ordinals"] == [1]
    assert supervisor["complete"] and supervisor["state"] == "EXITED"
    assert identity["worker"]["ppid"] == identity["supervisor"]["pid"]
    assert identity["supervisor"]["ppid"] == identity["parent"]["pid"]
    assert all(type(identity[role]["start_ticks"]) is int and identity[role]["start_ticks"] > 0
               for role in ("worker", "supervisor", "parent"))
    lane = payload["lane_id"]
    helpers = {row[1 if lane == "cuda" else 2]: site for site, row in NATIVE_SCOPE_HELPERS.items()}
    expected = set()
    descriptor = None
    for event in payload["lifecycle"]["transition_trace"]:
        if event["operation"] == "descriptor_observed":
            descriptor = event["details"]["descriptor"]
        elif event["operation"] == "low_level_leaf_entered":
            operation = descriptor["operation_id"]
            site = helpers.get(descriptor["runtime_identity"])
            if operation == "indexed_runtime_stream":
                site = "runtime_stream"
            elif lane == "warp" and operation in {"indexed_upload_scope", "indexed_validation_scope"}:
                site = "upload_scope" if operation == "indexed_upload_scope" else "readback_scope"
            if site is not None:
                expected.update((site, edge, event["index"]) for edge in ("before", "after"))
        elif event["operation"] == "emergency_sync_started":
            expected.update(("terminal_sync", edge, event["index"]) for edge in ("before", "after"))
    binding = next(e for e in payload["journal"] if e["kind"] == "runtime_binding")
    expected.add(("runtime_binding", "point", binding["lifecycle_event_index"]))
    rows = [e["details"] for e in payload["journal"] if e["kind"] == "native_scope"]
    observed = [(r["site"], r["edge"], r["operation_event_index"]) for r in rows]
    assert len(observed) == len(set(observed)) and set(observed) == expected
    required = {"runtime_stream", "terminal_sync"}
    if lane == "warp":
        required.update(("upload_scope", "readback_scope"))
    for site in required:
        assert sum(r["site"] == site and r["edge"] == "before" for r in rows) == 1
    baseline_stream = baseline_context = None
    for row in rows:
        assert row["worker_pid"] == row["thread_id"] == identity["worker"]["pid"]
        state = row["state"]
        pristine = row["site"] == "runtime_stream" and row["edge"] == "before"
        assert len(state["primary_contexts"]) == 8
        assert [r["active"] for r in state["primary_contexts"]] == [False, not pristine] + [False] * 6
        if not pristine:
            if baseline_stream is None:
                baseline_stream = state["target_current_stream"]
            assert state["target_current_stream"] == baseline_stream
            assert state["held_stream"] == dict(zip(("ordinal", "handle"), binding["details"]["stream_identity"]))
        if lane == "cuda":
            assert state["initialized"] is (not pristine)
            assert state["logical_basis"] == ("cold_source_derived" if pristine else "native_return")
            assert state["logical_device"] == 0 and state["raw_device"] == (0 if pristine else 1)
        else:
            if not pristine and baseline_context is None:
                baseline_context = state["target_context"]
            assert state["target_context"] == (None if pristine else baseline_context)
            interior = row["site"] in NATIVE_SCOPE_HELPERS
            assert state["default_device"] == (1 if interior else 0)
            assert state["current_context"] == (baseline_context if interior else None)
            assert state["current_context_device"] == (1 if interior else None)
    return len(rows)
