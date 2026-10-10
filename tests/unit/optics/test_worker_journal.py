"""Actual journal framing under OS write failures; no backend or GPU use."""

import os
from types import SimpleNamespace

import pytest

from optical_worker import worker
from optical_worker.protocol import MAX_RESPONSE_BYTES, decode_canonical_message, decode_frame_header


def test_journal_completes_short_writes_before_fsync(tmp_path, monkeypatch):
    path = tmp_path / "journal.bin"
    original_write, original_fsync = os.write, os.fsync
    calls = []
    payload = {"kind": "worker_loop_fault", "message": "retained failure " * 30}

    def short_write(fd, data):
        count = original_write(fd, data[:97])
        calls.append(("write", count))
        return count

    def sync(fd):
        calls.append(("fsync", path.stat().st_size))
        original_fsync(fd)

    monkeypatch.setattr(worker.os, "write", short_write)
    monkeypatch.setattr(worker.os, "fsync", sync)
    with path.open("wb", buffering=0) as output:
        complete = worker._journal(output.fileno(), payload)
    raw = path.read_bytes()
    size = decode_frame_header(raw[:4], max_bytes=MAX_RESPONSE_BYTES)
    assert len(raw) == size + 4
    assert decode_canonical_message(raw[4:], max_bytes=MAX_RESPONSE_BYTES) == payload
    assert complete is True
    assert len(calls) > 2 and calls[-1] == ("fsync", len(raw))
    assert all(kind == "write" for kind, _ in calls[:-1])


@pytest.mark.parametrize("failure", ["zero", "raise", "interrupted", "fsync"])
def test_journal_retains_partial_bytes_and_does_not_replace_original_fault(tmp_path, monkeypatch, failure):
    path = tmp_path / "journal.bin"
    original_write = os.write
    writes, syncs = [], []

    def write(fd, data):
        writes.append(len(data))
        if len(writes) == 1:
            return original_write(fd, data if failure == "fsync" else data[:9])
        if failure == "zero":
            return 0
        if failure == "interrupted":
            raise InterruptedError("injected journal interruption")
        raise OSError("injected journal failure")

    def sync(fd):
        syncs.append(fd)
        raise OSError("injected persistence failure")

    monkeypatch.setattr(worker.os, "write", write)
    monkeypatch.setattr(worker.os, "fsync", sync)
    original_fault = ValueError("original dispatcher failure")
    with path.open("wb", buffering=0) as output:
        try:
            raise original_fault
        except ValueError as caught:
            complete = worker._journal(output.fileno(), {"kind": "worker_loop_fault", "message": str(caught)})
            assert caught is original_fault
    assert complete is False
    raw = path.read_bytes()
    assert raw
    if failure == "fsync":
        assert len(writes) == len(syncs) == 1
        assert len(raw) == decode_frame_header(raw[:4], max_bytes=MAX_RESPONSE_BYTES) + 4
    else:
        assert len(writes) == 2 and syncs == []
        assert len(raw) == 9


def test_journal_rejects_oversize_before_any_write(tmp_path, monkeypatch):
    calls = []

    def unexpected(*_):
        calls.append("write")
        raise AssertionError("oversize journal reached OS write")

    monkeypatch.setattr(worker.os, "write", unexpected)
    with (tmp_path / "journal.bin").open("wb", buffering=0) as output:
        assert worker._journal(output.fileno(), {"message": "x" * MAX_RESPONSE_BYTES}) is False
    assert calls == []


@pytest.mark.parametrize("failure", [None, "zero", "raise", "fsync"])
@pytest.mark.parametrize("failed_frame,handler_fails", [
    ("worker_loop_fault", False), ("worker_loop_fault", True), ("termination_handler_failure", True),
])
def test_worker_loop_does_not_append_after_failed_journal_frame(
    tmp_path, monkeypatch, failure, failed_frame, handler_fails,
):
    """Exercise the original loop and file writes, with CPU transport/session doubles."""
    from optical_worker.protocol import encode_framed_message

    original_fault = ValueError("original dispatcher error")
    observed = []

    def handle(_request):
        raise original_fault

    def terminate(**kwargs):
        observed.append(kwargs)
        if handler_fails:
            raise RuntimeError("secondary termination error")
        return {"status": "error", "sequence": 0, "payload": {"code": "ValueError"}}

    session = SimpleNamespace(
        handle=handle, terminal_fault_response=terminate, hard_exit_required=False,
        termination_evidence=SimpleNamespace(to_payload=lambda: {}, sha256="a" * 64),
    )
    source = SimpleNamespace(audit=lambda: None, open_worker=lambda manifest: session)
    frame = encode_framed_message({"sequence": 0}, max_bytes=MAX_RESPONSE_BYTES)
    reads = iter((frame[:4], frame[4:]))
    monkeypatch.setattr(worker, "_read_exact", lambda *args, **kwargs: next(reads))
    monkeypatch.setattr(worker, "_write_all", lambda *args, **kwargs: None)
    original_write, original_sync, original_journal = os.write, os.fsync, worker._journal
    writes, syncs, journal_calls = [], [], []

    def journal(fd, payload):
        journal_calls.append(payload["kind"])
        return original_journal(fd, payload)

    def write(fd, data):
        if failure is None or journal_calls[-1] != failed_frame:
            return original_write(fd, data)
        writes.append(bytes(data))
        if len(writes) == 1:
            return original_write(fd, data if failure == "fsync" else data[:9])
        if failure == "zero":
            return 0
        raise OSError("journal no longer writable")

    def sync(fd):
        if failure is None or journal_calls[-1] != failed_frame:
            return original_sync(fd)
        syncs.append(fd)
        raise OSError("journal not durable")

    monkeypatch.setattr(worker.os, "write", write)
    monkeypatch.setattr(worker.os, "fsync", sync)
    monkeypatch.setattr(worker, "_journal", journal)
    path = tmp_path / "loop_journal.bin"
    with path.open("wb", buffering=0) as output:
        result = worker.run_worker_loop(
            command_descriptor=-1, liveness_descriptor=-1, journal_descriptor=output.fileno(),
            abort_signal_descriptor=-1, abort_ack_descriptor=-1,
            manifest=SimpleNamespace(control_idle_timeout_ms=1000, response_write_timeout_ms=1000),
            _source_session=source,
        )
    assert result == 64 and len(observed) == 1
    assert observed[0]["message"] == str(original_fault)
    assert observed[0]["cause"] is original_fault
    expected = ["worker_loop_fault"]
    if handler_fails:
        expected.append("termination_handler_failure")
    expected.append("transport_termination")
    if failure is not None:
        expected = expected[:expected.index(failed_frame) + 1]
    assert journal_calls == expected
    raw = path.read_bytes()
    if failure is None:
        records = []
        while raw:
            length = decode_frame_header(raw[:4], max_bytes=MAX_RESPONSE_BYTES)
            records.append(decode_canonical_message(raw[4:4 + length], max_bytes=MAX_RESPONSE_BYTES))
            raw = raw[4 + length:]
        assert [record["kind"] for record in records] == expected
        assert records[0]["message"] == str(original_fault)
        return
    assert len(writes) == (1 if failure == "fsync" else 2)
    assert len(syncs) == (1 if failure == "fsync" else 0)
    if failed_frame == "termination_handler_failure":
        length = decode_frame_header(raw[:4], max_bytes=MAX_RESPONSE_BYTES)
        first = decode_canonical_message(raw[4:4 + length], max_bytes=MAX_RESPONSE_BYTES)
        assert first["message"] == str(original_fault)
        raw = raw[4 + length:]
    if failure == "fsync":
        record = decode_canonical_message(raw[4:], max_bytes=MAX_RESPONSE_BYTES)
        assert record["kind"] == failed_frame
    else:
        assert len(raw) == 9


@pytest.mark.parametrize("original_fault", [
    ValueError("original dispatcher exception identity"), EOFError("original EOF"),
    TimeoutError("original idle timeout"), BrokenPipeError("original liveness failure"),
    worker.ClosedWorkerProtocolError("original request failure"),
])
def test_worker_loop_passes_original_exception_to_native_failure_handling(
    tmp_path, monkeypatch, original_fault,
):
    from optical_worker import native_setup
    from optical_worker.protocol import encode_framed_message

    failures = []
    manifest = SimpleNamespace(protocol_version=2, control_idle_timeout_ms=1000,
                               response_write_timeout_ms=1000)
    session = object.__new__(worker.ClosedWorkerSession)
    source = SimpleNamespace(failed=False, audit=lambda: None, open_worker=lambda _: session)
    session._manifest, session._native_source = manifest, source
    session._termination = None
    session._lifecycle = SimpleNamespace(termination_evidence=None)

    def handle(_request):
        raise original_fault

    def fail_native(actual_source, error, **kwargs):
        assert actual_source is source
        failures.append(error)

    session.handle = handle
    monkeypatch.setattr(native_setup, "_fail_optical_preparation", fail_native)
    frame = encode_framed_message({"sequence": 0}, max_bytes=MAX_RESPONSE_BYTES)
    reads = iter((frame[:4], frame[4:]))
    monkeypatch.setattr(worker, "_read_exact", lambda *args, **kwargs: next(reads))
    monkeypatch.setattr(worker, "_write_all", lambda *args, **kwargs: None)
    with (tmp_path / "cause.bin").open("wb", buffering=0) as output:
        assert worker.run_worker_loop(
            command_descriptor=-1, liveness_descriptor=-1, journal_descriptor=output.fileno(),
            abort_signal_descriptor=-1, abort_ack_descriptor=-1, manifest=manifest,
            _source_session=source,
        ) == 64
    assert failures == [original_fault]
    assert session.terminal
