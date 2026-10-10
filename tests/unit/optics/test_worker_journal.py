"""Actual journal framing under OS write failures; no backend or GPU use."""

import os

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
