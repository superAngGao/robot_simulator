import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

root = Path(sys.argv[1])
out = Path(tempfile.mkdtemp(prefix='c4-review-parent-loss-'))
read_fd, write_fd = os.pipe()
child = subprocess.Popen([
    sys.executable, '-I', '-S', str(root / 'tools/optical_sacrificial_test_supervisor.py'),
    str(out), 'V', 'hang', '1234567890abcdef1234567890abcdef', str(read_fd)
], pass_fds=(read_fd,), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
os.close(read_fd)
try:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        journal = out / 'journal.jsonl'
        if journal.exists() and 'blocking_callable_entered' in journal.read_text():
            break
        time.sleep(0.005)
    else:
        raise AssertionError('validation wait did not enter')
finally:
    os.close(write_fd)
stdout, stderr = child.communicate(timeout=5)
assert child.returncode == 0, stderr
result = json.loads((out / 'supervisor.json').read_text())
entries = [json.loads(line) for line in (out / 'journal.jsonl').read_text().splitlines()]
assert result['reason'] == 'parent_lost', result
assert result['killed'] and result['terminal_deadline_armed'], result
assert not any(e['operation'] in ('owner_finalized', 'released', 'inspection_completed') for e in entries)
print('PASS parent loss during ACKed V wait', out, json.dumps(result, sort_keys=True))
