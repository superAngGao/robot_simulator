"""One reviewed CPU OS matrix, sequential and fail-stop; no GPU execution."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
PYTHON = '/home/ang.gao/miniforge3/envs/robot_sim/bin/python'


def main(directory, abi_path):
    assert sys.platform == 'linux' and sys.version_info[:3] == (3, 12, 13)
    assert str(Path(sys.executable)) == PYTHON
    sys.path.insert(0, str(ROOT))
    from tests.support.compiler_process_cases import CASES, read, write
    from tests.support.setuptools_dependency_process import _create_run_directory

    abi_bytes = abi_path.read_bytes()
    assert json.loads(abi_bytes) == {
        'pointer_size': 8, 'ulong_size': 8, 'size': 152, 'offsets': [0, 8, 136, 144],
        'sigset_size': 128, 'sigchld': 17, 'no_cldwait': 2, 'set_subreaper': 36,
        'get_subreaper': 37, 'default_handler': True, 'no_cldwait_active': False, 'subreaper': 0,
    }
    assert directory.is_absolute() and directory.resolve() == directory
    assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT).strip()
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    _create_run_directory(directory)
    report = {'revision': revision, 'pid': os.getpid(), 'cases': [],
              'abi_path': str(abi_path), 'abi_sha256': hashlib.sha256(abi_bytes).hexdigest()}
    write(directory, 'campaign.json', report)
    for case in CASES:
        case_dir = directory / case
        start = time.monotonic()
        with (directory / (case + '.stdout')).open('wb') as out, (directory / (case + '.stderr')).open('wb') as err:
            child = subprocess.Popen([PYTHON, '-I', '-S', '-B', str(ROOT / 'tests/support/compiler_process_cases.py'),
                                      'parent', str(case_dir), case], stdout=out, stderr=err,
                                     stdin=subprocess.DEVNULL, cwd=ROOT, start_new_session=True)
            report['active'] = {'case': case, 'parent_pid': child.pid}
            write(directory, 'campaign.json', report)
            code = child.wait()  # Parent has an observation timeout; never kill its wait owner.
        pids = {child.pid}
        for filename in ('parent-identity.json', 'parent.json'):
            path = case_dir / filename
            if path.exists():
                row = read(path)
                pids.update((row['pid'], row['supervisor_pid']))
                pids.add(row.get('sentinel', row.get('sentinel_before'))['pid'])
        identity = case_dir / 'supervisor_identity.json'
        if identity.exists():
            row = read(identity)
            pids.update((row['pid'], row['worker_pid']))
        for path in case_dir.glob('member-*.json'):
            if not path.name.startswith('member-failure-'):
                pids.add(read(path)['pid'])
        alive = sorted(pid for pid in pids if Path('/proc', str(pid)).exists())
        row = {'case': case, 'parent_exit': code, 'seconds': time.monotonic() - start,
               'pids': sorted(pids), 'remaining_pids': alive}
        report['cases'].append(row)
        del report['active']
        write(directory, 'campaign.json', report)
        print(json.dumps(row), flush=True)
        assert code == 0 and not alive and identity.exists(), 'stop and inspect the same owners; do not rerun'
        assert read(case_dir / 'validated.json')['closed'] is True
        assert read(case_dir / 'sentinel-exit.json')['exit'] == 0
    report['complete'] = True
    write(directory, 'campaign.json', report)


if __name__ == '__main__':
    assert len(sys.argv) == 3
    main(Path(sys.argv[1]), Path(sys.argv[2]))
