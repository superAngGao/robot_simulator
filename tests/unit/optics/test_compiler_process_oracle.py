"""CPU models discriminate fixture outcomes; never actual OS execution proof."""

import copy
from pathlib import Path

import pytest

from optical_worker.compiler_job import _CompilerObligation
from optical_worker.wait_protocol import GuardConfiguration
from optical_worker.wait_supervision import GuardSupervisorCore
from tests.support.compiler_process_cases import ROOT, SCHEMA, validate_result, write


def modeled_case(directory, case, *, ineffective=False):
    clock = [100]
    config = GuardConfiguration('a' * 32, 1, 1000, 2000)
    core = GuardSupervisorCore(config, clock=lambda: clock[0])
    def identity(pid):
        return {'pid': pid, 'sid': 2000000001, 'starttime_ticks': pid}
    worker, child, grandchild = (identity(pid) for pid in (2000000001, 2000000002, 2000000003))
    supervisor = {'pid': 2000000000, 'sid': 2000000000, 'starttime_ticks': 1}
    job = _CompilerObligation(core, supervisor)
    job.subreaper_ready()
    job.spawned(worker)
    def exchange(operation, sequence):
        response = core.accept(config.message(operation, sequence, 'setup'))
        job.sync_core()
        if case != 'leader_ack_exit' or sequence != 6:
            core.acknowledge_written(response)
            job.sync_core()
    for sequence in range(6):
        exchange('ARM', sequence)
        exchange('COMPLETE', sequence)
    exchange('ARM', 6)
    deadline = core.effective_deadline_ns
    job.reaped(worker, 0)
    members = [(worker, 'worker'), (child, 'child')]
    victim = child
    if case == 'double_fork':
        write(directory, 'worker-wait.json', {'pid': child['pid'], 'wait_status': 0})
    else:
        job.adopted(child)
    if case in {'double_fork', 'fork_during_cleanup'}:
        if case == 'fork_during_cleanup':
            job.reaped(child, 0)
        job.adopted(grandchild)
        members.append((grandchild, 'grandchild'))
        victim = grandchild
    clock[0] = deadline
    job.tick()
    job.signal_attempted(victim)
    clock[0] = 5_000_000_100 if ineffective else deadline + 1
    job.reaped(victim, 0 if ineffective else 9)
    job.no_children()
    job.pipes_eof()
    job.close()
    guard = core.evidence(require_final=True).to_payload()
    record = job.record(require_closed=True).to_payload()
    write(directory, 'request.json', {'schema_version': SCHEMA, 'root': str(ROOT), 'revision': 'a' * 40,
          'lane': 'cuda', 'guard_configuration': config.to_payload(), 'process_case': case})
    for member, role in members:
        assert not Path('/proc', str(member['pid'])).exists()
        write(directory, f"member-{member['pid']}.json", {'pid': member['pid'], 'ppid': supervisor['pid'],
              'sid': worker['sid'], 'pgid': member['pid'], 'role': role, 'at_ns': 100})
    reaped = [worker, victim] if case == 'double_fork' else [member for member, _ in members]
    result = {'artifact': {'supervisor': guard, 'worker': None}, 'compiler_job': record,
              'worker_returncode': 0, 'process_fixture': {'case': case, 'exited_core': guard,
              'pidfds': [{'pid': member['pid'], 'fd': n, 'fdinfo': f"Pid:\t{member['pid']}\n"}
                         for n, member in enumerate(reaped)]}}
    write(directory, 'supervisor.json', result)
    if ineffective:
        write(directory, f"natural-exit-{victim['pid']}.json", {'at_ns': clock[0]})
    return result, victim


@pytest.mark.parametrize('case', ['leader_exit', 'leader_ack_exit', 'double_fork', 'fork_during_cleanup', 'pipe_hang'])
@pytest.mark.parametrize('ineffective', [False, True])
def test_real_validator_requires_effective_descendant_termination(tmp_path, case, ineffective):
    modeled_case(tmp_path, case, ineffective=ineffective)
    if ineffective:
        with pytest.raises(AssertionError, match='required exit mechanism'):
            validate_result(tmp_path)
        assert not (tmp_path / 'validated.json').exists()
    else:
        validate_result(tmp_path)
        assert (tmp_path / 'validated.json').exists()


def test_real_validator_rejects_natural_fallback_even_with_claimed_kill_status(tmp_path):
    _, victim = modeled_case(tmp_path, 'leader_exit')
    write(tmp_path, f"natural-exit-{victim['pid']}.json", {'at_ns': 5_000_000_100})
    with pytest.raises(AssertionError, match='natural fallback'):
        validate_result(tmp_path)


def test_real_validator_rejects_relabelled_missing_grandchild(tmp_path):
    result, _ = modeled_case(tmp_path, 'fork_during_cleanup')
    # Keep real core/replay records fixed while deleting the fixture's intended role.
    path = tmp_path / 'member-2000000003.json'
    import json
    row = json.loads(path.read_bytes())
    row['role'] = 'child'
    write(tmp_path, path.name, row)
    before = copy.deepcopy(result['compiler_job'])
    with pytest.raises(AssertionError, match='topology'):
        validate_result(tmp_path)
    assert result['compiler_job'] == before
