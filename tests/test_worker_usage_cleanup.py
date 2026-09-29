"""Question accounts release their connection and exclusive lock on exit."""
import os
import sqlite3

import pytest

from selfplay_graph_flowsteer import worker_usage_ledger as ledger_module
from selfplay_graph_flowsteer.adaptive import AdaptiveWorkflowSolver
from selfplay_graph_flowsteer.application import create_adaptive_application, load_adaptive_config
from selfplay_graph_flowsteer.worker_usage_ledger import WorkerUsageLedger
from .test_alfworld_v3_usage import build
from .test_application import write_config


def account(path, **overrides):
    return WorkerUsageLedger(path, **{
        'question_attempt_id': 'close-test', 'threshold': 100,
        'max_unsettled_attempts': 2, **overrides,
    })


def assert_closed(connection, lock_fd):
    with pytest.raises(sqlite3.ProgrammingError, match='closed'):
        connection.execute('SELECT 1')
    with pytest.raises(OSError):
        os.fstat(lock_fd)


def test_application_close_releases_account_without_erasing_results(tmp_path):
    app = create_adaptive_application(load_adaptive_config(write_config(tmp_path)), mock=True)
    ledger = app.runtime.worker_usage_ledger = account(tmp_path / 'usage.sqlite3')
    attempt = ledger.begin(route='fixture', agent_id='a', execution_id='1', request={})
    ledger.settle(attempt, input_tokens=30, output_tokens=10)
    before, digest = ledger.status(), ledger.digest()
    connection, lock_fd = ledger._db, ledger._lock_fd
    artifact = object()
    app.runtime.artifacts['a'] = artifact
    app.close()
    app.close()
    ledger.close()
    assert app.runtime.worker_usage_ledger is None
    assert app.runtime.artifacts['a'] is artifact
    assert_closed(connection, lock_fd)
    reopened = account(ledger.path)
    try:
        assert reopened.status() == before and reopened.digest() == digest
    finally:
        reopened.close()


@pytest.mark.parametrize('error_type', [RuntimeError, KeyboardInterrupt])
def test_broken_backend_close_still_closes_other_backends_and_ledger(tmp_path, error_type):
    app = create_adaptive_application(load_adaptive_config(write_config(tmp_path)), mock=True)
    ledger = app.runtime.worker_usage_ledger = account(tmp_path / 'usage.sqlite3')
    connection, lock_fd = ledger._db, ledger._lock_fd
    calls = []

    class Backend:
        def __init__(self, name):
            self.name = name

        def close(self):
            calls.append(self.name)
            if self.name == 'broken':
                raise error_type('close failed')

    broken, normal = Backend('broken'), Backend('normal')
    app.owned_backends = (broken, normal, normal, broken)
    with pytest.raises(error_type, match='close failed'):
        app.close()
    assert calls == ['broken', 'normal']
    assert_closed(connection, lock_fd)
    app.close()
    assert calls == ['broken', 'normal']
    reopened = account(ledger.path)
    reopened.close()


@pytest.fixture
def acquired_resources(monkeypatch):
    """Keep strong references so garbage collection cannot hide a leak."""
    connections, lock_fds = [], []
    real_connect, real_open = sqlite3.connect, os.open

    def connect(*args, **kwargs):
        connection = real_connect(*args, **kwargs)
        connections.append(connection)
        return connection

    def open_lock(path, *args, **kwargs):
        fd = real_open(path, *args, **kwargs)
        if str(path).endswith('.sqlite3.lock'):
            lock_fds.append(fd)
        return fd

    monkeypatch.setattr(ledger_module.sqlite3, 'connect', connect)
    monkeypatch.setattr(ledger_module.os, 'open', open_lock)
    return connections, lock_fds


@pytest.mark.parametrize('override', [
    {'question_attempt_id': 'different'}, {'threshold': 101}, {'max_unsettled_attempts': 3},
])
def test_rejected_account_identity_or_policy_releases_all_handles(tmp_path, acquired_resources, override):
    path = tmp_path / 'usage.sqlite3'
    account(path).close()
    with pytest.raises(ValueError, match='another question or policy'):
        account(path, **override)
    connections, lock_fds = acquired_resources
    assert_closed(connections[-1], lock_fds[-1])
    reopened = account(path)
    reopened.close()


def test_failed_lock_acquisition_closes_new_fd_and_preserves_live_owner(tmp_path, acquired_resources):
    path = tmp_path / 'usage.sqlite3'
    owner = account(path)
    try:
        with pytest.raises(RuntimeError, match='already active'):
            account(path)
        with pytest.raises(OSError):
            os.fstat(acquired_resources[1][-1])
        # A rejected second opener must not release the first opener's lock.
        with pytest.raises(RuntimeError, match='already active'):
            account(path)
        assert owner.status()['can_dispatch']
    finally:
        owner.close()
    account(path).close()


def test_sqlite_open_failure_releases_acquired_lock(tmp_path, monkeypatch, acquired_resources):
    path = tmp_path / 'usage.sqlite3'
    with monkeypatch.context() as patch:
        def fail(*args, **kwargs):
            raise sqlite3.OperationalError('open failed')
        patch.setattr(ledger_module.sqlite3, 'connect', fail)
        with pytest.raises(sqlite3.OperationalError, match='open failed'):
            account(path)
    with pytest.raises(OSError):
        os.fstat(acquired_resources[1][-1])
    account(path).close()


def test_interrupted_initialization_rolls_back_and_releases_handles(tmp_path, monkeypatch, acquired_resources):
    original = WorkerUsageLedger._initialize_account

    def interrupted(self):
        original(self)
        with self._transaction():
            self._db.execute("INSERT INTO reconciliation VALUES ('interrupted', 1, 1, 0)")
            raise KeyboardInterrupt('initialization interrupted')

    path = tmp_path / 'usage.sqlite3'
    with monkeypatch.context() as patch:
        patch.setattr(WorkerUsageLedger, '_initialize_account', interrupted)
        with pytest.raises(KeyboardInterrupt, match='initialization interrupted'):
            account(path)
    assert_closed(acquired_resources[0][-1], acquired_resources[1][-1])
    reopened = account(path)
    try:
        assert reopened._db.execute('SELECT COUNT(*) FROM reconciliation').fetchone() == (0,)
    finally:
        reopened.close()


@pytest.mark.parametrize('error_type', [RuntimeError, KeyboardInterrupt])
def test_solver_exception_closes_prepared_environment_and_usage_account(tmp_path, monkeypatch, error_type):
    life, task, backend, registry, runtime, config = build(tmp_path, monkeypatch)
    captured = []

    class BrokenDirector:
        def generate(self, *args, **kwargs):
            ledger = runtime.worker_usage_ledger
            captured.append((ledger.path, ledger._db, ledger._lock_fd))
            raise error_type('director failed')

    solver = AdaptiveWorkflowSolver(director_backend=BrokenDirector(), runtime=runtime,
        action_registry=registry, canvas_config=config, director_prompt_variant='v3')
    with pytest.raises(error_type, match='director failed'):
        solver.solve(task, run_id='failed-director')
    assert life.client.created == 1 and life.client.closed == ['1']
    assert runtime.worker_usage_ledger is None
    path, connection, lock_fd = captured[0]
    assert_closed(connection, lock_fd)
    reopened = account(path, question_attempt_id='failed-director', threshold=15)
    reopened.close()


def test_repeated_application_close_does_not_accumulate_connections_or_locks(tmp_path):
    app = create_adaptive_application(load_adaptive_config(write_config(tmp_path)), mock=True)
    connections = []
    for index in range(100):
        ledger = app.runtime.worker_usage_ledger = account(tmp_path / f'usage-{index}.sqlite3')
        connection, lock_fd = ledger._db, ledger._lock_fd
        connections.append(connection)
        app.close()
        assert_closed(connection, lock_fd)
    assert len(connections) == 100
