"""Per-execution state. Context variables isolate concurrent question histories."""
from contextvars import ContextVar
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import wraps
import time
import uuid
from .output_recovery import OutputRecoveryState


@dataclass
class WorkerExecution:
    task_id: str
    agent_id: str
    task_result: bool
    deadline: float
    execution_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    events: dict = field(default_factory=dict)
    parent_attempt_id: str | None = None
    recovery_used: int = 0
    output_recovery: OutputRecoveryState = field(default_factory=OutputRecoveryState)
    dispatched_calls: set = field(default_factory=set)
    started_monotonic: float = field(default_factory=time.monotonic)

    def usage(self):
        successes = [e for e in self.events.values() if e.get('event') == 'backend_request_success']
        return tuple(sum(e.get('completion_usage', {}).get(key) or 0 for e in successes)
                     for key in ('token_in', 'token_out'))


CURRENT = ContextVar('aime_worker_execution', default=None)
QUESTION = ContextVar('aime_question', default=None)
REQUEST_BUDGET = ContextVar('aime_request_budget', default=None)
FORMAT_ANSWER = ContextVar('aime_format_answer', default=None)


@contextmanager
def protected_finalization(answer):
    token = FORMAT_ANSWER.set(answer)
    try:
        yield
    finally:
        FORMAT_ANSWER.reset(token)


def question_execution(function):
    @wraps(function)
    def solve(self, prompt, **kwargs):
        task_id = str(kwargs.get('task_id', 'task'))
        token = QUESTION.set(task_id)
        # Worker graph nodes may run in other threads; carry identity on the
        # per-question executor instead of relying on ContextVar propagation.
        self.runtime.executor.task_id = task_id
        try:
            return function(self, prompt, **kwargs)
        finally:
            QUESTION.reset(token)
    return solve


def worker_execution(function):
    @wraps(function)
    def execute(self, **kwargs):
        from .unified_contract import is_task_result
        node = kwargs['node']
        config = getattr(self.backend, 'config', None)
        scope = WorkerExecution(str(getattr(self, 'task_id', None) or QUESTION.get() or 'unassigned'), node.agent_id,
                                is_task_result(node), time.monotonic() + getattr(config, 'timeout_s', 600))
        token = CURRENT.set(scope)
        try:
            artifact = function(self, **kwargs)
            # Physical requests are authoritative, including earlier successful
            # generations whose later retry/parse failed.
            if scope.events:
                artifact.token_in, artifact.token_out = scope.usage()
                artifact.backend_request_events = list(scope.events.values())
            return artifact
        except Exception as exc:
            exc.worker_token_usage = scope.usage()
            exc.request_events = list(scope.events.values()) or getattr(exc, 'request_events', [])
            raise
        finally:
            CURRENT.reset(token)
    return execute


def capture_event(event):
    scope = CURRENT.get()
    if scope is not None and event.get('event_id'):
        scope.events[event['event_id']] = event


def claim_recovery(*, output_state=None, evidence_version=None):
    """One recovery allowance shared by content retry, tool and format repair."""
    scope = CURRENT.get()
    if scope is not None and scope.recovery_used >= 1:
        return False
    if output_state is not None and not output_state.claim(evidence_version):
        return False
    if scope is not None:
        scope.recovery_used += 1
    return True
