"""Durable request ledger: pending dispatches retain unknown provider usage."""
import hashlib
import json
import os
import time
import uuid
from pathlib import Path

FIELDS = ('model', 'messages', 'temperature', 'top_p', 'seed', 'max_tokens',
          'max_completion_tokens', 'response_format', 'extra_body', 'tools',
          'tool_choice', 'parallel_tool_calls')


def dump(value):
    if hasattr(value, 'model_dump'):
        return value.model_dump()
    if hasattr(value, '__dict__'):
        return vars(value)
    return value


def execution_scope():
    from .aime_protocol import CURRENT
    return CURRENT.get()


def execution_metadata():
    from .aime_protocol import CURRENT, QUESTION
    scope = CURRENT.get()
    if scope is None:
        return {'task_id': QUESTION.get(), 'agent_id': None, 'execution_id': None}
    return {'task_id': scope.task_id, 'agent_id': scope.agent_id,
            'execution_id': scope.execution_id, 'task_result': scope.task_result}


def persist(directory, payload):
    result = {key: payload.get(key) for key in ('attempt_id', 'parent_attempt_id',
              'retry_kind', 'provider_usage_known', 'event_id', 'status')}
    if directory:
        path = Path(directory)/(payload['attempt_id']+'.json')
        path.parent.mkdir(parents=True, exist_ok=True)
        data = (json.dumps(payload, ensure_ascii=False, indent=2, default=dump)+'\n').encode()
        temporary = path.with_suffix('.tmp')
        temporary.write_bytes(data)
        os.replace(temporary, path)
        result.update(raw_generation_path=str(path.resolve()),
                      raw_generation_sha256=hashlib.sha256(data).hexdigest())
    return result


def begin_request(directory, *, request, role, parent_attempt_id=None, context_budget=None,
                  attempt_id=None, api_surface='chat_completions'):
    payload = {'attempt_id': attempt_id or uuid.uuid4().hex, 'event_id': uuid.uuid4().hex,
               'parent_attempt_id': parent_attempt_id, 'retry_kind': 'initial',
               'role': role, 'api_surface': api_surface, 'execution': execution_metadata(),
               'request': {key: request[key] for key in (*FIELDS, 'input', 'max_output_tokens', 'text', 'reasoning') if key in request},
               'context_budget': context_budget, 'completion': None,
               'status': 'pending', 'provider_usage_known': False,
               'dispatch_time_unix': time.time()}
    persist(directory, payload)
    return payload


def record_completion(directory, *, request, completion, role, event_id=None,
                      retry_kind='initial', parent_attempt_id=None, context_budget=None,
                      pending=None):
    payload = pending or begin_request(directory, request=request, role=role,
                    parent_attempt_id=parent_attempt_id, context_budget=context_budget)
    choices = getattr(completion, 'choices', None) or []
    choice = choices[0] if choices else None
    usage = getattr(completion, 'usage', None)
    keys = ('input_tokens', 'output_tokens') if payload.get('api_surface') == 'responses' else ('prompt_tokens', 'completion_tokens')
    usage_known = all(type(getattr(usage, key, None)) is int and getattr(usage, key) >= 0 for key in keys)
    payload.update(status='completed', retry_kind=retry_kind,
        request_id=getattr(completion, 'id', None), settled_time_unix=time.time(),
        provider_usage_known=usage_known,
        completion={'model': getattr(completion, 'model', None),
                    'finish_reason': getattr(choice, 'finish_reason', None),
                    'stop_reason': getattr(choice, 'stop_reason', None),
                    'message': dump(getattr(choice, 'message', None)),
                    'token_ids': getattr(choice, 'token_ids', None),
                    'usage': dump(getattr(completion, 'usage', None)),
                    **({'output': dump(getattr(completion, 'output', None)),
                        'status': getattr(completion, 'status', None)}
                       if payload.get('api_surface') == 'responses' else {})})
    return persist(directory, payload)


def record_failed_request(directory, *, request, role, event_id, error_type,
                          parent_attempt_id=None, pending=None):
    payload = pending or begin_request(directory, request=request, role=role,
                                      parent_attempt_id=parent_attempt_id)
    payload.update(status='failed', retry_kind='http_failure', error_type=error_type,
                   provider_usage_known=False, settled_time_unix=time.time())
    return persist(directory, payload)
