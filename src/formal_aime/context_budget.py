"""Count the exact served chat template before every physical Qwen request."""
import json
import time
from urllib.request import Request
from .model_network import model_urlopen
from .aime_protocol import CURRENT, REQUEST_BUDGET

CONTEXT_LIMIT = 35000
SAFETY_MARGIN = 128


class ContextBudgetExceeded(RuntimeError):
    pass


def output_allowance(prompt_tokens, configured, *,
                     context_limit=CONTEXT_LIMIT, margin=SAFETY_MARGIN):
    allowance = min(configured, context_limit - prompt_tokens - margin)
    if allowance < min(configured, 128) or allowance <= 0:
        raise ContextBudgetExceeded('insufficient_model_context')
    return allowance


def count_prompt(config, request):
    body = {key: request[key] for key in ('model', 'messages', 'tools', 'tool_choice') if key in request}
    body.update(add_generation_prompt=True,
                chat_template_kwargs=request.get('extra_body', {}).get('chat_template_kwargs', {}))
    url = config.base_url.rstrip('/').removesuffix('/v1') + '/tokenize'
    scope = CURRENT.get()
    timeout = min(30, config.timeout_s)
    if scope is not None:
        timeout = min(timeout, scope.deadline - time.monotonic())
    if timeout <= 0:
        raise ContextBudgetExceeded('worker_execution_deadline_exhausted')
    wire = Request(url, data=json.dumps(body).encode(), headers={
        'Authorization': 'Bearer ' + config.api_key, 'Content-Type': 'application/json'})
    with model_urlopen(wire, timeout=timeout) as response:
        return int(json.load(response)['count'])


def compact_recovery(messages):
    """Keep the original public task and marked recent observations; raw audit is separate."""
    start = max(2, len(messages)-4)
    # Never retain a tool result without the native assistant call that owns it.
    while start > 2 and messages[start].get('role') == 'tool':
        start -= 1
    selected = [*messages[:2], *messages[start:]]
    result = []
    for index, message in enumerate(selected):
        message = dict(message)
        content = message.get('content')
        if index >= 2 and isinstance(content, str) and len(content) > 12000:
            message['content'] = content[:3000] + '\n[RECOVERY EXCERPT: middle omitted; not executed]\n' + content[-9000:]
        result.append(message)
    return result


def admit(config, request, *, role):
    if config.request_profile != 'qwen':
        return None
    key = 'max_completion_tokens' if 'max_completion_tokens' in request else 'max_tokens'
    configured = int(request.get(key, 2048))
    prompt = count_prompt(config, request)
    compacted = False
    try:
        allowed = output_allowance(prompt, configured)
    except ContextBudgetExceeded:
        if role != 'worker':
            raise  # Director append-only history is never silently rewritten.
        request['messages'] = compact_recovery(request['messages'])
        compacted = True
        prompt = count_prompt(config, request)
        allowed = output_allowance(prompt, configured)
    request[key] = allowed
    result = {'prompt_tokens': prompt, 'configured_output_tokens': configured,
              'max_output_tokens': allowed, 'context_limit': CONTEXT_LIMIT,
              'safety_margin': SAFETY_MARGIN, 'recovery_context_compacted': compacted,
              'tokenizer_source': 'same_service_/tokenize'}
    REQUEST_BUDGET.set(result)
    return result
