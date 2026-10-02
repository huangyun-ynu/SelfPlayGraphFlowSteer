"""Public candidate feedback; no grader or reference information."""
import ast
import hashlib
import json
import re


_SCALAR = r'[+-]?[0-9]+(?:\.0+)?'
_FINAL_CONCLUSION = re.compile(
    r'(?:\b(?:the\s+)?(?:final\s+)?answer\s+(?:is|equals)|\bfinal\s+answer\s*[:=]|'
    r'\bfinal\s+(?:(?:computed|calculated)\s+)?(?:value|result)'
    r'(?:\s+for\s+[^\n;]{1,80}?)?\s+(?:is|equals|=))'
    r'\s*(?:exactly\s+)?\$?\s*(?P<value>' + _SCALAR + r')(?![0-9a-zA-Z/+*^\-]|\.[0-9])|'
    r'\\boxed\s*\{\s*(?P<boxed>' + _SCALAR + r')\s*\}', re.I)
_REJECTED_CANDIDATE = re.compile(
    r'\b(?:size|n|answer|value|result)\s*(?:(?:of|is|=)\s*)?'
    r'(?P<value>' + _SCALAR + r')\s+(?:cannot|fails\s+to|is\s+(?:invalid|incorrect|impossible|insufficient))\b', re.I)
_CONDITIONAL = re.compile(r'\b(?:if|suppose|assuming|hypothetically|earlier|previously|initially|not|no\s+longer)\b', re.I)


def _integer_value(text):
    # Compare strings, so unrestricted answers cannot hit Python's int digit cap.
    value = str(text).strip()
    if not re.fullmatch(_SCALAR, value):
        return None
    negative = value.startswith('-')
    digits = value.lstrip('+-').split('.')[0].lstrip('0') or '0'
    return ('-' if negative and digits != '0' else '') + digits


def consistency_warnings(artifact):
    answer = str(artifact.answer).strip()
    candidate = _integer_value(answer)
    if candidate is None:
        return []
    text = '\n'.join([str(artifact.summary), *map(str, artifact.evidence)])
    conclusions = []
    for match in _FINAL_CONCLUSION.finditer(text):
        prefix = re.split(r'[.!?;\n]', text[max(0, match.start()-100):match.start()])[-1]
        if not _CONDITIONAL.search(prefix):
            conclusions.append(match.group('value') or match.group('boxed'))
    warnings = []
    # Earlier explicit values may have been corrected. Only the latest counts.
    if conclusions and _integer_value(conclusions[-1]) != candidate:
        warnings.append({'code': 'answer_consistency_warning', 'candidate_answer': answer,
            'explicit_conclusions': conclusions[-1:], 'semantics': 'warning_only_no_answer_replacement'})
    for match in _REJECTED_CANDIDATE.finditer(text):
        if _integer_value(match.group('value')) == candidate:
            warnings.append({'code': 'answer_consistency_warning', 'reason': 'candidate_explicitly_rejected',
                'candidate_answer': answer, 'evidence_excerpt': text[max(0, match.start()-60):match.end()+80],
                'semantics': 'warning_only_no_answer_replacement'})
            break
    return warnings


def tool_evidence_warnings(artifact):
    warnings = []
    for index, turn in enumerate(artifact.react_trace):
        action = turn.get('action', {})
        output = turn.get('observation', {}).get('output', {})
        if not isinstance(output, dict):
            continue
        stdout = str(output.get('stdout', ''))
        mismatch = re.search(r'\b(?:match|matches|equal|agrees?|verified)\s*[:=]\s*False\b', stdout, re.I)
        if mismatch:
            warnings.append({'code': 'tool_reported_comparison_mismatch', 'react_trace_index': index,
                'evidence_excerpt': stdout[max(0, mismatch.start()-60):mismatch.end()+100],
                'semantics': 'program_report_only_not_a_math_verdict'})
        if action.get('name') != 'python_exec':
            continue
        arguments = action.get('arguments', {})
        if not isinstance(arguments, dict) or not isinstance(arguments.get('code'), str):
            continue
        try:
            tree = ast.parse(arguments['code'])
        except (SyntaxError, ValueError):
            continue
        for call in ast.walk(tree):
            if (isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == 'print'
                    and call.args and all(isinstance(arg, ast.Constant) for arg in call.args)):
                literal = ' '.join(str(arg.value) for arg in call.args)
                if _FINAL_CONCLUSION.search(literal):
                    warnings.append({'code': 'literal_answer_printed', 'react_trace_index': index,
                        'evidence_excerpt': literal[:200], 'semantics': 'literal_output_does_not_verify_the_answer'})
                    break
    return warnings

def candidate_feedback(artifact, *, head=256, tail=1024, answer_limit=2000):
    answer, summary = str(artifact.answer), str(artifact.summary)
    model_evidence = json.dumps(artifact.evidence, ensure_ascii=False)
    evidence = artifact.runtime_tool_evidence
    complete = len(answer) <= answer_limit
    tools = []
    for index, item in enumerate(artifact.react_trace):
        observation = item.get('observation', {})
        raw = json.dumps(observation.get('output', observation), ensure_ascii=False)
        output = observation.get('output')
        response_status = observation.get('status')
        execution_status = output.get('status') if isinstance(output, dict) else None
        effective_status = execution_status if response_status == 'ok' and execution_status else response_status
        tools.append({'name': item.get('action', {}).get('name'), 'status': effective_status,
                      'response_status': response_status, 'execution_status': execution_status,
                      'semantics': 'recorded_program_output_not_proof_of_mathematical_correctness',
                      'output_head': raw[:256], 'output_tail': raw[-512:] if len(raw) > 256 else '',
                      'output_truncated': len(raw) > 768,
                      'raw_reference': {'artifact_id': artifact.artifact_id, 'react_trace_index': index,
                                        'sha256': hashlib.sha256(raw.encode()).hexdigest()}})
    return {
        "artifact_id": artifact.artifact_id,
        "artifact_hash": hashlib.sha256(artifact.raw_response.encode()).hexdigest(),
        "candidate_answer": answer if complete else answer[:answer_limit] + " [answer truncated]",
        "answer_complete": complete,
        "answer_sha256": hashlib.sha256(answer.encode()).hexdigest(),
        "summary_head": summary[:head],
        "summary_tail": summary[-tail:] if len(summary) > head else "",
        "summary_truncated": len(summary) > head + tail,
        "model_evidence_head": model_evidence[:head],
        "model_evidence_tail": model_evidence[-tail:] if len(model_evidence) > head else "",
        "model_evidence_truncated": len(model_evidence) > head + tail,
        "tool_attempted_count": evidence.get("attempted_count", 0),
        "tool_successful_count": evidence.get("successful_count", 0),
        "tool_failed_count": evidence.get("failed_count", 0),
        "trusted_tool_observations": tools[-4:],
        "answer_consistency_warnings": consistency_warnings(artifact),
        "tool_evidence_warnings": tool_evidence_warnings(artifact),
        "protocol_recovery_history": [{key: item.get(key) for key in ('stage', 'accepted', 'rejection_reason')}
                                     for item in artifact.protocol_diagnostics[-6:]],
        "integrity_risks": list(artifact.integrity_risks),
        "unresolved_issues": list(artifact.unresolved_issues),
        "model_reported_confidence": artifact.claimed_confidence,
        "readiness_semantics": "protocol_eligibility_only",
    }
