"""The existing final Artifact contract via standard response_format JSON schema.

Tools are handled entirely by Qwen's native tool parser, never by this schema.
"""
FINAL_FIELDS = ('summary', 'evidence', 'unresolved_issues', 'tool_summary', 'confidence', 'answer')


def artifact_response_format(*, protected_answer=None, closing=False):
    properties = {
        'summary': {'type': 'string'},
        'evidence': {'type': 'array', 'items': {'type': 'string'}},
        'unresolved_issues': {'type': 'array', 'items': {'type': 'string'}},
        'tool_summary': {'type': 'array', 'items': {'type': 'string'}},
        'confidence': {'type': 'number', 'minimum': 0, 'maximum': 1},
        # Preserve the full generated answer; the verifier owns AIME validity.
        # During format repair, retain an existing answer exactly, at any length.
        'answer': {'type': 'string', **({'enum': [protected_answer]} if protected_answer is not None else {})},
    }
    fields = FINAL_FIELDS
    if closing:
        # Only an initial, no-tool closing report uses this tested schema. The
        # parser already accepts integers losslessly; no AIME range/length bound.
        fields = ('answer', 'summary', 'evidence', 'unresolved_issues', 'tool_summary', 'confidence')
        if protected_answer is None:
            properties['answer'] = {'anyOf': [{'type': 'integer'}, {'type': 'string'}]}
        properties = {key: properties[key] for key in fields}
    return {'type': 'json_schema', 'json_schema': {'name': 'final_artifact', 'strict': True,
        'schema': {'type': 'object', 'properties': properties, 'required': list(fields),
                   'additionalProperties': False}}}
