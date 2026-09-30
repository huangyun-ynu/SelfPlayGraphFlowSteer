"""Deduplicate artifacts from recorded execution events; no model/environment calls."""


def artifacts(record):
    found = {}

    def walk(value):
        if isinstance(value, dict):
            if 'artifact_id' in value and 'react_trace' in value:
                found[(value.get('agent_id'), value['artifact_id'])] = value
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(record['trajectory']['events'])
    return list(found.values())
