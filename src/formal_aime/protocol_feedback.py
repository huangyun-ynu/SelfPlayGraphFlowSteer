"""Compact only newly appended feedback; never rewrite sampled history."""
import hashlib
import json

VERSION = "math_protocol_feedback_v1"


class FeedbackProjection:
    def __init__(self):
        self.seen = {}

    def project(self, text):
        # Full topology/state is already present in the authoritative snapshot.
        lines = [line for line in text.splitlines() if not line.startswith(
            ("Actual topology:", "Topology audit:", "Canvas state:", "Structural repair counters:"))]
        compact = "\n".join(lines).strip()
        key = hashlib.sha256(compact.encode()).hexdigest()
        count = self.seen.get(key, 0) + 1
        self.seen[key] = count
        if count > 1:
            return json.dumps({"unchanged_feedback_id": key[:16], "occurrence": count,
                               "instruction": "Use the current authoritative control snapshot; the previously recorded feedback is unchanged."})
        return compact + f"\nFeedback ID: {key[:16]}"


def prompt_profile(tokenizer, messages):
    """Exact tokenizer measurements for diagnostics, not admission estimates."""
    if tokenizer is None:
        return {}
    encoder = getattr(tokenizer, "tokenizer", tokenizer)
    groups = {"system": [], "history": [], "current": []}
    for i, message in enumerate(messages):
        group = "system" if message.get("role") == "system" else "current" if i == len(messages)-1 else "history"
        groups[group].append(len(encoder.encode(str(message.get("content", "")), add_special_tokens=False)))
    return {"version": VERSION, "content_token_counts": {k: sum(v) for k, v in groups.items()},
            "count_scope": "exact_message_content_excludes_chat_template", "message_count": len(messages)}
