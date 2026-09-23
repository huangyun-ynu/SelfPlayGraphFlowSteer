"""SkillFlow text prompt and action protocol, independent of the runtime."""

from __future__ import annotations

import re

NATIVE_POLICY = "skillflow_native_v1"
WEBSHOP_EXECUTION_POLICIES = frozenset({"graph_tools_v1", NATIVE_POLICY})

# Kept byte-for-byte with the no-skill raw templates in training/react_prompts.py.
TEMPLATE_NO_HISTORY = """You are an expert autonomous agent operating in the WebShop e-commerce environment.
Your task is to: {task_description}.
Your current observation is: {current_observation}.
Your admissible actions of the current situation are:
[
{available_actions}
].

Now it's your turn to take one action for the current step.
Return exactly one executable action string in the form search[keywords] or click[value].
For click actions, copy one value from the admissible action list exactly. Do not repeat these instructions.
"""

TEMPLATE_HISTORY = """You are an expert autonomous agent operating in the WebShop e-commerce environment.
Your task is to: {task_description}.
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}.
Your admissible actions of the current situation are:
[
{available_actions}
].

Return exactly one executable action string in the form search[keywords] or click[value].
For click actions, copy one value from the admissible action list exactly. Do not repeat these instructions.
"""

DIRECTOR_ENVIRONMENT_HINT = """This is a WebShop environment task. Agents are generic and receive
the same task-specific action interface. Each Agent's shopping session and action history are
isolated; only graph-authorized structured packets communicate findings. A purchase candidate
is provisional and can change after a graph edit. SET_OUTPUT selects the output without
executing a purchase; FINISH commits only that Agent's latest staged candidate after graph
validation and dirty-subgraph execution. No candidate reward is available during graph design.
An Agent without a staged candidate has not completed a purchase. Choose responsibilities,
layers and relations from the task and factual feedback; no role set or topology is prescribed.
Workers select their own actions. Do not prescribe queries, click targets or action sequences.
"""


def parse_native_action(text: str) -> str | None:
    """SkillFlow's action-tag then raw-action parsing, with skills disabled.

    Ignore a text think block when using raw fallback: provider reasoning is not
    an executable action. Preserve the chosen click value instead of guessing.
    """
    # An unterminated block may be a truncated completion: none of its text is executable.
    content = re.sub(r"<think>.*?(?:</think>|$)", "", text, flags=re.I | re.S).strip()
    tagged = re.search(r"<action>(.*?)</action>", content, flags=re.I | re.S)
    if tagged:
        candidate = tagged.group(1).strip()
        return candidate if re.fullmatch(r"(?:search|click)\[.+\]", candidate, re.S) else None
    found = re.search(r"(?:search|click)\[[^\]]+\]", content)
    return found.group(0) if found else None


def native_prompt(*, task: str, state: dict, history: list[dict]) -> str:
    actions = state.get("raw_available_actions")
    if not isinstance(actions, list):
        raise ValueError("native WebShop requires raw_available_actions from the sidecar")
    formatted = "\n".join(f"'{('search[<your query>]' if a == 'search' else a)}'," for a in actions)
    return (TEMPLATE_HISTORY if history else TEMPLATE_NO_HISTORY).format(
        task_description=task,
        current_observation=str(state.get("page_text", "")),
        available_actions=formatted,
        action_history="\n".join(
            f"[Observation {i}: '{row['observation']}', Action {i}: '{row['action']}']"
            + (f" [Result error: {row['error']}]" if row.get("error") else "")
            for i, row in enumerate(history, 1)
        ),
        step_count=len(history),
        history_length=len(history),
        current_step=len(history) + 1,
    )
