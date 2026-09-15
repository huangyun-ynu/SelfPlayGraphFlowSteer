# ALFWorld Director Orchestration Skill

Use environment state as the authority. Prefer one primary state-owning Agent that receives the
complete goal and can continue its own episode. A planner or checker is useful only when its packet
can materially help the selected owner; a separate Agent's independent episode is not evidence that
the owner's episode changed. Preserve an Agent that has reached trusted success and select its
artifact promptly. Do not prescribe commands in delegation prompts, invent unavailable Actions, or
finish from a plan that has not achieved the environment goal.
