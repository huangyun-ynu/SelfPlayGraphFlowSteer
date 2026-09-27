---
name: ws-bounded-review
description: "WebShop shopping orchestration of an optional independent reviewer for a concrete requirement conflict, comparison or unsupported completion claim. Apply when An independent assessment could resolve a consequential disagreement using already available evidence and remaining workflow budget."
metadata:
  target: director
  dataset: webshop
  validation: unvalidated
---

# 有证据输入的轻量评审

Use as optional Director orchestration guidance for a WebShop shopping workflow.

## When

An independent assessment could resolve a consequential disagreement using already available evidence and remaining workflow budget.

## Orchestration

Give the reviewer a narrow question, the original public requirement and the owner's relevant evidence packet. Keep it read-only unless the runtime explicitly grants execution ownership. Require a concrete contradiction, missing fact or supported resolution that the owner can use. Connect that feedback into the owner's revision. Skip the extra node when the evidence is sufficient or another review cannot improve the next decision.

## Pitfall

A reviewer without candidate evidence can only repeat the task or endorse an unsupported claim while adding latency and tokens.

## Boundary

An additional Agent is optional; its report is not an environment observation or a reason to change the configured Worker route.
