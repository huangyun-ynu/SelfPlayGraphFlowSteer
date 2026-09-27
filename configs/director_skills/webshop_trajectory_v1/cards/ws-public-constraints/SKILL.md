---
name: ws-public-constraints
description: "WebShop shopping delegation that preserves product identity, mandatory constraints, numerical specifications and allowed alternatives. Apply when A shopping responsibility is created or rewritten, especially when the request mixes product identity, specifications and alternatives."
metadata:
  target: director
  dataset: webshop
  validation: unvalidated
---

# 公开需求保真的职责委派

Use as optional Director orchestration guidance for a WebShop shopping workflow.

## When

A shopping responsibility is created or rewritten, especially when the request mixes product identity, specifications and alternatives.

## Orchestration

Keep the original public request authoritative. Give the session owner responsibility for both finding a suitable configuration and completing the environment transaction. Preserve required properties, quantities, price limits and allowed alternatives when assigning a narrower subgoal. Let a helper resolve ambiguity from public evidence without silently adding requirements or substituting a different product category. Reconcile the revised responsibility with the original request before resuming execution.

## Pitfall

A concise assignment can drop an option, turn an alternative into a requirement, or redefine what is being bought.

## Boundary

Unresolved ambiguity remains explicit; an evaluation outcome does not supply an additional user requirement.
