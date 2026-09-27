---
name: ws-variant-capability
description: "WebShop shopping candidate assessment when titles show default size, color, capacity, flavor or pack quantity but selectable variants are still unknown. Apply when A plausible product family appears in search, but its title's default specification differs from the requested configuration."
metadata:
  target: director
  dataset: webshop
  validation: unvalidated
---

# 区分标题默认规格与可选变体

Use as optional Director orchestration guidance for a WebShop shopping workflow.

## When

A plausible product family appears in search, but its title's default specification differs from the requested configuration.

## Orchestration

Frame the owner's next subgoal as establishing the candidate's available configuration, rather than rejecting it from the title alone. Keep unobserved variant availability unknown. Ask for a compact comparison of the requested size, color, capacity or pack against the options actually exposed by that product. Use the result to decide whether further catalog exploration is necessary, allowing the Worker to choose legal evidence-gathering actions.

## Pitfall

Exact title matching can discard a configurable match or favor a title that sounds right but exposes no matching option.

## Boundary

Do not prescribe a catalog rank, product identifier, brand or fixed exploration sequence.
