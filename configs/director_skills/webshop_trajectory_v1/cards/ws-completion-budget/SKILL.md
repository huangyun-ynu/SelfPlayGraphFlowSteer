---
name: ws-completion-budget
description: "WebShop shopping action-budget planning for verification, returning from details, selecting required options and completing a purchase. Apply when Additional exploration, detail inspection or review competes with the actions still needed to complete a candidate transaction."
metadata:
  target: director
  dataset: webshop
  validation: unvalidated
---

# 按完整购买路径分配动作预算

Use as optional Director orchestration guidance for a WebShop shopping workflow.

## When

Additional exploration, detail inspection or review competes with the actions still needed to complete a candidate transaction.

## Orchestration

Ask the owner to estimate the remaining legal work from the current page through any required return, option selection and purchase. Compare that path with the actual remaining action budget before authorizing another information-gathering subgoal. Keep a feasible completion path while resolving important gaps. If no satisfactory path remains, surface that limitation and the available public alternatives under the existing task policy.

## Pitfall

Using the final actions to read a detail and return can leave a product identified but no way to buy it.

## Boundary

Use current budgets and state, not a fixed action count; do not force a purchase or label a known mismatch as success.
