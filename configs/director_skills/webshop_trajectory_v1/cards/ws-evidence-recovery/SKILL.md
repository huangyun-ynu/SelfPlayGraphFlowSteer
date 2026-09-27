---
name: ws-evidence-recovery
description: "WebShop shopping memory compression, visited-product flags, missing evidence packets and legitimate return-to-product work. Apply when A packet omits earlier evidence, memory fields disagree, or a previously visited product is dismissed as having no further value."
metadata:
  target: director
  dataset: webshop
  validation: unvalidated
---

# 分离访问历史与证据恢复价值

Use as optional Director orchestration guidance for a WebShop shopping workflow.

## When

A packet omits earlier evidence, memory fields disagree, or a previously visited product is dismissed as having no further value.

## Orchestration

Distinguish whether the product was visited, whether the relevant evidence is currently visible, and whether returning is necessary to configure or complete the purchase. Request recovery or clarification only for the missing fact that affects the decision. Keep the request with the session owner and bind recovered information to the correct product. Preserve uncertainty when the available records cannot establish the historical state.

## Pitfall

An absent compressed record can be read as never visited; a seen flag can be read as no possible value in returning.

## Boundary

Treat auxiliary value flags as insufficient evidence, not permission to fabricate history or override current legal actions.
