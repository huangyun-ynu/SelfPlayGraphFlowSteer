---
name: ws-live-options
description: "WebShop shopping option state after leaving, reopening or revising a product session, including cleared selections and exact variant confirmation. Apply when A candidate is revisited, the current product changes, or a report relies on options selected earlier."
metadata:
  target: director
  dataset: webshop
  validation: unvalidated
---

# 回访后核验当前选项

Use as optional Director orchestration guidance for a WebShop shopping workflow.

## When

A candidate is revisited, the current product changes, or a report relies on options selected earlier.

## Orchestration

Have the session owner reconcile the current product identity and live selected options with the requested configuration. Treat earlier selections as history until the current observation confirms them. Preserve still-valid product evidence separately from transient option state. If configuration is incomplete, assign that concrete gap to the owner and obtain a fresh selection record before the final purchase check.

## Pitfall

A correct past selection or a matching title can be mistaken for the configuration currently active in the session.

## Boundary

Historical packets cannot restore options or certify another product's state; only current public session feedback can.
