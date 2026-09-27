---
name: ws-session-owner
description: "WebShop shopping graph ownership, stateful execution continuity and read-only helper connections. Apply when The graph adds a helper, changes its output, or revises an Agent after a shopping session already exists."
metadata:
  target: director
  dataset: webshop
  validation: unvalidated
---

# 保留单一会话所有者

Use as optional Director orchestration guidance for a WebShop shopping workflow.

## When

The graph adds a helper, changes its output, or revises an Agent after a shopping session already exists.

## Orchestration

Identify the Agent that owns the current mutable shopping session and retain it as the execution authority. Give helpers distinct reasoning responsibilities over its visible artifacts. Connect useful helper feedback back to the owner instead of treating independent reports as shared browser state. Preserve the owner and its dependencies while useful session state exists; transfer execution only through an explicitly supported runtime handoff.

## Pitfall

Replacing the owner or selecting a reviewer as the shopping output can separate conclusions from the actual session.

## Boundary

Use the runtime's ownership permissions; this guidance does not require extra Agents or a particular Worker route.
