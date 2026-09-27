---
name: ws-transaction-closure
description: "WebShop shopping output selection and finalization when an owner has a candidate, a staged purchase or a confirmed environment result. Apply when The Director is choosing the graph output or ending the task after an Agent reports shopping progress."
metadata:
  target: director
  dataset: webshop
  validation: unvalidated
---

# 区分候选、暂存购买和环境完成

Use as optional Director orchestration guidance for a WebShop shopping workflow.

## When

The Director is choosing the graph output or ending the task after an Agent reports shopping progress.

## Orchestration

Check the current owner's public transaction state, rather than recommendation prose alone. Distinguish an identified candidate, an executable staged purchase and confirmed environment completion. Keep output attached to the owner holding the applicable transaction so the runtime's existing commit protocol can finalize it. If a prerequisite is missing and budget permits, revise that prerequisite; once completion is confirmed, avoid reopening the transaction for redundant review.

## Pitfall

A confident report or a reviewer's agreement can be mistaken for a purchase, while premature output changes can strand staged work.

## Boundary

Follow the current runtime commit protocol; no invented completion flag, extra purchase or new output schema is authorized.
