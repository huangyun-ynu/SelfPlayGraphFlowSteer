---
name: ws-candidate-comparison
description: "WebShop shopping comparison of several observed candidates with different prices, options, evidence gaps and return costs. Apply when More than one candidate has useful evidence, or the latest product is about to replace an earlier feasible candidate."
metadata:
  target: director
  dataset: webshop
  validation: unvalidated
---

# 在同一约束下比较候选

Use as optional Director orchestration guidance for a WebShop shopping workflow.

## When

More than one candidate has useful evidence, or the latest product is about to replace an earlier feasible candidate.

## Orchestration

Request a short comparison using the same public requirements for every candidate: stable identity, supported configuration, contradictions, unknowns, price and the work needed to return and finish. Keep the strongest currently feasible candidate visible while investigating a remaining gap. If a reviewer is useful, let it compare the owner's packets and return a specific decision gap to that same owner.

## Pitfall

Recency, attractive wording or one matching attribute can displace a better supported earlier candidate.

## Boundary

Compare public feasibility, not an inferred evaluator score; do not declare an unresolved candidate a full match.
