---
name: ws-targeted-evidence
description: "WebShop shopping verification of unresolved attributes, suitability, material or compatibility using public evidence for the current product and variant. Apply when A proposed candidate has an unresolved factual requirement, conflicting evidence, or a full-match claim supported only by a partial checklist."
metadata:
  target: director
  dataset: webshop
  validation: unvalidated
---

# 围绕具体需求缺口核对证据

Use as optional Director orchestration guidance for a WebShop shopping workflow.

## When

A proposed candidate has an unresolved factual requirement, conflicting evidence, or a full-match claim supported only by a partial checklist.

## Orchestration

Specify the exact requirement the owner or reviewer must resolve. Require a concise supported, contradicted or unknown status tied to public evidence about the same candidate and applicable variant. Prefer verification that can change the candidate decision and still leave a feasible completion path. Once the relevant evidence is sufficient, move to configuration or comparison instead of repeating general review.

## Pitfall

Listing satisfied conditions can conceal an omitted condition; reading more pages does not itself establish a match.

## Boundary

Do not require every detail section or convert missing evidence into either proof or an automatic rejection.
