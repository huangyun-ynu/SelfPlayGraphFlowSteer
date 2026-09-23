"""Optional Worker instructions inspired by LASER's page-specific WebShop prompts.

Reference: Mayer123/LASER, dc50dafa1f88a4b889945393456b8960144be858,
prompt_library.py (search/select/verify prompts). This is a prompt-only adaptation,
not LASER's controller, action mapper, or environment implementation.
"""

WEBSHOP_WORKER_GUIDANCE_POLICIES = frozenset({"baseline", "laser_checklist_v1"})

LASER_PAGE_CHECKLIST = """
WebShop page checklist (laser_checklist_v1):
Before choosing an Action, compare the public request with the latest observation
using the applicable page checklist below. Use your existing reasoning mode;
keep the existing Action/final JSON format, without a separate Rationale response
or a think Action. These checks guide decisions; they do not grant extra Actions.
If commit_pending, purchased, done, or terminal is true, follow the existing
completion protocol instead of starting another shopping decision.

- search: Identify the requested product type, distinguishing attributes, price
  limit, and requested options. Form a query from relevant product terms. When
  prior searches are visible, identify what a revised query should clarify.
- search_results: Compare visible candidates against the requested product type,
  attributes, and price. Treat a title's color/size/default variant as provisional:
  inspect a relevant candidate to learn its actual customization options rather
  than assuming a title mismatch rules it out. Prefer inspections that resolve
  an open requirement; revisit an inspected item when completing its purchase or
  checking a still-unresolved fact, not merely to repeat the same inspection.
- product: Check each requested attribute and the price against observed evidence.
  Distinguish options the item offers from options currently selected; compare
  the requested values with both. Resolve a missing selection using current
  option Actions. If evidence is missing, choose a useful visible detail section;
  if a requirement is contradicted and cannot be fixed by customization, consider
  another candidate. Before Buy Now, recheck price, attributes, selected options,
  and unresolved requirements under the existing purchase-evidence/budget rules.
- product_section: Use the displayed description/features/reviews to resolve the
  specific open requirement. Missing information stays unknown. Avoid reopening
  an already-read section without an information need. Return to the product page
  when needed to select options or stage a purchase, accounting for that Action.

For other page types use the existing protocol. Use only public observations and
visible history; do not invent options, attribute matches, or successful purchases.
"""


def webshop_worker_guidance(policy: str) -> str:
    if policy not in WEBSHOP_WORKER_GUIDANCE_POLICIES:
        raise ValueError("webshop.worker_guidance_policy must be baseline or laser_checklist_v1")
    return LASER_PAGE_CHECKLIST if policy == "laser_checklist_v1" else ""
