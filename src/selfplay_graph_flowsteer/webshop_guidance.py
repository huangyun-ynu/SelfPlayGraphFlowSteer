"""Optional Worker instructions inspired by LASER's page-specific WebShop prompts.

Reference: Mayer123/LASER, dc50dafa1f88a4b889945393456b8960144be858,
prompt_library.py (search/select/verify prompts). This is a prompt-only adaptation,
not LASER's controller, action mapper, or environment implementation.
"""

MERGED_CHECKLIST_POLICY = "merged_checklist_v1"
WEBSHOP_WORKER_GUIDANCE_POLICIES = frozenset(
    {"baseline", "laser_checklist_v1", MERGED_CHECKLIST_POLICY}
)

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


MERGED_PAGE_CHECKLIST = """
WebShop decision order (merged_checklist_v1):
Use public observations and visible history to complete the purchase that best
satisfies the request within the existing Action budget. Keep the current
Action/final JSON format and reasoning mode; no extra rationale or think Action.

1. Completion: If commit_pending, purchased, done, or terminal is true, follow
   the existing completion protocol without starting another shopping decision.
2. Evidence: Compare the requested product type, attributes, price, and options
   with observed evidence. Search titles may show a default variant; an absent
   or conflicting option there remains unknown until product-page inspection.
   Product-page option Actions show available choices; selected_options and
   selected=true Action fields show current selections, even if page_text does
   not mark them. A price range spanning the budget is unresolved, not confirmed
   affordable. Do not invent matches or treat missing evidence as verification.
3. Decision, in this order:
   - If evidence supports the requirements and requested options are selected,
     stage Buy Now; do not explore merely to use the remaining budget.
   - Otherwise, if a useful search, comparison, inspection, or option selection
     leaves enough Actions to finish a purchase, address the unresolved or
     contradicted requirement. Prefer resolving known mismatches; compare
     confirmed affordable alternatives when the price remains uncertain.
   - Use a partial match only when further useful investigation would leave
     insufficient Actions to complete the best observed purchase. Select the
     closest available requested options and stage that candidate, recording
     remaining uncertainty honestly. Reserve navigation, option selection, and
     Buy Now costs. Return without staging only if no executable purchase path
     remains or no observed product is relevant.

Apply the decision using the current page:
- search: Use relevant product terms and distinguishing requirements; revise
  a prior query to clarify an unresolved requirement.
- search_results: Inspect relevant candidates to resolve open requirements.
  Revisit for missing evidence or purchase completion, not identical inspection.
- product: Use available option Actions to resolve missing selections; inspect
  a useful detail section for missing evidence, or consider another candidate
  when customization cannot resolve a contradiction.
- product_section: Use the displayed details for the open requirement; avoid
  rereading without an information need. Return to the product page when needed
  to select options or stage the purchase, accounting for that Action.

A Buy Now call must include purchase_evidence with concise, observed
verified_requirements and an honest unresolved_constraints list, which may be
nonempty. It stages a candidate; only the runtime commits one staged purchase
after Canvas selects the output Agent. A recommendation or an unstaged product
page is not commit-ready. Do not claim a completed purchase before that commit.
"""


def webshop_worker_guidance(policy: str) -> str:
    """Return append-only guidance; merged guidance replaces the base block in runtime."""
    if policy not in WEBSHOP_WORKER_GUIDANCE_POLICIES:
        raise ValueError(f"unknown webshop.worker_guidance_policy: {policy}")
    return LASER_PAGE_CHECKLIST if policy == "laser_checklist_v1" else ""
