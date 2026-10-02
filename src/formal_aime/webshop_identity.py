"""Resolve product identity from public WebShop actions without enriching observations."""
from __future__ import annotations

import re


def visible_product_asin(action: object) -> str:
    """Use structured identity or the ASIN already embedded in a live legacy action.

    The renderer ordinal is page-local; the ASIN survives result reordering.
    Never infer identity from labels, task requirements, or a product database.
    """
    if not isinstance(action, dict) or action.get("kind") != "open_product":
        return ""
    explicit = action.get("asin")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip().casefold()
    target = action.get("target_id")
    match = re.fullmatch(r"open_product:\d+:([A-Za-z0-9]{10})", target) if isinstance(target, str) else None
    return match.group(1).casefold() if match else ""
