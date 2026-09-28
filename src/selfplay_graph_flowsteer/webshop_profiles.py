"""Compatibility settings for promoted, frozen WebShop inference versions."""

from .webshop_evidence import EVIDENCE_PROFILES

M02_PROFILE = "m02_merged_identity_v1"
WEBSHOP_COMPATIBILITY_PROFILES = frozenset({"current", M02_PROFILE, *EVIDENCE_PROFILES})


def section_memory_limit(profile: str) -> int:
    if profile not in WEBSHOP_COMPATIBILITY_PROFILES:
        raise ValueError("unknown webshop.compatibility_profile")
    return 1400 if profile == M02_PROFILE or profile in EVIDENCE_PROFILES else 0
