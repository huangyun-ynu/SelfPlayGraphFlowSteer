"""Compatibility settings for promoted, frozen WebShop inference versions."""

M02_PROFILE = "m02_merged_identity_v1"
WEBSHOP_COMPATIBILITY_PROFILES = frozenset({"current", M02_PROFILE})


def section_memory_limit(profile: str) -> int:
    if profile not in WEBSHOP_COMPATIBILITY_PROFILES:
        raise ValueError("unknown webshop.compatibility_profile")
    return 1400 if profile == M02_PROFILE else 0
