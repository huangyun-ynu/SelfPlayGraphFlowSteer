"""SWE argument normalization shared by tools and duplicate-call detection."""

from pathlib import PurePosixPath


def normalize_swe_directory_path(value: object) -> object:
    """Only directory actions accept the empty-string alias for repository root."""
    if not isinstance(value, str):
        return value
    # Preserve absolute/parent paths for the tool's subsequent rejection while
    # canonicalizing equivalent relative spellings before duplicate detection.
    return str(PurePosixPath(value.strip().replace("\\", "/")))
