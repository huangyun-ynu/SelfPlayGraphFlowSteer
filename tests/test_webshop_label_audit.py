import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("label_audit", Path(__file__).parents[1] / "scripts/formal/audit_webshop_public_labels.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


@pytest.mark.parametrize("group,value,expected", [
    ("pack", "2 pack of lights", "2 pack of lights"),
    ("size", "large red", "large red"),
    ("color", "infrared", "infrared"),
    ("color", "bright blue", "blue"),
    ("colour", "RED", "red"),
])
def test_color_normalization_requires_group_and_whole_word(group, value, expected):
    assert audit.normalize_option(group, value, ["blue", "light", "red"]) == expected
