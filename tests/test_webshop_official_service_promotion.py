"""A healthy stale/quality sidecar must not be reused by official training."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('field', ['scorer_version', 'goals_sha256', 'implementation_sha256'])
def test_formal_service_rejects_other_scoring_data_or_old_code(field, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'scripts/formal'))
    spec = importlib.util.spec_from_file_location('official_service_check', ROOT / 'scripts/formal/check_webshop_official_service.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    expected = {'scorer_version': 'official', 'goals_sha256': 'official-goals', 'implementation_sha256': 'promoted-code'}
    health = {'status': 'ok', 'idempotency_protocol': 'webshop-request-v1', **expected}
    module.validate_health(health, expected)
    health[field] = 'other-version'
    with pytest.raises(ValueError, match=field):
        module.validate_health(health, expected)
