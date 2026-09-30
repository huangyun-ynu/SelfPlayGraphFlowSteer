#!/usr/bin/env python3
"""Verify original official data and the promoted WebShop service before training."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tomllib
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from selfplay_graph_flowsteer.webshop_sidecar import implementation_sha256
from webshop_dataset_index import public_goal_order, validate_records


def validate_health(health, expected):
    if health.get('status') != 'ok' or health.get('idempotency_protocol') != 'webshop-request-v1':
        raise ValueError('WebShop service is not healthy or has an incompatible protocol')
    for key, value in expected.items():
        if health.get(key) != value:
            raise ValueError(f'WebShop service differs from formal official version: {key}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path)
    parser.add_argument('--dataset', type=Path)
    parser.add_argument('--offline', action='store_true')
    args = parser.parse_args()
    goals = Path(os.environ.get('SPGFS_WEBSHOP_GOALS', ROOT / 'assets/webshop/prepared/goals.jsonl'))
    raw = goals.read_bytes()
    if any(json.loads(line).get('_quality_contract') for line in goals.open() if line.strip()):
        raise ValueError('Formal WebShop requires original official goals, not quality-rewritten goals')
    expected = {'goals_sha256': hashlib.sha256(raw).hexdigest(), 'scorer_version': 'official',
        'implementation_sha256': implementation_sha256(),
        'index_path': str(Path(os.environ.get('SPGFS_WEBSHOP_INDEX', ROOT / 'assets/webshop/source/search_engine/indexes')).resolve())}
    checked = 0
    if args.dataset:
        rows = [json.loads(line) for line in args.dataset.open() if line.strip()]
        rows = [r for r in rows if r.get('dataset', r.get('metadata', {}).get('dataset')) == 'webshop']
        if not rows or any(r.get('metadata', {}).get('webshop_quality') for r in rows):
            raise ValueError('Formal pool must contain original official WebShop tasks')
        if validate_records(rows, *public_goal_order(goals)):
            raise ValueError('WebShop task prompts and original official goal indices differ')
        checked = len(rows)
    url = 'http://127.0.0.1:' + os.environ.get('SPGFS_WEBSHOP_PORT', '18020')
    if args.config:
        configured = tomllib.loads(args.config.read_text())['webshop']['service_url'].rstrip('/')
        if configured != url:
            raise ValueError('Formal config and WebShop service port differ')
    if not args.offline:
        with urlopen(url + '/health', timeout=3) as response:
            validate_health(json.load(response), expected)
    print(json.dumps({'expected': expected, 'validated_webshop_rows': checked}, sort_keys=True))


if __name__ == '__main__':
    main()
