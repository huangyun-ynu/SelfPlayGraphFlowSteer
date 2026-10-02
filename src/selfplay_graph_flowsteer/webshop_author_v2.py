"""Load the author's unchanged V2 reward in the isolated WebShop worker.

Only the reward is upgraded: goal generation, prices, pages and actions retain
the deployed environment's behavior. No matching rules are implemented here.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
import threading
from functools import lru_cache
from pathlib import Path

COMMIT = "efc76f6474edd7f888b1b170f4f8ec5c7ab3e4da"
VERSION = "webshop-author-v2-efc76f6"
SCORER_SHA256 = "32c9782c0c39e98e794e9964e4a8729511112ccb4391667df907e23da829ecc2"
NORMALIZE_SHA256 = "1962113ac9e209a636d4f181b2c70544bbea59b03adca9847e48287158d794fb"
VENDOR = Path(__file__).with_name("_vendor") / "webshop_v2"
_LOAD_LOCK = threading.Lock()


def verify_sources() -> None:
    for name, expected in (("goal.py", SCORER_SHA256), ("normalize.py", NORMALIZE_SHA256)):
        if hashlib.sha256((VENDOR / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"official WebShop V2 source hash mismatch: {name}")


def _load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load official WebShop V2: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def load():
    verify_sources()
    # Resolve the upstream absolute import to its own unchanged normalizer.
    # Restore the deployed environment's module immediately after loading.
    with _LOAD_LOCK:
        key = "web_agent_site.engine.normalize"
        previous = sys.modules.get(key)
        sys.modules[key] = _load_file("_webshop_v2_normalize", VENDOR / "normalize.py")
        try:
            module = _load_file("_webshop_author_v2_goal", VENDOR / "goal.py")
        finally:
            if previous is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = previous
    return module


def install(env_module) -> None:
    original = env_module.get_reward
    if getattr(original, "_author_v2_installed", False):
        return
    upstream = load()

    def get_reward(product, goal, price, options, **kwargs):
        if "_quality_contract" in goal:
            raise ValueError("official V2 requires original goals, not quality labels")
        reward, parts = upstream.get_reward(product, goal, price, options, verbose=True)
        parts["reward_parts"] = dict(parts)
        parts.update(
            scorer_version=VERSION,
            scorer_sha256=SCORER_SHA256,
            scorer_commit=COMMIT,
            official_reward=original(product, goal, price, options),
        )
        return (reward, parts) if kwargs.get("verbose") else reward

    get_reward._author_v2_installed = True
    env_module.get_reward = get_reward
