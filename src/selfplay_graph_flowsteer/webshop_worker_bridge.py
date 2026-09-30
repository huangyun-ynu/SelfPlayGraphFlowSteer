"""Select public rendering and the declared versioned scoring contract.

Executed by the isolated WebShop interpreter; keep imports standard-library only.
The original worker owns actions, sessions, and its wire protocol. Unmodified
goal inventories retain official scoring; repaired inventories activate the
project's audited scorer, with the upstream reward kept as a diagnostic.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from collections import Counter
from pathlib import Path


def distinguish_option_actions(env):
    """Keep identical visible values in different option groups selectable."""
    original_actions = env.get_available_actions
    original_item = env.server.item_page

    def available_actions():
        result = original_actions()
        nodes = env._parse_html().select('input[type="radio"]')
        counts = Counter(node.get("value") for node in nodes)
        for node in nodes:
            raw = node.get("value")
            if counts[raw] > 1:
                env.text_to_clickable.pop(raw, None)
                env.text_to_clickable[f"{node['name']}: {raw}".lower()] = node
        result["clickables"] = list(env.text_to_clickable)
        return result

    def item_page(session_id, **kwargs):
        label = kwargs["clickable_name"]
        node = kwargs["text_to_clickable"][label]
        if node.get("name") is not None and node.get("value") is not None:
            raw = node["value"]
            if label != raw:
                kwargs = {
                    **kwargs,
                    "clickable_name": raw,
                    "text_to_clickable": {**kwargs["text_to_clickable"], raw: node},
                }
        return original_item(session_id, **kwargs)

    env.get_available_actions = available_actions
    env.server.item_page = item_page


def load_worker(path: Path, observation_mode: str):
    if observation_mode not in {"text", "text_rich"}:
        raise ValueError("unsupported WebShop observation mode")
    spec = importlib.util.spec_from_file_location("_webshop_official_worker", path)
    if spec is None or spec.loader is None:
        raise ValueError("cannot load official WebShop worker")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    create_environment = module.create_webshop_env

    def create_webshop_env(*args, **kwargs):
        env = create_environment(*args, **kwargs)
        env.observation_mode = observation_mode
        goals = getattr(getattr(env, "server", None), "goals", ())
        if any("_quality_contract" in goal for goal in goals):
            scoring_spec = importlib.util.spec_from_file_location(
                "_webshop_quality_scorer", Path(__file__).with_name("webshop_quality.py")
            )
            if scoring_spec is None or scoring_spec.loader is None:
                raise RuntimeError("quality goal inventory requires its scoring module")
            scorer = importlib.util.module_from_spec(scoring_spec)
            scoring_spec.loader.exec_module(scorer)
            simulator = sys.modules.get("web_agent_site.envs.web_agent_text_env")
            if simulator is None:
                raise RuntimeError("WebShop simulator module is not available for scoring")
            scorer.install(simulator)
            env.server.product_item_dict = scorer.PricedCatalog(
                env.server.product_item_dict, env.server.product_prices
            )
            env.server.all_products = scorer.PricedProducts(
                env.server.all_products, env.server.product_prices
            )
            distinguish_option_actions(env)
        return env

    module.create_webshop_env = create_webshop_env
    worker_class = getattr(module, "WebShopWorker", None)
    if worker_class is not None:
        original_step = worker_class.step

        def step(self, action):
            result = original_step(self, action)
            if result.get("terminal"):
                session = self.env.server.user_sessions[self.env.session]
                info = session.get("verbose_info", {})
                if info.get("scorer_version"):
                    result["scoring"] = {
                        k: info[k]
                        for k in ("scorer_version", "official_reward", "quality_checks")
                        if k in info
                    }
            return result

        worker_class.step = step
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-script", type=Path, required=True)
    parser.add_argument("--observation-mode", choices=("text", "text_rich"), required=True)
    parser.add_argument("--response-fd", type=int, required=True)
    args = parser.parse_args()
    # Preserve the sibling-module imports available when the original script
    # is executed directly (notably its disk-backed environment adapter).
    sys.path.insert(0, str(args.worker_script.resolve().parent))
    worker = load_worker(args.worker_script, args.observation_mode)
    sys.argv = [str(args.worker_script), "--response-fd", str(args.response_fd)]
    worker.main()


if __name__ == "__main__":
    main()
