"""Select the official public HTML-to-text renderer without modifying the simulator.

Executed by the isolated WebShop interpreter; keep imports standard-library only.
The original worker owns actions, sessions, rewards, and its wire protocol.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


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
        return env

    module.create_webshop_env = create_webshop_env
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
