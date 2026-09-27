import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from selfplay_graph_flowsteer.webshop_sidecar import build_parser
from selfplay_graph_flowsteer.webshop_worker_bridge import load_worker


@pytest.mark.parametrize("mode", ["text", "text_rich"])
def test_bridge_changes_only_renderer_and_preserves_environment(tmp_path, mode):
    path = tmp_path / "worker.py"
    path.write_text(
        "from types import SimpleNamespace\n"
        "calls = []\n"
        "def create_webshop_env(*args, **kwargs):\n"
        "    calls.append((args, kwargs))\n"
        "    return SimpleNamespace(observation_mode='text', sentinel=kwargs['sentinel'])\n"
        "def get_reward(): return 0.125\n"
    )
    worker = load_worker(path, mode)
    marker = SimpleNamespace()
    env = worker.create_webshop_env("source", sentinel=marker)
    assert env.observation_mode == mode
    assert env.sentinel is marker
    assert worker.calls == [(("source",), {"sentinel": marker})]
    assert worker.get_reward() == 0.125


def test_invalid_renderer_rejected_before_loading_external_module(tmp_path):
    with pytest.raises(ValueError, match="unsupported"):
        load_worker(tmp_path / "missing.py", "reward_aware")


def test_sidecar_default_renderer_is_unchanged():
    assert build_parser().get_default("observation_mode") == "text"


def test_bridge_script_preserves_sibling_imports_and_wire_arguments(tmp_path):
    import selfplay_graph_flowsteer.webshop_worker_bridge as bridge

    (tmp_path / "local_adapter.py").write_text("VALUE = 'sibling-loaded'\n")
    worker = tmp_path / "worker.py"
    worker.write_text(
        "import sys\nfrom types import SimpleNamespace\nfrom local_adapter import VALUE\n"
        "def create_webshop_env(): return SimpleNamespace(observation_mode='text')\n"
        "def main():\n"
        "    assert sys.argv[1:] == ['--response-fd', '1']\n"
        "    print(VALUE, create_webshop_env().observation_mode)\n"
    )
    result = subprocess.run(
        [
            sys.executable,
            str(Path(bridge.__file__)),
            "--worker-script",
            str(worker),
            "--observation-mode",
            "text_rich",
            "--response-fd",
            "1",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "sibling-loaded text_rich"


@pytest.mark.parametrize("marker", ["button", "clicked button"])
def test_rich_search_page_preserves_public_product_preview(marker):
    from selfplay_graph_flowsteer.webshop_sidecar import _search_products

    rich = f"[{marker}] B012345678 [{marker}_]\nExample product\n$12.50\n"
    simple = "B012345678 [SEP] Example product [SEP] $12.50"
    assert _search_products(rich) == _search_products(simple)
