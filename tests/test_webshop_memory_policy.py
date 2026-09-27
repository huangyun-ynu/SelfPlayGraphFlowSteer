import pytest

from selfplay_graph_flowsteer.application import WebShopConfig
from selfplay_graph_flowsteer.llm import MockBackend
from selfplay_graph_flowsteer.runtime import ModelAgentExecutor, RoutedModelAgentExecutor


@pytest.mark.parametrize("policy", ["skillflow_history_v1", "unknown_memory"])
def test_unsupported_webshop_memory_is_rejected_at_every_entrypoint(policy):
    with pytest.raises(ValueError, match="worker_memory_policy"):
        WebShopConfig(worker_memory_policy=policy).validate()
    with pytest.raises(ValueError, match="worker_memory_policy"):
        ModelAgentExecutor(MockBackend([]), webshop_worker_memory_policy=policy)
    with pytest.raises(ValueError, match="worker_memory_policy"):
        RoutedModelAgentExecutor(
            {"deepseek": MockBackend([])}, ("deepseek",), webshop_worker_memory_policy=policy
        )
