from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from selfplay_graph_flowsteer.observability import TaskSpec
from selfplay_graph_flowsteer.webshop import _trusted_goal_id
from selfplay_graph_flowsteer.webshop_sidecar import ProductStore, WebShopSession


class FakeWorker:
    def __init__(self) -> None:
        self.operations: list[tuple[str, dict]] = []

    def request(self, operation: str, payload: dict) -> dict:
        self.operations.append((operation, payload))
        if operation == "reset":
            return {
                "available_actions": ["search", "click[search]"],
                "instruction_text": "find the requested product",
                "observation_text": "WebShop [SEP] Search",
            }
        action = payload["action"]
        if action.startswith("search["):
            return {
                "available_actions": ["click[back to search]", "click[B012345678]"],
                "observation_text": (
                    "Page 1 [SEP] B012345678 [SEP] Example Product [SEP] $12.50"
                ),
                "reward": 0.0,
                "terminal": False,
            }
        if action == "click[B012345678]":
            return {
                "available_actions": [
                    "click[buy now]",
                    "click[Small]",
                    "click[Large]",
                ],
                "observation_text": "size [SEP] Small [SEP] Large [SEP] Buy Now",
                "reward": 0.0,
                "terminal": False,
            }
        if action == "click[Large]":
            return {
                "available_actions": ["click[buy now]", "click[Small]", "click[Large]"],
                "observation_text": "size [SEP] Small [SEP] Large [SEP] Buy Now",
                "reward": 0.0,
                "terminal": False,
            }
        assert action == "click[buy now]"
        return {
            "available_actions": [],
            "observation_text": "Done",
            "reward": 1.0,
            "terminal": True,
        }

    def close(self) -> None:
        return None


def product_store(tmp_path: Path) -> ProductStore:
    path = tmp_path / "products.sqlite3"
    product = {
        "asin": "B012345678",
        "name": "Example Product",
        "pricing": [12.5],
        "customization_options": {
            "Size": [
                {"is_selected": True, "url": None, "value": "Small"},
                {"is_selected": False, "url": None, "value": "Large"},
            ]
        },
    }
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE products (asin TEXT NOT NULL, product_json TEXT NOT NULL)"
        )
        connection.execute("CREATE UNIQUE INDEX products_asin_uq ON products(asin)")
        connection.execute(
            "INSERT INTO products(asin, product_json) VALUES (?, ?)",
            (product["asin"], json.dumps(product)),
        )
    return ProductStore(path)


def test_session_projects_official_actions_and_versions(tmp_path: Path) -> None:
    worker = FakeWorker()
    session = WebShopSession("session", worker, product_store(tmp_path), "webshop/goal-1")

    reset = session.reset()
    assert reset["page_type"] == "search"
    assert reset["state_version"] == 1

    search = session.search("example")
    product = next(item for item in search["valid_subactions"] if item["kind"] == "open_product")
    assert product == {
        "asin": "B012345678",
        "kind": "open_product",
        "label": "Example Product",
        "price": 12.5,
        "target_id": "open_product:1:B012345678",
        "title": "Example Product",
    }

    page = session.click(product["target_id"])
    assert page["product"] == {
        "asin": "B012345678",
        "price": 12.5,
        "title": "Example Product",
    }
    assert page["selected_options"] == {"size": "Small"}
    assert page["state_version"] == 3

    large = next(
        item
        for item in page["valid_subactions"]
        if item.get("option_value") == "Large"
    )
    selected = session.click(large["target_id"])
    assert selected["selected_options"] == {"size": "Large"}


def test_commit_id_is_idempotent(tmp_path: Path) -> None:
    worker = FakeWorker()
    session = WebShopSession("session", worker, product_store(tmp_path), "webshop/goal-1")
    session.reset()
    search = session.search("example")
    product = next(item for item in search["valid_subactions"] if item["kind"] == "open_product")
    page = session.click(product["target_id"])
    purchase = next(item for item in page["valid_subactions"] if item["kind"] == "purchase")

    first = session.commit(purchase["target_id"], "commit-1")
    second = session.commit(purchase["target_id"], "commit-1")

    assert first == second
    assert first["purchased"] is True
    assert first["reward"] == 1.0
    assert [payload["action"] for op, payload in worker.operations if op == "step"].count(
        "click[buy now]"
    ) == 1


@pytest.mark.parametrize(
    ("metadata", "expected"),
    [
        ({"goal_id": "goal:17"}, "goal:17"),
        ({"source_id": "webshop/goal-10931"}, "webshop/goal-10931"),
        ({"source_task_id": "webshop/goal-42"}, "webshop/goal-42"),
        ({"source_task_id": "webshop/not-a-goal"}, ""),
    ],
)
def test_trusted_goal_id_accepts_only_pinned_webshop_identities(
    metadata: dict[str, str], expected: str
) -> None:
    task = TaskSpec("task", "shop", metadata=metadata)
    assert _trusted_goal_id(task) == expected
