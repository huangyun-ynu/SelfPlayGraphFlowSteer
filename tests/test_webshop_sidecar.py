from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from selfplay_graph_flowsteer.observability import TaskSpec
from selfplay_graph_flowsteer.webshop import _trusted_goal_id
from selfplay_graph_flowsteer.webshop_sidecar import ProductStore, WebShopSession


@pytest.mark.parametrize("low,high,status", [(40,100,"within_public_ceiling"), (40,200,"uncertain_price_range"), (190,200,"exceeds_public_ceiling")])
def test_public_price_range_and_budget(low: float, high: float, status: str) -> None:
    from selfplay_graph_flowsteer.webshop_sidecar import _public_price_fields, _search_products
    from selfplay_graph_flowsteer.runtime import _webshop_public_constraint_matrix

    text = f"${low} to ${high}"
    fields = _public_price_fields({"pricing": [low, high], "Price": text})
    assert fields["price"] is None
    preview = _search_products(f"B012345678 [SEP] Example [SEP] {text}")
    assert preview["B012345678"]["price_max"] == high
    result = _webshop_public_constraint_matrix(
        "price lower than 180.00 dollars", {"product": fields}, progress={}
    )
    assert result["price"]["status"] == status
    assert result["current_product"]["price_max"] == high


def test_fixed_and_unknown_public_price() -> None:
    from selfplay_graph_flowsteer.webshop_sidecar import _public_price_fields

    assert _public_price_fields({"pricing": [9.5]}) == {"price": 9.5}
    assert _public_price_fields({}) == {"price": None}


def test_official_normalized_option_is_tracked() -> None:
    product = {
        "options": {"color": ["thyme | white"]},
        "customization_options": {"Color": [{"value": "Thyme/White"}]},
    }
    session = WebShopSession(
        "session", FakeWorker(), SimpleNamespace(product=lambda asin: product),
        "webshop/goal-1", current_asin="B084FWQF1F",
    )
    actions, targets = session._subactions(
        ["click[thyme | white]"], "", "product"
    )
    assert actions[0]["kind"] == "select_option"
    target = actions[0]["target_id"]
    session._track_click("select_option", targets[target])
    assert session.selected_options == {"color": "thyme | white"}


@pytest.mark.parametrize("from_product", [True, False])
def test_previous_page_clears_identity_only_when_leaving_product(tmp_path: Path, from_product: bool) -> None:
    worker = SimpleNamespace(request=lambda *args: {
        "available_actions": ["click[B012345678]"] if from_product else ["click[buy now]"],
        "observation_text": "Page 1" if from_product else "Buy Now",
    })
    session = WebShopSession("session", worker, product_store(tmp_path), "webshop/goal-1")
    session.current_asin = "B012345678"
    session.selected_options = {"size": "Large"}
    session.targets = {"previous_page:1": "< prev"}
    if from_product:
        session.targets["purchase:B012345678"] = "buy now"
    result = session.click("previous_page:1")
    if from_product:
        assert result["page_type"] == "search_results"
        assert "product" not in result
        assert session.selected_options == {}
    else:
        assert result["page_type"] == "product"
        assert result["selected_options"] == {"size": "Large"}


@pytest.mark.parametrize("mode", ["legacy", "structured_only", "structured_plus_raw"])
def test_all_product_options_remain_executable(mode: str) -> None:
    from selfplay_graph_flowsteer.webshop import (
        WebShopSessionLifecycle, _webshop_current_action_surface,
    )

    actions = [
        {"target_id": f"select_option:size:{i}", "kind": "select_option", "label": str(i)}
        for i in range(150)
    ]
    lifecycle = SimpleNamespace(search_observation_mode=mode, max_observation_chars=100)
    result = WebShopSessionLifecycle._bounded(
        lifecycle, {"page_type": "product", "valid_subactions": actions}
    )
    assert result["valid_subactions"] == actions
    assert len(_webshop_current_action_surface(result)["current_actions"]) == 150


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
                    "click[B099999999]",
                ],
                "observation_text": "size [SEP] Small [SEP] Large [SEP] Buy Now",
                "reward": 0.0,
                "terminal": False,
            }
        if action == "click[Large]":
            return {
                "available_actions": [
                    "click[buy now]",
                    "click[Small]",
                    "click[Large]",
                    "click[B099999999]",
                ],
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
                {
                    "is_selected": False,
                    "url": "https://example.test/dp/B099999999/",
                    "value": "Large",
                },
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
    assert page["page_type"] == "product"
    assert page["selected_options"] == {}
    assert page["unselected_option_groups"] == ["size"]
    assert all(
        item.get("selected") is False
        for item in page["valid_subactions"]
        if item.get("kind") == "select_option"
    )
    assert page["state_version"] == 3

    large = next(
        item
        for item in page["valid_subactions"]
        if item.get("option_value") == "Large"
    )
    selected = session.click(large["target_id"])
    assert selected["page_type"] == "product"
    assert selected["product"]["asin"] == "B012345678"
    assert selected["selected_options"] == {"size": "Large"}
    assert selected["unselected_option_groups"] == []
    assert next(
        item for item in selected["valid_subactions"] if item["kind"] == "purchase"
    )["target_id"] == "purchase:B012345678"


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
