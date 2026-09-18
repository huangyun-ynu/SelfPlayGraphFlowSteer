from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import select
import signal
import sqlite3
import subprocess
import threading
import uuid
from collections import OrderedDict
from contextlib import suppress
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

_PROTOCOL = "skillev-official-environment-worker@1"
_IDEMPOTENCY_PROTOCOL = "webshop-request-v1"
_GOAL_ID = re.compile(r"^(?:webshop/)?goal[-/:](\d+)$", re.IGNORECASE)
_ASIN = re.compile(r"^[A-Z0-9]{10}$", re.IGNORECASE)
_PRICE = re.compile(r"\$\s*([0-9]+(?:\.[0-9]+)?)")
_MAX_RESPONSE_BYTES = 2 * 1024 * 1024


class SidecarError(RuntimeError):
    def __init__(self, message: str, status: HTTPStatus = HTTPStatus.BAD_REQUEST) -> None:
        super().__init__(message)
        self.status = status


@dataclass
class OfficialWorker:
    interpreter: Path
    worker_script: Path
    source_root: Path
    source_revision: str
    store_path: Path
    goals_path: Path
    index_path: Path
    goal_index: int
    seed: int
    timeout_s: float
    java_home: Path | None = None
    _process: subprocess.Popen[bytes] = field(init=False, repr=False)
    _response_fd: int = field(init=False, repr=False)
    _request_id: int = field(default=0, init=False)
    _closed: bool = field(default=False, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        read_fd, write_fd = os.pipe()
        environment = dict(os.environ)
        environment.pop("PYTHONPATH", None)
        if self.java_home is not None:
            environment["JAVA_HOME"] = str(self.java_home)
            environment["JVM_PATH"] = str(self.java_home / "lib/server/libjvm.so")
            environment["PATH"] = f"{self.java_home / 'bin'}:{environment.get('PATH', '')}"
        self._process = subprocess.Popen(  # noqa: S603
            (str(self.interpreter), str(self.worker_script), "--response-fd", str(write_fd)),
            cwd=self.source_root,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            pass_fds=(write_fd,),
            env=environment,
        )
        os.close(write_fd)
        self._response_fd = read_fd
        try:
            self.request(
                "initialize",
                {
                    "benchmark": "webshop",
                    "deployment": {
                        "goals_path": str(self.goals_path),
                        "index_path": str(self.index_path),
                        "kind": "sqlite",
                        "seed": self.seed,
                        "store_path": str(self.store_path),
                    },
                    "source_revision": self.source_revision,
                    "source_root": str(self.source_root),
                    "task": {"goal_index": self.goal_index},
                },
            )
        except Exception:
            self.abort()
            raise

    def request(self, operation: str, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            if self._closed:
                raise SidecarError("official WebShop worker is closed", HTTPStatus.GONE)
            self._request_id += 1
            message = {
                "operation": operation,
                "payload": payload,
                "protocol_version": _PROTOCOL,
                "request_id": self._request_id,
            }
            encoded = json.dumps(message, separators=(",", ":")).encode() + b"\n"
            stdin = self._process.stdin
            if stdin is None or self._process.poll() is not None:
                raise SidecarError("official WebShop worker exited", HTTPStatus.BAD_GATEWAY)
            try:
                stdin.write(encoded)
                stdin.flush()
                ready, _, _ = select.select((self._response_fd,), (), (), self.timeout_s)
                if not ready:
                    raise SidecarError("official WebShop worker timed out", HTTPStatus.GATEWAY_TIMEOUT)
                response = json.loads(self._read_line())
            except SidecarError:
                raise
            except (BrokenPipeError, OSError, ValueError) as exc:
                raise SidecarError(
                    f"official WebShop worker failed: {exc}", HTTPStatus.BAD_GATEWAY
                ) from exc
            if response.get("protocol_version") != _PROTOCOL:
                raise SidecarError("official WebShop protocol mismatch", HTTPStatus.BAD_GATEWAY)
            if response.get("request_id") != self._request_id:
                raise SidecarError("official WebShop response ID mismatch", HTTPStatus.BAD_GATEWAY)
            result = response.get("result")
            if not isinstance(result, dict):
                raise SidecarError("official WebShop response is invalid", HTTPStatus.BAD_GATEWAY)
            return result

    def _read_line(self) -> bytes:
        output = bytearray()
        while True:
            chunk = os.read(self._response_fd, 1)
            if not chunk:
                raise SidecarError("official WebShop worker returned no data", HTTPStatus.BAD_GATEWAY)
            if chunk == b"\n":
                return bytes(output)
            output.extend(chunk)
            if len(output) > _MAX_RESPONSE_BYTES:
                raise SidecarError("official WebShop response is too large", HTTPStatus.BAD_GATEWAY)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
        try:
            self.request("close", {})
        finally:
            self.abort()

    def abort(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            process = self._process
            stdin = process.stdin
            if stdin is not None:
                stdin.close()
            with suppress(OSError):
                os.close(self._response_fd)
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)


@dataclass
class ProductStore:
    path: Path
    _cache: dict[str, dict[str, Any]] = field(default_factory=dict, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def product(self, asin: str) -> dict[str, Any]:
        key = str(asin).upper()
        with self._lock:
            cached = self._cache.get(key)
        if cached is not None:
            return dict(cached)
        with sqlite3.connect(f"file:{self.path}?mode=ro", uri=True) as connection:
            row = connection.execute(
                "SELECT product_json FROM products INDEXED BY products_asin_uq WHERE asin = ?",
                (key,),
            ).fetchone()
        value = json.loads(row[0]) if row else {}
        if not isinstance(value, dict):
            value = {}
        with self._lock:
            if len(self._cache) >= 4096:
                self._cache.pop(next(iter(self._cache)))
            self._cache[key] = value
        return dict(value)


@dataclass
class WebShopSession:
    session_id: str
    worker: OfficialWorker
    products: ProductStore
    goal_id: str
    instruction: str = ""
    state_version: int = 0
    step_count: int = 0
    current_asin: str = ""
    selected_options: dict[str, str] = field(default_factory=dict)
    targets: dict[str, str] = field(default_factory=dict)
    commit_results: dict[str, dict[str, Any]] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def reset(self) -> dict[str, Any]:
        with self._lock:
            result = self.worker.request("reset", {})
            self.instruction = str(result.get("instruction_text", ""))
            return self._project(result, action_kind="reset")

    def search(self, query: str) -> dict[str, Any]:
        query = " ".join(str(query).split())
        if not query:
            raise SidecarError("query must be non-empty")
        with self._lock:
            self.current_asin = ""
            self.selected_options = {}
            result = self.worker.request("step", {"action": f"search[{query}]"})
            return self._project(result, action_kind="search", action_value=query)

    def click(self, target_id: str) -> dict[str, Any]:
        with self._lock:
            raw_action = self.targets.get(str(target_id))
            if raw_action is None:
                raise SidecarError("target_id is not valid in the current WebShop state")
            kind = str(target_id).split(":", 1)[0]
            returns_to_results = kind == "previous_page" and "buy now" in self.targets.values()
            self._track_click(kind, raw_action)
            result = self.worker.request("step", {"action": f"click[{raw_action}]"})
            if returns_to_results:
                self.current_asin = ""
                self.selected_options = {}
            return self._project(result, action_kind=kind, action_value=raw_action)

    def commit(self, target_id: str, commit_id: str) -> dict[str, Any]:
        with self._lock:
            cached = self.commit_results.get(commit_id)
            if cached is not None:
                return dict(cached)
            raw_action = self.targets.get(str(target_id))
            if raw_action != "buy now":
                raise SidecarError("commit target is not the current Buy Now action")
            result = self.worker.request("step", {"action": "click[buy now]"})
            projected = self._project(result, action_kind="purchase", action_value="buy now")
            self.commit_results[commit_id] = dict(projected)
            return projected

    def close(self) -> None:
        self.worker.close()

    def _track_click(self, kind: str, raw_action: str) -> None:
        if kind == "open_product":
            self.current_asin = raw_action.upper()
            # The official environment records only options explicitly clicked in
            # this session. Catalog display defaults are not purchase selections.
            self.selected_options = {}
        elif kind == "select_option":
            product = self.products.product(self.current_asin)
            for name, values in _option_groups(product).items():
                if any(str(item.get("value", "")).casefold() == raw_action.casefold() for item in values):
                    self.selected_options[name.casefold()] = raw_action
                    break
        elif kind == "back_to_search":
            self.current_asin = ""
            self.selected_options = {}

    def _project(
        self,
        result: dict[str, Any],
        *,
        action_kind: str,
        action_value: str = "",
    ) -> dict[str, Any]:
        actions = [str(value) for value in result.get("available_actions", [])]
        text = str(result.get("observation_text", ""))
        terminal = bool(result.get("terminal", False))
        self.state_version += 1
        if action_kind != "reset":
            self.step_count += 1
        page_type = self._page_type(actions, terminal)
        valid, targets = self._subactions(actions, text, page_type)
        self.targets = targets
        payload: dict[str, Any] = {
            "done": terminal,
            "page_text": text,
            "page_type": page_type,
            "purchased": terminal and action_kind == "purchase",
            "reward": float(result.get("reward", 0.0)),
            "state_version": self.state_version,
            "steps": self.step_count,
            "termination_reason": "purchase_completed" if terminal else "active",
            "valid_subactions": valid,
        }
        if action_kind != "reset":
            payload["action_effect"] = {"kind": action_kind, "value": action_value}
        if page_type in {"product", "product_section"} and self.current_asin:
            product = self.products.product(self.current_asin)
            payload.update(
                {
                    "product": {
                        "asin": self.current_asin,
                        **_public_price_fields(product),
                        "title": str(product.get("name") or product.get("Title") or ""),
                    },
                    "purchase_visible": "click[buy now]" in actions,
                    "selected_options": dict(self.selected_options),
                    "unselected_option_groups": [
                        name.casefold()
                        for name in _option_groups(product)
                        if name.casefold() not in self.selected_options
                    ],
                }
            )
        if terminal:
            payload["exact_success"] = payload["reward"] >= 1.0
        return payload

    def _page_type(self, actions: list[str], terminal: bool) -> str:
        if terminal:
            return "done"
        raw = {action.removeprefix("click[").removesuffix("]") for action in actions}
        if self.current_asin and "buy now" in raw:
            return "product"
        if self.current_asin:
            return "product_section"
        if any(_ASIN.fullmatch(value) for value in raw):
            return "search_results"
        return "search"

    def _subactions(
        self, actions: list[str], text: str, page_type: str
    ) -> tuple[list[dict[str, Any]], dict[str, str]]:
        output: list[dict[str, Any]] = []
        targets: dict[str, str] = {}
        search_products = _search_products(text)
        product = self.products.product(self.current_asin) if self.current_asin else {}
        option_lookup = {
            str(item.get("value", "")).casefold(): (name, str(item.get("value", "")))
            for name, values in _option_groups(product).items()
            for item in values
        }
        product_ordinal = 0
        for raw in actions:
            if raw == "search" or not raw.startswith("click["):
                continue
            value = raw[6:-1]
            lower = value.casefold()
            item: dict[str, Any]
            if _ASIN.fullmatch(value):
                product_ordinal += 1
                asin = value.upper()
                preview = search_products.get(asin, {})
                target = f"open_product:{product_ordinal}:{asin}"
                item = {
                    "asin": asin,
                    "kind": "open_product",
                    "label": str(preview.get("title", asin)),
                    "price": preview.get("price"),
                    **{key: preview[key] for key in ("price_min", "price_max", "price_text") if key in preview},
                    "target_id": target,
                    "title": str(preview.get("title", asin)),
                }
            elif lower == "buy now":
                target = f"purchase:{self.current_asin}"
                item = {"kind": "purchase", "label": "Buy Now", "target_id": target}
            elif lower in {"description", "features", "reviews"}:
                target = f"view_{lower}:{self.current_asin}"
                item = {"kind": "view_section", "label": value, "target_id": target}
            elif lower == "back to search":
                target = f"back_to_search:{self.state_version}"
                item = {"kind": "navigate", "label": value, "target_id": target}
            elif lower == "< prev":
                target = f"previous_page:{self.state_version}"
                item = {"kind": "navigate", "label": value, "target_id": target}
            elif lower == "next >":
                target = f"next_page:{self.state_version}"
                item = {"kind": "navigate", "label": value, "target_id": target}
            elif lower in option_lookup:
                name, public_value = option_lookup[lower]
                target = f"select_option:{name.casefold()}:{public_value}"
                item = {
                    "kind": "select_option",
                    "label": public_value,
                    "option_name": name.casefold(),
                    "option_value": public_value,
                    "selected": self.selected_options.get(name.casefold(), "").casefold()
                    == public_value.casefold(),
                    "target_id": target,
                }
            else:
                target = f"click:{len(output) + 1}:{value}"
                item = {"kind": "navigate", "label": value, "target_id": target}
            output.append(item)
            targets[target] = value
        return output, targets

def _option_groups(product: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    # The official environment renders normalized options, not catalog labels.
    normalized = product.get("options")
    if isinstance(normalized, dict):
        return {
            str(name): [{"value": str(value)} for value in values]
            for name, values in normalized.items()
            if isinstance(values, list)
        }
    value = product.get("customization_options", {})
    if not isinstance(value, dict):
        return {}
    return {
        str(name): [item for item in values if isinstance(item, dict)]
        for name, values in value.items()
        if isinstance(values, list)
    }


def _product_price(product: dict[str, Any]) -> float | None:
    return _public_price_fields(product)["price"]


def _public_price_fields(product: dict[str, Any]) -> dict[str, Any]:
    """Expose only public catalog prices, never the evaluator's sampled price."""
    pricing = product.get("pricing")
    text = str(product.get("Price", ""))
    values = (
        [float(value) for value in pricing if isinstance(value, (int, float)) and not isinstance(value, bool)]
        if isinstance(pricing, list) else []
    )
    if not values:
        values = [float(value) for value in _PRICE.findall(text)]
    if not values:
        return {"price": None}
    low, high = min(values), max(values)
    if low == high:
        return {"price": low}
    return {"price": None, "price_min": low, "price_max": high,
            "price_text": text or f"${low} to ${high}"}


def _search_products(text: str) -> dict[str, dict[str, Any]]:
    parts = [part.strip() for part in text.split("[SEP]") if part.strip()]
    output: dict[str, dict[str, Any]] = {}
    for index, part in enumerate(parts):
        if not _ASIN.fullmatch(part):
            continue
        title = parts[index + 1] if index + 1 < len(parts) else part
        price_text = parts[index + 2] if index + 2 < len(parts) else ""
        output[part.upper()] = {
            **_public_price_fields({"Price": price_text}),
            "title": title,
        }
    return output


@dataclass
class SidecarState:
    args: argparse.Namespace
    epoch: str = field(default_factory=lambda: uuid.uuid4().hex)
    sessions: dict[str, WebShopSession] = field(default_factory=dict)
    idempotency: OrderedDict[tuple[str, str, str], tuple[int, dict[str, Any]]] = field(
        default_factory=OrderedDict
    )
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    initializer_gate: threading.Semaphore = field(init=False, repr=False)
    products: ProductStore = field(init=False)
    goal_fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        self.initializer_gate = threading.Semaphore(self.args.max_initializers)
        self.products = ProductStore(self.args.store)
        digest = hashlib.sha256()
        for path in (self.args.goals, self.args.store, self.args.index):
            stat = path.stat()
            digest.update(f"{path}:{stat.st_size}:{stat.st_mtime_ns}".encode())
        self.goal_fingerprint = digest.hexdigest()

    def create_session(self, goal_id: str, seed: int) -> dict[str, Any]:
        match = _GOAL_ID.fullmatch(goal_id.strip())
        if match is None:
            raise SidecarError("goal_id must identify webshop/goal-N")
        with self.lock:
            if len(self.sessions) >= self.args.max_sessions:
                raise SidecarError("WebShop session capacity exhausted", HTTPStatus.TOO_MANY_REQUESTS)
        with self.initializer_gate:
            worker = OfficialWorker(
                interpreter=self.args.interpreter,
                worker_script=self.args.worker_script,
                source_root=self.args.source_root,
                source_revision=self.args.source_revision,
                store_path=self.args.store,
                goals_path=self.args.goals,
                index_path=self.args.index,
                goal_index=int(match.group(1)),
                seed=self.args.seed,
                timeout_s=self.args.worker_timeout,
                java_home=self.args.java_home,
            )
            session_id = uuid.uuid4().hex
            session = WebShopSession(session_id, worker, self.products, goal_id)
            try:
                payload = session.reset()
            except Exception:
                worker.abort()
                raise
        with self.lock:
            self.sessions[session_id] = session
        return {"session_id": session_id, **payload}

    def session(self, session_id: str) -> WebShopSession:
        with self.lock:
            session = self.sessions.get(session_id)
        if session is None:
            raise SidecarError("WebShop session does not exist", HTTPStatus.NOT_FOUND)
        return session

    def close_session(self, session_id: str) -> None:
        with self.lock:
            session = self.sessions.pop(session_id, None)
        if session is None:
            return
        session.close()

    def close_all(self) -> None:
        with self.lock:
            sessions = list(self.sessions.values())
            self.sessions.clear()
        for session in sessions:
            try:
                session.close()
            except Exception:
                session.worker.abort()


class WebShopSidecarServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], state: SidecarState) -> None:
        super().__init__(address, WebShopRequestHandler)
        self.state = state


class WebShopRequestHandler(BaseHTTPRequestHandler):
    server: WebShopSidecarServer

    def do_GET(self) -> None:  # noqa: N802
        if urlparse(self.path).path != "/health":
            self._send(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return
        state = self.server.state
        with state.lock:
            session_count = len(state.sessions)
        self._send(
            HTTPStatus.OK,
            {
                "goal_fingerprint": state.goal_fingerprint,
                "index_path": str(state.args.index.resolve()),
                "idempotency_protocol": _IDEMPOTENCY_PROTOCOL,
                "request_epoch": state.epoch,
                "session_count": session_count,
                "status": "ok",
            },
        )

    def do_POST(self) -> None:  # noqa: N802
        self._mutate("POST")

    def do_DELETE(self) -> None:  # noqa: N802
        self._mutate("DELETE")

    def _mutate(self, method: str) -> None:
        state = self.server.state
        path = urlparse(self.path).path
        epoch = self.headers.get("X-Webshop-Epoch", "")
        if epoch and epoch != state.epoch:
            self._send(HTTPStatus.CONFLICT, {"error": "request_epoch_mismatch"})
            return
        key = self.headers.get("Idempotency-Key", "")
        cache_key = (method, path, key)
        if key:
            with state.lock:
                cached = state.idempotency.get(cache_key)
            if cached is not None:
                self._send(HTTPStatus(cached[0]), cached[1])
                return
        try:
            payload = self._read_json()
            status, response = self._dispatch(method, path, payload)
        except SidecarError as exc:
            status, response = exc.status, {"error": str(exc)}
        except Exception as exc:  # noqa: BLE001
            status, response = HTTPStatus.INTERNAL_SERVER_ERROR, {
                "error": f"{type(exc).__name__}: {exc}"
            }
        if key:
            with state.lock:
                state.idempotency[cache_key] = (int(status), dict(response))
                while len(state.idempotency) > state.args.idempotency_cache_size:
                    state.idempotency.popitem(last=False)
        self._send(status, response)

    def _dispatch(
        self, method: str, path: str, payload: dict[str, Any]
    ) -> tuple[HTTPStatus, dict[str, Any]]:
        state = self.server.state
        if method == "POST" and path == "/sessions":
            goal_id = str(payload.get("goal_id", ""))
            seed = payload.get("seed", 0)
            if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
                raise SidecarError("seed must be a non-negative integer")
            return HTTPStatus.CREATED, state.create_session(goal_id, seed)
        match = re.fullmatch(r"/sessions/([a-f0-9]{32})(?:/(search|click|commit))?", path)
        if match is None:
            raise SidecarError("endpoint does not exist", HTTPStatus.NOT_FOUND)
        session_id, operation = match.groups()
        if method == "DELETE" and operation is None:
            state.close_session(session_id)
            return HTTPStatus.OK, {"closed": True}
        if method != "POST" or operation is None:
            raise SidecarError("method is not allowed", HTTPStatus.METHOD_NOT_ALLOWED)
        session = state.session(session_id)
        if operation == "search":
            return HTTPStatus.OK, session.search(str(payload.get("query", "")))
        if operation == "click":
            return HTTPStatus.OK, session.click(str(payload.get("target_id", "")))
        commit_id = str(payload.get("commit_id", "")).strip()
        if not commit_id:
            raise SidecarError("commit_id must be non-empty")
        return HTTPStatus.OK, session.commit(str(payload.get("target_id", "")), commit_id)

    def _read_json(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise SidecarError("invalid Content-Length") from exc
        if not 0 <= length <= 1_000_000:
            raise SidecarError("request body is too large", HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
        if length == 0:
            return {}
        try:
            value = json.loads(self.rfile.read(length))
        except (UnicodeDecodeError, ValueError) as exc:
            raise SidecarError("request body must be valid JSON") from exc
        if not isinstance(value, dict):
            raise SidecarError("request body must be a JSON object")
        return value

    def _send(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
        self.send_response(int(status))
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        print(f"webshop-sidecar {self.address_string()} {format % args}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve the pinned official WebShop environment")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8020)
    parser.add_argument("--interpreter", type=Path, required=True)
    parser.add_argument("--worker-script", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--goals", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--java-home", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--worker-timeout", type=float, default=180.0)
    parser.add_argument("--max-sessions", type=int, default=48)
    parser.add_argument("--max-initializers", type=int, default=4)
    parser.add_argument("--idempotency-cache-size", type=int, default=8192)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    files = (args.interpreter, args.worker_script, args.store, args.goals)
    directories = (args.source_root, args.index)
    if any(not path.is_file() for path in files):
        raise ValueError("WebShop sidecar file dependency is missing")
    if any(not path.is_dir() for path in directories):
        raise ValueError("WebShop sidecar directory dependency is missing")
    if args.java_home is not None and not args.java_home.is_dir():
        raise ValueError("WebShop sidecar JAVA_HOME is missing")
    if min(args.port, args.max_sessions, args.max_initializers, args.idempotency_cache_size) <= 0:
        raise ValueError("WebShop sidecar limits must be positive")
    if args.seed < 0:
        raise ValueError("WebShop sidecar seed must be non-negative")
    if args.worker_timeout <= 0:
        raise ValueError("WebShop sidecar worker timeout must be positive")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_args(args)
    state = SidecarState(args)
    server = WebShopSidecarServer((args.host, args.port), state)

    def stop(_signum: int, _frame: object) -> None:
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        server.serve_forever(poll_interval=0.25)
    finally:
        state.close_all()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
