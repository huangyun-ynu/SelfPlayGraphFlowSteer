"""Live HTTP/Worker-tool concurrency probe; no model calls or benchmark labels."""
import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from http.cookiejar import CookieJar
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import HTTPCookieProcessor, Request, build_opener

from selfplay_graph_flowsteer.agent_tools import SearchServiceTool
from selfplay_graph_flowsteer.retrieval_service import RetrievalHandler, RetrievalServer
from selfplay_graph_flowsteer.wikipedia_retrieval_service import WikipediaIndex

QUERIES = [
    "Marie Curie", "Apollo 11", "Mount Everest", "Charles Darwin",
    "Great Barrier Reef", "Ada Lovelace", "Panama Canal", "Ludwig van Beethoven",
    "Rosetta Stone", "James Webb Space Telescope", "Machu Picchu", "Alan Turing",
    "Suez Canal", "Isaac Newton", "Lake Baikal", "Voyager 1",
    "Mona Lisa", "Antarctica", "Johannes Gutenberg", "International Space Station",
    "Galapagos Islands", "Nikola Tesla", "Angkor Wat", "Hubble Space Telescope",
]


class TrackedIndex(WikipediaIndex):
    def __init__(self, cache):
        super().__init__(cache)
        self.condition = threading.Condition()
        self.active = 0

    def search(self, query, top_k):
        with self.condition:
            self.active += 1
        try:
            return super().search(query, top_k)
        finally:
            with self.condition:
                self.active -= 1
                self.condition.notify_all()


class AuditedSession:
    def __init__(self, session, path):
        self.session = session
        self.path = path
        self.lock = threading.Lock()

    def open(self, request, **kwargs):
        started = time.monotonic()
        row = {}
        try:
            response = self.session.open(request, **kwargs)
            row["status"] = response.status
            return response
        except HTTPError as error:
            row.update(status=error.code, retry_after=error.headers.get("Retry-After"))
            raise
        except Exception as error:
            row["error_type"] = type(error).__name__
            raise
        finally:
            row.update(seconds=round(time.monotonic() - started, 3),
                       cookie_sent=request.has_header("Cookie"))
            with self.lock, self.path.open("a") as target:
                target.write(json.dumps(row) + "\n")


def login_session(path, user_agent):
    credentials = json.loads(path.read_text())
    session = build_opener(HTTPCookieProcessor(CookieJar()))
    api = "https://en.wikipedia.org/w/api.php"

    def call(params, post=False):
        encoded = urlencode({"format": "json", "formatversion": 2, **params}).encode()
        request = Request(api if post else api + "?" + encoded.decode(),
                          data=encoded if post else None,
                          headers={"User-Agent": user_agent, "Accept": "application/json"})
        with session.open(request, timeout=30) as response:
            return json.load(response)

    token = call({"action": "query", "meta": "tokens", "type": "login"})
    time.sleep(1)
    result = call({"action": "login", "lgname": credentials["username"],
                   "lgpassword": credentials["password"],
                   "lgtoken": token["query"]["tokens"]["logintoken"]}, post=True)
    if result.get("login", {}).get("result") != "Success":
        raise RuntimeError("Bot Password login did not succeed")
    time.sleep(1)
    user = call({"action": "query", "meta": "userinfo"}).get("query", {}).get("userinfo", {})
    if not user.get("id") or "anon" in user:
        raise RuntimeError("Authenticated session could not be verified")
    return session


def run_phase(url, output, phase, queries=QUERIES):
    barrier = threading.Barrier(24)

    def call(slot, query):
        tool = SearchServiceTool(service_url=url, top_k=5, timeout_s=120)
        barrier.wait()
        start = time.monotonic()
        row = {"slot": slot, "query": query}
        try:
            payload = json.loads(tool.execute({"query": query}))
            hits = payload["result"][0]
            row.update(ok=True, hits=len(hits),
                       nonempty=bool(hits) and all(h["document"].get("text") for h in hits))
            (output / f"{phase}-{slot:02d}.json").write_text(json.dumps(payload, ensure_ascii=False))
        except Exception as error:
            row.update(ok=False, error=str(error))
        row["seconds"] = round(time.monotonic() - start, 3)
        return row

    started = time.monotonic()
    rows = []
    with ThreadPoolExecutor(max_workers=24) as pool:
        futures = [pool.submit(call, i, q) for i, q in enumerate(queries)]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(json.dumps({"phase": phase, **row}), flush=True)
            (output / f"{phase}-progress.json").write_text(json.dumps(rows, indent=2))
    durations = sorted(r["seconds"] for r in rows)
    summary = {"phase": phase, "requests": 24, "successful": sum(r["ok"] for r in rows),
               "nonempty": sum(bool(r.get("nonempty")) for r in rows),
               "five_hits": sum(r.get("hits") == 5 for r in rows),
               "wall_seconds": round(time.monotonic() - started, 3),
               "p50_seconds": durations[11], "p95_seconds": durations[22],
               "max_seconds": durations[-1], "rows": sorted(rows, key=lambda r: r["slot"])}
    (output / f"{phase}.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cached-only-cache", type=Path)
    parser.add_argument("--credentials", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    index = TrackedIndex(args.cached_only_cache or args.output / "cache")
    if args.credentials:
        try:
            session = login_session(args.credentials, index.user_agent)
        except Exception as error:
            report = {"authenticated": False, "error_type": type(error).__name__,
                      "http_status": getattr(error, "code", None), "load_test_started": False}
            (args.output / "report.json").write_text(json.dumps(report, indent=2))
            print(json.dumps(report), flush=True)
            return
        index.opener = AuditedSession(session, args.output / "upstream.jsonl")
        print("Authenticated cookie session verified; starting 24 cold requests", flush=True)
    with RetrievalServer(("127.0.0.1", 0), RetrievalHandler) as server:
        server.index = index
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            url = f"http://127.0.0.1:{server.server_port}/retrieve"
            if args.cached_only_cache:
                available = [json.loads(path.read_text()) for path in args.cached_only_cache.glob("*.json")]
                queries = [row["query"] for row in available
                           if row.get("schema") == index.schema and row.get("top_k") == 5]
                if not queries:
                    raise ValueError("No matching cached queries")

                def reject_fetch(*args):
                    raise RuntimeError("Cache-only probe encountered a cache miss")

                index._fetch = reject_fetch
                report = run_phase(url, args.output, "cached_only",
                                   [queries[i % len(queries)] for i in range(24)])
                report.update(scope="24 cached HTTP/tool requests; no upstream access allowed",
                              unique_queries=min(24, len(queries)))
                (args.output / "report.json").write_text(json.dumps(report, indent=2))
                return
            cold = run_phase(url, args.output, "cold")
            print("Waiting for server requests to drain before cache phase", flush=True)
            with index.condition:
                drained = index.condition.wait_for(lambda: index.active == 0, timeout=300)
            cache_entries = len(list((args.output / "cache").glob("*.json")))
            warm = run_phase(url, args.output, "warm") if drained and cache_entries == 24 else None
            report = {"scope": "24 simultaneous retrieval slots only; not model/SOTA evaluation",
                      "authenticated": bool(args.credentials),
                      "top_k": 5, "client_timeout_s": 120,
                      "upstream_concurrency": index.upstream_concurrency,
                      "request_interval_s": index.request_interval_s,
                      "cold": cold, "warm": warm, "server_drained": drained,
                      "cache_entries_after_cold": cache_entries,
                      "warm_skipped_reason": None if warm else "Not all 24 queries cached, or requests still active"}
            (args.output / "report.json").write_text(json.dumps(report, indent=2))
            print(json.dumps({"report": str(args.output / "report.json"), "drained": drained}), flush=True)
        finally:
            server.shutdown()
            thread.join()


if __name__ == "__main__":
    main()
