"""Low-volume Bot Password login probe; never persist credentials or session responses."""
from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from http.cookiejar import CookieJar
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import HTTPCookieProcessor, Request, build_opener

API = "https://en.wikipedia.org/w/api.php"


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--credentials", type=Path, default=root / "state/private/wikipedia-auth.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    credentials = json.loads(args.credentials.read_text())
    contact = (root / "state/private/wikipedia-contact.txt").read_text().strip()
    user_agent = f"SelfPlayGraphFlowSteer/0.1 ({contact})"
    cookies = CookieJar()
    session = build_opener(HTTPCookieProcessor(cookies))
    anonymous = build_opener()
    report = {"started_at": datetime.now(UTC).isoformat(),
              "scope": "One login and low-frequency search checks, not a concurrency test",
              "stages": [], "authenticated": False}

    def save():
        (args.output / "report.json").write_text(json.dumps(report, indent=2))

    def call(stage, params, *, post=False, opener=session):
        encoded = urlencode({"format": "json", "formatversion": 2, **params}).encode()
        request = Request(API if post else API + "?" + encoded.decode(),
                          data=encoded if post else None,
                          headers={"User-Agent": user_agent, "Accept": "application/json"})
        start = time.monotonic()
        row = {"stage": stage}
        try:
            with opener.open(request, timeout=30) as response:
                row["http_status"] = response.status
                row["retry_after"] = response.headers.get("Retry-After")
                payload = json.load(response)
            row["api_error_code"] = payload.get("error", {}).get("code")
        except HTTPError as error:
            row.update(http_status=error.code, retry_after=error.headers.get("Retry-After"))
            payload = None
        except (URLError, TimeoutError, OSError, ValueError) as error:
            row["error_type"] = type(error).__name__
            payload = None
        row["seconds"] = round(time.monotonic() - start, 3)
        report["stages"].append(row)
        print(json.dumps(row), flush=True)
        save()
        return payload

    token_payload = call("login_token", {"action": "query", "meta": "tokens", "type": "login"})
    token = (token_payload or {}).get("query", {}).get("tokens", {}).get("logintoken")
    if not token:
        report["outcome"] = "Could not obtain login token; credentials not submitted"
        save()
        return
    time.sleep(1)
    login = call("login", {"action": "login", "lgname": credentials["username"],
                           "lgpassword": credentials["password"], "lgtoken": token}, post=True)
    result = (login or {}).get("login", {}).get("result")
    report["login_result"] = result
    if result != "Success":
        report["outcome"] = "Login did not succeed; no further login attempts"
        save()
        print(json.dumps({"login_result": result}), flush=True)
        return
    time.sleep(1)
    user = call("verify_session", {"action": "query", "meta": "userinfo"})
    info = (user or {}).get("query", {}).get("userinfo", {})
    report["authenticated"] = bool(info.get("id", 0)) and "anon" not in info
    if not report["authenticated"]:
        report["outcome"] = "Could not verify authenticated cookie session"
        save()
        return
    for i, query in enumerate(["Ada Lovelace", "Apollo 11", "Marie Curie"]):
        time.sleep(2)
        payload = call(f"authenticated_search_{i + 1}", {
            "action": "query", "list": "search", "srsearch": query, "srlimit": 5})
        if payload is None or payload.get("error"):
            report["outcome"] = "Authenticated search failed; stopped without retry"
            save()
            return
        report["stages"][-1]["hits"] = len(payload.get("query", {}).get("search", []))
        save()
    time.sleep(2)
    payload = call("anonymous_comparison", {
        "action": "query", "list": "search", "srsearch": "Ada Lovelace", "srlimit": 5},
        opener=anonymous)
    if payload is not None:
        report["stages"][-1]["hits"] = len(payload.get("query", {}).get("search", []))
    report["outcome"] = "Low-volume authenticated search completed; no claim of higher quota"
    save()
    print(json.dumps({"authenticated": report["authenticated"], "outcome": report["outcome"]}), flush=True)


if __name__ == "__main__":
    main()
