"""One synchronized wave of short gateway requests; no retries or secret logging."""
import argparse
import json
import math
import threading
import time
import tomllib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--concurrency", type=int, default=32)
    args = parser.parse_args()
    concurrency = args.concurrency
    if not 1 <= concurrency <= 512:
        parser.error("concurrency must be between 1 and 512")
    config = tomllib.loads(Path("configs/formal_training.toml").read_text())["runtimes"]["gpt_student"]
    key = Path(config["api_key_file"]).expanduser().read_text().strip()
    output = Path("state/api-probes") / time.strftime(f"student-concurrency{concurrency}-%Y%m%d-%H%M%S")
    output.mkdir(parents=True, exist_ok=False)
    barrier = threading.Barrier(concurrency)
    lock = threading.Lock()
    active = peak = 0

    def call(index):
        nonlocal active, peak
        expected = f"OK-{index:02d}"
        body = {"model": config["served_model"], "input": f"Reply with exactly {expected} and nothing else.",
                "reasoning": {"effort": "low"}, "max_output_tokens": 128, "store": False}
        request = Request(config["base_url"].rstrip("/") + "/responses",
                          data=json.dumps(body).encode(), headers={
                              "Authorization": "Bearer " + key,
                              "User-Agent": config["user_agent"], "Content-Type": "application/json"})
        opener = build_opener(ProxyHandler({}), NoRedirect())
        barrier.wait(timeout=30)
        started = time.monotonic()
        with lock:
            active += 1
            peak = max(peak, active)
        row = {"index": index, "ok": False, "http_status": None}
        try:
            with opener.open(request, timeout=config["timeout_s"]) as response:
                row["http_status"] = response.status
                payload = json.load(response)
            text = "".join(part.get("text", "") for item in payload.get("output", [])
                           if item.get("type") == "message" for part in item.get("content", [])
                           if part.get("type") == "output_text").strip()
            row.update(response_status=payload.get("status"), response_id=payload.get("id"),
                       exact_match=text == expected, usage=payload.get("usage", {}))
            row["ok"] = payload.get("status") == "completed" and text == expected
        except HTTPError as error:
            row.update(http_status=error.code, error_type="HTTPError")
            error.close()
        except Exception as error:
            row["error_type"] = type(error).__name__
            reason = getattr(error, "reason", None)
            if reason is not None:
                row["reason_type"] = type(reason).__name__
        finally:
            row["seconds"] = round(time.monotonic() - started, 3)
            with lock:
                active -= 1
        return row

    started = time.monotonic()
    rows = []
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(call, i) for i in range(concurrency)]
        with (output / "requests.jsonl").open("w") as target:
            for future in as_completed(futures):
                row = future.result()
                rows.append(row)
                target.write(json.dumps(row) + "\n")
                target.flush()
                print(json.dumps({k: row.get(k) for k in ("index", "ok", "http_status", "seconds", "error_type")}), flush=True)
    elapsed = time.monotonic() - started
    latencies = sorted(row["seconds"] for row in rows)
    summary = {"model": config["served_model"], "concurrency": concurrency, "requests": len(rows),
               "peak_client_inflight": peak, "passed": sum(row["ok"] for row in rows),
               "http_200": sum(row["http_status"] == 200 for row in rows),
               "http_429": sum(row["http_status"] == 429 for row in rows),
               "seconds": round(elapsed, 3), "retries": 0, "max_output_tokens": 128,
               "latency_p50_s": latencies[math.ceil(len(rows) * .5) - 1],
               "latency_p95_s": latencies[math.ceil(len(rows) * .95) - 1],
               "latency_max_s": max(latencies),
               "input_tokens": sum(row.get("usage", {}).get("input_tokens", 0) for row in rows),
               "output_tokens": sum(row.get("usage", {}).get("output_tokens", 0) for row in rows),
               "note": "One short-request wave; not a sustained or long-context workload test."}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"output": str(output), **summary}), flush=True)


if __name__ == "__main__":
    main()
