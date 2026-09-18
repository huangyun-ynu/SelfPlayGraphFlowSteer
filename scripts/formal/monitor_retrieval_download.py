"""Record local download progress without modifying or restarting the downloader."""
import argparse
import fcntl
import json
import time
from collections import deque
from datetime import UTC, datetime
from pathlib import Path

from prepare_searchr1_retrieval import FILES


def file_size(path):
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return 0


def progress(root):
    result = {}
    for _, _, name, size, _ in FILES:
        final = file_size(root / name)
        prefix = file_size(root / (name + ".partial"))
        # Assembly temporarily duplicates chunks; never count the assembled copy.
        ranges = root / (name + ".ranges")
        chunks = {}
        for path in ranges.glob("*"):
            if path.suffix in (".chunk", ".tmp") and "-" in path.stem:
                chunks[path.stem] = max(chunks.get(path.stem, 0), file_size(path))
        result[name] = min(size, max(final, prefix + sum(chunks.values())))
    return result


def alive(pid):
    try:
        proc = Path(f"/proc/{pid}")
        return (proc.joinpath("stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
                and b"prepare_searchr1_retrieval.py" in proc.joinpath("cmdline").read_bytes())
    except FileNotFoundError:
        return False


def monitor(root, state, interval):
    pid = int((state / "retrieval-download-segmented.pid").read_text())
    total = sum(item[3] for item in FILES)
    history = deque()
    last_growth = time.monotonic()
    high_water = 0
    previous_alert = None
    with (state / "retrieval-download-speed.jsonl").open("a", buffering=1) as log:
        while True:
            now = time.monotonic()
            files = progress(root)
            downloaded = sum(files.values())
            if downloaded > high_water:
                last_growth, high_water = now, downloaded
            history.append((now, downloaded))
            while len(history) > 2 and now - history[1][0] >= 300:
                history.popleft()
            elapsed = now - history[0][0]
            average = max(0, downloaded - history[0][1]) / elapsed if elapsed else None
            recent = (max(0, downloaded - history[-2][1]) / (now - history[-2][0])
                      if len(history) > 1 else None)
            running = alive(pid)
            complete = (root / "manifest.json").exists() and downloaded == total
            status = ("complete" if complete else "exited_incomplete" if not running
                      else "postprocessing" if downloaded == total else "downloading")
            alert = None
            if status == "exited_incomplete":
                alert = "Downloader exited before completion"
            elif status == "downloading":
                if now - last_growth >= 300:
                    alert = "No net download progress for 5 minutes"
                elif elapsed >= 240 and average < 500_000:
                    alert = "Rolling download speed below 0.5 MB/s"
            row = {"time": datetime.now(UTC).isoformat(), "pid": pid,
                   "status": status, "files": files, "bytes": downloaded, "total_bytes": total,
                   "percent": downloaded / total * 100,
                   "recent_MB_s": recent / 1e6 if recent is not None else None,
                   "rolling_MB_s": average / 1e6 if average is not None else None,
                   "eta_download_hours": (total - downloaded) / average / 3600 if average else None,
                   "alert": alert}
            line = json.dumps(row)
            log.write(line + "\n")
            temporary = state / "retrieval-download-speed.tmp"
            temporary.write_text(json.dumps(row, indent=2) + "\n")
            temporary.replace(state / "retrieval-download-speed.json")
            if alert != previous_alert or status in ("complete", "exited_incomplete"):
                with (state / "retrieval-download-alerts.jsonl").open("a") as alerts:
                    alerts.write(line + "\n")
            previous_alert = alert
            print(line, flush=True)
            if status in ("complete", "exited_incomplete"):
                break
            time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval", type=float, default=60)
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("interval must be positive")
    state = Path(__file__).resolve().parents[2] / "state"
    with (state / "retrieval-download-monitor.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        monitor(state / "formal-data/retrieval/searchr1", state, args.interval)


if __name__ == "__main__":
    main()
