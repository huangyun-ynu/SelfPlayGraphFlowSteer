"""Keep spare memory on one explicitly selected GPU for subsequent user work.

Run only on user request. SIGTERM releases this reservation without stopping
the model server. A small free-memory margin remains for runtime allocations.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--headroom-mib", type=int, default=1024)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--headroom-control", type=Path,
                        help="Optional file containing a live headroom value in MiB")
    args = parser.parse_args()
    if args.gpu < 0 or args.headroom_mib < 512:
        parser.error("GPU must be nonnegative and headroom must be at least 512 MiB")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import torch

    stopping = False

    def stop(_signum, _frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    chunk = 256 * 1024**2
    headroom = args.headroom_mib * 1024**2
    allocations = []
    args.state.parent.mkdir(parents=True, exist_ok=True)

    def save(status):
        free, total = torch.cuda.mem_get_info()
        payload = {
            "status": status, "pid": os.getpid(), "physical_gpu": args.gpu,
            "reserved_tensor_bytes": sum(x.numel() for x in allocations),
            "process_reserved_bytes": torch.cuda.memory_reserved(),
            "device_free_bytes": free, "device_total_bytes": total,
            "headroom_mib": args.headroom_mib, "updated_at_unix": time.time(),
            "release_command": f"kill -TERM {os.getpid()}",
        }
        temporary = args.state.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n")
        temporary.replace(args.state)

    try:
        torch.cuda.init()
        while not stopping:
            if args.headroom_control and args.headroom_control.exists():
                try:
                    requested = int(args.headroom_control.read_text().strip())
                    if requested >= 512:
                        args.headroom_mib = requested
                        headroom = requested * 1024**2
                except (OSError, ValueError):
                    pass  # Keep the last valid reservation on an incomplete write.
            free, _ = torch.cuda.mem_get_info()
            while allocations and free < headroom:
                allocations.pop()
                torch.cuda.empty_cache()
                free, _ = torch.cuda.mem_get_info()
            while not stopping and free >= headroom + chunk:
                try:
                    allocations.append(torch.empty(chunk, dtype=torch.uint8, device="cuda"))
                except torch.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    break
                free, _ = torch.cuda.mem_get_info()
            save("holding")
            time.sleep(5)
    finally:
        allocations.clear()
        torch.cuda.empty_cache()
        save("released")


if __name__ == "__main__":
    main()
