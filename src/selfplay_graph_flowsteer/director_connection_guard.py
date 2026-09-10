"""Conservative pre-send retries; durable endpoint quarantine, never answer repair."""

import errno
import hashlib
import json
import os
import socket
import threading
import time
import uuid
from contextlib import suppress
from functools import wraps
from pathlib import Path

ACTIVE = threading.local()


def director_guard(function):
    @wraps(function)
    def wrapped(self, messages, *args, **kwargs):
        root = os.environ.get("SPGFS_DIRECTOR_CONNECTION_GUARD_DIR")
        if not root or kwargs.get("role") != "graph-director":
            return function(self, messages, *args, **kwargs)
        return guarded_call(
            self.config.base_url, root, messages, lambda: function(self, messages, *args, **kwargs)
        )

    return wrapped


class DirectorServicePaused(RuntimeError):
    pass


def definitely_not_sent(error):
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        if isinstance(error, socket.gaierror):
            return True
        if isinstance(error, OSError) and error.errno == errno.ECONNREFUSED:
            return True
        error = error.__cause__ or error.__context__
    return False


def save(path, payload):
    # Requests contain task context: private permissions, no headers/API keys.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(payload, f, ensure_ascii=False, default=str)


def guarded_call(endpoint, root, messages, call):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    key = hashlib.sha256(endpoint.encode()).hexdigest()[:20]
    paused = root / (key + ".paused.json")
    incident = uuid.uuid4().hex
    started = time.monotonic()
    for attempt in range(1, 4):
        if paused.exists():
            save(
                root / (incident + ".blocked.json"),
                {
                    "status": "service_paused",
                    "messages": messages,
                    "pause_record": str(paused),
                    "attempt": attempt,
                },
            )
            raise DirectorServicePaused(
                "Director endpoint paused; successful probe required before resume"
            )
        try:
            ACTIVE.enabled = True
            return call()
        except Exception as error:
            safe = definitely_not_sent(error)
            record = {
                "status": "not_sent" if safe else "execution_state_uncertain",
                "attempt": attempt,
                "messages": messages,
                "exception_type": type(error).__name__,
                "elapsed_wall_s": time.monotonic() - started,
                "retry": safe and attempt < 3,
                "reward_assigned": False,
            }
            save(root / f"{incident}.{attempt}.json", record)
            if not safe:
                raise
            if attempt == 3:
                with suppress(FileExistsError):
                    save(
                        paused,
                        {
                            "incident": incident,
                            "status": "paused",
                            "reason": "three_confirmed_pre_send_failures",
                        },
                    )
                raise
            time.sleep(0.25)
        finally:
            ACTIVE.enabled = False


def resume_after_probe(endpoint, root, probe):
    """An actual successful probe is mandatory; failure leaves quarantine intact."""
    root = Path(root)
    key = hashlib.sha256(endpoint.encode()).hexdigest()[:20]
    paused = root / (key + ".paused.json")
    evidence = probe()
    if evidence is not True:
        raise DirectorServicePaused("Probe did not attest success")
    if paused.exists():
        paused.rename(root / (key + ".resumed." + uuid.uuid4().hex + ".json"))
