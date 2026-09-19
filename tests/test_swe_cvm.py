from __future__ import annotations

from selfplay_graph_flowsteer.swebench import TencentCVMLease


class _FakeCVM:
    def __init__(self, state: str) -> None:
        self.current = state
        self.starts = 0
        self.stops = 0

    def state(self) -> str:
        return self.current

    def start(self, *, timeout_s: float, poll_s: float) -> None:
        self.starts += 1
        self.current = "running"

    def stop(self, *, timeout_s: float, poll_s: float) -> None:
        self.stops += 1
        self.current = "stopped"


def test_cvm_lease_shares_start_and_stops_after_last_user() -> None:
    client = _FakeCVM("stopped")
    lease = TencentCVMLease(client, timeout_s=10, poll_s=0.1, stop_when_idle=True)

    lease.acquire()
    lease.acquire()
    assert client.starts == 1
    assert client.stops == 0

    lease.release()
    assert client.stops == 0
    lease.release()
    assert client.stops == 1
    assert client.current == "stopped"


def test_cvm_lease_does_not_stop_server_it_found_running() -> None:
    client = _FakeCVM("running")
    lease = TencentCVMLease(client, timeout_s=10, poll_s=0.1, stop_when_idle=True)

    lease.acquire()
    lease.release()

    assert client.starts == 0
    assert client.stops == 0
