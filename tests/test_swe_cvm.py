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


def test_cvm_failed_start_cleans_up_instance_that_actually_started():
    import pytest
    client = _FakeCVM("stopped")
    def start(**kwargs):
        client.current = "running"
        raise TimeoutError("start response lost")
    client.start = start
    lease = TencentCVMLease(client, timeout_s=10, poll_s=0.1, stop_when_idle=True)
    with pytest.raises(TimeoutError, match="start response lost"):
        lease.acquire()
    assert client.current == "stopped"
    assert client.stops == 1
    assert lease._users == 0


def test_cvm_failed_stop_retains_ownership_for_cleanup_retry():
    import pytest
    client = _FakeCVM("stopped")
    lease = TencentCVMLease(client, timeout_s=10, poll_s=0.1, stop_when_idle=True)
    lease.acquire()
    original_stop = client.stop
    def fail(**kwargs):
        raise TimeoutError("stop response lost")
    client.stop = fail
    with pytest.raises(TimeoutError):
        lease.release()
    assert lease._started_by_us
    # A later acquisition must not mistake our own still-running VM for an
    # externally owned VM and discard cleanup responsibility.
    client.stop = original_stop
    lease.acquire()
    lease.release()
    assert client.current == "stopped"
    assert not lease._started_by_us


def test_already_stopped_cvm_still_requires_confirmed_stop_charging(monkeypatch):
    import pytest
    from selfplay_graph_flowsteer.swebench import TencentCVMClient, TencentCVMError
    client=object.__new__(TencentCVMClient)
    client.instance_id='fixture-instance'
    monkeypatch.setattr(client,'state',lambda:'stopped')
    calls=[]
    def request(action,payload):
        calls.append((action,payload))
        return {'InstanceSet':[{'StopChargingMode':'KEEP_CHARGING'}]}
    monkeypatch.setattr(client,'_request',request)
    with pytest.raises(TencentCVMError,match='without STOP_CHARGING'):
        client.stop(timeout_s=1,poll_s=0.1)
    assert [action for action,_ in calls]==['DescribeInstances']
    monkeypatch.setattr(client,'_request',lambda *a:{'InstanceSet':[{'StopChargingMode':'STOP_CHARGING'}]})
    client.stop(timeout_s=1,poll_s=0.1)
