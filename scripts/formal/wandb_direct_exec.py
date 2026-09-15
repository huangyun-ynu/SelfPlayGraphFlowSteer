#!/usr/bin/env python3
"""Run a command with project-local DNS entries for W&B's static Go core."""

from __future__ import annotations

import argparse
import ipaddress
import os
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlparse

_READY_ENV = "SPGFS_WANDB_DIRECT_DNS_READY"
_DEFAULT_DNS_SERVER = "8.8.8.8"


def _dns_name(name: str) -> bytes:
    return b"".join(bytes((len(part),)) + part.encode("ascii") for part in name.split(".")) + b"\0"


def _skip_dns_name(packet: bytes, offset: int) -> int:
    while True:
        if offset >= len(packet):
            raise ValueError("truncated DNS name")
        length = packet[offset]
        if length & 0xC0 == 0xC0:
            if offset + 1 >= len(packet):
                raise ValueError("truncated DNS pointer")
            return offset + 2
        offset += 1
        if length == 0:
            return offset
        offset += length


def _resolve_ipv4(
    hostname: str, dns_server: str, timeout: float = 3.0, attempts: int = 5
) -> list[str]:
    packet = b""
    request_id = 0
    last_error: OSError | None = None
    for _ in range(attempts):
        request_id = int.from_bytes(os.urandom(2), "big")
        query = struct.pack("!HHHHHH", request_id, 0x0100, 1, 0, 0, 0)
        query += _dns_name(hostname) + struct.pack("!HH", 1, 1)
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                sock.settimeout(timeout)
                sock.sendto(query, (dns_server, 53))
                packet, _ = sock.recvfrom(65535)
            break
        except OSError as error:
            last_error = error
    else:
        raise RuntimeError(
            f"DNS server {dns_server} did not answer for {hostname} after {attempts} attempts"
        ) from last_error

    if len(packet) < 12:
        raise ValueError("truncated DNS response")
    response_id, flags, questions, answers, _, _ = struct.unpack("!HHHHHH", packet[:12])
    if response_id != request_id or flags & 0x000F:
        raise ValueError("invalid DNS response")
    offset = 12
    for _ in range(questions):
        offset = _skip_dns_name(packet, offset) + 4

    addresses: list[str] = []
    for _ in range(answers):
        offset = _skip_dns_name(packet, offset)
        if offset + 10 > len(packet):
            raise ValueError("truncated DNS answer")
        record_type, record_class, _, length = struct.unpack("!HHIH", packet[offset : offset + 10])
        offset += 10
        data = packet[offset : offset + length]
        offset += length
        if record_type == 1 and record_class == 1 and length == 4:
            addresses.append(socket.inet_ntoa(data))
    if not addresses:
        raise RuntimeError(f"DNS server {dns_server} returned no IPv4 address for {hostname}")
    return sorted(set(addresses))


def _wandb_hostname() -> str:
    base_url = os.environ.get("WANDB_BASE_URL", "https://api.wandb.ai")
    hostname = urlparse(base_url).hostname
    if not hostname:
        raise ValueError(f"WANDB_BASE_URL has no hostname: {base_url!r}")
    return hostname


def _resolve_systemd_ipv4(hostname: str, timeout: float = 20.0) -> list[str]:
    resolvectl = shutil.which("resolvectl")
    if not resolvectl:
        return []
    result = subprocess.run(
        [resolvectl, "query", "--legend=no", "--type=A", hostname],
        capture_output=True,
        check=False,
        text=True,
        timeout=timeout,
    )
    if result.returncode:
        return []
    addresses: list[str] = []
    for token in result.stdout.replace(":", " ").split():
        try:
            address = ipaddress.ip_address(token)
        except ValueError:
            continue
        if address.version == 4:
            addresses.append(str(address))
    return sorted(set(addresses))


def _resolve_for_hosts(hostname: str, dns_server: str) -> list[str]:
    addresses = _resolve_systemd_ipv4(hostname)
    return addresses or _resolve_ipv4(hostname, dns_server)


def _make_hosts_file(hostname: str, addresses: list[str]) -> Path:
    descriptor, raw_path = tempfile.mkstemp(prefix="spgfs-wandb-hosts-", text=True)
    path = Path(raw_path)
    with os.fdopen(descriptor, "w", encoding="utf-8") as temp:
        temp.write(Path("/etc/hosts").read_text(encoding="utf-8"))
        temp.write("\n# Project-local W&B direct DNS mapping.\n")
        for address in addresses:
            temp.write(f"{address} {hostname}\n")
    path.chmod(0o644)
    return path


def _inside_namespace(hosts_file: Path, command: list[str]) -> None:
    subprocess.run(["mount", "--bind", str(hosts_file), "/etc/hosts"], check=True)
    hosts_file.unlink(missing_ok=True)
    os.environ[_READY_ENV] = "1"
    os.execvpe(command[0], command, os.environ)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a command with direct W&B DNS isolated to its mount namespace."
    )
    parser.add_argument("--inside", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a command is required after --")

    if args.inside is not None:
        _inside_namespace(args.inside, command)
    if os.environ.get(_READY_ENV) == "1":
        os.execvpe(command[0], command, os.environ)

    unshare = shutil.which("unshare")
    if not unshare:
        raise RuntimeError("unshare is required for isolated W&B direct DNS")
    dns_server = os.environ.get("SPGFS_WANDB_DNS_SERVER", _DEFAULT_DNS_SERVER)
    ipaddress.ip_address(dns_server)
    hostname = _wandb_hostname()
    hosts_file = _make_hosts_file(hostname, _resolve_for_hosts(hostname, dns_server))
    try:
        os.execvpe(
            unshare,
            [
                unshare,
                "--user",
                "--map-root-user",
                "--mount",
                sys.executable,
                str(Path(__file__).resolve()),
                "--inside",
                str(hosts_file),
                "--",
                *command,
            ],
            os.environ,
        )
    except BaseException:
        hosts_file.unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    main()
