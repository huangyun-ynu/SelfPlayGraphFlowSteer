from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import tempfile
import threading
import time
import uuid
from datetime import UTC, datetime
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .contracts import CodeArtifactRef
from .observability import TaskSpec, VerificationResult

SWE_ACTION_NAMES = (
    "swe_list",
    "swe_search",
    "swe_read",
    "swe_edit",
    "swe_apply_artifact",
    "swe_test",
    "swe_status",
)

_FORBIDDEN_PUBLIC_KEYS = frozenset(
    {
        "patch",
        "test_patch",
        "fail_to_pass",
        "pass_to_pass",
        "gold_patch",
        "gold_files",
        "code_files",
        "target_answers",
        "ads_target",
        "reference",
    }
)
_PUBLIC_METADATA_KEYS = frozenset(
    {
        "dataset",
        "instance_id",
        "source_id",
        "source_split",
        "experiment_split",
        "split_manifest_id",
        "split_seed",
        "repo",
        "base_commit",
        "version",
        "environment_setup_commit",
        "pool_id",
        "cluster_id",
        "difficulty_score",
        "rank_in_cluster",
        "cluster_size",
        "ads_sample_id",
        "ads_preprocessing",
        "reward_capable",
        "verifier",
        "validated_pool_entry",
        "validated_pool_version",
        "validated_pool_manifest_sha256",
        "validated_pool_sha256",
        "verifier_contract_version",
        "adapter_contract_version",
    }
)
_REPO_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{7,64}$")
_INSTANCE_RE = re.compile(r"^[A-Za-z0-9_.-]+__[A-Za-z0-9_.-]+-[0-9]+$")
_CACHE_FETCH_LOCKS: dict[Path, threading.Lock] = {}
_CACHE_FETCH_LOCKS_GUARD = threading.Lock()
_CVM_LEASES: dict[tuple[str, str, str, str, str], "TencentCVMLease"] = {}
_CVM_LEASES_GUARD = threading.Lock()


class SWEWorkspaceProvisioningError(RuntimeError):
    """A trusted SWE workspace could not be provisioned from its local cache."""


def _normalized_key(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().casefold()).strip("_")


def find_forbidden_swe_keys(value: object, *, prefix: str = "$") -> list[str]:
    """Return every public serialization path containing private evaluator data."""

    found: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{prefix}.{key}"
            normalized = _normalized_key(key)
            prefix_forbidden = _FORBIDDEN_PUBLIC_KEYS - {"patch"}
            if normalized in _FORBIDDEN_PUBLIC_KEYS or any(
                normalized.startswith(f"{forbidden}_") for forbidden in prefix_forbidden
            ):
                found.append(child_path)
            found.extend(find_forbidden_swe_keys(child, prefix=child_path))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            found.extend(find_forbidden_swe_keys(child, prefix=f"{prefix}[{index}]"))
    return found


def assert_public_swe_payload(value: object) -> None:
    leaks = find_forbidden_swe_keys(value)
    if leaks:
        raise ValueError("public SWE payload contains private fields: " + ", ".join(leaks))


def _validate_identity(instance_id: str, repo: str, base_commit: str) -> None:
    if not _INSTANCE_RE.fullmatch(instance_id):
        raise ValueError(f"invalid SWE instance_id: {instance_id!r}")
    if not _REPO_RE.fullmatch(repo) or repo.startswith((".", "/")):
        raise ValueError(f"invalid SWE repo: {repo!r}")
    if not _COMMIT_RE.fullmatch(base_commit):
        raise ValueError(f"invalid SWE base_commit: {base_commit!r}")


class TencentCVMError(RuntimeError):
    """A Tencent Cloud CVM lifecycle operation failed."""


class TencentCVMClient:
    """Minimal TC3-SHA256 CVM client used for the SWE verifier lease.

    The implementation intentionally uses the official HTTP API directly so
    enabling this feature does not add a cloud SDK (and its transitive
    dependencies) to every training environment.
    """

    def __init__(
        self,
        *,
        secret_id: str,
        secret_key: str,
        region: str,
        instance_id: str,
        endpoint: str = "cvm.tencentcloudapi.com",
        timeout_s: float = 30.0,
    ) -> None:
        self.secret_id = str(secret_id).strip()
        self.secret_key = str(secret_key).strip()
        self.region = str(region).strip()
        self.instance_id = str(instance_id).strip()
        self.endpoint = str(endpoint).strip()
        self.timeout_s = float(timeout_s)
        if not all((self.secret_id, self.secret_key, self.region, self.instance_id)):
            raise ValueError("Tencent CVM credentials, region, and instance_id are required")
        if not re.fullmatch(r"[A-Za-z0-9.-]+", self.endpoint):
            raise ValueError("invalid Tencent CVM API endpoint")
        if self.timeout_s <= 0:
            raise ValueError("Tencent CVM API timeout must be positive")

    @staticmethod
    def _hmac(key: bytes, value: str) -> bytes:
        import hmac

        return hmac.new(key, value.encode("utf-8"), hashlib.sha256).digest()

    def _request(self, action: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        timestamp = int(time.time())
        date = datetime.fromtimestamp(timestamp, UTC).strftime("%Y-%m-%d")
        service = "cvm"
        content_type = "application/json; charset=utf-8"
        payload_hash = hashlib.sha256(body).hexdigest()
        canonical_headers = f"content-type:{content_type}\nhost:{self.endpoint}\n"
        signed_headers = "content-type;host"
        canonical_request = "\n".join(
            ("POST", "/", "", canonical_headers, signed_headers, payload_hash)
        )
        credential_scope = f"{date}/{service}/tc3_request"
        string_to_sign = "\n".join(
            (
                "TC3-HMAC-SHA256",
                str(timestamp),
                credential_scope,
                hashlib.sha256(canonical_request.encode("utf-8")).hexdigest(),
            )
        )
        secret_date = self._hmac(("TC3" + self.secret_key).encode("utf-8"), date)
        secret_service = self._hmac(secret_date, service)
        secret_signing = self._hmac(secret_service, "tc3_request")
        import hmac

        signature = hmac.new(
            secret_signing, string_to_sign.encode("utf-8"), hashlib.sha256
        ).hexdigest()
        authorization = (
            "TC3-HMAC-SHA256 "
            f"Credential={self.secret_id}/{credential_scope}, "
            f"SignedHeaders={signed_headers}, Signature={signature}"
        )
        request = Request(
            f"https://{self.endpoint}",
            data=body,
            headers={
                "Authorization": authorization,
                "Content-Type": content_type,
                "Host": self.endpoint,
                "X-TC-Action": action,
                "X-TC-Version": "2017-03-12",
                "X-TC-Region": self.region,
                "X-TC-Timestamp": str(timestamp),
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_s) as response:
                result = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            raise TencentCVMError(f"{action} request failed: {type(exc).__name__}") from exc
        if not isinstance(result, dict):
            raise TencentCVMError(f"{action} returned a non-object response")
        error = result.get("Response", {}).get("Error")
        if error:
            code = str(error.get("Code", "unknown"))
            message = str(error.get("Message", ""))[:300]
            raise TencentCVMError(f"{action} failed: {code}: {message}")
        return result.get("Response", {})

    def state(self) -> str:
        response = self._request(
            "DescribeInstances", {"InstanceIds": [self.instance_id]}
        )
        instances = response.get("InstanceSet") or []
        if not instances:
            raise TencentCVMError("DescribeInstances returned no matching instance")
        state = str(instances[0].get("InstanceState", "")).strip().casefold()
        return state

    def wait_for(self, expected: str, *, timeout_s: float, poll_s: float) -> None:
        deadline = time.monotonic() + timeout_s
        expected = expected.casefold()
        while time.monotonic() < deadline:
            if self.state() == expected:
                return
            time.sleep(max(0.1, poll_s))
        raise TencentCVMError(f"CVM did not reach {expected} before timeout")

    def start(self, *, timeout_s: float, poll_s: float) -> None:
        state = self.state()
        if state == "running":
            return
        if state not in {"stopped", "stopping"}:
            if state == "starting":
                self.wait_for("running", timeout_s=timeout_s, poll_s=poll_s)
                return
            raise TencentCVMError(f"cannot start CVM from state {state or 'unknown'}")
        if state == "stopping":
            self.wait_for("stopped", timeout_s=timeout_s, poll_s=poll_s)
        self._request("StartInstances", {"InstanceIds": [self.instance_id]})
        self.wait_for("running", timeout_s=timeout_s, poll_s=poll_s)

    def stop(self, *, timeout_s: float, poll_s: float) -> None:
        state = self.state()
        if state == "stopped":
            return
        if state not in {"running", "starting"}:
            raise TencentCVMError(f"cannot stop CVM from state {state or 'unknown'}")
        if state == "starting":
            self.wait_for("running", timeout_s=timeout_s, poll_s=poll_s)
        self._request("StopInstances", {"InstanceIds": [self.instance_id]})
        self.wait_for("stopped", timeout_s=timeout_s, poll_s=poll_s)


class TencentCVMLease:
    """Process-shared reference-counted lease for concurrent SWE evaluations."""

    def __init__(
        self,
        client: TencentCVMClient,
        *,
        timeout_s: float,
        poll_s: float,
        stop_when_idle: bool,
    ) -> None:
        self.client = client
        self.timeout_s = float(timeout_s)
        self.poll_s = float(poll_s)
        self.stop_when_idle = bool(stop_when_idle)
        self._users = 0
        self._started_by_us = False
        self._lock = threading.RLock()

    def acquire(self) -> None:
        with self._lock:
            if self._users == 0:
                before = self.client.state()
                if before != "running":
                    self.client.start(timeout_s=self.timeout_s, poll_s=self.poll_s)
                self._started_by_us = before != "running"
            self._users += 1

    def release(self) -> None:
        with self._lock:
            if self._users <= 0:
                return
            self._users -= 1
            if self._users == 0 and self._started_by_us and self.stop_when_idle:
                try:
                    self.client.stop(timeout_s=self.timeout_s, poll_s=self.poll_s)
                finally:
                    self._started_by_us = False


def shared_tencent_cvm_lease(
    *,
    secret_id: str,
    secret_key: str,
    region: str,
    instance_id: str,
    endpoint: str,
    timeout_s: float,
    poll_s: float,
    stop_when_idle: bool = True,
) -> TencentCVMLease:
    key = (secret_id, region, instance_id, endpoint, str(bool(stop_when_idle)))
    with _CVM_LEASES_GUARD:
        lease = _CVM_LEASES.get(key)
        if lease is None:
            lease = TencentCVMLease(
                TencentCVMClient(
                    secret_id=secret_id,
                    secret_key=secret_key,
                    region=region,
                    instance_id=instance_id,
                    endpoint=endpoint,
                    timeout_s=min(timeout_s, 60.0),
                ),
                timeout_s=timeout_s,
                poll_s=poll_s,
                stop_when_idle=stop_when_idle,
            )
            _CVM_LEASES[key] = lease
        return lease


@dataclass(frozen=True)
class PublicSWETask:
    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    source_split: str
    version: str = ""
    environment_setup_commit: str = ""

    def __post_init__(self) -> None:
        _validate_identity(self.instance_id, self.repo, self.base_commit)
        if not self.problem_statement.strip():
            raise ValueError("SWE problem_statement cannot be empty")
        if self.source_split.casefold() not in {"train", "dev", "test", "verified"}:
            raise ValueError(f"unsupported SWE source_split: {self.source_split!r}")
        if self.environment_setup_commit and not _COMMIT_RE.fullmatch(
            self.environment_setup_commit
        ):
            raise ValueError("invalid SWE environment_setup_commit")

    def to_task_spec(self, *, reward_capable: bool) -> TaskSpec:
        metadata = {
            "dataset": "swe_bench",
            "instance_id": self.instance_id,
            "source_id": self.instance_id,
            "source_split": self.source_split,
            "repo": self.repo,
            "base_commit": self.base_commit,
            "version": self.version,
            "environment_setup_commit": self.environment_setup_commit,
            "reward_capable": bool(reward_capable),
            "verifier": "swe_outcome",
        }
        assert_public_swe_payload(metadata)
        return TaskSpec(
            task_id=f"swe_bench:{self.instance_id}",
            prompt=self.problem_statement,
            reference=None,
            task_type="swe_bench",
            metadata=metadata,
        )


@dataclass(frozen=True)
class PrivateSWEVerifierManifest:
    instance_id: str
    repo: str
    base_commit: str
    test_patch: str
    fail_to_pass: tuple[str, ...]
    pass_to_pass: tuple[str, ...]
    dataset_revision: str

    def __post_init__(self) -> None:
        _validate_identity(self.instance_id, self.repo, self.base_commit)
        if not self.test_patch.strip():
            raise ValueError("private SWE manifest requires test_patch")
        if not self.fail_to_pass:
            raise ValueError("private SWE manifest requires FAIL_TO_PASS tests")
        if not self.dataset_revision.strip():
            raise ValueError("private SWE manifest requires a pinned dataset revision")


@dataclass(frozen=True)
class RemoteSWEVerifierManifest:
    """Trusted identity attestation for a verifier that keeps tests remotely."""

    instance_id: str
    repo: str
    base_commit: str
    dataset_revision: str

    def __post_init__(self) -> None:
        _validate_identity(self.instance_id, self.repo, self.base_commit)
        if not _COMMIT_RE.fullmatch(self.dataset_revision):
            raise ValueError("remote SWE manifest requires a pinned dataset revision")


class TrustedSWEVerifierRegistry:
    """Local allowlist with identities only; no tests or gold localization data."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).resolve()
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        assert_public_swe_payload(payload)
        if not isinstance(payload, dict) or payload.get("schema_version") != 1:
            raise ValueError("invalid trusted SWE verifier registry schema")
        revision = str(payload.get("dataset_revision", ""))
        if not _COMMIT_RE.fullmatch(revision):
            raise ValueError("trusted SWE registry requires a pinned dataset revision")
        rows = payload.get("instances")
        if not isinstance(rows, list) or not rows:
            raise ValueError("trusted SWE registry requires at least one instance")
        manifests: dict[str, RemoteSWEVerifierManifest] = {}
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("trusted SWE registry instances must be objects")
            manifest = RemoteSWEVerifierManifest(
                instance_id=str(row.get("instance_id", "")),
                repo=str(row.get("repo", "")),
                base_commit=str(row.get("base_commit", "")),
                dataset_revision=revision,
            )
            if manifest.instance_id in manifests:
                raise ValueError(f"duplicate trusted SWE instance: {manifest.instance_id}")
            manifests[manifest.instance_id] = manifest
        self.dataset_revision = revision
        self._manifests = manifests

    def resolve(self, public: PublicSWETask) -> RemoteSWEVerifierManifest | None:
        manifest = self._manifests.get(public.instance_id)
        if manifest is None:
            return None
        if (manifest.repo, manifest.base_commit) != (public.repo, public.base_commit):
            raise ValueError("trusted SWE registry identity mismatch")
        return manifest


@dataclass(frozen=True)
class TrustedSWETaskBinding:
    public: PublicSWETask
    private: PrivateSWEVerifierManifest | RemoteSWEVerifierManifest | None = None

    def __post_init__(self) -> None:
        if self.private is None:
            return
        public_identity = (
            self.public.instance_id,
            self.public.repo,
            self.public.base_commit,
        )
        private_identity = (
            self.private.instance_id,
            self.private.repo,
            self.private.base_commit,
        )
        if public_identity != private_identity:
            raise ValueError("public/private SWE task identity mismatch")

    @property
    def reward_capable(self) -> bool:
        return self.private is not None


def sanitize_swe_pool_row(row: dict[str, Any]) -> dict[str, Any]:
    """Convert a legacy SWE pool row into the only metadata visible online."""

    metadata = dict(row.get("metadata") or {})
    instance_id = str(
        metadata.get("instance_id")
        or metadata.get("source_id")
        or row.get("instance_id")
        or str(row.get("id", "")).removeprefix("swe_bench:")
    ).strip()
    repo = str(metadata.get("repo", row.get("repo", ""))).strip()
    base_commit = str(metadata.get("base_commit", row.get("base_commit", ""))).strip()
    _validate_identity(instance_id, repo, base_commit)
    clean = {
        key: value
        for key, value in metadata.items()
        if _normalized_key(key) in _PUBLIC_METADATA_KEYS
    }
    clean.update(
        {
            "dataset": "swe_bench",
            "instance_id": instance_id,
            "source_id": instance_id,
            "source_split": str(metadata.get("source_split", row.get("split", "train"))),
            "repo": repo,
            "base_commit": base_commit,
            # The verifier name is a public capability selector, not private
            # test evidence. Preserve it so generic runners do not mistake a
            # SWE task for reference-answer code generation.
            "verifier": "swe_outcome",
            # A public pool row cannot self-authorize reward. The lifecycle
            # grants this only after a trusted private registry identity match.
            "reward_capable": False,
        }
    )
    assert_public_swe_payload(clean)
    return clean


def swe_task_is_training_split(metadata: dict[str, Any]) -> bool:
    """Return whether a SWE task belongs to an explicitly authorized train pool.

    Historical official-train rows retain their legacy behaviour.  Public
    Verified instances are training-authorized only when a deterministic
    internal split manifest marks them as train; an arbitrary Verified canary
    cannot self-authorize merely by changing one field.
    """

    source_split = str(metadata.get("source_split", "")).strip().casefold()
    if source_split == "train":
        return True
    return bool(
        source_split == "verified"
        and str(metadata.get("experiment_split", "")).strip().casefold() == "train"
        and str(metadata.get("split_manifest_id", "")).strip()
    )


def safe_repo_path(value: object) -> PurePosixPath:
    raw = str(value).strip().replace("\\", "/")
    if raw == ".":
        return PurePosixPath(".")
    path = PurePosixPath(raw)
    if not raw or path.is_absolute() or ".." in path.parts or ".git" in path.parts:
        raise SWEActionError("path_outside_workspace", "path must be repository-relative")
    if any(part in {"", "."} for part in path.parts):
        raise SWEActionError("invalid_path", "path contains an empty or dot component")
    return path


def _is_test_or_harness_path(path: PurePosixPath) -> bool:
    lowered = tuple(part.casefold() for part in path.parts)
    name = lowered[-1]
    return (
        any(part in {"test", "tests", "testing", "swebench", "harness"} for part in lowered)
        or name.startswith("test_")
        or name.endswith(("_test.py", ".patch"))
    )


class SWEActionError(ValueError):
    def __init__(self, code: str, message: str, *, details: dict[str, Any] | None = None) -> None:
        self.code = str(code)
        self.details = dict(details or {})
        super().__init__(message)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": "error",
            "error": {
                "code": self.code,
                "message": str(self),
                "details": self.details,
            },
        }


class CodeArtifactStore:
    """Content-addressed immutable patch store outside ephemeral workspaces."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def put(
        self,
        patch: bytes,
        *,
        instance_id: str,
        repo: str,
        base_commit: str,
        changed_files: tuple[str, ...],
    ) -> CodeArtifactRef:
        if not patch.strip():
            raise ValueError("cannot store an empty code patch")
        digest = hashlib.sha256(patch).hexdigest()
        target = self.root / f"{digest}.patch"
        with self._lock:
            if target.exists():
                if target.read_bytes() != patch:
                    raise RuntimeError("content-addressed patch collision")
            else:
                temporary = self.root / f".{digest}.{os.getpid()}.tmp"
                temporary.write_bytes(patch)
                temporary.replace(target)
        return CodeArtifactRef(
            artifact_sha256=digest,
            instance_id=instance_id,
            repo=repo,
            base_commit=base_commit,
            patch_bytes=len(patch),
            changed_files=changed_files,
        )

    def read(self, ref: CodeArtifactRef) -> bytes:
        payload = (self.root / f"{ref.artifact_sha256}.patch").read_bytes()
        if hashlib.sha256(payload).hexdigest() != ref.artifact_sha256:
            raise RuntimeError("stored code artifact failed integrity validation")
        return payload


@dataclass
class _WorkspaceAttempt:
    agent_id: str
    workspace: Path
    version: int = 0
    action_index: int = 0
    visible_artifacts: dict[str, CodeArtifactRef] = field(default_factory=dict)


class SWEHarnessBackend(Protocol):
    def evaluate(self, binding: TrustedSWETaskBinding, patch: bytes) -> dict[str, Any]: ...


class SSHSWEHarnessBackend:
    """Submit a sealed patch to the forced-command official verifier over SSH."""

    _HOST_RE = re.compile(r"^[A-Za-z0-9.-]+$")
    _USER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]*$")
    _RESPONSE_KEYS = frozenset(
        {
            "schema_version",
            "request_id",
            "instance_id",
            "repo",
            "base_commit",
            "dataset_revision",
            "status",
            "environment_completed",
            "official",
            "synthetic",
            "patch_applied",
            "fail_to_pass_complete",
            "pass_to_pass_complete",
            "f2p_success_count",
            "f2p_failure_count",
            "p2p_success_count",
            "p2p_failure_count",
            "detail",
        }
    )

    def __init__(
        self,
        *,
        host: str,
        user: str,
        identity_file: str | Path,
        known_hosts_file: str | Path,
        dataset_revision: str,
        connect_timeout_s: float = 10.0,
        request_timeout_s: float = 720.0,
        max_patch_bytes: int = 2_000_000,
        ssh_executable: str = "ssh",
        log_path: str | Path | None = None,
        server_lease: TencentCVMLease | None = None,
    ) -> None:
        self.host = str(host).strip()
        self.user = str(user).strip()
        self.identity_file = Path(identity_file).resolve()
        self.known_hosts_file = Path(known_hosts_file).resolve()
        self.dataset_revision = str(dataset_revision).strip().casefold()
        self.connect_timeout_s = float(connect_timeout_s)
        self.request_timeout_s = float(request_timeout_s)
        self.max_patch_bytes = int(max_patch_bytes)
        self.ssh_executable = str(ssh_executable)
        self.log_path = Path(log_path).resolve() if log_path is not None else None
        self.server_lease = server_lease
        if not self._HOST_RE.fullmatch(self.host):
            raise ValueError("invalid SWE verifier SSH host")
        if not self._USER_RE.fullmatch(self.user):
            raise ValueError("invalid SWE verifier SSH user")
        if not self.identity_file.is_file():
            raise ValueError("SWE verifier SSH identity file does not exist")
        if not self.known_hosts_file.is_file():
            raise ValueError("SWE verifier known_hosts file does not exist")
        if not _COMMIT_RE.fullmatch(self.dataset_revision):
            raise ValueError("SWE verifier requires a pinned dataset revision")
        if min(self.connect_timeout_s, self.request_timeout_s, self.max_patch_bytes) <= 0:
            raise ValueError("SWE verifier SSH timeouts and patch limit must be positive")

    def evaluate(self, binding: TrustedSWETaskBinding, patch: bytes) -> dict[str, Any]:
        request_id = uuid.uuid4().hex
        started = time.monotonic()
        private = binding.private
        if private is None or private.dataset_revision != self.dataset_revision:
            return self._infrastructure_result("untrusted_task_binding")
        if len(patch) > self.max_patch_bytes:
            return self._completed_failure("patch_apply_failed", "patch_too_large")
        request = {
            "schema_version": 1,
            "request_id": request_id,
            "instance_id": binding.public.instance_id,
            "repo": binding.public.repo,
            "base_commit": binding.public.base_commit,
            "dataset_revision": self.dataset_revision,
            "patch_b64": base64.b64encode(patch).decode("ascii"),
        }
        lease_acquired = False
        if self.server_lease is not None:
            try:
                self.server_lease.acquire()
                lease_acquired = True
            except (TencentCVMError, ValueError) as exc:
                result = self._infrastructure_result(
                    f"cvm_control_error:{type(exc).__name__}"
                )
                result["control_error"] = str(exc)[:500]
                self._log(request, patch, result, time.monotonic() - started)
                return result
        command = [
            self.ssh_executable,
            "-T",
            "-i",
            str(self.identity_file),
            "-o",
            "IdentitiesOnly=yes",
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=yes",
            "-o",
            f"UserKnownHostsFile={self.known_hosts_file}",
            "-o",
            f"ConnectTimeout={max(1, int(self.connect_timeout_s))}",
            "-o",
            "ServerAliveInterval=15",
            "-o",
            "ServerAliveCountMax=4",
            f"{self.user}@{self.host}",
        ]
        try:
            try:
                completed = subprocess.run(
                    command,
                    input=json.dumps(request, separators=(",", ":")).encode() + b"\n",
                    capture_output=True,
                    timeout=self.request_timeout_s,
                    check=False,
                    start_new_session=True,
                )
            except subprocess.TimeoutExpired:
                result = self._infrastructure_result("ssh_request_timeout", status="timeout")
                self._log(request, patch, result, time.monotonic() - started)
                return result
            except OSError as exc:
                result = self._infrastructure_result(f"ssh_process_error:{type(exc).__name__}")
                self._log(request, patch, result, time.monotonic() - started)
                return result
            if completed.returncode != 0:
                result = self._infrastructure_result(f"ssh_exit_{completed.returncode}")
                result["transport_stderr"] = completed.stderr.decode("utf-8", errors="replace")[-2000:]
                self._log(request, patch, result, time.monotonic() - started)
                return result
            if len(completed.stdout) > 100_000:
                result = self._infrastructure_result("oversized_verifier_response")
                self._log(request, patch, result, time.monotonic() - started)
                return result
            try:
                response = json.loads(completed.stdout)
                result = self._validated_response(request, response)
            except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
                result = self._infrastructure_result(f"invalid_verifier_response:{type(exc).__name__}")
            self._log(request, patch, result, time.monotonic() - started)
            return result
        finally:
            if lease_acquired:
                self.server_lease.release()

    def _validated_response(self, request: dict[str, Any], response: object) -> dict[str, Any]:
        if not isinstance(response, dict) or set(response) != self._RESPONSE_KEYS:
            raise ValueError("unexpected verifier response schema")
        identity_keys = (
            "request_id",
            "instance_id",
            "repo",
            "base_commit",
            "dataset_revision",
        )
        if response.get("schema_version") != 1 or any(
            response.get(key) != request[key] for key in identity_keys
        ):
            raise ValueError("verifier response identity mismatch")
        status = SWEEvaluationStatus(str(response.get("status", "")))
        if response.get("official") is not True or response.get("synthetic") is not False:
            raise ValueError("verifier response is not official outcome evidence")
        bool_keys = (
            "environment_completed",
            "patch_applied",
            "fail_to_pass_complete",
            "pass_to_pass_complete",
        )
        if any(not isinstance(response.get(key), bool) for key in bool_keys):
            raise ValueError("verifier response contains invalid booleans")
        count_keys = (
            "f2p_success_count",
            "f2p_failure_count",
            "p2p_success_count",
            "p2p_failure_count",
        )
        if any(
            not isinstance(response.get(key), int)
            or isinstance(response.get(key), bool)
            or not 0 <= int(response[key]) <= 100_000
            for key in count_keys
        ):
            raise ValueError("verifier response contains invalid result counts")
        result = dict(response)
        result["status"] = status.value
        result["detail"] = str(response.get("detail", ""))[:500]
        if status is SWEEvaluationStatus.RESOLVED and not (
            result["environment_completed"]
            and result["patch_applied"]
            and result["fail_to_pass_complete"]
            and result["pass_to_pass_complete"]
            and result["f2p_failure_count"] == 0
            and result["p2p_failure_count"] == 0
        ):
            raise ValueError("invalid resolved verifier response")
        if (
            status
            in {
                SWEEvaluationStatus.INFRASTRUCTURE_ERROR,
                SWEEvaluationStatus.TIMEOUT,
                SWEEvaluationStatus.CANCELLED,
            }
            and result["environment_completed"]
        ):
            raise ValueError("infrastructure status cannot be completed")
        return result

    @staticmethod
    def _infrastructure_result(
        detail: str, *, status: str = "infrastructure_error"
    ) -> dict[str, Any]:
        return {
            "status": status,
            "environment_completed": False,
            "official": True,
            "synthetic": False,
            "patch_applied": False,
            "fail_to_pass_complete": False,
            "pass_to_pass_complete": False,
            "f2p_success_count": 0,
            "f2p_failure_count": 0,
            "p2p_success_count": 0,
            "p2p_failure_count": 0,
            "detail": detail,
        }

    @staticmethod
    def _completed_failure(status: str, detail: str) -> dict[str, Any]:
        result = SSHSWEHarnessBackend._infrastructure_result(detail, status=status)
        result["environment_completed"] = True
        return result

    def _log(
        self,
        request: dict[str, Any],
        patch: bytes,
        result: dict[str, Any],
        elapsed_s: float,
    ) -> None:
        if self.log_path is None:
            return
        record = {
            "timestamp": time.time(),
            "request_id": request["request_id"],
            "instance_id": request["instance_id"],
            "repo": request["repo"],
            "base_commit": request["base_commit"],
            "dataset_revision": request["dataset_revision"],
            "patch_sha256": hashlib.sha256(patch).hexdigest(),
            "patch_bytes": len(patch),
            "status": result.get("status", "infrastructure_error"),
            "environment_completed": result.get("environment_completed", False),
            # This log is experiment-private.  Preserve the bounded harness
            # classification detail so an infrastructure_error can be repaired
            # without guessing whether SSH, image provisioning, patch setup, or
            # container execution failed.  Public evaluation payloads continue
            # to omit this field.
            "detail": str(result.get("detail", ""))[:500],
            "transport_stderr": str(result.get("transport_stderr", ""))[-2000:],
            "elapsed_s": round(elapsed_s, 6),
        }
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


class SWEWorkspaceLifecycle:
    """Offline lifecycle with the production isolation contract.

    This implementation is deliberately local-only and never downloads a
    repository. Production SWE execution must replace it with the official
    instance-image backend, while preserving this lifecycle contract.
    """

    def __init__(
        self,
        *,
        repo_cache_root: str | Path,
        workspace_root: str | Path,
        artifact_store: CodeArtifactStore,
        log_path: str | Path | None = None,
        test_profiles: dict[str, tuple[str, ...]] | None = None,
        test_timeout_s: float = 60.0,
        max_output_chars: int = 12_000,
        max_file_chars: int = 200_000,
        harness_backend: SWEHarnessBackend | None = None,
        verifier_registry: TrustedSWEVerifierRegistry | None = None,
    ) -> None:
        self.repo_cache_root = Path(repo_cache_root).resolve()
        self.workspace_root = Path(workspace_root).resolve()
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        self.artifact_store = artifact_store
        self.log_path = Path(log_path).resolve() if log_path is not None else None
        self.test_profiles = dict(test_profiles or {})
        self.test_timeout_s = float(test_timeout_s)
        self.max_output_chars = int(max_output_chars)
        self.max_file_chars = int(max_file_chars)
        self.harness_backend = harness_backend
        self.verifier_registry = verifier_registry
        self._binding: TrustedSWETaskBinding | None = None
        self._active: _WorkspaceAttempt | None = None
        self._results: dict[str, dict[str, Any]] = {}
        self._visible_artifacts: dict[str, CodeArtifactRef] = {}
        self._attempt_counter = 0
        self.created_workspace_count = 0
        self.cleaned_workspace_count = 0
        self.environment_fingerprint = ""

    def bind_task(
        self,
        task: TaskSpec,
        *,
        private: PrivateSWEVerifierManifest | RemoteSWEVerifierManifest | None = None,
    ) -> None:
        if str(task.metadata.get("dataset", "")).casefold() not in {
            "swe_bench",
            "swe-bench",
            "swebench",
        }:
            raise ValueError("SWE lifecycle can bind only a SWE-bench task")
        assert_public_swe_payload(task.metadata)
        public = PublicSWETask(
            instance_id=str(task.metadata.get("instance_id", task.metadata.get("source_id", ""))),
            repo=str(task.metadata.get("repo", "")),
            base_commit=str(task.metadata.get("base_commit", "")),
            problem_statement=task.prompt,
            source_split=str(task.metadata.get("source_split", "train")),
            version=str(task.metadata.get("version", "")),
            environment_setup_commit=str(task.metadata.get("environment_setup_commit", "")),
        )
        if private is None and self._binding is not None:
            existing = self._binding
            if (
                existing.public.instance_id,
                existing.public.repo,
                existing.public.base_commit,
            ) == (public.instance_id, public.repo, public.base_commit):
                private = existing.private
        if private is None and self.verifier_registry is not None:
            private = self.verifier_registry.resolve(public)
        if bool(task.metadata.get("reward_capable", False)) and private is None:
            raise ValueError("reward-capable SWE task is absent from trusted verifier registry")
        self._binding = TrustedSWETaskBinding(public, private)
        # Reward capability is granted only after the trusted local registry
        # matches the public identity; a pool row cannot self-authorize it.
        task.metadata["reward_capable"] = self._binding.reward_capable
        self.environment_fingerprint = hashlib.sha256(
            f"{public.instance_id}\0{public.repo}\0{public.base_commit}".encode()
        ).hexdigest()

    def set_visible_artifacts(self, refs: list[CodeArtifactRef]) -> None:
        self._visible_artifacts = {ref.artifact_sha256: ref for ref in refs}

    def begin_execution(self, *, agent_id: str, seed: int, revision: bool) -> dict[str, Any]:
        del seed
        if self._binding is None:
            raise RuntimeError("SWE lifecycle has no bound trusted task")
        if self._active is not None:
            raise RuntimeError("SWE lifecycle already has an active workspace")
        public = self._binding.public
        source = self._resolve_repo(public.repo)
        canonical_commit = self._verify_commit(source, public.base_commit)
        # A cache populated by ``git fetch <full-sha>`` can contain the
        # commit object without any branch or tag pointing at it.  A normal
        # clone copies refs, not arbitrary dangling objects, so the later
        # checkout would otherwise fail even though ``cat-file`` succeeds.
        # Keep an internal, immutable-by-convention cache ref for every
        # verified commit before cloning.  This does not expose an arbitrary
        # revision: ``canonical_commit`` was just resolved by git in the
        # trusted local cache and still has to match the bound task identity.
        self._ensure_cloneable_commit(source, canonical_commit)
        self._attempt_counter += 1
        attempt_name = (
            f"{public.instance_id}-{agent_id}-{self._attempt_counter}-"
            f"{'revision' if revision else 'initial'}-"
        )
        workspace = Path(tempfile.mkdtemp(prefix=attempt_name, dir=self.workspace_root))
        try:
            self._run_git(
                [
                    "clone",
                    "--quiet",
                    "--no-hardlinks",
                    "--no-checkout",
                    str(source),
                    str(workspace),
                ],
                cwd=self.workspace_root,
                timeout_s=60.0,
                safe_directories=(source,),
            )
            self._run_git(
                ["checkout", "--quiet", "--detach", canonical_commit],
                cwd=workspace,
                timeout_s=30.0,
            )
        except Exception:
            shutil.rmtree(workspace, ignore_errors=True)
            raise
        self.created_workspace_count += 1
        self._active = _WorkspaceAttempt(
            agent_id=agent_id,
            workspace=workspace,
            visible_artifacts=dict(self._visible_artifacts),
        )
        self._log(
            {
                "event": "workspace_created",
                "instance_id": public.instance_id,
                "agent_id": agent_id,
                "attempt": self._attempt_counter,
                "revision": bool(revision),
                "workspace": str(workspace),
                "base_commit": public.base_commit,
            }
        )
        return self.status()

    def end_execution(self) -> None:
        attempt = self._active
        if attempt is None or self._binding is None:
            return
        result: dict[str, Any] = {
            "adapter": "swe_bench",
            "workspace_completed": True,
            "environment_completed": False,
            "official": False,
            "synthetic": False,
            "workspace_version": attempt.version,
            "code_artifact_ref": None,
        }
        try:
            self._run_git(
                ["add", "--intent-to-add", "--", "."],
                cwd=attempt.workspace,
                timeout_s=10.0,
            )
            patch = self._git_bytes(
                ["diff", "--binary", "--no-ext-diff", "HEAD"], attempt.workspace
            )
            changed_files = self._changed_files(attempt.workspace)
            if patch.strip():
                ref = self.artifact_store.put(
                    patch,
                    instance_id=self._binding.public.instance_id,
                    repo=self._binding.public.repo,
                    base_commit=self._binding.public.base_commit,
                    changed_files=changed_files,
                )
                result["code_artifact_ref"] = ref.to_dict()
                result["patch_sha256"] = ref.artifact_sha256
                result["changed_files"] = list(ref.changed_files)
        except Exception as exc:  # noqa: BLE001 - preserve cleanup and mark infra failure
            result.update(
                {
                    "workspace_completed": False,
                    "infrastructure_error": type(exc).__name__,
                    "infrastructure_detail": str(exc)[:1000],
                }
            )
        finally:
            self._results[attempt.agent_id] = result
            try:
                self._cleanup_workspace(attempt.workspace)
            except Exception as exc:  # noqa: BLE001 - cleanup failure is infrastructure failure
                result.update(
                    {
                        "workspace_completed": False,
                        "infrastructure_error": "WorkspaceCleanupError",
                        "infrastructure_detail": str(exc)[:1000],
                        "orphan_workspace": str(attempt.workspace),
                    }
                )
                self._results[attempt.agent_id] = result
            finally:
                self._active = None
                self._visible_artifacts = {}

    def close_all(self) -> None:
        if self._active is not None:
            self.end_execution()

    def result_for(self, agent_id: str | None) -> dict[str, Any]:
        if not agent_id:
            return {}
        return dict(self._results.get(agent_id, {}))

    def evaluate_artifact(self, agent_id: str) -> dict[str, Any]:
        if self._binding is None:
            raise RuntimeError("SWE lifecycle has no bound task")
        if self.harness_backend is None:
            raise RuntimeError("SWE official harness backend is not configured")
        result = self._results.get(agent_id, {})
        raw_ref = result.get("code_artifact_ref")
        if not isinstance(raw_ref, dict):
            evaluation = dict(self.harness_backend.evaluate(self._binding, b""))
        else:
            ref = CodeArtifactRef.from_dict(raw_ref)
            evaluation = dict(
                self.harness_backend.evaluate(self._binding, self.artifact_store.read(ref))
            )
        self._results[agent_id] = {**result, "evaluation": evaluation}
        self._log(
            {
                "event": "private_harness_evaluation",
                "agent_id": agent_id,
                "evaluation": evaluation,
            }
        )
        return dict(evaluation)

    def status(self) -> dict[str, Any]:
        attempt = self._require_active()
        public = self._require_binding().public
        return {
            "status": "ok",
            "instance_id": public.instance_id,
            "repo": public.repo,
            "base_commit": public.base_commit,
            "workspace_version": attempt.version,
            "changed_files": list(self._changed_files(attempt.workspace)),
        }

    def list_files(
        self, path: object, *, workspace_version: object, max_entries: int
    ) -> dict[str, Any]:
        attempt = self._check_version(workspace_version)
        relative, target = self._resolve_path(path, require_exists=True)
        if not target.is_dir():
            raise SWEActionError("not_a_directory", f"{relative} is not a directory")
        entries = []
        for child in sorted(target.iterdir(), key=lambda value: value.name):
            if child.name == ".git":
                continue
            entries.append(
                {
                    "path": str(relative / child.name),
                    "kind": "directory" if child.is_dir() else "file",
                }
            )
            if len(entries) >= max_entries:
                break
        return self._ok(attempt, entries=entries, truncated=len(entries) >= max_entries)

    def search(
        self, query: object, path: object, *, workspace_version: object, max_results: int
    ) -> dict[str, Any]:
        attempt = self._check_version(workspace_version)
        needle = str(query)
        if not needle or len(needle) > 500:
            raise SWEActionError("invalid_query", "query must contain 1..500 characters")
        _, root = self._resolve_path(path, require_exists=True)
        candidates = [root] if root.is_file() else root.rglob("*")
        results: list[dict[str, Any]] = []
        for candidate in candidates:
            if len(results) >= max_results:
                break
            if not candidate.is_file() or ".git" in candidate.parts:
                continue
            try:
                if candidate.stat().st_size > self.max_file_chars:
                    continue
                lines = candidate.read_text(encoding="utf-8").splitlines()
            except (OSError, UnicodeDecodeError):
                continue
            for line_number, line in enumerate(lines, start=1):
                if needle in line:
                    results.append(
                        {
                            "path": candidate.relative_to(attempt.workspace).as_posix(),
                            "line": line_number,
                            "text": line[:500],
                        }
                    )
                    if len(results) >= max_results:
                        break
        return self._ok(attempt, matches=results, truncated=len(results) >= max_results)

    def read_file(
        self, path: object, *, workspace_version: object, start_line: int, end_line: int
    ) -> dict[str, Any]:
        attempt = self._check_version(workspace_version)
        relative, target = self._resolve_path(path, require_exists=True)
        if not target.is_file() or target.stat().st_size > self.max_file_chars:
            raise SWEActionError("file_not_readable", "file is absent, non-regular, or too large")
        try:
            lines = target.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError as exc:
            raise SWEActionError("binary_file", "binary files cannot be read") from exc
        start = max(1, int(start_line))
        end = min(len(lines), int(end_line))
        if end < start or end - start > 400:
            raise SWEActionError("invalid_line_range", "read range must contain at most 401 lines")
        content = "\n".join(lines[start - 1 : end])
        return self._ok(
            attempt,
            path=str(relative),
            start_line=start,
            end_line=end,
            content=content[: self.max_output_chars],
            file_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
            truncated=len(content) > self.max_output_chars,
        )

    def edit(self, arguments: dict[str, Any]) -> dict[str, Any]:
        attempt = self._check_version(arguments.get("workspace_version"))
        operation = str(arguments.get("operation", "")).strip().casefold()
        relative, target = self._resolve_path(
            arguments.get("path"), require_exists=operation not in {"create"}
        )
        if _is_test_or_harness_path(relative):
            raise SWEActionError("protected_path", "tests and evaluator files are read-only")
        expected_sha = str(arguments.get("expected_sha256", "")).strip()
        if operation in {"replace", "delete"}:
            actual_sha = hashlib.sha256(target.read_bytes()).hexdigest()
            if expected_sha != actual_sha:
                raise SWEActionError(
                    "stale_file_sha",
                    "file changed since it was read",
                    details={"expected": expected_sha, "actual": actual_sha},
                )
        if operation == "replace":
            old = arguments.get("old_content")
            new = arguments.get("new_content")
            if not isinstance(old, str) or not old:
                raise SWEActionError("invalid_edit", "replace requires non-empty old_content")
            if not isinstance(new, str):
                raise SWEActionError("invalid_edit", "replace requires string new_content")
            content = target.read_text(encoding="utf-8")
            if content.count(old) != 1:
                raise SWEActionError("ambiguous_edit", "old_content must match exactly once")
            updated = content.replace(old, new, 1)
            if len(updated) > self.max_file_chars:
                raise SWEActionError("file_too_large", "edited file exceeds the size limit")
            target.write_text(updated, encoding="utf-8")
        elif operation == "create":
            if target.exists():
                raise SWEActionError("path_exists", "create target already exists")
            content = arguments.get("new_content")
            if not isinstance(content, str) or len(content) > self.max_file_chars:
                raise SWEActionError("invalid_edit", "create requires bounded string new_content")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        elif operation == "delete":
            target.unlink()
        elif operation == "restore":
            self._run_git(["restore", "--", str(relative)], cwd=attempt.workspace, timeout_s=10.0)
        else:
            raise SWEActionError(
                "invalid_operation", "operation must be replace/create/delete/restore"
            )
        attempt.version += 1
        return self._ok(
            attempt,
            path=str(relative),
            operation=operation,
            changed_files=list(self._changed_files(attempt.workspace)),
        )

    def apply_artifact(
        self, artifact_sha256: object, *, workspace_version: object
    ) -> dict[str, Any]:
        attempt = self._check_version(workspace_version)
        digest = str(artifact_sha256).strip().casefold()
        ref = attempt.visible_artifacts.get(digest)
        if ref is None:
            raise SWEActionError(
                "artifact_not_visible", "artifact is not from a visible upstream packet"
            )
        binding = self._require_binding().public
        if (ref.instance_id, ref.repo, ref.base_commit) != (
            binding.instance_id,
            binding.repo,
            binding.base_commit,
        ):
            raise SWEActionError(
                "artifact_binding_mismatch", "artifact belongs to another task or base commit"
            )
        patch = self.artifact_store.read(ref)
        completed = self._run_process(
            ["git", "apply", "--index", "--whitespace=nowarn", "-"],
            cwd=attempt.workspace,
            timeout_s=20.0,
            input_bytes=patch,
        )
        if completed[0] != 0:
            self._run_git(["reset", "--quiet"], cwd=attempt.workspace, timeout_s=10.0)
            raise SWEActionError(
                "patch_conflict",
                "upstream patch did not apply atomically",
                details={"stderr": completed[2][:1000]},
            )
        # Leave modifications unstaged; the index was used solely to obtain
        # git apply's atomic all-or-nothing behavior.
        self._run_git(["reset", "--quiet"], cwd=attempt.workspace, timeout_s=10.0)
        attempt.version += 1
        return self._ok(
            attempt,
            applied_artifact_sha256=digest,
            changed_files=list(self._changed_files(attempt.workspace)),
        )

    def test(self, profile: object, target: object, *, workspace_version: object) -> dict[str, Any]:
        attempt = self._check_version(workspace_version)
        profile_name = str(profile).strip()
        command = self.test_profiles.get(profile_name)
        if command is None:
            raise SWEActionError(
                "test_profile_not_allowed",
                "select a runtime-configured test profile",
                details={"allowed_profiles": sorted(self.test_profiles)},
            )
        resolved_command = list(command)
        if target not in (None, ""):
            relative, _ = self._resolve_path(target, require_exists=True)
            resolved_command.append(str(relative))
        returncode, stdout, stderr, timed_out = self._run_process(
            resolved_command,
            cwd=attempt.workspace,
            timeout_s=self.test_timeout_s,
        )
        return self._ok(
            attempt,
            profile=profile_name,
            target=str(target or ""),
            returncode=returncode,
            timed_out=timed_out,
            stdout=stdout[: self.max_output_chars],
            stderr=stderr[: self.max_output_chars],
            truncated=len(stdout) > self.max_output_chars or len(stderr) > self.max_output_chars,
        )

    def _ok(self, attempt: _WorkspaceAttempt, **payload: Any) -> dict[str, Any]:
        attempt.action_index += 1
        result = {
            "status": "ok",
            "workspace_version": attempt.version,
            "action_index": attempt.action_index,
            **payload,
        }
        self._log(
            {
                "event": "action",
                "agent_id": attempt.agent_id,
                "workspace_version": attempt.version,
                "action_index": attempt.action_index,
                "result": result,
            }
        )
        return result

    def _resolve_repo(self, repo: str) -> Path:
        relative = PurePosixPath(*repo.split("/"))
        candidate = (self.repo_cache_root / Path(*relative.parts)).resolve()
        if self.repo_cache_root not in candidate.parents or not candidate.is_dir():
            raise RuntimeError(f"trusted local repository cache is missing {repo}")
        return candidate

    def _verify_commit(self, source: Path, commit: str) -> str:
        canonical_commit = self._resolve_cached_commit(source, commit)
        if canonical_commit:
            return canonical_commit

        # A curriculum can contain several cycles, while a preflight may have
        # populated only the first cycle's commits.  Fetching this *already
        # trusted, bound* object from the cache's configured origin is safe and
        # avoids turning a cache omission into five simultaneous failed workers.
        # The per-repository lock also prevents concurrent sibling rollouts from
        # racing on git's ref and packed-refs locks.
        with self._cache_fetch_lock(source):
            canonical_commit = self._resolve_cached_commit(source, commit)
            if canonical_commit:
                return canonical_commit
            completed = subprocess.run(
                [
                    "git",
                    "-c",
                    f"safe.directory={source}",
                    "-C",
                    str(source),
                    "fetch",
                    "--no-tags",
                    "origin",
                    commit,
                ],
                capture_output=True,
                text=True,
                timeout=120.0,
                check=False,
            )
            canonical_commit = self._resolve_cached_commit(source, commit)
            if completed.returncode == 0 and canonical_commit:
                return canonical_commit
            detail = (completed.stderr or completed.stdout or "git fetch failed").strip()
            raise SWEWorkspaceProvisioningError(
                f"trusted base commit is missing and could not be provisioned: {commit}; {detail[:500]}"
            )

    @staticmethod
    def _resolve_cached_commit(source: Path, commit: str) -> str:
        completed = subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={source}",
                "-C",
                str(source),
                "rev-parse",
                "--verify",
                f"{commit}^{{commit}}",
            ],
            capture_output=True,
            text=True,
            timeout=10.0,
            check=False,
        )
        canonical_commit = completed.stdout.strip()
        return canonical_commit if completed.returncode == 0 and canonical_commit else ""

    @staticmethod
    def _cache_fetch_lock(source: Path) -> threading.Lock:
        with _CACHE_FETCH_LOCKS_GUARD:
            return _CACHE_FETCH_LOCKS.setdefault(source.resolve(), threading.Lock())

    def _ensure_cloneable_commit(self, source: Path, commit: str) -> None:
        """Make a verified cache object reachable to an isolated clone.

        The namespace is deliberately a local branch because ``git clone``
        transfers branch/tag refs by default, whereas custom namespaces are
        not guaranteed to be copied.  The full object id makes the operation
        idempotent and prevents two cached commits from sharing a ref.
        """
        self._run_git(
            ["update-ref", f"refs/heads/spgfs-cache-{commit}", commit],
            cwd=source,
            timeout_s=10.0,
        )

    def _resolve_path(self, value: object, *, require_exists: bool) -> tuple[PurePosixPath, Path]:
        attempt = self._require_active()
        relative = safe_repo_path(value)
        target = attempt.workspace.joinpath(*relative.parts)
        parent = target if target.exists() else target.parent
        resolved_parent = parent.resolve()
        if (
            attempt.workspace != resolved_parent
            and attempt.workspace not in resolved_parent.parents
        ):
            raise SWEActionError("symlink_escape", "path resolves outside the workspace")
        if require_exists and not target.exists():
            raise SWEActionError("path_not_found", f"path does not exist: {relative}")
        if target.is_symlink():
            resolved = target.resolve()
            if attempt.workspace != resolved and attempt.workspace not in resolved.parents:
                raise SWEActionError("symlink_escape", "symlink resolves outside the workspace")
        return relative, target

    def _check_version(self, value: object) -> _WorkspaceAttempt:
        attempt = self._require_active()
        try:
            requested = int(value)
        except (TypeError, ValueError) as exc:
            raise SWEActionError(
                "invalid_workspace_version", "workspace_version must be an integer"
            ) from exc
        if requested != attempt.version:
            raise SWEActionError(
                "stale_workspace_version",
                f"expected workspace version {attempt.version}, got {requested}",
                details={"current_workspace_version": attempt.version},
            )
        return attempt

    def _require_active(self) -> _WorkspaceAttempt:
        if self._active is None:
            raise SWEActionError("workspace_not_active", "no SWE workspace is active")
        return self._active

    def _require_binding(self) -> TrustedSWETaskBinding:
        if self._binding is None:
            raise RuntimeError("SWE lifecycle has no bound task")
        return self._binding

    def _changed_files(self, workspace: Path) -> tuple[str, ...]:
        output = self._git_bytes(
            ["status", "--porcelain=v1", "-z", "--untracked-files=all"], workspace
        )
        fields = output.decode("utf-8", errors="replace").split("\0")
        paths: list[str] = []
        index = 0
        while index < len(fields):
            field = fields[index]
            if not field:
                break
            code = field[:2]
            path = field[3:]
            if code[0] in {"R", "C"}:
                index += 1
                if index < len(fields):
                    path = fields[index]
            paths.append(path)
            index += 1
        return tuple(sorted(dict.fromkeys(paths)))

    def _cleanup_workspace(self, workspace: Path) -> None:
        if self.workspace_root not in workspace.resolve().parents:
            raise RuntimeError("refusing to clean a workspace outside the configured root")
        shutil.rmtree(workspace, ignore_errors=False)
        self.cleaned_workspace_count += 1
        self._log({"event": "workspace_cleaned", "workspace": str(workspace)})

    def orphan_workspaces(self) -> tuple[str, ...]:
        return tuple(sorted(str(path) for path in self.workspace_root.iterdir() if path.is_dir()))

    def _run_git(
        self,
        arguments: list[str],
        *,
        cwd: Path,
        timeout_s: float,
        safe_directories: tuple[Path, ...] = (),
    ) -> None:
        trusted_paths = (cwd.resolve(), *(path.resolve() for path in safe_directories))
        git_environment = {
            "GIT_CONFIG_COUNT": "1",
            "GIT_CONFIG_KEY_0": "safe.directory",
            "GIT_CONFIG_VALUE_0": str(trusted_paths[0]),
        }
        temporary_global_config: Path | None = None
        if safe_directories:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                prefix="spgfs-git-safe-",
                suffix=".config",
                dir=self.workspace_root,
                delete=False,
            ) as handle:
                handle.write("[safe]\n")
                for path in trusted_paths:
                    handle.write(f"\tdirectory = {json.dumps(str(path))}\n")
                temporary_global_config = Path(handle.name)
            git_environment["GIT_CONFIG_GLOBAL"] = str(temporary_global_config)
        try:
            returncode, _stdout, stderr, timed_out = self._run_process(
                ["git", *arguments],
                cwd=cwd,
                timeout_s=timeout_s,
                environment_overrides=git_environment,
            )
        finally:
            if temporary_global_config is not None:
                temporary_global_config.unlink(missing_ok=True)
        if timed_out or returncode != 0:
            raise RuntimeError(f"git operation failed: {stderr[:1000]}")

    def _git_bytes(self, arguments: list[str], cwd: Path) -> bytes:
        completed = subprocess.run(
            ["git", "-c", f"safe.directory={cwd.resolve()}", *arguments],
            cwd=cwd,
            capture_output=True,
            timeout=20.0,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr.decode("utf-8", errors="replace")[:1000])
        return completed.stdout

    @staticmethod
    def _run_process(
        command: list[str],
        *,
        cwd: Path,
        timeout_s: float,
        input_bytes: bytes | None = None,
        environment_overrides: dict[str, str] | None = None,
    ) -> tuple[int, str, str, bool]:
        environment = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "PYTHONIOENCODING": "utf-8",
            "NO_PROXY": "*",
            "no_proxy": "*",
            "HTTP_PROXY": "http://127.0.0.1:9",
            "HTTPS_PROXY": "http://127.0.0.1:9",
        }
        environment.update(environment_overrides or {})
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=environment,
            stdin=subprocess.PIPE if input_bytes is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        try:
            stdout, stderr = process.communicate(input=input_bytes, timeout=timeout_s)
            timed_out = False
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate()
            timed_out = True
        return (
            int(process.returncode),
            stdout.decode("utf-8", errors="replace"),
            stderr.decode("utf-8", errors="replace"),
            timed_out,
        )

    def _log(self, payload: dict[str, Any]) -> None:
        if self.log_path is None:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        record = {"timestamp": time.time(), **payload}
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


@dataclass(frozen=True)
class _SWEToolBase:
    lifecycle: SWEWorkspaceLifecycle
    stateful: bool = True

    def _execute(self, operation: Any) -> str:
        try:
            return json.dumps(operation(), ensure_ascii=False)
        except SWEActionError as exc:
            return json.dumps(exc.to_dict(), ensure_ascii=False)


@dataclass(frozen=True)
class SWEListTool(_SWEToolBase):
    name: str = "swe_list"
    description: str = (
        "List one repository directory in the current isolated workspace. Repeating the same "
        "path at an unchanged workspace_version yields no new evidence and is rejected."
    )
    parameters: dict[str, Any] = field(
        default_factory=lambda: _schema(
            {"path": {"type": "string"}, "workspace_version": {"type": "integer", "minimum": 0}},
            ("path", "workspace_version"),
        )
    )

    def execute(self, arguments: dict[str, Any]) -> str:
        return self._execute(
            lambda: self.lifecycle.list_files(
                arguments["path"], workspace_version=arguments["workspace_version"], max_entries=200
            )
        )


@dataclass(frozen=True)
class SWESearchTool(_SWEToolBase):
    name: str = "swe_search"
    description: str = (
        "Search literal source text under one repository-relative path. Use a new query or path "
        "to advance localization; identical searches at one workspace_version are rejected."
    )
    parameters: dict[str, Any] = field(
        default_factory=lambda: _schema(
            {
                "query": {"type": "string", "minLength": 1, "maxLength": 500},
                "path": {"type": "string"},
                "workspace_version": {"type": "integer", "minimum": 0},
            },
            ("query", "path", "workspace_version"),
        )
    )

    def execute(self, arguments: dict[str, Any]) -> str:
        return self._execute(
            lambda: self.lifecycle.search(
                arguments["query"],
                arguments["path"],
                workspace_version=arguments["workspace_version"],
                max_results=100,
            )
        )


@dataclass(frozen=True)
class SWEReadTool(_SWEToolBase):
    name: str = "swe_read"
    description: str = (
        "Read a bounded line range and return its SHA-256 for guarded editing. Use the returned "
        "file_sha256 as expected_sha256 for replace/delete; identical reads are rejected until "
        "the workspace version changes."
    )
    parameters: dict[str, Any] = field(
        default_factory=lambda: _schema(
            {
                "path": {"type": "string"},
                "start_line": {"type": "integer", "minimum": 1},
                "end_line": {"type": "integer", "minimum": 1},
                "workspace_version": {"type": "integer", "minimum": 0},
            },
            ("path", "start_line", "end_line", "workspace_version"),
        )
    )

    def execute(self, arguments: dict[str, Any]) -> str:
        return self._execute(
            lambda: self.lifecycle.read_file(
                arguments["path"],
                workspace_version=arguments["workspace_version"],
                start_line=arguments["start_line"],
                end_line=arguments["end_line"],
            )
        )


@dataclass(frozen=True)
class SWEEditTool(_SWEToolBase):
    name: str = "swe_edit"
    description: str = (
        "Atomically replace/create/delete/restore one non-test repository file. replace/delete "
        "require the exact expected_sha256 returned by swe_read; replace also requires one "
        "exact old_content match and new_content. A successful edit advances workspace_version."
    )
    parameters: dict[str, Any] = field(
        default_factory=lambda: _schema(
            {
                "operation": {"type": "string", "enum": ["replace", "create", "delete", "restore"]},
                "path": {"type": "string"},
                "expected_sha256": {"type": "string"},
                "old_content": {"type": "string"},
                "new_content": {"type": "string"},
                "workspace_version": {"type": "integer", "minimum": 0},
            },
            ("operation", "path", "workspace_version"),
        )
    )

    def execute(self, arguments: dict[str, Any]) -> str:
        return self._execute(lambda: self.lifecycle.edit(arguments))


@dataclass(frozen=True)
class SWEApplyArtifactTool(_SWEToolBase):
    name: str = "swe_apply_artifact"
    description: str = (
        "Atomically apply one content-addressed patch from a visible upstream packet."
    )
    parameters: dict[str, Any] = field(
        default_factory=lambda: _schema(
            {
                "artifact_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                "workspace_version": {"type": "integer", "minimum": 0},
            },
            ("artifact_sha256", "workspace_version"),
        )
    )

    def execute(self, arguments: dict[str, Any]) -> str:
        return self._execute(
            lambda: self.lifecycle.apply_artifact(
                arguments["artifact_sha256"], workspace_version=arguments["workspace_version"]
            )
        )


@dataclass(frozen=True)
class SWETestTool(_SWEToolBase):
    name: str = "swe_test"

    @property
    def description(self) -> str:
        profiles = ", ".join(sorted(self.lifecycle.test_profiles)) or "none"
        return (
            "Run one runtime-configured test profile after a workspace change; raw commands "
            f"and flags are forbidden. Available profiles: {profiles}."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        profiles = sorted(self.lifecycle.test_profiles)
        profile_schema: dict[str, Any] = {"type": "string"}
        if profiles:
            profile_schema["enum"] = profiles
        return _schema(
            {
                "profile": profile_schema,
                "target": {"type": "string"},
                "workspace_version": {"type": "integer", "minimum": 0},
            },
            ("profile", "workspace_version"),
        )

    def execute(self, arguments: dict[str, Any]) -> str:
        return self._execute(
            lambda: self.lifecycle.test(
                arguments["profile"],
                arguments.get("target"),
                workspace_version=arguments["workspace_version"],
            )
        )


@dataclass(frozen=True)
class SWEStatusTool(_SWEToolBase):
    name: str = "swe_status"
    description: str = (
        "Inspect current workspace version and changed file paths. This does not inspect source "
        "or advance the fix; an identical status call at the same version is rejected."
    )
    parameters: dict[str, Any] = field(
        default_factory=lambda: _schema(
            {"workspace_version": {"type": "integer", "minimum": 0}},
            ("workspace_version",),
        )
    )

    def execute(self, arguments: dict[str, Any]) -> str:
        return self._execute(
            lambda: (
                self.lifecycle._check_version(arguments["workspace_version"]),
                self.lifecycle.status(),
            )[1]
        )


def swe_tools(lifecycle: SWEWorkspaceLifecycle) -> dict[str, _SWEToolBase]:
    tools = (
        SWEListTool(lifecycle),
        SWESearchTool(lifecycle),
        SWEReadTool(lifecycle),
        SWEEditTool(lifecycle),
        SWEApplyArtifactTool(lifecycle),
        SWETestTool(lifecycle),
        SWEStatusTool(lifecycle),
    )
    return {tool.name: tool for tool in tools}


def swe_lifecycles(tools: dict[str, Any]) -> tuple[SWEWorkspaceLifecycle, ...]:
    values: dict[int, SWEWorkspaceLifecycle] = {}
    for tool in tools.values():
        lifecycle = getattr(tool, "lifecycle", None)
        if isinstance(lifecycle, SWEWorkspaceLifecycle):
            values[id(lifecycle)] = lifecycle
    return tuple(values.values())


def _schema(properties: dict[str, Any], required: tuple[str, ...]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(required),
        "additionalProperties": False,
    }


class SWEEvaluationStatus(StrEnum):
    RESOLVED = "resolved"
    UNRESOLVED = "unresolved"
    EMPTY_PATCH = "empty_patch"
    PATCH_APPLY_FAILED = "patch_apply_failed"
    INFRASTRUCTURE_ERROR = "infrastructure_error"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class SWEHarnessResult:
    status: SWEEvaluationStatus | str
    environment_completed: bool
    official: bool
    synthetic: bool
    patch_applied: bool = False
    fail_to_pass_complete: bool = False
    pass_to_pass_complete: bool = False
    fail_to_pass_success: tuple[str, ...] = ()
    fail_to_pass_failure: tuple[str, ...] = ()
    pass_to_pass_success: tuple[str, ...] = ()
    pass_to_pass_failure: tuple[str, ...] = ()
    detail: str = ""

    def __post_init__(self) -> None:
        status = SWEEvaluationStatus(self.status)
        object.__setattr__(self, "status", status)
        if self.official == self.synthetic:
            raise ValueError("SWE harness result must be exactly one of official or synthetic")
        if (
            status
            in {
                SWEEvaluationStatus.INFRASTRUCTURE_ERROR,
                SWEEvaluationStatus.TIMEOUT,
                SWEEvaluationStatus.CANCELLED,
            }
            and self.environment_completed
        ):
            raise ValueError("infrastructure termination cannot be environment_completed")
        if status is SWEEvaluationStatus.RESOLVED and (
            not self.environment_completed
            or not self.patch_applied
            or not self.fail_to_pass_complete
            or not self.pass_to_pass_complete
            or self.fail_to_pass_failure
            or self.pass_to_pass_failure
        ):
            raise ValueError("resolved requires a completed harness with no F2P/P2P failures")

    @classmethod
    def unresolved(cls, detail: str, *, synthetic: bool = True) -> SWEHarnessResult:
        return cls(
            SWEEvaluationStatus.UNRESOLVED,
            environment_completed=True,
            official=not synthetic,
            synthetic=synthetic,
            patch_applied=True,
            detail=detail,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["status"] = self.status.value
        payload["fail_to_pass_success"] = list(self.fail_to_pass_success)
        payload["fail_to_pass_failure"] = list(self.fail_to_pass_failure)
        payload["pass_to_pass_success"] = list(self.pass_to_pass_success)
        payload["pass_to_pass_failure"] = list(self.pass_to_pass_failure)
        return payload


class SWEOutcomeVerifier:
    """Strict final-only verifier over sealed runtime harness evidence."""

    name = "swe_outcome"
    supports_intermediate_scoring = False

    def __init__(self, *, allow_synthetic: bool = False) -> None:
        self.allow_synthetic = bool(allow_synthetic)

    def verify(self, task: TaskSpec, prediction: str) -> VerificationResult:
        del prediction
        raw = task.metadata.get("swe_environment_result")
        if not isinstance(raw, dict):
            raise RuntimeError("SWE verifier requires runtime-owned harness evidence")
        if str(raw.get("status", "")) == "typed_policy_failure":
            if not (
                raw.get("environment_completed") is True
                and raw.get("runtime_owned") is True
                and raw.get("attribution") == "model_policy"
                and raw.get("patch_applied") is False
                and raw.get("official") is False
                and raw.get("synthetic") is False
                and raw.get("failure_code") == "read_only_policy_stall"
            ):
                raise RuntimeError("invalid runtime-owned SWE policy failure evidence")
            return VerificationResult(
                score=0.0,
                passed=False,
                verifier=self.name,
                detail=(
                    "status=typed_policy_failure; official=False; synthetic=False; "
                    "attribution=model_policy"
                ),
            )
        result = SWEHarnessResult(
            status=str(raw.get("status", "infrastructure_error")),
            environment_completed=bool(raw.get("environment_completed", False)),
            official=bool(raw.get("official", False)),
            synthetic=bool(raw.get("synthetic", False)),
            patch_applied=bool(raw.get("patch_applied", False)),
            fail_to_pass_complete=bool(
                raw.get("f2p_complete", raw.get("fail_to_pass_complete", False))
            ),
            pass_to_pass_complete=bool(
                raw.get("p2p_complete", raw.get("pass_to_pass_complete", False))
            ),
            fail_to_pass_success=tuple(str(value) for value in raw.get("fail_to_pass_success", ())),
            fail_to_pass_failure=(
                tuple(str(value) for value in raw.get("fail_to_pass_failure", ()))
                or tuple("<redacted>" for _ in range(int(raw.get("f2p_failure_count", 0))))
            ),
            pass_to_pass_success=tuple(str(value) for value in raw.get("pass_to_pass_success", ())),
            pass_to_pass_failure=(
                tuple(str(value) for value in raw.get("pass_to_pass_failure", ()))
                or tuple("<redacted>" for _ in range(int(raw.get("p2p_failure_count", 0))))
            ),
            detail=str(raw.get("detail", "")),
        )
        if result.synthetic and not self.allow_synthetic:
            raise RuntimeError("synthetic SWE harness evidence is forbidden for production reward")
        if not result.environment_completed:
            raise RuntimeError(f"SWE harness did not complete: {result.status.value}")
        resolved = (
            result.status is SWEEvaluationStatus.RESOLVED
            and result.patch_applied
            and result.fail_to_pass_complete
            and result.pass_to_pass_complete
            and not result.fail_to_pass_failure
            and not result.pass_to_pass_failure
        )
        return VerificationResult(
            score=float(resolved),
            passed=resolved,
            verifier=self.name,
            detail=f"status={result.status.value}; official={result.official}; synthetic={result.synthetic}",
        )


def public_swe_evaluation(result: dict[str, Any]) -> dict[str, Any]:
    """Redact private test identities before attaching outcome evidence to TaskSpec."""

    def count(name: str, values_name: str) -> int:
        raw = result.get(name)
        if isinstance(raw, int) and not isinstance(raw, bool) and raw >= 0:
            return raw
        return len(result.get(values_name, ()))

    payload = {
        "status": str(result.get("status", "infrastructure_error")),
        "environment_completed": bool(result.get("environment_completed", False)),
        "official": bool(result.get("official", False)),
        "synthetic": bool(result.get("synthetic", False)),
        "patch_applied": bool(result.get("patch_applied", False)),
        "f2p_complete": bool(result.get("fail_to_pass_complete", False)),
        "p2p_complete": bool(result.get("pass_to_pass_complete", False)),
        "f2p_success_count": count("f2p_success_count", "fail_to_pass_success"),
        "f2p_failure_count": count("f2p_failure_count", "fail_to_pass_failure"),
        "p2p_success_count": count("p2p_success_count", "pass_to_pass_success"),
        "p2p_failure_count": count("p2p_failure_count", "pass_to_pass_failure"),
        "detail": str(result.get("detail", ""))[:500],
    }
    assert_public_swe_payload(payload)
    return payload


class SyntheticSWEHarness:
    """Explicitly non-official fake-repository backend for lifecycle unit tests."""

    def __init__(self, evaluator: Any) -> None:
        self.evaluator = evaluator

    def evaluate(self, binding: TrustedSWETaskBinding, patch: bytes) -> dict[str, Any]:
        result = self.evaluator(binding, patch)
        if not isinstance(result, SWEHarnessResult) or not result.synthetic or result.official:
            raise TypeError("synthetic harness evaluator must return a synthetic SWEHarnessResult")
        return result.to_dict()
