"""Audit complete WebShop Worker requests; never trim shopping facts.

Endpoint limits must come from service metadata, not the logical route name.
An endpoint without published limits remains explicitly provider-enforced.
The UTF-8 estimate is an upper bound, not a tokenizer measurement.
"""
from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path


class WebShopContextCapacityExceeded(RuntimeError):
    def __init__(self, audit: dict):
        self.context_audit = audit
        super().__init__("Cannot prove complete WebShop request fits verified endpoint capacity: conservative bound exceeds the limit; facts were not truncated")


def is_memory_request(request: dict, *, dataset: str, role: str) -> bool:
    # Thread-pool workers may lack the parent's dataset/credit ContextVars.
    # The physical request still contains the trusted memory revision marker.
    return role == "worker" and (dataset == "webshop" or
        "structured_facts_v1" in json.dumps({k: request[k] for k in
            ("messages", "input") if k in request}, ensure_ascii=False))


def audit_request(request: dict, *, route: str, capabilities: dict | None = None,
                  root: Path | None = None) -> dict:
    if capabilities is None:
        path = os.environ.get("SPGFS_WEBSHOP_CONTEXT_CAPABILITIES")
        capabilities = json.loads(Path(path).read_text()) if path else {}
    endpoint = capabilities.get(route, {})
    # Include Responses instructions and structured output schemas as well as
    # messages/input and tools. Output allowance is checked independently.
    body = {k: request[k] for k in ("messages", "input", "instructions", "tools",
                                  "response_format", "text") if k in request}
    raw = json.dumps(body, ensure_ascii=False).encode("utf-8")
    input_bound = len(raw) + 2048
    output = int(request.get("max_output_tokens", request.get(
        "max_completion_tokens", request.get("max_tokens", 2048))))
    input_limit = endpoint.get("max_input_tokens")
    output_limit = endpoint.get("max_output_tokens")
    context_limit = endpoint.get("context_tokens")
    exceeds = ((input_limit is not None and input_bound > input_limit)
               or (output_limit is not None and output > output_limit)
               or (context_limit is not None and input_bound + output > context_limit))
    audit = {"method": "utf8_request_upper_bound_v1", "route": route,
             "model": request.get("model"), "input_upper_bound": input_bound,
             "output_allowance": output, "combined_upper_bound": input_bound + output,
             "endpoint_capacity": endpoint,
             "capacity_status": "conservative_bound_exceeds_capacity" if exceeds else
                 "verified_upper_bound_fits" if input_limit is not None or context_limit is not None else
                 "unpublished_provider_enforced",
             "facts_truncated": False, "input_sha256": hashlib.sha256(raw).hexdigest()}
    directory = root or (Path(os.environ.get("SPGFS_WEBSHOP_MEMORY_DIR", "state/webshop-memory"))
                         / "request-inputs")
    if directory is not None:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / (audit["input_sha256"] + ".json")
        # Different agents may share an identical public request. Atomic rename
        # keeps their audit file valid without adding any model/tool operation.
        temporary = path.with_suffix("." + uuid.uuid4().hex + ".tmp")
        temporary.write_text(json.dumps({"input": body, "audit": audit}, ensure_ascii=False))
        temporary.replace(path)
        audit["input_ref"] = str(path)
    if exceeds:
        raise WebShopContextCapacityExceeded(audit)
    return audit
