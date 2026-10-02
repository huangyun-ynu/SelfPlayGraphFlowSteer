"""Select the complete, frozen AIME application for formal collection and replay."""

from __future__ import annotations

import ast
from dataclasses import replace
from functools import lru_cache
import hashlib
import importlib
from pathlib import Path
import sys

VERSION = "aime-no-code-comments-shared-usage-default-20261002"
_ROOT = Path(__file__).parent
_ENGINE = _ROOT.parent / "formal_aime"
_SHARED_MODULES = (
    "actions", "action_protocol", "graph", "deadline", "backend_failures",
    "observability", "outcome_admission", "rollouts", "unified_contract",
    "director_connection_guard", "route_health",
)
_SHARED_TYPES = (
    "RelationType", "StructuralOperator", "AgentNode", "Relation",
    "CodeArtifactRef", "RelayPacket", "ExecutionReport",
)
_SHARED_GATE_STATE = (
    "_REQUEST_GATE_LOCK", "_REQUEST_GATES", "_REQUEST_PRIORITY", "_REQUEST_DATASET",
)
_SHARED_APPLICATION_ERRORS = ("GraphEvaluationIncompleteError", "GraphEvaluationBackendError")


def engine_contract():
    return {
        "version": VERSION,
        "module_namespace": "formal_aime",
        "source_sha256": {
            str(path.relative_to(_ENGINE)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(_ENGINE.rglob("*"))
            if path.is_file() and "__pycache__" not in path.parts
            and path.suffix not in {".pyc", ".pyo"}
        },
        "shared_contract_modules": [*_SHARED_MODULES, "submission_contract"],
        "shared_contract_types": list(_SHARED_TYPES),
        "shared_request_gate_state": list(_SHARED_GATE_STATE),
        "shared_application_errors": list(_SHARED_APPLICATION_ERRORS),
        "shared_observability_compatibility": "same_ast_except_qa_trace_replay_v1",
        "primary_and_counterfactual_engine": "complete_frozen_application",
    }


def _observability_core(path):
    """QA replay may evolve; all shared AIME trace and verifier definitions match."""
    tree = ast.parse(path.read_text())
    tree.body = [node for node in tree.body if not (
        isinstance(node, ast.FunctionDef) and node.name == "replay_trace"
    )]
    return ast.dump(tree, include_attributes=False)


@lru_cache(maxsize=1)
def _load_application():
    namespace = "formal_aime"
    for name in _SHARED_MODULES:
        current_path = _ROOT / (name + ".py")
        frozen_path = _ENGINE / (name + ".py")
        same = (
            _observability_core(current_path) == _observability_core(frozen_path)
            if name == "observability" else current_path.read_bytes() == frozen_path.read_bytes()
        )
        if not same:
            raise ValueError(f"AIME shared contract changed: {name}")
        sys.modules[namespace + "." + name] = importlib.import_module("." + name, __package__)
    sys.modules[namespace + ".submission_contract"] = importlib.import_module(
        ".submission_contract", __package__)
    current = importlib.import_module(".contracts", __package__)
    frozen = importlib.import_module(namespace + ".contracts")
    definitions = []
    for root in (_ROOT, _ENGINE):
        definitions.append({node.name: ast.dump(node, include_attributes=False)
                            for node in ast.parse((root / "contracts.py").read_text()).body
                            if isinstance(node, ast.ClassDef)})
    for name in _SHARED_TYPES:
        if definitions[0][name] != definitions[1][name]:
            raise ValueError(f"AIME shared type changed: {name}")
        setattr(frozen, name, getattr(current, name))
    application = importlib.import_module(namespace + ".application")
    current_application = importlib.import_module(".application", __package__)
    for name in _SHARED_APPLICATION_ERRORS:
        setattr(application, name, getattr(current_application, name))
    gateway = importlib.import_module(namespace + ".llm")
    current_gateway = importlib.import_module(".llm", __package__)
    for name in _SHARED_GATE_STATE:
        setattr(gateway, name, getattr(current_gateway, name))
    return application


class FormalAIMEApplication:
    """Keep one full application per implementation behind the formal factory."""

    def __init__(self, application, factory_options):
        self._base = application
        self._active = application
        self._aime = None
        self._options = dict(factory_options)
        self._deadline = None

    def __getattr__(self, name):
        return getattr(self._active, name)

    @property
    def config(self):
        return self._base.config

    def _select(self, dataset):
        from .config import canonical_dataset_name

        if canonical_dataset_name(dataset) != "aime":
            self._active = self._base
            return self._base
        if self._aime is None:
            module = _load_application()
            config = self.config
            construction = replace(
                config,
                retrieval=replace(config.retrieval, enabled=False, nq_evidence_mode=None),
                webshop=replace(config.webshop, enabled=False),
                alfworld=replace(config.alfworld, enabled=False),
                swe=replace(config.swe, enabled=False),
                skillbank_enabled=False,
            )
            options = {**self._options, "swe_lifecycle": None}
            self._aime = module.create_adaptive_application(construction, **options)
            self._aime.config = config
            self._aime.skillbank = self._base.skillbank
            self._aime.skill_lifecycle = self._base.skill_lifecycle
            self._aime.solver.skillbank = self._base.skillbank
            self._aime.set_rollout_deadline(self._deadline)
        self._active = self._aime
        return self._aime

    def solve(self, prompt, *, task_id="task", task_type="general", reference=None,
              run_id=None, metadata=None, private_verifier_payload=None):
        target = self._select((metadata or {}).get("dataset", task_type))
        return target.solve(prompt, task_id=task_id, task_type=task_type, reference=reference,
                            run_id=run_id, metadata=metadata,
                            private_verifier_payload=private_verifier_payload)

    def configure_scoped_worker_routes(self, *, dataset, request_role="primary"):
        return self._select(dataset).configure_scoped_worker_routes(
            dataset=dataset, request_role=request_role)

    def supports_graph_counterfactual(self, task):
        return self._select(task.metadata.get("dataset", task.task_type)).supports_graph_counterfactual(task)

    def evaluate_graph(self, task, graph, *, seed, initial_artifacts=None,
                       dirty_agents=None, return_verification=False):
        return self._select(task.metadata.get("dataset", task.task_type)).evaluate_graph(
            task, graph, seed=seed, initial_artifacts=initial_artifacts,
            dirty_agents=dirty_agents, return_verification=return_verification)

    def set_rollout_deadline(self, deadline):
        self._deadline = deadline
        self._base.set_rollout_deadline(deadline)
        if self._aime is not None:
            self._aime.set_rollout_deadline(deadline)

    def close(self):
        error = None
        for application in (self._aime, self._base):
            if application is not None:
                try:
                    application.close()
                except BaseException as issue:
                    error = error or issue
        if error is not None:
            raise error
