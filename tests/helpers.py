from __future__ import annotations

from selfplay_graph_flowsteer.contracts import AgentArtifact, AgentNode, RelayPacket


class RecordingExecutor:
    version = "recording-v1"

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def execute(
        self,
        *,
        task: str,
        node: AgentNode,
        upstream: list[RelayPacket],
        peers: list[RelayPacket],
        revision: bool,
        seed: int,
        prior: RelayPacket | None = None,
    ) -> AgentArtifact:
        self.calls.append(
            {
                "agent_id": node.agent_id,
                "task": task,
                "allowed_tools": list(node.allowed_tools),
                "upstream": [packet.sender for packet in upstream],
                "peers": [packet.sender for packet in peers],
                "prior": prior.sender if prior is not None else None,
                "prior_answer": prior.answer if prior is not None else None,
                "revision": revision,
            }
        )
        context = ",".join(packet.sender for packet in [*upstream, *peers]) or "none"
        answer = f"{node.agent_id}:{node.prompt}:context={context}:revision={revision}"
        return AgentArtifact(
            artifact_id="pending",
            agent_id=node.agent_id,
            answer=answer,
            summary=answer,
            confidence=1.0,
            token_in=2,
            token_out=3,
            revision=revision,
        )


class NumericRecordingExecutor(RecordingExecutor):
    """Recording-only fixture with a syntactically valid AIME final answer."""

    def execute(self, **kwargs):
        artifact = super().execute(**kwargs)
        artifact.answer = "35"
        return artifact


def install_numeric_mock_worker(monkeypatch):
    """Give AIME scheduling tests valid syntax without consulting task targets."""
    from selfplay_graph_flowsteer import application
    from selfplay_graph_flowsteer.llm import MockBackend

    original = application._mock_adaptive_backends

    def backends(**kwargs):
        director, _, distiller = original(**kwargs)
        worker = MockBackend(handler=lambda *_: '{"answer":"0","summary":"mock numeric output"}')
        return director, worker, distiller

    monkeypatch.setattr(application, "_mock_adaptive_backends", backends)
