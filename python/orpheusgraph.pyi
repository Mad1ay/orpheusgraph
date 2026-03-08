"""Type stubs for orpheusgraph — Rust knowledge graph engine."""

from __future__ import annotations

class OrpheusGraph:
    """Immutable knowledge graph with traversal and scoring."""

    def node_count(self) -> int: ...
    def edge_count(self) -> int: ...
    def get_node(self, name: str) -> NodeResult | None: ...
    def outgoing_edges(self, name: str) -> list[EdgeResult]: ...
    def incoming_edges(self, name: str) -> list[EdgeResult]: ...
    def beam_traverse(
        self, start: str, k: int, depth: int, ctx: DynamicContext
    ) -> list[NodeResult]: ...
    def find_path(
        self, start: str, end: str, ctx: DynamicContext
    ) -> list[PathStep] | None: ...
    def contextual_subgraph(
        self, ctx: DynamicContext, k: int
    ) -> SubGraph: ...
    def to_rkyv(self) -> bytes: ...
    def close(self) -> None: ...

class DynamicContext:
    semantic_boosts: dict[str, float]
    weight_overrides: dict[str, float]
    noise_tags: set[str]
    max_fan_out: int | None
    w_base: float
    w_semantic: float
    w_noise: float
    w_override: float

    def __init__(
        self,
        *,
        semantic_boosts: dict[str, float] | None = None,
        weight_overrides: dict[str, float] | None = None,
        noise_tags: set[str] | None = None,
        max_fan_out: int | None = None,
        w_base: float = 1.0,
        w_semantic: float = 1.5,
        w_noise: float = 1.0,
        w_override: float = 1.0,
        overlay_nodes: list[dict[str, str]] | None = None,
        overlay_edges: list[dict[str, str]] | None = None,
    ) -> None: ...

class NodeResult:
    name: str
    kind: str
    weight: float
    base_component: float
    semantic_component: float
    noise_component: float
    override_component: float
    def explain_score(self) -> dict[str, float]: ...

class EdgeResult:
    source: str
    target: str
    kind: str
    field_name: str | None
    weight: float

class PathStep:
    node: str
    edge_kind: str
    field_name: str
    direction: str

class SubGraph:
    nodes: list[NodeResult]
    edges: list[EdgeResult]

def build_graph(
    nodes: list[dict], edges: list[dict]
) -> OrpheusGraph: ...

def from_rkyv(data: bytes) -> OrpheusGraph: ...
