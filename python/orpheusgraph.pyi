"""Type stubs for orpheusgraph — Rust knowledge graph engine."""

from __future__ import annotations

class OrpheusGraph:
    """Immutable knowledge graph with traversal and scoring."""

    def node_count(self) -> int: ...
    def edge_count(self) -> int: ...
    def get_node(self, name: str) -> NodeResult | None: ...
    def outgoing_edges(self, name: str) -> list[EdgeResult]: ...
    def incoming_edges(self, name: str) -> list[EdgeResult]: ...
    # NOTE: outgoing_edges/incoming_edges are RAW, UNFILTERED adjacency: they take
    # no DynamicContext and apply NO temporal (`valid_from`/`valid_to`) or ACL
    # (`acl`) visibility filtering. Every edge is returned with its full `acl`
    # tag list. Temporal/ACL visibility applies ONLY to the ctx-taking traversal
    # methods (beam_traverse, find_path, contextual_subgraph,
    # multi_beam_intersection). Do NOT surface these raw lists to an end user
    # under an access-control assumption.
    def beam_traverse(
        self, start: str, k: int, depth: int, ctx: DynamicContext
    ) -> list[NodeResult]: ...
    def find_path(
        self, start: str, end: str, ctx: DynamicContext
    ) -> list[PathStep] | None: ...
    def contextual_subgraph(
        self, ctx: DynamicContext, k: int
    ) -> SubGraph: ...
    def multi_beam_intersection(
        self,
        start_nodes: list[str],
        k: int,
        depth: int,
        ctx: DynamicContext,
        threshold: int | None = None,
    ) -> SubGraph: ...
    def to_rkyv(self) -> bytes: ...
    def close(self) -> None: ...

class PersistentGraph:
    """Durable delta store: an immutable base + a crash-safe mutable delta.

    The read API mirrors OrpheusGraph but runs over the live base-plus-delta
    view at the current seq. Mutations go through apply(); durability is
    controlled by flush()/compact() and the fsync policy.
    """

    seq: int
    """Current durable commit sequence — the CAS token and cache generation."""
    epoch: int
    """Current incarnation id (changes across an unclean reopen)."""

    def node_count(self) -> int: ...
    def edge_count(self) -> int: ...
    def apply(
        self, ops: list[dict], expected_seq: int | None = None
    ) -> int:
        """Apply a batch atomically (one WAL frame); return the new seq.

        Each op is a dict with an ``"op"`` discriminant:
          - ``{"op": "upsert_node", "name": str, "kind": str, "base_weight": float,
             "noise_penalty": float, "metadata": dict}``
          - ``{"op": "remove_node", "name": str}``
          - ``{"op": "add_edge", "from": str, "to": str, "kind": str,
             "field": str | None, "base_weight": float,
             "valid_from": int | None, "valid_to": int | None,
             "acl": list[str] | None}``
          - ``{"op": "remove_edge", "from": str, "to": str, "kind": str}``

        ``expected_seq`` enables optimistic CAS: if the store advanced past it,
        ``ConflictError`` is raised and nothing is written. Weights must be
        pre-normalized to [0.0, 1.0] or ``ValueError`` is raised.

        ``valid_from``/``valid_to`` (opaque caller-defined valid-time instants,
        half-open ``[valid_from, valid_to)``) and ``acl`` (tag list; empty/absent
        = public) default to unbounded/public when omitted. A malformed
        ``valid_from > valid_to`` range raises ``ValueError`` (the whole batch is
        rejected, nothing written). These fields are QUERY-TIME ONLY filters
        (via ``DynamicContext.as_of``/``principals``) — they never affect
        PageRank or any other base metric, which are always computed over the
        full, unfiltered graph.

        Durability: a returned seq means the batch is DURABLE (its WAL frame and a
        covering COMMIT marker are both fsync'd) — it survives process death and
        power loss. A raised error means durability is INDETERMINATE: a COMMIT
        fsync can fail while its slot still persists, so the batch MAY be visible
        or invisible after a reopen. Recovery always reopens to a consistent
        committed prefix and never loses an acked batch; reconcile an error by
        re-opening the store and reading ``seq`` (the recovered committed prefix),
        NOT by blindly retrying (a non-idempotent batch could double-apply).
        """
    def flush(self) -> None: ...
    def compact(self) -> None: ...
    def get_node(self, name: str) -> NodeResult | None: ...
    def outgoing_edges(self, name: str) -> list[EdgeResult]: ...
    def incoming_edges(self, name: str) -> list[EdgeResult]: ...
    # NOTE: outgoing_edges/incoming_edges are RAW, UNFILTERED adjacency: they take
    # no DynamicContext and apply NO temporal (`valid_from`/`valid_to`) or ACL
    # (`acl`) visibility filtering. Every edge is returned with its full `acl`
    # tag list. Temporal/ACL visibility applies ONLY to the ctx-taking traversal
    # methods (beam_traverse, find_path, contextual_subgraph,
    # multi_beam_intersection). Do NOT surface these raw lists to an end user
    # under an access-control assumption.
    def beam_traverse(
        self, start: str, k: int, depth: int, ctx: DynamicContext
    ) -> list[NodeResult]: ...
    def find_path(
        self, start: str, end: str, ctx: DynamicContext
    ) -> list[PathStep] | None: ...
    def contextual_subgraph(self, ctx: DynamicContext, k: int) -> SubGraph: ...
    def multi_beam_intersection(
        self,
        start_nodes: list[str],
        k: int,
        depth: int,
        ctx: DynamicContext,
        threshold: int | None = None,
    ) -> SubGraph: ...
    def set_fsync_policy(self, policy: str, every_n: int | None = None) -> None: ...
    def set_auto_compact_threshold(self, threshold: int | None) -> None: ...
    def close(self) -> None:
        """Flush, mark a clean shutdown, and release the store (idempotent).

        Under multithreading this may raise a transient ``RuntimeError``
        ("Already borrowed") if another thread is mid-traversal on the same
        object — treat it as retryable, or close only when no read is in flight.
        """
    def __enter__(self) -> PersistentGraph: ...
    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> bool: ...

class ConflictError(Exception):
    """Raised by apply(expected_seq=...) when the store moved past expected_seq."""

class CorruptError(Exception):
    """Raised on structural corruption or an unsupported on-disk format version."""

class DynamicContext:
    semantic_boosts: dict[str, float]
    weight_overrides: dict[str, float]
    noise_tags: set[str]
    max_fan_out: int | None
    w_base: float
    w_semantic: float
    w_noise: float
    w_override: float
    as_of: int | None
    """Valid-time instant for temporal edge filtering. None = no filtering."""
    principals: list[str]
    """ACL principals held by the caller. Normalized (sorted+deduped) on use.
    Empty means an ACL-tagged edge is never visible (fail-closed)."""

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
        as_of: int | None = None,
        principals: list[str] | None = None,
    ) -> None:
        """overlay_edges do NOT support valid_from/valid_to/acl (the dict
        schema is string-valued only) — overlay edges are always
        unbounded/public regardless of as_of/principals."""
    def add_boost(self, name: str, value: float) -> None: ...
    def add_override(self, name: str, value: float) -> None: ...
    def add_noise_tag(self, tag: str) -> None: ...

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
    valid_from: int | None
    valid_to: int | None
    acl: list[str]

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
) -> OrpheusGraph:
    """Edge dicts accept optional ``valid_from``/``valid_to`` (int | None) and
    ``acl`` (list[str] | None), defaulting to unbounded/public. A malformed
    ``valid_from > valid_to`` range is silently DROPPED (the edge is skipped,
    matching the existing behavior for an edge referencing an unknown node) —
    it is not an error here, unlike ``PersistentGraph.apply()``.
    """

def from_rkyv(data: bytes) -> OrpheusGraph: ...

def open(
    dir: str,
    create: bool = False,
    mmap: bool = True,
    validate: str = "full",
    prefault: bool = False,
) -> PersistentGraph:
    """Open (or, with create=True, initialize) a durable store at ``dir``."""

def create_persistent(
    dir: str, nodes: list[dict], edges: list[dict]
) -> PersistentGraph:
    """Create a new durable store, seeding its base from build_graph inputs."""
