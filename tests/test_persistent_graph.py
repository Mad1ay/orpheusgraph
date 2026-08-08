"""Integration tests for the PersistentGraph Python API (persistence spec §4.6).

Run with the project venv (system Python 3.14 needs the PyO3 forward-compat
flag only at BUILD time; at runtime the built extension just imports):

    /home/madjay/dev/osds/.venv/bin/python -m pytest tests/test_persistent_graph.py -v

Every test terminates on its own — the threading test uses a fixed iteration
count and joins with a timeout — so the suite can never hang.
"""

from __future__ import annotations

import threading

import pytest

import orpheusgraph as og


# --------------------------------------------------------------------------
# Helpers / fixtures
# --------------------------------------------------------------------------

def node(name, kind="model", base_weight=0.5, noise_penalty=0.0, **md):
    op = {"op": "upsert_node", "name": name, "kind": kind,
          "base_weight": base_weight, "noise_penalty": noise_penalty}
    if md:
        op["metadata"] = md
    return op


def edge(frm, to, kind="relates_to", base_weight=0.8, field=None):
    op = {"op": "add_edge", "from": frm, "to": to, "kind": kind,
          "base_weight": base_weight}
    if field is not None:
        op["field"] = field
    return op


@pytest.fixture
def store(tmp_path):
    """A fresh empty store; closed on teardown if still open."""
    g = og.open(str(tmp_path / "s"), create=True)
    yield g
    try:
        g.close()
    except Exception:
        pass


def seed_triangle(g):
    """a -> b -> c, a -> c. Returns the new seq."""
    return g.apply([
        node("a"), node("b"), node("c"),
        edge("a", "b", field="ab"), edge("b", "c", field="bc"),
        edge("a", "c", field="ac"),
    ])


# --------------------------------------------------------------------------
# Lifecycle & durability
# --------------------------------------------------------------------------

def test_open_create_empty(tmp_path):
    g = og.open(str(tmp_path / "s"), create=True)
    assert g.seq == 0
    assert g.node_count() == 0
    assert g.edge_count() == 0
    assert isinstance(g.epoch, int) and g.epoch > 0
    g.close()


def test_open_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        og.open(str(tmp_path / "does_not_exist"), create=False)


def test_apply_returns_monotonic_seq(store):
    assert store.apply([node("x")]) == 1
    assert store.apply([node("y")]) == 2
    assert store.apply([edge("x", "y")]) == 3
    assert store.seq == 3


def test_reopen_preserves_committed_data(tmp_path):
    p = str(tmp_path / "s")
    g = og.open(p, create=True)
    seed_triangle(g)
    seq = g.seq
    g.close()

    g2 = og.open(p, create=False)
    assert g2.seq == seq
    assert g2.node_count() == 3
    assert g2.edge_count() == 3
    assert g2.get_node("a") is not None
    assert {e.target for e in g2.outgoing_edges("a")} == {"b", "c"}
    g2.close()


def test_reopen_after_many_batches_with_compaction(tmp_path):
    """Force auto-compaction mid-stream, then reopen and verify the logical
    graph survived the fold."""
    p = str(tmp_path / "s")
    g = og.open(p, create=True)
    g.set_auto_compact_threshold(5)  # compact aggressively
    for i in range(50):
        g.apply([node(f"n{i}", base_weight=0.4)])
    for i in range(0, 49):
        g.apply([edge(f"n{i}", f"n{i+1}")])
    expected_nodes = g.node_count()
    expected_edges = g.edge_count()
    seq = g.seq
    g.close()

    g2 = og.open(p, create=False)
    assert g2.seq == seq
    assert g2.node_count() == expected_nodes == 50
    assert g2.edge_count() == expected_edges == 49
    assert g2.get_node("n0") is not None
    assert g2.get_node("n49") is not None
    g2.close()


# --------------------------------------------------------------------------
# Op semantics
# --------------------------------------------------------------------------

def test_upsert_insert_then_update(store):
    store.apply([node("x", base_weight=0.2)])
    n1 = store.get_node("x")
    store.apply([node("x", base_weight=0.9)])  # update in place
    n2 = store.get_node("x")
    assert store.node_count() == 1  # not duplicated
    assert n2.base_component != n1.base_component


def test_remove_node_masks_incident_edges(store):
    seed_triangle(store)
    assert store.node_count() == 3 and store.edge_count() == 3
    store.apply([{"op": "remove_node", "name": "b"}])
    assert store.node_count() == 2
    # edges a->b and b->c are gone; only a->c remains
    assert store.edge_count() == 1
    assert store.get_node("b") is None
    assert {e.target for e in store.outgoing_edges("a")} == {"c"}


def test_remove_then_readd_node_symmetric(store):
    store.apply([node("x")])
    store.apply([{"op": "remove_node", "name": "x"}])
    assert store.get_node("x") is None
    store.apply([node("x", base_weight=0.7)])
    assert store.get_node("x") is not None
    assert store.node_count() == 1


def test_remove_then_readd_edge_symmetric(store):
    store.apply([node("a"), node("b"), edge("a", "b")])
    assert store.edge_count() == 1
    store.apply([{"op": "remove_edge", "from": "a", "to": "b", "kind": "relates_to"}])
    assert store.edge_count() == 0
    store.apply([edge("a", "b")])
    assert store.edge_count() == 1


# --------------------------------------------------------------------------
# CAS (optimistic concurrency)
# --------------------------------------------------------------------------

def test_cas_matching_proceeds(store):
    s0 = store.seq
    new = store.apply([node("x")], expected_seq=s0)
    assert new == s0 + 1


def test_cas_stale_raises_and_writes_nothing(store):
    store.apply([node("x")])  # seq -> 1
    before = store.seq
    with pytest.raises(og.ConflictError):
        store.apply([node("y")], expected_seq=0)  # stale
    assert store.seq == before  # nothing written
    assert store.get_node("y") is None


def test_cas_retry_loop_converges(store):
    # Simulate a read-modify-write that lost a race once, then retried.
    store.apply([node("counter", base_weight=0.1)])
    stale = 0  # deliberately stale token
    for _ in range(5):
        try:
            store.apply([node("counter", base_weight=0.2)], expected_seq=stale)
            break
        except og.ConflictError:
            stale = store.seq  # re-read and retry
    assert store.get_node("counter") is not None


# --------------------------------------------------------------------------
# Batch atomicity & validation
# --------------------------------------------------------------------------

def test_batch_atomic_on_bad_endpoint(store):
    store.apply([node("a")])
    before = store.seq
    with pytest.raises(ValueError):
        # good op followed by an edge to a missing node -> whole batch rejected
        store.apply([node("z"), edge("z", "missing_target")])
    assert store.seq == before
    assert store.get_node("z") is None  # the good op did not land


def test_out_of_range_weight_rejected(store):
    with pytest.raises(ValueError):
        store.apply([node("bad", base_weight=-0.5)])
    with pytest.raises(ValueError):
        store.apply([node("bad", base_weight=1.5)])
    with pytest.raises(ValueError):
        store.apply([edge("a", "b", base_weight=2.0)])


def test_nan_weight_rejected(store):
    with pytest.raises(ValueError):
        store.apply([node("bad", base_weight=float("nan"))])


def test_missing_required_field_raises(store):
    with pytest.raises((KeyError, ValueError)):
        store.apply([{"op": "upsert_node", "base_weight": 0.5}])  # no name
    with pytest.raises(ValueError):
        store.apply([{"op": "upsert_node", "name": "", "base_weight": 0.5}])  # empty name


def test_unknown_op_raises(store):
    with pytest.raises(ValueError):
        store.apply([{"op": "frobnicate", "name": "x"}])


def test_create_persistent_refuses_to_clobber(tmp_path):
    p = str(tmp_path / "s")
    g = og.open(p, create=True)
    g.apply([node("x")])
    g.close()
    with pytest.raises(Exception):
        og.create_persistent(p, [{"name": "a", "kind": "model", "base_weight": 0.5}], [])


# --------------------------------------------------------------------------
# Read parity: delta view == build_graph(materialized)
# --------------------------------------------------------------------------

def test_read_parity_with_build_graph(tmp_path):
    nodes = [
        {"name": "a", "kind": "model", "base_weight": 0.6},
        {"name": "b", "kind": "model", "base_weight": 0.5},
        {"name": "c", "kind": "model", "base_weight": 0.4},
    ]
    edges = [
        {"from": "a", "to": "b", "kind": "relates_to", "base_weight": 0.8},
        {"from": "b", "to": "c", "kind": "relates_to", "base_weight": 0.7},
    ]
    ephemeral = og.build_graph(nodes, edges)

    # Same graph via the persistent delta path.
    p = str(tmp_path / "s")
    g = og.open(p, create=True)
    g.apply([node("a", base_weight=0.6), node("b", base_weight=0.5), node("c", base_weight=0.4),
             edge("a", "b", base_weight=0.8), edge("b", "c", base_weight=0.7)])

    ctx = og.DynamicContext()
    e_beam = {r.name for r in ephemeral.beam_traverse("a", k=5, depth=3, ctx=ctx)}
    p_beam = {r.name for r in g.beam_traverse("a", k=5, depth=3, ctx=ctx)}
    assert e_beam == p_beam

    e_path = ephemeral.find_path("a", "c", ctx)
    p_path = g.find_path("a", "c", ctx)
    assert (e_path is None) == (p_path is None)
    assert [s.node for s in e_path] == [s.node for s in p_path]

    assert (ephemeral.get_node("a") is None) == (g.get_node("a") is None)
    g.close()


def test_create_persistent_seeds_base(tmp_path):
    p = str(tmp_path / "s")
    g = og.create_persistent(
        p,
        [{"name": "a", "kind": "model", "base_weight": 0.5},
         {"name": "b", "kind": "model", "base_weight": 0.5}],
        [{"from": "a", "to": "b", "kind": "relates_to", "base_weight": 1.0}],
    )
    assert g.node_count() == 2
    assert g.edge_count() == 1
    # base can be extended by delta
    g.apply([node("c"), edge("b", "c")])
    assert g.node_count() == 3
    g.close()


# --------------------------------------------------------------------------
# compact()
# --------------------------------------------------------------------------

def test_compact_preserves_graph_and_seq(store):
    seed_triangle(store)
    seq = store.seq
    nc, ec = store.node_count(), store.edge_count()
    store.compact()
    assert store.seq == seq  # compaction folds AT seq, does not mint a new one
    assert store.node_count() == nc
    assert store.edge_count() == ec


def test_compact_empty_delta_is_noop(store):
    seq = store.seq
    store.compact()
    assert store.seq == seq


# --------------------------------------------------------------------------
# fsync policy
# --------------------------------------------------------------------------

def test_every_batch_policy_durable(tmp_path):
    p = str(tmp_path / "s")
    g = og.open(p, create=True)
    g.set_fsync_policy("every_batch")
    g.apply([node("x"), node("y")])
    g.close()
    g2 = og.open(p, create=False)
    assert g2.get_node("x") is not None
    g2.close()


def test_every_n_requires_n(store):
    with pytest.raises(ValueError):
        store.set_fsync_policy("every_n")  # missing every_n
    store.set_fsync_policy("every_n", every_n=4)  # ok


def test_unknown_fsync_policy_raises(store):
    with pytest.raises(ValueError):
        store.set_fsync_policy("whenever")


# --------------------------------------------------------------------------
# close / context manager
# --------------------------------------------------------------------------

def test_use_after_close_raises(tmp_path):
    g = og.open(str(tmp_path / "s"), create=True)
    g.close()
    with pytest.raises(RuntimeError):
        g.node_count()
    with pytest.raises(RuntimeError):
        g.apply([node("x")])


def test_double_close_is_noop(tmp_path):
    g = og.open(str(tmp_path / "s"), create=True)
    g.close()
    g.close()  # must not raise


def test_context_manager_closes(tmp_path):
    p = str(tmp_path / "s")
    with og.open(p, create=True) as g:
        g.apply([node("x")])
    # after the with-block the store is closed
    with pytest.raises(RuntimeError):
        g.node_count()
    # and the data is durable
    g2 = og.open(p, create=False)
    assert g2.get_node("x") is not None
    g2.close()


# --------------------------------------------------------------------------
# Concurrency / GIL (bounded — cannot hang)
# --------------------------------------------------------------------------

def test_concurrent_readers_during_writes(tmp_path):
    """Readers traverse in a loop (fixed iterations) while the main thread
    applies + compacts. The lock-free reader path (ArcSwap snapshot) must never
    crash or observe a torn graph, and every thread must finish within the join
    timeout (a deadlock would trip the assert)."""
    p = str(tmp_path / "s")
    g = og.open(p, create=True)
    seed_triangle(g)
    g.set_auto_compact_threshold(3)

    ctx = og.DynamicContext()
    errors: list[BaseException] = []
    stop = threading.Event()

    def reader():
        try:
            for _ in range(300):
                if stop.is_set():
                    break
                res = g.beam_traverse("a", k=5, depth=3, ctx=ctx)
                # every observed state must be internally consistent
                assert all(r.name for r in res)
                _ = g.node_count()
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=reader) for _ in range(4)]
    for t in threads:
        t.start()

    # meanwhile mutate + compact
    for i in range(60):
        g.apply([node(f"m{i}", base_weight=0.3), edge("a", f"m{i}")])
        if i % 10 == 0:
            g.compact()

    stop.set()
    for t in threads:
        t.join(timeout=30)
        assert not t.is_alive(), "reader thread deadlocked"

    assert not errors, f"reader errors: {errors[:3]}"
    g.close()
