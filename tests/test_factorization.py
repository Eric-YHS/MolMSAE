from dataclasses import asdict

import numpy as np
import pytest

from molmsae.factorization import (
    GraphFactorization,
    GraphValidationError,
    assemble_graph,
    factorize_graph,
)


def make_graph(max_nodes, active_nodes, edges, *, node_dtype=np.uint8):
    node_mask = np.zeros(max_nodes, dtype=bool)
    node_mask[list(active_nodes)] = True
    node_labels = np.full(max_nodes, 255, dtype=node_dtype)
    for position, node in enumerate(active_nodes):
        node_labels[node] = 6 + (position % 5)
    edge_labels = np.zeros((max_nodes, max_nodes), dtype=np.uint8)
    for left, right, label in edges:
        edge_labels[left, right] = label
        edge_labels[right, left] = label
    return node_mask, node_labels, edge_labels


def assert_round_trip(inputs):
    factorization = factorize_graph(*inputs)
    reconstructed = assemble_graph(factorization)
    np.testing.assert_array_equal(reconstructed.node_mask, inputs[0])
    np.testing.assert_array_equal(reconstructed.node_labels, inputs[1])
    np.testing.assert_array_equal(reconstructed.edge_labels, inputs[2])
    assert reconstructed.node_mask.dtype == inputs[0].dtype
    assert reconstructed.node_labels.dtype == inputs[1].dtype
    assert reconstructed.edge_labels.dtype == inputs[2].dtype
    return factorization


def test_ring_with_tail_has_ring_and_attachment():
    ring_edges = [(index, (index + 1) % 6, 4) for index in range(6)]
    inputs = make_graph(10, range(7), ring_edges + [(5, 6, 1)])

    factorization = assert_round_trip(inputs)

    assert [item.fragment_kind for item in factorization.fragment_inventory] == [
        "ring_system",
        "singleton",
    ]
    assert factorization.fragment_inventory[0].cycle_rank == 1
    assert factorization.algorithm_trace.bridge_edges_original == ((5, 6),)
    assert len(factorization.cross_fragment_attachments) == 1
    attachment = factorization.cross_fragment_attachments[0]
    assert (attachment.left_fragment_id, attachment.right_fragment_id) == (0, 1)
    assert attachment.edge_label == 1


def test_chain_splits_at_internal_single_bridge_and_records_bridges():
    inputs = make_graph(
        8,
        [0, 1, 2, 3, 4],
        [(0, 1, 1), (1, 2, 2), (2, 3, 1), (3, 4, 3)],
    )

    factorization = assert_round_trip(inputs)

    assert len(factorization.fragment_inventory) == 2
    assert [item.num_nodes for item in factorization.fragment_inventory] == [3, 2]
    assert [item.num_internal_edges for item in factorization.fragment_inventory] == [2, 1]
    assert all(
        item.fragment_kind == "acyclic_component"
        for item in factorization.fragment_inventory
    )
    assert len(factorization.cross_fragment_attachments) == 1
    assert factorization.cross_fragment_attachments[0].edge_label == 1
    assert factorization.algorithm_trace.bridge_edges_original == (
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
    )


def test_branched_tree_round_trip():
    inputs = make_graph(
        9,
        [0, 1, 2, 3, 4, 5],
        [(0, 1, 1), (1, 2, 1), (1, 3, 1), (3, 4, 1), (3, 5, 1)],
    )

    factorization = assert_round_trip(inputs)

    assert all(
        item.fragment_kind in {"acyclic_component", "singleton"}
        for item in factorization.fragment_inventory
    )
    assert len(factorization.fragment_inventory) > 1
    assert len(factorization.algorithm_trace.bridge_edges_original) == 5


def test_disconnected_ring_chain_and_singleton_with_sparse_mask():
    active = [0, 1, 2, 4, 5, 7]
    inputs = make_graph(
        12,
        active,
        [(0, 1, 4), (1, 2, 4), (0, 2, 4), (4, 5, 1)],
    )

    factorization = assert_round_trip(inputs)

    assert [item.fragment_kind for item in factorization.fragment_inventory] == [
        "ring_system",
        "acyclic_component",
        "singleton",
    ]
    assert factorization.coarse_topology.fragment_component_ids == (0, 1, 2)
    assert factorization.coarse_topology.edges == ()


def test_spiro_like_cycles_merge_but_remain_two_biconnected_blocks():
    inputs = make_graph(
        10,
        range(5),
        [
            (0, 1, 1),
            (1, 2, 1),
            (0, 2, 1),
            (2, 3, 1),
            (3, 4, 1),
            (2, 4, 1),
        ],
    )

    factorization = assert_round_trip(inputs)

    assert len(factorization.fragment_inventory) == 1
    assert factorization.fragment_inventory[0].fragment_kind == "ring_system"
    cyclic_blocks = [
        block
        for block in factorization.algorithm_trace.biconnected_blocks_original
        if block.is_cyclic
    ]
    assert len(cyclic_blocks) == 2


def test_decoder_view_cannot_access_roundtrip_or_trace_metadata():
    inputs = make_graph(6, [0, 1, 2], [(0, 1, 1), (1, 2, 1)])
    factorization = factorize_graph(*inputs)

    targets = factorization.decoder_targets()
    serialized = asdict(targets)

    assert set(serialized) == {
        "fragment_inventory",
        "coarse_topology",
        "fragment_contents",
        "cross_fragment_attachments",
    }
    assert not hasattr(targets, "roundtrip_residual")
    assert not hasattr(targets, "algorithm_trace")
    assert (
        GraphFactorization.__dataclass_fields__["roundtrip_residual"].metadata[
            "decoder_visible"
        ]
        is False
    )


@pytest.mark.parametrize("seed", range(20))
def test_random_graphs_are_deterministic_and_lossless(seed):
    rng = np.random.default_rng(seed)
    max_nodes = 32
    num_active = int(rng.integers(0, 25))
    active_nodes = sorted(
        int(value)
        for value in rng.choice(max_nodes, size=num_active, replace=False)
    )
    edges = []
    for offset, left in enumerate(active_nodes):
        for right in active_nodes[offset + 1 :]:
            if rng.random() < 0.12:
                edges.append((left, right, int(rng.integers(1, 5))))
    inputs = make_graph(max_nodes, active_nodes, edges)

    first = assert_round_trip(inputs)
    second = factorize_graph(*inputs)

    assert first == second
    assigned = [
        node
        for fragment in first.roundtrip_residual.original_node_indices_by_fragment
        for node in fragment
    ]
    assert sorted(assigned) == active_nodes
    assert len(assigned) == len(set(assigned))


def test_rejects_asymmetric_edges():
    node_mask, node_labels, edge_labels = make_graph(4, [0, 1], [])
    edge_labels[0, 1] = 1

    with pytest.raises(GraphValidationError, match="symmetric"):
        factorize_graph(node_mask, node_labels, edge_labels)


def test_rejects_edges_to_padding():
    node_mask, node_labels, edge_labels = make_graph(4, [0, 1], [])
    edge_labels[1, 3] = edge_labels[3, 1] = 1

    with pytest.raises(GraphValidationError, match="masked-out"):
        factorize_graph(node_mask, node_labels, edge_labels)

