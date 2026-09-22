"""Lossless, deterministic factorization of a padded undirected graph.

The decoder-facing representation contains four sections: fragment inventory,
coarse topology, fragment contents, and cross-fragment attachments. Original
node positions are kept separately to support strict round-trip validation with
fixed-width molecular graph tensors.
"""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

DECODER_VISIBLE_SECTIONS = (
    "fragment_inventory",
    "coarse_topology",
    "fragment_contents",
    "cross_fragment_attachments",
)

DECODER_FORBIDDEN_SECTIONS = {
    "roundtrip_residual": (
        "Contains original padded-slot indices and padding values. It is for "
        "exact serialization tests only and must never be an encoder or decoder input."
    ),
    "algorithm_trace": (
        "Contains bridge and biconnected-block diagnostics in original node indices. "
        "It is for auditing the tokenizer only and must never be a decoder input."
    ),
}


class GraphValidationError(ValueError):
    """Raised when an input is not a valid padded undirected graph tensor."""


@dataclass(frozen=True)
class InternalEdge:
    local_u: int
    local_v: int
    edge_label: int


@dataclass(frozen=True)
class FragmentInventoryItem:
    fragment_id: int
    fragment_kind: str
    num_nodes: int
    num_internal_edges: int
    cycle_rank: int


@dataclass(frozen=True)
class FragmentContent:
    fragment_id: int
    fragment_kind: str
    node_labels: tuple[int, ...]
    internal_edges: tuple[InternalEdge, ...]


@dataclass(frozen=True)
class CoarseEdge:
    coarse_edge_id: int
    left_fragment_id: int
    right_fragment_id: int
    attachment_count: int


@dataclass(frozen=True)
class CoarseTopology:
    num_fragments: int
    fragment_component_ids: tuple[int, ...]
    edges: tuple[CoarseEdge, ...]


@dataclass(frozen=True)
class CrossFragmentAttachment:
    attachment_id: int
    coarse_edge_id: int
    left_fragment_id: int
    left_local_node: int
    right_fragment_id: int
    right_local_node: int
    edge_label: int


@dataclass(frozen=True)
class RoundTripResidual:
    """Serialization-only metadata that is forbidden as a model input."""

    max_nodes: int
    original_node_mask: tuple[bool, ...]
    original_node_indices_by_fragment: tuple[tuple[int, ...], ...]
    inactive_node_labels: tuple[tuple[int, int], ...]
    none_edge_label: int
    node_mask_dtype: str
    node_labels_dtype: str
    edge_labels_dtype: str


@dataclass(frozen=True)
class BiconnectedBlockTrace:
    """Audit record in original node coordinates; not decoder-visible."""

    original_nodes: tuple[int, ...]
    original_edges: tuple[tuple[int, int], ...]
    is_cyclic: bool


@dataclass(frozen=True)
class DecompositionTrace:
    """Tokenizer diagnostics in original node coordinates; not decoder-visible."""

    bridge_edges_original: tuple[tuple[int, int], ...]
    biconnected_blocks_original: tuple[BiconnectedBlockTrace, ...]


@dataclass(frozen=True)
class DecoderTargets:
    """The only four sections that may be exposed to the staged decoder."""

    fragment_inventory: tuple[FragmentInventoryItem, ...]
    coarse_topology: CoarseTopology
    fragment_contents: tuple[FragmentContent, ...]
    cross_fragment_attachments: tuple[CrossFragmentAttachment, ...]


@dataclass(frozen=True)
class GraphFactorization:
    fragment_inventory: tuple[FragmentInventoryItem, ...] = field(
        metadata={"decoder_visible": True}
    )
    coarse_topology: CoarseTopology = field(metadata={"decoder_visible": True})
    fragment_contents: tuple[FragmentContent, ...] = field(
        metadata={"decoder_visible": True}
    )
    cross_fragment_attachments: tuple[CrossFragmentAttachment, ...] = field(
        metadata={"decoder_visible": True}
    )
    roundtrip_residual: RoundTripResidual = field(
        metadata={"decoder_visible": False}
    )
    algorithm_trace: DecompositionTrace = field(metadata={"decoder_visible": False})

    def decoder_targets(self) -> DecoderTargets:
        """Return a view that makes serialization-only metadata inaccessible."""

        return DecoderTargets(
            fragment_inventory=self.fragment_inventory,
            coarse_topology=self.coarse_topology,
            fragment_contents=self.fragment_contents,
            cross_fragment_attachments=self.cross_fragment_attachments,
        )


@dataclass(frozen=True)
class ReconstructedGraph:
    node_mask: np.ndarray
    node_labels: np.ndarray
    edge_labels: np.ndarray


@dataclass(frozen=True)
class _TarjanAnalysis:
    bridges: tuple[tuple[int, int], ...]
    blocks: tuple[tuple[tuple[int, int], ...], ...]


class _DisjointSet:
    def __init__(self, values: Iterable[int]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: int) -> int:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if left_root < right_root:
            self.parent[right_root] = left_root
        else:
            self.parent[left_root] = right_root


def _edge_key(left: int, right: int) -> tuple[int, int]:
    return (left, right) if left < right else (right, left)


def _require_integer_array(name: str, value: Any, ndim: int) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != ndim:
        raise GraphValidationError(f"{name} must have {ndim} dimensions; got {array.shape}")
    if not np.issubdtype(array.dtype, np.integer):
        raise GraphValidationError(f"{name} must use an integer dtype; got {array.dtype}")
    return array


def _normalize_inputs(
    node_mask: Sequence[bool] | np.ndarray,
    node_labels: Sequence[int] | np.ndarray,
    edge_labels: Sequence[Sequence[int]] | np.ndarray,
    none_edge_label: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask_array = np.asarray(node_mask)
    if mask_array.ndim != 1:
        raise GraphValidationError(
            f"node_mask must have one dimension; got {mask_array.shape}"
        )
    if not (
        np.issubdtype(mask_array.dtype, np.bool_)
        or np.issubdtype(mask_array.dtype, np.integer)
    ):
        raise GraphValidationError(
            f"node_mask must be boolean or 0/1 integer; got {mask_array.dtype}"
        )
    if np.issubdtype(mask_array.dtype, np.integer):
        unique_mask_values = set(int(value) for value in np.unique(mask_array))
        if not unique_mask_values.issubset({0, 1}):
            raise GraphValidationError("integer node_mask values must be 0 or 1")

    node_array = _require_integer_array("node_labels", node_labels, ndim=1)
    edge_array = _require_integer_array("edge_labels", edge_labels, ndim=2)
    max_nodes = mask_array.shape[0]
    if node_array.shape != (max_nodes,):
        raise GraphValidationError(
            f"node_labels must have shape {(max_nodes,)}; got {node_array.shape}"
        )
    if edge_array.shape != (max_nodes, max_nodes):
        raise GraphValidationError(
            "edge_labels must be square and match node_mask; "
            f"expected {(max_nodes, max_nodes)}, got {edge_array.shape}"
        )
    if not np.array_equal(edge_array, edge_array.T):
        raise GraphValidationError("edge_labels must be symmetric for an undirected graph")
    if np.any(np.diag(edge_array) != none_edge_label):
        raise GraphValidationError("self-edges are not supported")

    mask_bool = mask_array.astype(bool, copy=False)
    inactive = ~mask_bool
    if inactive.any() and np.any(edge_array[inactive, :] != none_edge_label):
        raise GraphValidationError(
            "edges incident to masked-out nodes are not allowed"
        )
    return mask_array, node_array, edge_array


def _build_adjacency(
    active_nodes: tuple[int, ...],
    edge_labels: np.ndarray,
    none_edge_label: int,
) -> tuple[dict[int, tuple[int, ...]], tuple[tuple[int, int, int], ...]]:
    adjacency_lists: dict[int, list[int]] = {node: [] for node in active_nodes}
    edges: list[tuple[int, int, int]] = []
    for offset, left in enumerate(active_nodes):
        for right in active_nodes[offset + 1 :]:
            label = int(edge_labels[left, right])
            if label == none_edge_label:
                continue
            adjacency_lists[left].append(right)
            adjacency_lists[right].append(left)
            edges.append((left, right, label))
    adjacency = {
        node: tuple(sorted(neighbors)) for node, neighbors in adjacency_lists.items()
    }
    return adjacency, tuple(edges)


def _tarjan_analysis(
    active_nodes: tuple[int, ...], adjacency: dict[int, tuple[int, ...]]
) -> _TarjanAnalysis:
    discovery: dict[int, int] = {}
    low: dict[int, int] = {}
    edge_stack: list[tuple[int, int]] = []
    bridges: set[tuple[int, int]] = set()
    blocks: list[tuple[tuple[int, int], ...]] = []
    clock = 0

    def visit(node: int, parent: int | None) -> None:
        nonlocal clock
        discovery[node] = clock
        low[node] = clock
        clock += 1

        for neighbor in adjacency[node]:
            edge = _edge_key(node, neighbor)
            if neighbor not in discovery:
                edge_stack.append(edge)
                visit(neighbor, node)
                low[node] = min(low[node], low[neighbor])

                if low[neighbor] > discovery[node]:
                    bridges.add(edge)
                if low[neighbor] >= discovery[node]:
                    block_edges: list[tuple[int, int]] = []
                    while edge_stack:
                        popped = edge_stack.pop()
                        block_edges.append(popped)
                        if popped == edge:
                            break
                    blocks.append(tuple(sorted(set(block_edges))))
            elif neighbor != parent and discovery[neighbor] < discovery[node]:
                edge_stack.append(edge)
                low[node] = min(low[node], discovery[neighbor])

    for root in active_nodes:
        if root in discovery:
            continue
        visit(root, None)
        if edge_stack:
            blocks.append(tuple(sorted(set(edge_stack))))
            edge_stack.clear()

    normalized_blocks = tuple(
        sorted(
            (block for block in blocks if block),
            key=lambda block: (min(min(edge) for edge in block), block),
        )
    )
    return _TarjanAnalysis(
        bridges=tuple(sorted(bridges)),
        blocks=normalized_blocks,
    )


def _block_nodes(block: tuple[tuple[int, int], ...]) -> tuple[int, ...]:
    return tuple(sorted({node for edge in block for node in edge}))


def _is_cyclic_block(block: tuple[tuple[int, int], ...]) -> bool:
    return len(block) >= len(_block_nodes(block))


def _ring_systems(
    analysis: _TarjanAnalysis,
) -> tuple[tuple[int, ...], ...]:
    cyclic_blocks = [block for block in analysis.blocks if _is_cyclic_block(block)]
    ring_nodes = sorted(
        {node for block in cyclic_blocks for edge in block for node in edge}
    )
    if not ring_nodes:
        return ()

    disjoint_set = _DisjointSet(ring_nodes)
    for block in cyclic_blocks:
        nodes = _block_nodes(block)
        anchor = nodes[0]
        for node in nodes[1:]:
            disjoint_set.union(anchor, node)

    grouped: dict[int, list[int]] = defaultdict(list)
    for node in ring_nodes:
        grouped[disjoint_set.find(node)].append(node)
    return tuple(sorted((tuple(nodes) for nodes in grouped.values()), key=lambda x: x))


def _non_ring_components(
    active_nodes: tuple[int, ...],
    adjacency: dict[int, tuple[int, ...]],
    ring_nodes: set[int],
    edge_labels: np.ndarray,
    bridge_edges: set[tuple[int, int]],
    single_edge_label: int,
) -> tuple[tuple[int, ...], ...]:
    """Return rigid-ish acyclic fragments separated at internal single bonds.

    Cutting only bridge single bonds whose two endpoints are both non-terminal
    approximates the usual rotatable-bond split without requiring RDKit.  Ring
    bonds, multiple bonds, and terminal substituents remain inside a fragment.
    """

    remaining = set(active_nodes) - ring_nodes
    components: list[tuple[int, ...]] = []
    while remaining:
        root = min(remaining)
        remaining.remove(root)
        queue = deque([root])
        component = [root]
        while queue:
            node = queue.popleft()
            for neighbor in adjacency[node]:
                if neighbor in ring_nodes or neighbor not in remaining:
                    continue
                edge = _edge_key(node, neighbor)
                is_internal_single_bridge = (
                    edge in bridge_edges
                    and int(edge_labels[node, neighbor]) == single_edge_label
                    and len(adjacency[node]) > 1
                    and len(adjacency[neighbor]) > 1
                )
                if is_internal_single_bridge:
                    continue
                remaining.remove(neighbor)
                component.append(neighbor)
                queue.append(neighbor)
        components.append(tuple(sorted(component)))
    return tuple(components)


def _fragment_components(
    num_fragments: int, coarse_edges: tuple[CoarseEdge, ...]
) -> tuple[int, ...]:
    adjacency: dict[int, list[int]] = {index: [] for index in range(num_fragments)}
    for edge in coarse_edges:
        adjacency[edge.left_fragment_id].append(edge.right_fragment_id)
        adjacency[edge.right_fragment_id].append(edge.left_fragment_id)

    component_ids = [-1] * num_fragments
    component_id = 0
    for root in range(num_fragments):
        if component_ids[root] != -1:
            continue
        component_ids[root] = component_id
        queue = deque([root])
        while queue:
            fragment = queue.popleft()
            for neighbor in sorted(adjacency[fragment]):
                if component_ids[neighbor] != -1:
                    continue
                component_ids[neighbor] = component_id
                queue.append(neighbor)
        component_id += 1
    return tuple(component_ids)


def factorize_graph(
    node_mask: Sequence[bool] | np.ndarray,
    node_labels: Sequence[int] | np.ndarray,
    edge_labels: Sequence[Sequence[int]] | np.ndarray,
    *,
    none_edge_label: int = 0,
    single_edge_label: int = 1,
) -> GraphFactorization:
    """Factorize a padded graph into deterministic, lossless graph sections.

    Ring atoms are detected from cyclic biconnected blocks. Cyclic blocks that
    share an articulation atom are merged into one ring system. Remaining
    non-ring atoms form maximal connected acyclic/linker fragments. Every edge
    crossing two fragments becomes an explicit attachment.
    """

    mask_array, node_array, edge_array = _normalize_inputs(
        node_mask, node_labels, edge_labels, none_edge_label
    )
    mask_bool = mask_array.astype(bool, copy=False)
    active_nodes = tuple(int(index) for index in np.flatnonzero(mask_bool))
    adjacency, original_edges = _build_adjacency(
        active_nodes, edge_array, none_edge_label
    )
    analysis = _tarjan_analysis(active_nodes, adjacency)

    ring_systems = _ring_systems(analysis)
    ring_node_set = {node for system in ring_systems for node in system}
    non_ring_components = _non_ring_components(
        active_nodes,
        adjacency,
        ring_node_set,
        edge_array,
        set(analysis.bridges),
        single_edge_label,
    )

    raw_fragments: list[tuple[str, tuple[int, ...]]] = [
        ("ring_system", nodes) for nodes in ring_systems
    ]
    raw_fragments.extend(
        ("singleton" if len(nodes) == 1 else "acyclic_component", nodes)
        for nodes in non_ring_components
    )
    raw_fragments.sort(key=lambda item: item[1])

    original_to_local: dict[int, tuple[int, int]] = {}
    fragment_contents: list[FragmentContent] = []
    fragment_inventory: list[FragmentInventoryItem] = []
    for fragment_id, (fragment_kind, original_nodes) in enumerate(raw_fragments):
        for local_index, original_index in enumerate(original_nodes):
            original_to_local[original_index] = (fragment_id, local_index)

        internal_edges = tuple(
            InternalEdge(
                local_u=original_to_local[left][1],
                local_v=original_to_local[right][1],
                edge_label=edge_label,
            )
            for left, right, edge_label in original_edges
            if original_to_local.get(left, (-1, -1))[0] == fragment_id
            and original_to_local.get(right, (-2, -2))[0] == fragment_id
        )
        cycle_rank = len(internal_edges) - len(original_nodes) + 1
        content = FragmentContent(
            fragment_id=fragment_id,
            fragment_kind=fragment_kind,
            node_labels=tuple(int(node_array[index]) for index in original_nodes),
            internal_edges=internal_edges,
        )
        fragment_contents.append(content)
        fragment_inventory.append(
            FragmentInventoryItem(
                fragment_id=fragment_id,
                fragment_kind=fragment_kind,
                num_nodes=len(original_nodes),
                num_internal_edges=len(internal_edges),
                cycle_rank=cycle_rank,
            )
        )

    grouped_attachments: dict[
        tuple[int, int], list[tuple[int, int, int]]
    ] = defaultdict(list)
    for original_left, original_right, edge_label in original_edges:
        left_fragment, left_local = original_to_local[original_left]
        right_fragment, right_local = original_to_local[original_right]
        if left_fragment == right_fragment:
            continue
        if left_fragment > right_fragment:
            left_fragment, right_fragment = right_fragment, left_fragment
            left_local, right_local = right_local, left_local
        grouped_attachments[(left_fragment, right_fragment)].append(
            (left_local, right_local, edge_label)
        )

    coarse_edges: list[CoarseEdge] = []
    attachments: list[CrossFragmentAttachment] = []
    for coarse_edge_id, fragment_pair in enumerate(sorted(grouped_attachments)):
        candidates = sorted(grouped_attachments[fragment_pair])
        left_fragment, right_fragment = fragment_pair
        coarse_edges.append(
            CoarseEdge(
                coarse_edge_id=coarse_edge_id,
                left_fragment_id=left_fragment,
                right_fragment_id=right_fragment,
                attachment_count=len(candidates),
            )
        )
        for left_local, right_local, edge_label in candidates:
            attachments.append(
                CrossFragmentAttachment(
                    attachment_id=len(attachments),
                    coarse_edge_id=coarse_edge_id,
                    left_fragment_id=left_fragment,
                    left_local_node=left_local,
                    right_fragment_id=right_fragment,
                    right_local_node=right_local,
                    edge_label=edge_label,
                )
            )

    coarse_edge_tuple = tuple(coarse_edges)
    topology = CoarseTopology(
        num_fragments=len(raw_fragments),
        fragment_component_ids=_fragment_components(
            len(raw_fragments), coarse_edge_tuple
        ),
        edges=coarse_edge_tuple,
    )

    block_traces = tuple(
        BiconnectedBlockTrace(
            original_nodes=_block_nodes(block),
            original_edges=block,
            is_cyclic=_is_cyclic_block(block),
        )
        for block in analysis.blocks
    )
    residual = RoundTripResidual(
        max_nodes=len(mask_bool),
        original_node_mask=tuple(bool(value) for value in mask_bool),
        original_node_indices_by_fragment=tuple(
            nodes for _, nodes in raw_fragments
        ),
        inactive_node_labels=tuple(
            (int(index), int(node_array[index]))
            for index in np.flatnonzero(~mask_bool)
        ),
        none_edge_label=int(none_edge_label),
        node_mask_dtype=mask_array.dtype.str,
        node_labels_dtype=node_array.dtype.str,
        edge_labels_dtype=edge_array.dtype.str,
    )
    return GraphFactorization(
        fragment_inventory=tuple(fragment_inventory),
        coarse_topology=topology,
        fragment_contents=tuple(fragment_contents),
        cross_fragment_attachments=tuple(attachments),
        roundtrip_residual=residual,
        algorithm_trace=DecompositionTrace(
            bridge_edges_original=analysis.bridges,
            biconnected_blocks_original=block_traces,
        ),
    )


def _contents_by_id(
    factorization: GraphFactorization,
) -> dict[int, FragmentContent]:
    contents = {item.fragment_id: item for item in factorization.fragment_contents}
    expected_ids = set(range(len(factorization.fragment_contents)))
    if set(contents) != expected_ids:
        raise GraphValidationError("fragment content ids must be contiguous from zero")
    if len(factorization.fragment_inventory) != len(contents):
        raise GraphValidationError("inventory and fragment content counts differ")
    for inventory in factorization.fragment_inventory:
        content = contents.get(inventory.fragment_id)
        if content is None:
            raise GraphValidationError("inventory references an unknown fragment")
        if inventory.fragment_kind != content.fragment_kind:
            raise GraphValidationError("inventory/content fragment kinds differ")
        if inventory.num_nodes != len(content.node_labels):
            raise GraphValidationError("inventory/content node counts differ")
        if inventory.num_internal_edges != len(content.internal_edges):
            raise GraphValidationError("inventory/content edge counts differ")
    return contents


def assemble_graph(factorization: GraphFactorization) -> ReconstructedGraph:
    """Reassemble the exact padded arrays represented by ``factorization``."""

    contents = _contents_by_id(factorization)
    residual = factorization.roundtrip_residual
    num_fragments = len(contents)
    if factorization.coarse_topology.num_fragments != num_fragments:
        raise GraphValidationError("coarse topology has the wrong fragment count")
    if len(residual.original_node_indices_by_fragment) != num_fragments:
        raise GraphValidationError("residual fragment mapping has the wrong length")

    max_nodes = residual.max_nodes
    if len(residual.original_node_mask) != max_nodes:
        raise GraphValidationError("residual node mask has the wrong length")
    active_indices = {
        index
        for fragment_indices in residual.original_node_indices_by_fragment
        for index in fragment_indices
    }
    expected_active = {
        index for index, active in enumerate(residual.original_node_mask) if active
    }
    flattened_count = sum(
        len(indices) for indices in residual.original_node_indices_by_fragment
    )
    if active_indices != expected_active or flattened_count != len(active_indices):
        raise GraphValidationError(
            "original node index mapping must cover each active slot exactly once"
        )

    node_dtype = np.dtype(residual.node_labels_dtype)
    edge_dtype = np.dtype(residual.edge_labels_dtype)
    mask_dtype = np.dtype(residual.node_mask_dtype)
    reconstructed_mask = np.asarray(residual.original_node_mask, dtype=mask_dtype)
    reconstructed_nodes = np.zeros(max_nodes, dtype=node_dtype)
    assigned_nodes: set[int] = set()

    inactive_labels = dict(residual.inactive_node_labels)
    if set(inactive_labels) != set(range(max_nodes)) - expected_active:
        raise GraphValidationError("inactive node labels do not cover every padded slot")
    for original_index, label in inactive_labels.items():
        reconstructed_nodes[original_index] = label
        assigned_nodes.add(original_index)

    for fragment_id, content in contents.items():
        original_indices = residual.original_node_indices_by_fragment[fragment_id]
        if len(original_indices) != len(content.node_labels):
            raise GraphValidationError("fragment node mapping and labels differ in length")
        for local_index, label in enumerate(content.node_labels):
            original_index = original_indices[local_index]
            reconstructed_nodes[original_index] = label
            assigned_nodes.add(original_index)
    if assigned_nodes != set(range(max_nodes)):
        raise GraphValidationError("not every padded node slot received a label")

    reconstructed_edges = np.full(
        (max_nodes, max_nodes), residual.none_edge_label, dtype=edge_dtype
    )
    assigned_edges: set[tuple[int, int]] = set()

    def assign_edge(original_u: int, original_v: int, label: int) -> None:
        edge = _edge_key(original_u, original_v)
        if original_u == original_v:
            raise GraphValidationError("self-edges are not supported")
        if edge in assigned_edges:
            raise GraphValidationError(f"edge {edge} is encoded more than once")
        if label == residual.none_edge_label:
            raise GraphValidationError("an encoded edge cannot use the non-edge label")
        assigned_edges.add(edge)
        reconstructed_edges[original_u, original_v] = label
        reconstructed_edges[original_v, original_u] = label

    for fragment_id, content in contents.items():
        original_indices = residual.original_node_indices_by_fragment[fragment_id]
        for edge in content.internal_edges:
            if not (
                0 <= edge.local_u < edge.local_v < len(original_indices)
            ):
                raise GraphValidationError("invalid fragment-local internal edge")
            assign_edge(
                original_indices[edge.local_u],
                original_indices[edge.local_v],
                edge.edge_label,
            )

    coarse_by_id = {
        edge.coarse_edge_id: edge for edge in factorization.coarse_topology.edges
    }
    if set(coarse_by_id) != set(range(len(coarse_by_id))):
        raise GraphValidationError("coarse edge ids must be contiguous from zero")
    attachment_counts: dict[int, int] = defaultdict(int)
    expected_attachment_ids = set(
        range(len(factorization.cross_fragment_attachments))
    )
    if {
        attachment.attachment_id
        for attachment in factorization.cross_fragment_attachments
    } != expected_attachment_ids:
        raise GraphValidationError("attachment ids must be contiguous from zero")

    for attachment in factorization.cross_fragment_attachments:
        coarse_edge = coarse_by_id.get(attachment.coarse_edge_id)
        if coarse_edge is None:
            raise GraphValidationError("attachment references an unknown coarse edge")
        fragment_pair = (
            attachment.left_fragment_id,
            attachment.right_fragment_id,
        )
        if fragment_pair != (
            coarse_edge.left_fragment_id,
            coarse_edge.right_fragment_id,
        ):
            raise GraphValidationError("attachment and coarse edge fragment pairs differ")
        left_indices = residual.original_node_indices_by_fragment[
            attachment.left_fragment_id
        ]
        right_indices = residual.original_node_indices_by_fragment[
            attachment.right_fragment_id
        ]
        if not 0 <= attachment.left_local_node < len(left_indices):
            raise GraphValidationError("invalid left attachment endpoint")
        if not 0 <= attachment.right_local_node < len(right_indices):
            raise GraphValidationError("invalid right attachment endpoint")
        assign_edge(
            left_indices[attachment.left_local_node],
            right_indices[attachment.right_local_node],
            attachment.edge_label,
        )
        attachment_counts[attachment.coarse_edge_id] += 1

    for coarse_edge_id, coarse_edge in coarse_by_id.items():
        if attachment_counts[coarse_edge_id] != coarse_edge.attachment_count:
            raise GraphValidationError("coarse edge attachment count is inconsistent")

    return ReconstructedGraph(
        node_mask=reconstructed_mask,
        node_labels=reconstructed_nodes,
        edge_labels=reconstructed_edges,
    )

