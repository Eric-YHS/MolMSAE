"""Dataset and hierarchical target construction for padded molecular graphs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .factorization import factorize_graph


FRAGMENT_KIND_TO_ID = {
    "padding": 0,
    "singleton": 1,
    "acyclic_component": 2,
    "ring_system": 3,
}


@dataclass(frozen=True)
class MoleculeArrays:
    atom_types: np.ndarray
    bond_types: np.ndarray
    node_mask: np.ndarray


class MolecularGraphDataset(Dataset[MoleculeArrays]):
    """Read fixed-width molecular graph tensors from an ``.npz`` archive.

    Required arrays are ``atom_types [M,N]``, ``bond_types [M,N,N]`` and
    ``node_mask [M,N]``. Bond label zero denotes a non-edge.
    """

    def __init__(self, path: str | Path) -> None:
        archive = np.load(Path(path), allow_pickle=False)
        required = {"atom_types", "bond_types", "node_mask"}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"dataset is missing arrays: {sorted(missing)}")
        self.atom_types = np.asarray(archive["atom_types"])
        self.bond_types = np.asarray(archive["bond_types"])
        self.node_mask = np.asarray(archive["node_mask"], dtype=bool)
        count, width = self.atom_types.shape
        if self.bond_types.shape != (count, width, width):
            raise ValueError("bond_types must have shape [molecules, nodes, nodes]")
        if self.node_mask.shape != (count, width):
            raise ValueError("node_mask must have shape [molecules, nodes]")

    def __len__(self) -> int:
        return int(self.atom_types.shape[0])

    def __getitem__(self, index: int) -> MoleculeArrays:
        return MoleculeArrays(
            atom_types=self.atom_types[index],
            bond_types=self.bond_types[index],
            node_mask=self.node_mask[index],
        )


def hierarchical_targets(example: MoleculeArrays) -> dict[str, np.ndarray]:
    """Convert one molecule into the four scale-specific supervision targets."""

    factorization = factorize_graph(
        example.node_mask,
        example.atom_types,
        example.bond_types,
    )
    width = len(example.node_mask)
    fragment_ids = np.full(width, width, dtype=np.int64)
    fragment_kind = np.zeros(width, dtype=np.int64)
    fragment_size = np.zeros(width, dtype=np.int64)

    nodes_by_fragment = factorization.roundtrip_residual.original_node_indices_by_fragment
    for item, original_nodes in zip(factorization.fragment_inventory, nodes_by_fragment):
        nodes = np.asarray(original_nodes, dtype=np.int64)
        fragment_ids[nodes] = item.fragment_id
        fragment_kind[nodes] = FRAGMENT_KIND_TO_ID[item.fragment_kind]
        fragment_size[nodes] = item.num_nodes

    active_pairs = example.node_mask[:, None] & example.node_mask[None, :]
    same_fragment = (
        fragment_ids[:, None] == fragment_ids[None, :]
    ) & active_pairs
    coarse_topology = np.zeros((width, width), dtype=bool)
    for edge in factorization.coarse_topology.edges:
        left_nodes = np.asarray(nodes_by_fragment[edge.left_fragment_id], dtype=np.int64)
        right_nodes = np.asarray(nodes_by_fragment[edge.right_fragment_id], dtype=np.int64)
        coarse_topology[np.ix_(left_nodes, right_nodes)] = True
        coarse_topology[np.ix_(right_nodes, left_nodes)] = True

    internal_bond = np.where(same_fragment, example.bond_types, 0)
    attachment_bond = np.where(active_pairs & ~same_fragment, example.bond_types, 0)
    return {
        "node_presence": example.node_mask.astype(np.float32),
        "fragment_ids": fragment_ids,
        "fragment_kind": fragment_kind,
        "fragment_size": fragment_size,
        "same_fragment": same_fragment,
        "coarse_topology": coarse_topology,
        "atom_types": np.asarray(example.atom_types, dtype=np.int64),
        "internal_bond": np.asarray(internal_bond, dtype=np.int64),
        "attachment_bond": np.asarray(attachment_bond, dtype=np.int64),
    }


def collate_molecules(examples: list[MoleculeArrays]) -> dict[str, Any]:
    """Collate molecular inputs and deterministic multi-scale targets."""

    targets = [hierarchical_targets(example) for example in examples]
    return {
        "atom_types": torch.as_tensor(np.stack([x.atom_types for x in examples])).long(),
        "bond_types": torch.as_tensor(np.stack([x.bond_types for x in examples])).long(),
        "node_mask": torch.as_tensor(np.stack([x.node_mask for x in examples])).bool(),
        "targets": {
            key: torch.as_tensor(np.stack([target[key] for target in targets]))
            for key in targets[0]
        },
    }

