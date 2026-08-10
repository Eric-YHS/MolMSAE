"""Multi-scale molecular graph autoencoder with role-constrained latents."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class MolMSAEConfig:
    """Architecture configuration for MolMSAE."""

    max_nodes: int = 32
    num_atom_types: int = 32
    num_bond_types: int = 6
    num_fragment_kinds: int = 4
    hidden_dim: int = 128
    latent_dim: int = 128
    encoder_layers: int = 6
    attention_heads: int = 8
    dropout: float = 0.1


class DenseGraphLayer(nn.Module):
    """Edge-aware message passing on padded dense molecular graphs."""

    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.source = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.edge = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.update = nn.Sequential(
            nn.Linear(2 * hidden_dim, 2 * hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        nodes: torch.Tensor,
        edges: torch.Tensor,
        adjacency: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> torch.Tensor:
        pair_mask = adjacency.unsqueeze(-1).to(nodes.dtype)
        messages = self.source(nodes).unsqueeze(1) + self.edge(edges)
        degree = pair_mask.sum(dim=2).clamp_min(1.0)
        aggregated = (messages * pair_mask).sum(dim=2) / degree
        updated = self.update(torch.cat([nodes, aggregated], dim=-1))
        nodes = self.norm(nodes + updated)
        return nodes * node_mask.unsqueeze(-1).to(nodes.dtype)


class MolecularGraphEncoder(nn.Module):
    """Encode atom and bond tensors into contextualized node states."""

    def __init__(self, config: MolMSAEConfig) -> None:
        super().__init__()
        self.atom_embedding = nn.Embedding(config.num_atom_types, config.hidden_dim)
        self.bond_embedding = nn.Embedding(config.num_bond_types, config.hidden_dim)
        self.layers = nn.ModuleList(
            DenseGraphLayer(config.hidden_dim, config.dropout)
            for _ in range(config.encoder_layers)
        )
        self.final_norm = nn.LayerNorm(config.hidden_dim)

    def forward(
        self,
        atom_types: torch.Tensor,
        bond_types: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> torch.Tensor:
        nodes = self.atom_embedding(atom_types)
        edges = self.bond_embedding(bond_types)
        adjacency = (bond_types > 0) & node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
        nodes = nodes * node_mask.unsqueeze(-1).to(nodes.dtype)
        for layer in self.layers:
            nodes = layer(nodes, edges, adjacency, node_mask)
        return self.final_norm(nodes) * node_mask.unsqueeze(-1).to(nodes.dtype)


class MultiScaleTokenizer(nn.Module):
    """Read a node set into four learnable molecular scale tokens."""

    ROLE_NAMES = (
        "fragment_inventory",
        "coarse_topology",
        "fragment_contents",
        "cross_fragment_attachments",
    )

    def __init__(self, config: MolMSAEConfig) -> None:
        super().__init__()
        self.queries = nn.Parameter(torch.empty(4, config.latent_dim))
        self.source_projection = nn.Linear(config.hidden_dim, config.latent_dim)
        self.attention = nn.MultiheadAttention(
            config.latent_dim,
            config.attention_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(config.latent_dim)
        nn.init.normal_(self.queries, std=0.02)

    def forward(self, nodes: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        source = self.source_projection(nodes)
        queries = self.queries.unsqueeze(0).expand(nodes.shape[0], -1, -1)
        tokens, _ = self.attention(
            queries,
            source,
            source,
            key_padding_mask=~node_mask.bool(),
            need_weights=False,
        )
        return self.norm(tokens + queries)


class PairHead(nn.Module):
    """Predict symmetric pair labels from node-slot states."""

    def __init__(self, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        left = states.unsqueeze(2)
        right = states.unsqueeze(1)
        left = left.expand(-1, -1, states.shape[1], -1)
        right = right.expand(-1, states.shape[1], -1, -1)
        pair = torch.cat([left + right, torch.abs(left - right), left * right], dim=-1)
        return self.network(pair)


class RoleConditionedStage(nn.Module):
    """Decode one chemical scale from exactly one assigned latent token."""

    def __init__(self, config: MolMSAEConfig, use_context: bool) -> None:
        super().__init__()
        self.use_context = use_context
        self.token_projection = nn.Linear(config.latent_dim, config.hidden_dim)
        self.context_projection = (
            nn.Linear(config.hidden_dim, config.hidden_dim) if use_context else None
        )
        self.block = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, 2 * config.hidden_dim),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(2 * config.hidden_dim, config.hidden_dim),
        )

    def forward(
        self,
        token: torch.Tensor,
        node_queries: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        states = node_queries + self.token_projection(token).unsqueeze(1)
        if self.use_context:
            if context is None or self.context_projection is None:
                raise ValueError("this decoder stage requires earlier-scale context")
            states = states + self.context_projection(context.detach())
        return states + self.block(states)


class MultiScaleDecoder(nn.Module):
    """Hierarchical decoder with hard routing at the latent boundary."""

    def __init__(self, config: MolMSAEConfig) -> None:
        super().__init__()
        h = config.hidden_dim
        self.node_queries = nn.Parameter(torch.empty(config.max_nodes, h))
        self.stages = nn.ModuleList(
            [RoleConditionedStage(config, use_context=index > 0) for index in range(4)]
        )

        self.node_presence = nn.Linear(h, 1)
        self.fragment_kind = nn.Linear(h, config.num_fragment_kinds)
        self.fragment_size = nn.Linear(h, config.max_nodes + 1)
        self.coarse_topology = PairHead(h, 1)
        self.atom_type = nn.Linear(h, config.num_atom_types)
        self.internal_bond = PairHead(h, config.num_bond_types)
        self.attachment_bond = PairHead(h, config.num_bond_types)
        nn.init.normal_(self.node_queries, std=0.02)

    def forward(self, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        if tokens.ndim != 3 or tokens.shape[1] != 4:
            raise ValueError(f"expected [batch, 4, dim] tokens, got {tuple(tokens.shape)}")
        queries = self.node_queries.unsqueeze(0).expand(tokens.shape[0], -1, -1)
        stage1 = self.stages[0](tokens[:, 0], queries)
        stage2 = self.stages[1](tokens[:, 1], queries, stage1)
        stage3 = self.stages[2](tokens[:, 2], queries, stage2)
        stage4 = self.stages[3](tokens[:, 3], queries, stage3)

        return {
            "node_presence_logits": self.node_presence(stage1).squeeze(-1),
            "fragment_kind_logits": self.fragment_kind(stage1),
            "fragment_size_logits": self.fragment_size(stage1),
            "coarse_topology_logits": self.coarse_topology(stage2).squeeze(-1),
            "atom_type_logits": self.atom_type(stage3),
            "internal_bond_logits": self.internal_bond(stage3),
            "attachment_bond_logits": self.attachment_bond(stage4),
            "stage_states": torch.stack([stage1, stage2, stage3, stage4], dim=1),
        }


class MolMSAE(nn.Module):
    """Molecular Multi-Scale AutoEncoder."""

    def __init__(self, config: MolMSAEConfig | None = None) -> None:
        super().__init__()
        self.config = config or MolMSAEConfig()
        self.encoder = MolecularGraphEncoder(self.config)
        self.tokenizer = MultiScaleTokenizer(self.config)
        self.decoder = MultiScaleDecoder(self.config)

    def encode(
        self,
        atom_types: torch.Tensor,
        bond_types: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> torch.Tensor:
        nodes = self.encoder(atom_types, bond_types, node_mask)
        return self.tokenizer(nodes, node_mask)

    def decode(self, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.decoder(tokens)

    def forward(
        self,
        atom_types: torch.Tensor,
        bond_types: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        tokens = self.encode(atom_types, bond_types, node_mask)
        outputs = self.decode(tokens)
        outputs["tokens"] = tokens
        return outputs


def token_diversity_regularizer(tokens: torch.Tensor) -> torch.Tensor:
    """Penalize redundant directions while preserving token magnitudes."""

    normalized = F.normalize(tokens, dim=-1)
    similarity = normalized @ normalized.transpose(1, 2)
    identity = torch.eye(tokens.shape[1], device=tokens.device, dtype=tokens.dtype)
    return ((similarity - identity.unsqueeze(0)) ** 2).mean()

