"""Multi-scale reconstruction objectives for MolMSAE."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .model import token_diversity_regularizer


def _masked_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    selected = mask.bool()
    if not torch.any(selected):
        return logits.sum() * 0.0
    return F.cross_entropy(logits[selected], targets.long()[selected])


def _balanced_binary_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    selected = mask.bool()
    if not torch.any(selected):
        return logits.sum() * 0.0
    values = targets.bool()[selected]
    scores = logits[selected]
    positive = values
    negative = ~values
    losses = []
    if torch.any(positive):
        losses.append(F.softplus(-scores[positive]).mean())
    if torch.any(negative):
        losses.append(F.softplus(scores[negative]).mean())
    return torch.stack(losses).mean()


def multiscale_reconstruction_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    stage_weights: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
    diversity_weight: float = 0.02,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute role-aligned losses for inventory, topology, content and attachment."""

    active = targets["node_presence"].bool()
    pair_mask = active.unsqueeze(1) & active.unsqueeze(2)
    diagonal = torch.eye(active.shape[1], device=active.device, dtype=torch.bool)
    pair_mask = pair_mask & ~diagonal.unsqueeze(0)
    same_fragment = targets["same_fragment"].bool()

    presence = F.binary_cross_entropy_with_logits(
        outputs["node_presence_logits"], targets["node_presence"].float()
    )
    fragment_kind = _masked_cross_entropy(
        outputs["fragment_kind_logits"], targets["fragment_kind"], active
    )
    fragment_size = _masked_cross_entropy(
        outputs["fragment_size_logits"], targets["fragment_size"], active
    )
    stage1 = presence + fragment_kind + fragment_size

    stage2 = _balanced_binary_cross_entropy(
        outputs["coarse_topology_logits"],
        targets["coarse_topology"],
        pair_mask & ~same_fragment,
    )

    atom_type = _masked_cross_entropy(
        outputs["atom_type_logits"], targets["atom_types"], active
    )
    internal_bond = _masked_cross_entropy(
        outputs["internal_bond_logits"],
        targets["internal_bond"],
        pair_mask & same_fragment,
    )
    stage3 = atom_type + internal_bond

    stage4 = _masked_cross_entropy(
        outputs["attachment_bond_logits"],
        targets["attachment_bond"],
        pair_mask & ~same_fragment,
    )
    diversity = token_diversity_regularizer(outputs["tokens"])
    total = sum(
        weight * loss
        for weight, loss in zip(stage_weights, (stage1, stage2, stage3, stage4))
    ) + diversity_weight * diversity
    metrics = {
        "loss/total": total.detach(),
        "loss/inventory": stage1.detach(),
        "loss/topology": stage2.detach(),
        "loss/fragment_content": stage3.detach(),
        "loss/attachment": stage4.detach(),
        "loss/token_diversity": diversity.detach(),
    }
    return total, metrics

