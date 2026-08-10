import torch

from molmsae.model import MolMSAE, MolMSAEConfig


def compact_config() -> MolMSAEConfig:
    return MolMSAEConfig(
        max_nodes=8,
        num_atom_types=12,
        num_bond_types=5,
        hidden_dim=32,
        latent_dim=32,
        encoder_layers=2,
        attention_heads=4,
        dropout=0.0,
    )


def graph_batch():
    atom_types = torch.tensor([[6, 6, 8, 0, 0, 0, 0, 0], [6, 7, 6, 8, 0, 0, 0, 0]])
    node_mask = torch.tensor(
        [[True, True, True, False, False, False, False, False],
         [True, True, True, True, False, False, False, False]]
    )
    bonds = torch.zeros(2, 8, 8, dtype=torch.long)
    bonds[0, 0, 1] = bonds[0, 1, 0] = 1
    bonds[0, 1, 2] = bonds[0, 2, 1] = 2
    bonds[1, 0, 1] = bonds[1, 1, 0] = 1
    bonds[1, 1, 2] = bonds[1, 2, 1] = 1
    bonds[1, 2, 3] = bonds[1, 3, 2] = 2
    return atom_types, bonds, node_mask


def test_forward_shapes():
    model = MolMSAE(compact_config())
    atom_types, bonds, node_mask = graph_batch()
    outputs = model(atom_types, bonds, node_mask)

    assert outputs["tokens"].shape == (2, 4, 32)
    assert outputs["node_presence_logits"].shape == (2, 8)
    assert outputs["fragment_kind_logits"].shape == (2, 8, 4)
    assert outputs["coarse_topology_logits"].shape == (2, 8, 8)
    assert outputs["atom_type_logits"].shape == (2, 8, 12)
    assert outputs["attachment_bond_logits"].shape == (2, 8, 8, 5)


def test_decoder_stage_gradients_are_role_isolated():
    model = MolMSAE(compact_config())
    objectives = [
        "node_presence_logits",
        "coarse_topology_logits",
        "atom_type_logits",
        "attachment_bond_logits",
    ]
    for expected_token, objective in enumerate(objectives):
        tokens = torch.randn(2, 4, 32, requires_grad=True)
        model.decode(tokens)[objective].sum().backward()
        token_gradient = tokens.grad.abs().sum(dim=(0, 2))
        assert token_gradient[expected_token] > 0
        assert torch.count_nonzero(token_gradient > 0).item() == 1

