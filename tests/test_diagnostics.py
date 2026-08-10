import numpy as np
import torch

from molmsae.diagnostics import (
    effective_rank,
    intervene_tokens,
    linear_cka,
    token_cosine_similarity,
)


def test_linear_cka_detects_shared_geometry():
    rng = np.random.default_rng(7)
    features = rng.normal(size=(128, 16))
    assert linear_cka(features, features @ rng.normal(size=(16, 16))) > 0.2
    np.testing.assert_allclose(linear_cka(features, features), 1.0, atol=1e-10)


def test_effective_rank_matches_axis_aligned_basis():
    features = np.tile(np.eye(4), (32, 1))
    assert 2.9 < effective_rank(features) <= 4.0


def test_token_interventions_preserve_shape():
    tokens = torch.randn(12, 4, 8)
    replaced = intervene_tokens(tokens, "mean_replace", token_index=2)
    kept = intervene_tokens(tokens, "keep_only", token_index=1)
    permuted = intervene_tokens(
        tokens, "permute", permutation=torch.tensor([3, 2, 1, 0])
    )
    assert replaced.shape == kept.shape == permuted.shape == tokens.shape
    assert torch.allclose(permuted[:, 0], tokens[:, 3])
    assert token_cosine_similarity(tokens).shape == (4, 4)

