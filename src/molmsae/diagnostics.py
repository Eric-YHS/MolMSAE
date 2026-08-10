"""Diagnostics for multi-scale latent organization and decoder sensitivity."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
import torch.nn.functional as F


def linear_cka(left: np.ndarray, right: np.ndarray) -> float:
    """Centered linear CKA between two sample-by-feature matrices."""

    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
        raise ValueError("CKA inputs must be 2-D and share the sample axis")
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    cross = np.linalg.norm(x.T @ y, ord="fro") ** 2
    scale = np.linalg.norm(x.T @ x, ord="fro") * np.linalg.norm(y.T @ y, ord="fro")
    return float(cross / scale) if scale > 0 else 0.0


def effective_rank(features: np.ndarray) -> float:
    """Entropy-based effective rank of centered representations."""

    matrix = np.asarray(features, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("features must be a 2-D matrix")
    matrix = matrix - matrix.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    energy = singular_values**2
    if float(energy.sum()) == 0.0:
        return 0.0
    probabilities = energy / energy.sum()
    probabilities = probabilities[probabilities > 0]
    return float(np.exp(-(probabilities * np.log(probabilities)).sum()))


def token_cosine_similarity(tokens: torch.Tensor) -> torch.Tensor:
    """Return the batch-mean cosine matrix between latent roles."""

    if tokens.ndim != 3:
        raise ValueError("tokens must have shape [batch, scales, dim]")
    normalized = F.normalize(tokens, dim=-1)
    return torch.einsum("bkd,bld->bkl", normalized, normalized).mean(dim=0)


def intervene_tokens(
    tokens: torch.Tensor,
    intervention: str,
    token_index: int | None = None,
    permutation: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply controlled token ablations used in representation diagnostics."""

    output = tokens.clone()
    if intervention == "permute":
        if permutation is None:
            raise ValueError("permute intervention requires a permutation")
        return output[:, permutation]
    if token_index is None or not 0 <= token_index < tokens.shape[1]:
        raise ValueError("a valid token_index is required")
    token_mean = tokens[:, token_index].mean(dim=0, keepdim=True)
    if intervention == "mean_replace":
        output[:, token_index] = token_mean
    elif intervention == "batch_shuffle":
        donors = torch.roll(torch.arange(tokens.shape[0], device=tokens.device), 1)
        output[:, token_index] = tokens[donors, token_index]
    elif intervention == "keep_only":
        means = tokens.mean(dim=0, keepdim=True)
        output = means.expand_as(tokens).clone()
        output[:, token_index] = tokens[:, token_index]
    else:
        raise ValueError(f"unknown intervention: {intervention}")
    return output


def projected_decoder_jacobian(
    decoder_observable: Callable[[torch.Tensor], torch.Tensor],
    tokens: torch.Tensor,
    projections: int = 32,
    seed: int = 0,
) -> torch.Tensor:
    """Sketch decoder-sensitive latent directions using random-output VJPs."""

    if projections <= 0:
        raise ValueError("projections must be positive")
    latent = tokens.detach().clone().requires_grad_(True)
    generator = torch.Generator(device=latent.device)
    generator.manual_seed(seed)
    rows: list[torch.Tensor] = []
    for _ in range(projections):
        observable = decoder_observable(latent).reshape(latent.shape[0], -1)
        direction = torch.randn(
            observable.shape,
            generator=generator,
            device=observable.device,
            dtype=observable.dtype,
        ) / np.sqrt(observable.shape[-1])
        scalar = (observable * direction).sum()
        gradient = torch.autograd.grad(scalar, latent, retain_graph=False)[0]
        rows.append(gradient.detach().flatten(start_dim=1))
    return torch.stack(rows, dim=1)


def jacobian_token_summary(sketch: torch.Tensor, num_tokens: int = 4) -> list[dict[str, float]]:
    """Summarize energy and effective rank for each latent role."""

    if sketch.ndim != 3 or sketch.shape[-1] % num_tokens:
        raise ValueError("sketch must be [batch, projections, num_tokens * dim]")
    token_dim = sketch.shape[-1] // num_tokens
    reshaped = sketch.reshape(sketch.shape[0], sketch.shape[1], num_tokens, token_dim)
    energy = reshaped.square().sum(dim=(0, 1, 3))
    total = energy.sum().clamp_min(torch.finfo(energy.dtype).eps)
    rows: list[dict[str, float]] = []
    for token_index in range(num_tokens):
        matrix = reshaped[:, :, token_index].reshape(-1, token_dim).cpu().numpy()
        rows.append(
            {
                "token_index": token_index,
                "energy_fraction": float((energy[token_index] / total).item()),
                "effective_rank": effective_rank(matrix),
            }
        )
    return rows

