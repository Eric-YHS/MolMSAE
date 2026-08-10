"""MolMSAE: multi-scale representation learning for molecular graphs."""

from .diagnostics import effective_rank, linear_cka, token_cosine_similarity
from .factorization import (
    DecoderTargets,
    GraphFactorization,
    GraphValidationError,
    assemble_graph,
    factorize_graph,
)
from .model import MolMSAE, MolMSAEConfig

__all__ = [
    "DecoderTargets",
    "GraphFactorization",
    "GraphValidationError",
    "MolMSAE",
    "MolMSAEConfig",
    "assemble_graph",
    "effective_rank",
    "factorize_graph",
    "linear_cka",
    "token_cosine_similarity",
]

__version__ = "0.1.0"

