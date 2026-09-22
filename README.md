# MolMSAE

![CI](https://github.com/Eric-YHS/MolMSAE/workflows/CI/badge.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)

**Multi-Scale Representation Learning and Autoencoding for Molecular Graphs**

MolMSAE is a molecular graph autoencoder designed to study how structural
information is organized across multiple latent variables. It decomposes graph
reconstruction into chemically meaningful scales and assigns a dedicated latent
role to fragment inventory, fragment-level topology, intra-fragment structure,
and cross-fragment attachments. The resulting representation is intended to
support accurate reconstruction, transferable molecular features, and an
interpretable latent organization.

## Motivation

Multi-token graph autoencoders are commonly optimized through a single
reconstruction objective. Although the full latent set may reconstruct the
input graph successfully, the objective alone does not require individual
tokens to develop stable or complementary semantics. This can lead to highly
shared latent directions, order-sensitive downstream interfaces, and a latent
geometry that is difficult to interpret through chemical edits.

MolMSAE introduces explicit scale-aware roles at the decoder boundary while
retaining a set-based graph encoder. Each latent variable is responsible for a
different level of molecular structure:

| Latent | Structural scale | Information represented |
| --- | --- | --- |
| $z_0$ | Fragment inventory | Fragment types, sizes, and membership |
| $z_1$ | Coarse topology | Connectivity between molecular fragments |
| $z_2$ | Fragment contents | Atom types and intra-fragment bonds |
| $z_3$ | Cross-fragment attachments | Attachment sites, bond types, and local constraints |

~~~mermaid
flowchart LR
    A["Molecular graph"] --> B["Edge-aware graph encoder"]
    B --> C["Multi-scale latent tokenizer"]
    C --> Z0["z0: fragment inventory"]
    C --> Z1["z1: coarse topology"]
    C --> Z2["z2: fragment contents"]
    C --> Z3["z3: attachments"]
    Z0 --> D["Role-constrained hierarchical decoder"]
    Z1 --> D
    Z2 --> D
    Z3 --> D
    D --> E["Reconstructed molecular graph"]
~~~

The decoder follows a hierarchical reconstruction process. Every stage reads
only its assigned continuous latent token and receives detached context from
earlier structural scales. This preserves dependencies between scales while
preventing later tokens from bypassing their assigned roles by directly
copying the complete molecule.

## Key Components

- **Lossless hierarchical factorization** uses graph bridges, biconnected
  components, and ring-system merging to decompose a molecule into fragment
  inventory, coarse topology, fragment contents, and cross-fragment
  attachments.
- **Multi-scale latent tokenization** uses four learnable queries to extract
  fixed-width graph-level representations from contextualized atom states.
- **Role-constrained decoding** reconstructs molecular structure in the order
  of fragment inventory, fragment topology, intra-fragment structure, and
  cross-fragment connectivity.
- **Latent-space diagnostics** include token permutation and ablation, linear
  CKA, effective rank, cosine similarity, and random-projection decoder
  Jacobian analysis.
- **Strict graph-tensor auditing** supports exact round-trip assembly while
  separating serialization-only node indices from all decoder-visible model
  inputs.

## Repository Structure

~~~text
MolMSAE/
├── configs/                 # Model and training configurations
├── scripts/                 # Latent-space analysis utilities
├── src/molmsae/
│   ├── factorization.py     # Lossless multi-scale graph decomposition
│   ├── model.py             # Encoder, latent tokenizer, and decoder
│   ├── data.py              # Dataset interface and target construction
│   ├── losses.py            # Multi-scale reconstruction objectives
│   └── diagnostics.py       # Representation and sensitivity diagnostics
├── tests/                   # Factorization, routing, and diagnostic tests
└── train.py                 # Training entry point
~~~

## Installation

~~~bash
git clone https://github.com/Eric-YHS/MolMSAE.git
cd MolMSAE
pip install -e ".[dev]"
~~~

## Data Format

The training pipeline reads a fixed-width NumPy graph archive with the
following arrays:

~~~text
atom_types  [num_molecules, max_nodes]
bond_types  [num_molecules, max_nodes, max_nodes]
node_mask   [num_molecules, max_nodes]
~~~

Bond label <code>0</code> represents a non-edge. Bond matrices must be
symmetric, and padded nodes cannot be incident to active edges. Multi-scale
supervision is generated deterministically from each molecular graph.

## Training

~~~bash
python train.py \
  --config configs/pcqm4m32.yaml \
  --data /path/to/molecular_graphs.npz \
  --output outputs/pcqm4m32
~~~

## Latent-Space Analysis

After exporting encoded representations as a NumPy array with shape
<code>[samples, 4, latent_dim]</code>, latent organization can be summarized
with:

~~~bash
python scripts/analyze_latent.py outputs/latents.npy \
  --output outputs/latent_report.json
~~~

The report contains joint and per-token effective rank, pairwise linear CKA,
and token cosine similarity. These measurements characterize semantic overlap,
complementarity, and effective latent capacity.

## Tests

~~~bash
pytest
~~~

The test suite covers rings, chains, branched molecules, disconnected graphs,
and sparse padded tensors. It also verifies deterministic factorization, exact
round-trip reconstruction, and gradient isolation across decoder roles.

## Development

~~~bash
pip install -e ".[dev]"
ruff check .        # lint (import order, unused code, modern typing)
pytest              # unit tests
~~~

