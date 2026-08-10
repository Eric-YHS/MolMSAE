"""Train MolMSAE on a fixed-width molecular graph archive."""

from __future__ import annotations

import argparse
from dataclasses import fields
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, random_split

from molmsae.data import MolecularGraphDataset, collate_molecules
from molmsae.losses import multiscale_reconstruction_loss
from molmsae.model import MolMSAE, MolMSAEConfig


def move_to_device(value, device: torch.device):
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if torch.is_tensor(value):
        return value.to(device)
    return value


def evaluate(model, loader, device: torch.device) -> float:
    model.eval()
    total = 0.0
    batches = 0
    with torch.no_grad():
        for batch in loader:
            batch = move_to_device(batch, device)
            outputs = model(batch["atom_types"], batch["bond_types"], batch["node_mask"])
            loss, _ = multiscale_reconstruction_loss(outputs, batch["targets"])
            total += float(loss.item())
            batches += 1
    return total / max(batches, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pcqm4m32.yaml")
    parser.add_argument("--data", required=True, help="NPZ molecular graph archive")
    parser.add_argument("--output", default="outputs/molmsae")
    args = parser.parse_args()

    settings = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    model_keys = {item.name for item in fields(MolMSAEConfig)}
    model_config = MolMSAEConfig(
        **{key: value for key, value in settings["model"].items() if key in model_keys}
    )
    dataset = MolecularGraphDataset(args.data)
    valid_size = max(1, int(len(dataset) * settings["training"]["validation_fraction"]))
    train_size = len(dataset) - valid_size
    train_set, valid_set = random_split(
        dataset,
        [train_size, valid_size],
        generator=torch.Generator().manual_seed(settings["training"]["seed"]),
    )
    loader_options = {
        "batch_size": settings["training"]["batch_size"],
        "num_workers": settings["training"]["num_workers"],
        "collate_fn": collate_molecules,
    }
    train_loader = DataLoader(train_set, shuffle=True, **loader_options)
    valid_loader = DataLoader(valid_set, shuffle=False, **loader_options)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MolMSAE(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(settings["training"]["learning_rate"]),
        weight_decay=float(settings["training"]["weight_decay"]),
    )
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_validation = float("inf")

    for epoch in range(1, int(settings["training"]["epochs"]) + 1):
        model.train()
        for batch in train_loader:
            batch = move_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch["atom_types"], batch["bond_types"], batch["node_mask"])
            loss, _ = multiscale_reconstruction_loss(outputs, batch["targets"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        validation = evaluate(model, valid_loader, device)
        print(f"epoch={epoch:03d} validation_loss={validation:.6f}")
        if validation < best_validation:
            best_validation = validation
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": settings,
                    "epoch": epoch,
                    "validation_loss": validation,
                },
                output_dir / "best.pt",
            )


if __name__ == "__main__":
    main()

