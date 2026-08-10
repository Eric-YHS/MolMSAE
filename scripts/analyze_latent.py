"""Summarize role similarity and effective capacity from exported latents."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from molmsae.diagnostics import effective_rank, linear_cka, token_cosine_similarity


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("latents", help="NumPy file with shape [samples, scales, dim]")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    latents = np.load(args.latents)
    if latents.ndim != 3:
        raise ValueError("latents must have shape [samples, scales, dim]")
    report = {
        "shape": list(latents.shape),
        "joint_effective_rank": effective_rank(latents.reshape(len(latents), -1)),
        "token_effective_rank": [effective_rank(latents[:, i]) for i in range(latents.shape[1])],
        "token_cosine_similarity": token_cosine_similarity(torch.from_numpy(latents)).tolist(),
        "token_cka": [
            [linear_cka(latents[:, i], latents[:, j]) for j in range(latents.shape[1])]
            for i in range(latents.shape[1])
        ],
    }
    serialized = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(serialized + "\n", encoding="utf-8")
    else:
        print(serialized)


if __name__ == "__main__":
    main()

