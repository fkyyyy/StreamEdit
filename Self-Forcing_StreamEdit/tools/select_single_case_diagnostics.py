#!/usr/bin/env python3
"""Select one diagnostic frame/query tuple from saved Full-run evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F


ROLE_NAMES = ("q_object", "q_interface", "q_hand", "q_background")


def _map(value, frames):
    value = torch.as_tensor(value).detach().float().cpu()
    if value.ndim == 5 and value.shape[0] == 1 and value.shape[2] == 1:
        value = value[0, :, 0]
    elif value.ndim == 4 and value.shape[0] == 1:
        value = value[0]
    elif value.ndim == 4 and value.shape[1] == 1:
        value = value[:, 0]
    if value.ndim != 3 or value.shape[0] != frames:
        raise ValueError(f"Expected [T,H,W], got {tuple(value.shape)}")
    return value


def _roles_on_grid(payload, shape, frames):
    first = payload["first_response"]
    roles = torch.stack([_map(first[name], frames) for name in ROLE_NAMES], 1)
    if tuple(roles.shape[-2:]) != tuple(shape):
        roles = F.interpolate(
            roles,
            size=shape,
            mode="bilinear",
            align_corners=False,
        )
        roles = roles.clamp_min(0)
        roles = roles / roles.sum(1, keepdim=True).clamp_min(1e-6)
    return roles


def select(responsibility_dir: Path):
    candidates = []
    for path in sorted(responsibility_dir.glob("block_*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        indices = torch.as_tensor(payload["latent_frame_indices"]).tolist()
        frames = len(indices)
        support = _map(payload["initial"]["connected_support"], frames) > 0
        hand = _map(payload["initial"]["hand_occupancy"], frames).clamp(0, 1)
        overlap = (support.float() * hand).flatten(1).sum(1)
        roles = _roles_on_grid(payload, support.shape[-2:], frames)
        valid = _map(payload["valid_token"], frames) > 0
        for local, global_index in enumerate(indices):
            candidates.append((
                float(overlap[local]),
                int(global_index),
                int(payload["block_index"]),
                local,
                path,
                roles,
                support,
                hand,
                valid,
            ))
    if not candidates:
        raise RuntimeError("No responsibility block artifacts found")
    selected = max(candidates, key=lambda item: (item[0], -item[1]))
    overlap_score, global_index, block_index, local, path, roles, support, hand, valid = selected
    height, width = support.shape[-2:]
    connected = support[local] & valid[local]
    interface_region = connected & (hand[local] > 0)
    if not interface_region.any():
        raise RuntimeError("Selected frame has no hand-supported connected object token")
    object_region = connected & (hand[local] <= 0.10)
    if not object_region.any():
        raise RuntimeError("Selected frame has no low-hand object-core token")
    hand_region = valid[local] & (hand[local] > 0)
    if not hand_region.any():
        raise RuntimeError("Selected frame has no valid hand token")

    def argmax_in(field, region):
        score = roles[local, field].masked_fill(~region, -torch.inf)
        flat = int(score.argmax())
        row, col = divmod(flat, width)
        query = local * height * width + flat
        return {
            "query_index": query,
            "local_frame": local,
            "row": row,
            "column": col,
            "score": float(score[row, col]),
            **{
                name: float(roles[local, role_index, row, col])
                for role_index, name in enumerate(ROLE_NAMES)
            },
        }

    queries = {
        "interface_query": argmax_in(1, interface_region),
        "object_query": argmax_in(0, object_region),
        "hand_query": argmax_in(2, hand_region),
    }
    return {
        "selection_rule": "maximum hand occupancy times connected-support area; no edited-output inspection",
        "source_artifact": str(path.resolve()),
        "block_index": block_index,
        "global_latent_frame": global_index,
        "local_frame": local,
        "token_grid": [height, width],
        "hand_object_overlap_mass": overlap_score,
        "queries": queries,
        "query_indices": [
            queries[name]["query_index"]
            for name in ("interface_query", "object_query", "hand_query")
        ],
        "role_sum_max_error": float(
            (roles.sum(1) - 1).abs().max()
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--responsibility-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = select(args.responsibility_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
