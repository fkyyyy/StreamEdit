"""Local-weight torchvision RAFT backend."""

from __future__ import annotations

from pathlib import Path

import torch
from torchvision.models.optical_flow import (
    Raft_Large_Weights,
    raft_large,
)


class TorchvisionRAFT:
    """Estimate dense flow without downloading weights at runtime."""

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        device: str | torch.device = "cuda",
    ) -> None:
        checkpoint = Path(checkpoint).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"RAFT checkpoint not found: {checkpoint}")
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested for RAFT, but torch.cuda.is_available() is false"
            )

        model = raft_large(weights=None, progress=False)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=True)
        self.model = model.eval().to(self.device)
        self.transforms = Raft_Large_Weights.DEFAULT.transforms()

    @torch.inference_mode()
    def estimate_bidirectional(
        self,
        frame_a: torch.Tensor,
        frame_b: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return A->B and B->A flow for uint8/float RGB batches."""

        if frame_a.shape != frame_b.shape:
            raise ValueError("RAFT frame batches must have identical shapes")
        if frame_a.ndim != 4 or frame_a.shape[1] != 3:
            raise ValueError(
                "RAFT frames must have shape [B,3,H,W]"
            )
        if frame_a.shape[-2] % 8 or frame_a.shape[-1] % 8:
            raise ValueError(
                "RAFT frame height and width must be divisible by 8"
            )
        count = frame_a.shape[0]
        image_1 = torch.cat((frame_a, frame_b), dim=0).to(
            self.device, non_blocking=True
        )
        image_2 = torch.cat((frame_b, frame_a), dim=0).to(
            self.device, non_blocking=True
        )
        image_1, image_2 = self.transforms(image_1, image_2)
        prediction = self.model(image_1, image_2)[-1].float()
        forward, backward = prediction.split(count, dim=0)
        return forward, backward
