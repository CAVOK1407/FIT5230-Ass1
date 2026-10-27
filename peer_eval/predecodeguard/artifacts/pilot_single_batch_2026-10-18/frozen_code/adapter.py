"""CPU reference wrapper around the peer's verified frozen architecture."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

from .contract import SHAPE, package_manifest


class FrozenGuard:
    def __init__(self, package: Path):
        self.package = Path(package).resolve()
        self.manifest = package_manifest(self.package)
        torch.set_num_threads(4)  # Same reference execution as the supplied scorer.
        # model.py was read/reviewed and is byte-pinned before it is imported.
        spec = importlib.util.spec_from_file_location("predecodeguard_verified_model", self.package / "model.py")
        module = importlib.util.module_from_spec(spec)
        previous_bytecode = sys.dont_write_bytecode
        try:
            sys.dont_write_bytecode = True
            spec.loader.exec_module(module)
        finally:
            sys.dont_write_bytecode = previous_bytecode
        state = torch.load(self.package / self.manifest["checkpoint_file"], map_location="cpu", weights_only=True)
        self.model = module.SpatialLatentCNN(dropout=0.2).eval()
        self.model.load_state_dict(state["state_dict"], strict=True)
        self.model.requires_grad_(False)
        self.mean = state["channel_mean"].reshape(1, 16, 1, 1).to(dtype=torch.float32, device="cpu")
        self.std = state["channel_std"].reshape(1, 16, 1, 1).to(dtype=torch.float32, device="cpu")
        if not torch.isfinite(self.mean).all() or not torch.isfinite(self.std).all() or not torch.all(self.std > 0):
            raise ValueError("Invalid frozen channel statistics")
        self.threshold = self.manifest["threshold"]

    def logits(self, raw: torch.Tensor) -> torch.Tensor:
        if tuple(raw.shape) != SHAPE or raw.device.type != "cpu":
            raise ValueError("Expected one CPU latent with the package shape")
        if self.model.training:
            raise RuntimeError("Frozen guard must remain in eval mode")
        return self.model((raw.to(torch.float32) - self.mean) / self.std)

    def score(self, array: np.ndarray) -> float:
        if array.dtype != np.float16 or array.shape != SHAPE or not np.isfinite(array).all():
            raise ValueError("Reference scores require a finite saved-representation float16 latent")
        with torch.inference_mode():
            return float(torch.sigmoid(self.logits(torch.from_numpy(array.astype(np.float32)))).item())

    def flagged(self, score: float) -> bool:
        return score >= self.threshold
