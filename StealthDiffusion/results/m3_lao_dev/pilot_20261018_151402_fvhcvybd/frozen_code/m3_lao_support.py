"""Small, model-independent helpers for opt-in LAO architecture experiments."""
import math

import numpy as np
import torch


def validate_control_scale(value):
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("--control_scale must be finite and in [0, 1]")
    return value


def scale_control_features(features, scale):
    """Keep the exact original objects and arithmetic when scale is one."""
    scale = validate_control_scale(scale)
    if scale == 1.0:
        return features
    return [feature * scale for feature in features]


def require_finite(value, name):
    if value is None:
        raise RuntimeError(f"LAO {name} is missing; check the gradient path")
    if not bool(torch.isfinite(value).all().item()):
        raise FloatingPointError(f"LAO {name} contains NaN or infinity")


def quantize_like_save(image):
    """Exact CPU NumPy uint8 truncation used by m2_stealth.to_uint8, per batch.

    Returned values are those obtained by reloading the PNG into [0, 1]. This
    helper intentionally does not use torch.round or a GPU multiplication.
    """
    require_finite(image, "image before saved-image quantization")
    pixels = np.uint8(image.detach().clamp(0, 1).cpu().numpy() * 255.0)
    return torch.from_numpy(pixels.astype(np.float32) / 255.0).to(
        device=image.device, dtype=image.dtype)


class _SavedImageSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, image):
        # A custom Function avoids cancellation error in x + (q - x).detach().
        return quantize_like_save(image)

    @staticmethod
    def backward(ctx, gradient):
        return gradient


def saved_image_ste(image):
    """Use saved-image values forward, and an identity surrogate backward."""
    return _SavedImageSTE.apply(image)


def make_latent_optimizer(latent, lr, weight_decay=0.01):
    if not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("--lao_weight_decay must be finite and non-negative")
    return torch.optim.AdamW([latent], lr=lr, weight_decay=weight_decay)


def tensor_norms(value, name):
    require_finite(value, name)
    value = value.detach().float()
    return {"l2": float(torch.linalg.vector_norm(value).item()),
            "linf": float(value.abs().max().item())}


def changed_pixel_fraction(first_u8, second_u8):
    """Fraction of spatial pixels whose saved RGB values differ in any channel."""
    if first_u8.shape != second_u8.shape or first_u8.ndim != 3:
        raise ValueError("saved-image comparison requires matching HWC arrays")
    return float(np.any(first_u8 != second_u8, axis=-1).mean())
