"""LIU JUNPENG's Spectral Fingerprint Alignment (SFA), integrated for M3.

Adapted from FIT5230_M3_Individual_LIU_JUNPENG_SFA_real_real.ipynb.
Method: minimize the mean surrogate AI margin plus lambda times radial-spectrum
MSE to a real-image reference. PGD and spectrum-only use the same optimizer.
Integration fixes: dynamic sizes, explicit class convention and surrogate set,
no detector calls in SPEC, consistent objective scaling, device-safe integer
projection, and actual disk PNG readback. This module does not train models.

Caller supplies frozen, eval-mode detectors and a reference fitted only on the
reference/development split. The input is quantized once; all budget/quality
claims are relative to that canonical uint8 RGB baseline. Model inputs are
float32 RGB in [0,1], BEFORE the model-specific preprocessing in logits_fn.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
from PIL import Image
import torch

STRATEGY_AUTHOR = "LIU JUNPENG"
STRATEGY_NAME = "Spectral Fingerprint Alignment (SFA)"
DEFAULTS = {"budget_levels": 8, "alpha": 2 / 255, "steps": 50, "seed": 5230}
LogitsFn = Callable[[torch.Tensor, Sequence[str]], Mapping[str, torch.Tensor]]


def _check_rgb(images: torch.Tensor) -> None:
    if not isinstance(images, torch.Tensor) or images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("Expected RGB tensor [N,3,H,W].")
    if images.shape[0] < 1 or min(images.shape[-2:]) < 2:
        raise ValueError("Expected a nonempty batch and H,W >= 2.")
    if not torch.is_floating_point(images):
        raise TypeError("Expected floating RGB values in [0,1], not raw uint8.")
    if not bool(torch.isfinite(images).all()):
        raise ValueError("Non-finite RGB input.")
    if bool((images < 0).any()) or bool((images > 1).any()):
        raise ValueError("RGB input must be in [0,1]; preprocessing belongs in logits_fn.")


def quantize_rgb(images: torch.Tensor) -> torch.Tensor:
    """CPU uint8 [N,3,H,W], using nearest-integer (torch round) quantization."""
    _check_rgb(images)
    return (images.detach().to(device="cpu", dtype=torch.float32) * 255).round().to(torch.uint8)


def canonical_rgb(images: torch.Tensor) -> torch.Tensor:
    """Canonical original on the same device, float32 multiples of 1/255."""
    return quantize_rgb(images).to(device=images.device, dtype=torch.float32) / 255


def radial_logpower(images: torch.Tensor) -> torch.Tensor:
    """Differentiable luminance radial log power, [N,min(H,W)//2].

    Preserves Liu's unnormalized rfft2, rounded normalized-radius bins and
    log1p(mean power) definition. No fixed-size cache or device global is used.
    Consequently reference and generated images MUST have the same H,W.
    FFT is performed in float32 for half inputs, retaining input gradients.
    """
    if images.ndim != 4 or images.shape[1] != 3 or min(images.shape[-2:]) < 2:
        raise ValueError("Expected RGB [N,3,H,W] with H,W >= 2.")
    # Torch 2.2.2 has no native MPS FFT. Copy operations preserve autograd;
    # CUDA and CPU retain their native spectral path.
    output_device = images.device
    x = images.to("cpu") if images.device.type == "mps" else images
    x = x if x.dtype in (torch.float32, torch.float64) else x.float()
    h, w = x.shape[-2:]
    k = min(h, w) // 2
    lum = 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]
    power = torch.fft.rfft2(lum).abs().square()
    fy = torch.fft.fftfreq(h, device=x.device, dtype=x.dtype)
    fx = torch.fft.rfftfreq(w, device=x.device, dtype=x.dtype)
    radius = torch.sqrt(fy[:, None].square() + fx[None, :].square())
    idx = (radius / radius.max() * (k - 1)).round().long().clamp(0, k - 1).reshape(-1)
    counts = torch.zeros(k, device=x.device, dtype=x.dtype).index_add(
        0, idx, torch.ones(idx.numel(), device=x.device, dtype=x.dtype)
    ).clamp_min(1)
    radial = torch.zeros(x.shape[0], k, device=x.device, dtype=x.dtype).index_add(
        1, idx, power.flatten(1)
    )
    return torch.log1p(radial / counts[None]).to(output_device)


def ai_margin(logits: torch.Tensor, ai_class: int = 0) -> torch.Tensor:
    """Higher means more AI: logit_AI - logit_real (native model ordering)."""
    if ai_class not in (0, 1) or logits.ndim != 2 or logits.shape[1] != 2:
        raise ValueError("Expected binary [N,2] logits and ai_class in {0,1}.")
    return logits[:, ai_class] - logits[:, 1 - ai_class]


@dataclass
class AttackResult:
    images: torch.Tensor
    trace: list[dict]
    config: dict


def attack(
    images: torch.Tensor,
    logits_fn: LogitsFn | None,
    target_spec: torch.Tensor | None,
    *,
    mode: str = "SFA",
    surrogate_names: Sequence[str] = ("E", "R"),
    ai_class: int = 0,
    lam: float = 1.0,
    budget_levels: int = 8,
    alpha: float = 2 / 255,
    steps: int = 50,
    seed: int = 5230,
) -> AttackResult:
    """Matched deterministic zero-start projected sign-gradient descent.

    logits_fn(x, names) must query ONLY the requested frozen models and return
    their differentiable native logits. SPEC never invokes logits_fn. The
    caller evaluates other detectors only AFTER choosing parameters.

    Loss = mean(model AI margins)                     for PGD
         = mean(model AI margins) + lam * spectrumMSE for SFA
         = spectrumMSE                               for SPEC.

    No redundant 0.5 is applied after averaging the surrogate margins.
    All modes take the full fixed step count; there is no best-iterate or
    early-stop selection. Seed is recorded; zero initialization introduces no
    random draws. Backend nondeterminism remains the caller's responsibility.
    Returned images must still pass save_png_batch before evaluation.
    """
    _check_rgb(images)
    mode = mode.upper()
    names = tuple(surrogate_names)
    if mode not in {"PGD", "SFA", "SPEC"}:
        raise ValueError("mode must be PGD, SFA or SPEC.")
    if isinstance(budget_levels, bool) or not isinstance(budget_levels, int) or not 0 <= budget_levels <= 255:
        raise ValueError("budget_levels must be an integer in [0,255].")
    if steps < 0 or not isinstance(steps, int) or alpha < 0 or not np.isfinite(alpha):
        raise ValueError("steps must be a nonnegative integer and alpha finite/nonnegative.")
    if ai_class not in (0, 1) or lam < 0 or not np.isfinite(lam):
        raise ValueError("Invalid AI class or spectral weight.")
    if mode != "SPEC" and (logits_fn is None or not names or len(set(names)) != len(names)):
        raise ValueError("PGD/SFA require explicit nonempty unique surrogate names and logits_fn.")
    x0 = canonical_rgb(images)
    x = x0.clone()
    eps = budget_levels / 255
    lower, upper = (x0 - eps).clamp_min(0), (x0 + eps).clamp_max(1)
    target = None
    if mode != "PGD":
        if target_spec is None:
            raise ValueError("SFA/SPEC require a fixed reference spectrum.")
        target = target_spec.detach().to(device=x.device, dtype=x.dtype)
        if target.ndim != 1 or target.shape[0] != min(x.shape[-2:]) // 2:
            raise ValueError("Reference spectrum shape does not match input dimensions.")
        if not bool(torch.isfinite(target).all()):
            raise ValueError("Non-finite reference spectrum.")
    trace = []
    with torch.enable_grad():
        for step in range(steps):
            x = x.detach().requires_grad_(True)
            margin = None
            spec = None
            if mode != "SPEC":
                logits = logits_fn(x, names)
                for name in names:
                    if name not in logits or logits[name].shape != (len(x), 2):
                        raise ValueError(f"Missing/invalid logits for requested surrogate {name!r}.")
                margin = torch.stack([ai_margin(logits[n], ai_class).mean() for n in names]).mean()
            if mode != "PGD":
                spec = (radial_logpower(x) - target[None]).square().mean()
            loss = margin if mode == "PGD" else spec if mode == "SPEC" else margin + lam * spec
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"Non-finite loss at step {step}.")
            grad, = torch.autograd.grad(loss, x)
            if not bool(torch.isfinite(grad).all()):
                raise FloatingPointError(f"Non-finite gradient at step {step}.")
            trace.append({"step": step, "objective_before_update": float(loss.detach().cpu()),
                          "mean_ai_margin": None if margin is None else float(margin.detach().cpu()),
                          "spectral_mse": None if spec is None else float(spec.detach().cpu())})
            with torch.no_grad():
                x = torch.minimum(torch.maximum(x - alpha * grad.sign(), lower), upper)
    return AttackResult(x.detach(), trace, {
        "strategy_author": STRATEGY_AUTHOR, "mode": mode,
        "surrogate_names": [] if mode == "SPEC" else list(names), "ai_class": ai_class,
        "lambda": float(lam) if mode == "SFA" else None,
        "budget_levels": budget_levels, "eps": eps, "alpha": alpha,
        "steps": steps, "seed": seed, "initialization": "zero; no random draws",
        "baseline": "canonical quantized RGB uint8", "loss_scaling": "mean surrogate AI margin",
        "trace_semantics": "objective at each iterate before its update; no extra final detector query",
    })


def load_png(path: str | Path) -> torch.Tensor:
    """Read actual PNG into CPU float32 [1,3,H,W], with no resize."""
    with Image.open(path) as im:
        if im.format != "PNG":
            raise ValueError(f"Not a PNG: {path}")
        rgb = np.array(im.convert("RGB"), dtype=np.uint8, copy=True)
    return torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float() / 255


def save_png_batch(
    candidate: torch.Tensor,
    baseline: torch.Tensor,
    paths: Sequence[str | Path],
    *,
    budget_levels: int = 8,
    overwrite: bool = False,
) -> tuple[torch.Tensor, list[dict]]:
    """Project integers, write each PNG to disk, read back and verify exactness.

    The budget is checked in integer levels on the actual reloaded files,
    avoiding float-tolerance claims. All serialization/checking is on CPU.
    Returns CPU reloaded float32 tensors and per-file SHA-256/budget records.
    """
    if isinstance(budget_levels, bool) or not isinstance(budget_levels, int) or not 0 <= budget_levels <= 255:
        raise ValueError("budget_levels must be an integer in [0,255].")
    c8, b8 = quantize_rgb(candidate), quantize_rgb(baseline)
    if c8.shape != b8.shape or len(paths) != c8.shape[0]:
        raise ValueError("Candidate, baseline and output path counts/shapes must match.")
    paths = [Path(p) for p in paths]
    if len({p.resolve() for p in paths}) != len(paths):
        raise ValueError("Output paths must be unique.")
    for path in paths:
        if path.suffix.lower() != ".png":
            raise ValueError("Output filenames must end in .png.")
        if path.exists() and not overwrite:
            raise FileExistsError(path)
    b16 = b8.to(torch.int16)
    projected = torch.minimum(torch.maximum(c8.to(torch.int16), b16 - budget_levels), b16 + budget_levels).clamp(0, 255).to(torch.uint8)
    outputs, records = [], []
    for i, path in enumerate(paths):
        path.parent.mkdir(parents=True, exist_ok=True)
        rgb = projected[i].permute(1, 2, 0).contiguous().numpy()
        Image.fromarray(rgb).save(path, format="PNG")
        reloaded = load_png(path)
        actual = quantize_rgb(reloaded)[0]
        if not torch.equal(actual, projected[i]):
            raise AssertionError(f"PNG readback mismatch: {path}")
        distance = (actual.to(torch.int16) - b16[i]).abs()
        linf_levels = int(distance.max())
        if linf_levels > budget_levels:
            raise AssertionError(f"Saved PNG violates integer budget: {path}")
        outputs.append(reloaded)
        records.append({"filename": path.name, "path": str(path.resolve()),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "linf_levels": linf_levels, "linf": linf_levels / 255,
                        "budget_levels": budget_levels, "budget_pass": True,
                        "height": int(actual.shape[1]), "width": int(actual.shape[2]),
                        "changed_values": int((distance != 0).sum()),
                        "rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest()})
    return torch.cat(outputs), records


def image_quality(baseline: torch.Tensor, saved: torch.Tensor) -> list[dict]:
    """Per-image exact saved-grid distortion; no optional metric dependencies.

    PSNR is +inf for identical images, not an arbitrary capped value. This does
    not calculate SSIM/LPIPS or make any perceptual-quality guarantee.
    """
    b8, a8 = quantize_rgb(baseline), quantize_rgb(saved)
    if b8.shape != a8.shape:
        raise ValueError("Quality inputs must have matching shapes.")
    d = (a8.to(torch.float64) - b8.to(torch.float64)) / 255
    rows = []
    for i in range(len(d)):
        mse = float(d[i].square().mean())
        levels = int((a8[i].to(torch.int16) - b8[i].to(torch.int16)).abs().max())
        rows.append({"mse": mse, "psnr_db": float("inf") if mse == 0 else float(-10 * np.log10(mse)),
                     "linf_levels": levels, "linf": levels / 255,
                     "l2": float(torch.linalg.vector_norm(d[i])), "mae": float(d[i].abs().mean())})
    return rows


def _flat_numpy(values) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    return np.asarray(values).reshape(-1)


def conditional_asr(base_is_ai, adv_is_ai) -> dict:
    """Rate among originally detected images under the SAME evaluation transform.

    Pass decisions from logits.argmax(1) == ai_class to preserve tie behavior.
    """
    original, attacked = _flat_numpy(base_is_ai).astype(bool), _flat_numpy(adv_is_ai).astype(bool)
    if original.shape != attacked.shape:
        raise ValueError("Original and adversarial decisions must be aligned.")
    eligible = int(original.sum())
    flips = int((original & ~attacked).sum())
    return {"eligible": eligible, "flips": flips, "asr": flips / eligible if eligible else float("nan")}


def auc_adv_vs_real(adv_ai_scores, real_ai_scores) -> float:
    """AUC with adversarial samples positive and higher score meaning more AI.

    Equals P(score_adv > score_real) + .5*P(tie). A value near .5 alone does
    NOT prove equality of distributions; near 0 suggests reversed ranking.
    """
    a, r = _flat_numpy(adv_ai_scores).astype(float), _flat_numpy(real_ai_scores).astype(float)
    if not len(a) or not len(r):
        return float("nan")
    if not np.isfinite(a).all() or not np.isfinite(r).all():
        raise ValueError("AUC scores must be finite.")
    real = np.sort(r)
    below = np.searchsorted(real, a, side="left")
    through = np.searchsorted(real, a, side="right")
    return float(np.mean((below + 0.5 * (through - below)) / len(real)))
