"""Raw-latent perturbation methods, independent of any peer data or model.

The caller supplies a frozen evaluation-mode, differentiable CPU logit function
and an independent CPU scorer.  Only the scorer sees FP16 arrays, exactly as a
saved-and-reloaded latent would be represented.  This module performs no file
access and does not authorize or start experiments on a real package.
"""

from __future__ import annotations

from numbers import Integral, Real
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F


def _real(name: str, value: float, *, minimum: float = 0.0) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    value = float(value)
    if not np.isfinite(value) or value < minimum:
        raise ValueError(f"{name} must be finite and >= {minimum}")
    return value


def _original(original: np.ndarray) -> np.ndarray:
    if not isinstance(original, np.ndarray) or original.dtype != np.float16:
        raise ValueError("original must be a float16 numpy array")
    if original.ndim == 0 or not original.size or not np.isfinite(original).all():
        raise ValueError("original must be a nonempty finite array")
    return original


def quantize_within_budget(
    candidate: np.ndarray, original: np.ndarray, epsilon: float
) -> np.ndarray:
    """Project and round to finite FP16, with a strict raw-unit L-infinity bound.

    Nearest FP16 rounding can move an otherwise valid candidate outside the
    budget.  Any such value is moved one representable FP16 step toward the
    original.  The *difference* is checked in FP64, avoiding cancellation in
    bounds such as ``original + epsilon``.  No RGB range clamp is applied.
    """
    reference = _original(original)
    epsilon = _real("epsilon", epsilon)
    if not isinstance(candidate, np.ndarray) or candidate.shape != reference.shape:
        raise ValueError("candidate must be a numpy array with original's shape")
    if not (
        np.issubdtype(candidate.dtype, np.floating)
        or np.issubdtype(candidate.dtype, np.integer)
    ):
        raise ValueError("candidate must contain real numeric values")
    if not np.isfinite(candidate).all():
        raise ValueError("candidate must be finite")

    source = reference.astype(np.float64)
    projected = np.clip(candidate.astype(np.float64), source - epsilon, source + epsilon)
    # Saturate to representable *finite* values, not to an image intensity range.
    half_max = float(np.finfo(np.float16).max)
    rounded = np.clip(projected, -half_max, half_max).astype(np.float16)
    outside = np.abs(rounded.astype(np.float64) - source) > epsilon
    if outside.any():
        rounded[outside] = np.nextafter(rounded[outside], reference[outside])
    if (np.abs(rounded.astype(np.float64) - source) > epsilon).any():
        raise RuntimeError("FP16 projection failed its exact budget check")
    return np.ascontiguousarray(rounded)


def generate(
    original: np.ndarray,
    logit_fn: Callable[[torch.Tensor], torch.Tensor],
    *,
    method: str,
    epsilon: float,
    steps: int,
    alpha: float,
    seed: int,
    threshold: float,
    score_fn: Callable[[np.ndarray], float],
    early_stop_margin: float = 0.0,
) -> tuple[np.ndarray, dict]:
    """Return a bounded FP16 candidate and a complete per-update score trace.

    ``lpgd`` minimizes BCE-with-logits with target zero for exactly ``steps``
    updates. ``lstop`` differs only by checking the independently evaluated,
    quantized candidate after each update and stopping strictly below
    ``threshold - early_stop_margin``. Official predictions always use the
    original ``threshold``. An initially benign input remains unmodified even
    when its score does not meet the margin target.
    ``random`` draws uniform noise in [-epsilon, epsilon] once with a local RNG.
    The optimization iterate remains FP32; the returned artifact and every
    score/early-stop decision use the quantized FP16 candidate.

    ``logit_fn`` must return exactly one finite harmful-content logit, preserve
    the input gradient, and use a frozen model in evaluation mode.  Model
    preparation belongs in the adapter because an arbitrary callable may wrap
    a model. ``autograd.grad`` requests only the input gradient and does not
    accumulate parameter gradients. ``score_fn`` must return one probability
    from its own CPU evaluation of the supplied FP16 array.
    """
    source = _original(original).copy()
    epsilon = _real("epsilon", epsilon)
    alpha = _real("alpha", alpha)
    threshold = _real("threshold", threshold)
    if threshold > 1.0:
        raise ValueError("threshold must be in [0, 1]")
    early_stop_margin = _real("early_stop_margin", early_stop_margin)
    if early_stop_margin > 0.0 and early_stop_margin >= threshold:
        raise ValueError("positive early_stop_margin must be less than threshold")
    early_stop_target = threshold - early_stop_margin
    if isinstance(steps, (bool, np.bool_)) or not isinstance(steps, Integral) or steps <= 0:
        raise ValueError("steps must be a positive integer")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    steps, seed = int(steps), int(seed)
    if method not in {"random", "lpgd", "lstop"}:
        raise ValueError("method must be random, lpgd, or lstop")
    if method != "lstop" and early_stop_margin > 0.0:
        raise ValueError("positive early_stop_margin is only supported for lstop")
    if not callable(logit_fn) or not callable(score_fn):
        raise ValueError("logit_fn and score_fn must be callable")
    if method != "random" and epsilon > 0.0 and alpha == 0.0:
        raise ValueError("alpha must be positive when iterative epsilon is positive")
    if alpha > np.finfo(np.float32).max or epsilon > np.finfo(np.float32).max:
        raise ValueError("epsilon and alpha must be representable in float32")

    trace: list[dict] = []

    def evaluate(candidate: np.ndarray, step: int) -> float:
        # A defensive copy prevents a scoring callback from changing the output.
        score_array = np.asarray(score_fn(candidate.copy()))
        if score_array.shape != () or not np.issubdtype(score_array.dtype, np.number):
            raise ValueError("score_fn must return one finite probability")
        if np.iscomplexobj(score_array):
            raise ValueError("score_fn must return a real probability")
        score = float(score_array)
        if not np.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError("score_fn must return a probability in [0, 1]")
        delta = candidate.astype(np.float64) - source.astype(np.float64)
        trace.append(
            {"step": step, "score": score, "linf": float(np.abs(delta).max()),
             "l2": float(np.linalg.norm(delta.ravel()))}
        )
        return score

    candidate = source.copy()
    initial_score = evaluate(candidate, 0)
    stop_reason = "completed"
    stopped_early = False
    actual_steps = 0

    if epsilon == 0.0:
        stop_reason = "zero_budget"
    elif method == "lstop" and initial_score < threshold:
        stop_reason = "initially_below_threshold"
        stopped_early = True
    elif method == "random":
        rng = np.random.default_rng(seed)
        noise = rng.uniform(-epsilon, epsilon, size=source.shape)
        candidate = quantize_within_budget(source.astype(np.float64) + noise, source, epsilon)
        actual_steps = 1
        evaluate(candidate, actual_steps)
    else:
        reference = torch.from_numpy(source.astype(np.float32))
        current = reference.clone()
        # All optimization and CPU reference callbacks are deliberately local.
        for step in range(1, steps + 1):
            current = current.detach().requires_grad_(True)
            with torch.enable_grad():
                logit = logit_fn(current)
                if not isinstance(logit, torch.Tensor) or logit.numel() != 1:
                    raise ValueError("logit_fn must return exactly one torch logit")
                if logit.device.type != "cpu" or not torch.isfinite(logit).all():
                    raise ValueError("logit_fn must return a finite CPU logit")
                if not logit.is_floating_point() or not logit.requires_grad:
                    raise ValueError("logit_fn must preserve a floating input gradient")
                loss = F.binary_cross_entropy_with_logits(logit, torch.zeros_like(logit))
                gradient = torch.autograd.grad(loss, current, only_inputs=True)[0]
            if not torch.isfinite(gradient).all():
                raise ValueError("logit_fn produced a nonfinite input gradient")
            with torch.no_grad():
                current = reference + torch.clamp(
                    current - alpha * gradient.sign() - reference, min=-epsilon, max=epsilon
                )
            candidate = quantize_within_budget(current.numpy(), source, epsilon)
            actual_steps = step
            score = evaluate(candidate, step)
            if method == "lstop" and score < early_stop_target:
                stop_reason = (
                    "below_early_stop_target" if early_stop_margin > 0.0 else "below_threshold"
                )
                stopped_early = step < steps
                break

    info = {
        "method": method,
        "steps": actual_steps,
        "requested_steps": steps,
        "stopped_early": stopped_early,
        "stop_reason": stop_reason,
        "epsilon": epsilon,
        "alpha": alpha,
        "seed": seed,
        "threshold": threshold,
        "early_stop_margin": early_stop_margin,
        "early_stop_target": early_stop_target,
        "initial_score": initial_score,
        "final_score": trace[-1]["score"],
        "initial_prediction": int(initial_score >= threshold),
        "final_prediction": int(trace[-1]["score"] >= threshold),
        "trace": trace,
    }
    return candidate, info
