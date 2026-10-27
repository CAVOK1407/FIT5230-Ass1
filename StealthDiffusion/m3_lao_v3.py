"""Isolated saved-PNG low-distortion P/L/H experiment, with optional JPEG EOT.

All gradients are surrogates through the exact discrete PNG/JPEG forward. The
latent renderer is imported unchanged from frozen v2. No held-out model selects
a candidate. Model-independent cores permit CPU-only tests without model files.
"""
import argparse
from datetime import datetime
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, __version__ as PILLOW_VERSION
import torch
import torch.nn.functional as F

from m3_lao_budget import (canonical_source, image_paths, load_rgb, pixels_sha256,
                           save_png_verified, sha256, validate_rgb_u8, write_json_exclusive)
from m3_lao_pgd import (image_tensor, normalized_logits, projected_png_ste,
                        projected_uint8, require_finite, synchronize)
from m3_lao_support import tensor_norms
from m3_lao_v2 import ResidualRenderer, assert_frozen, quality

BUDGET = 8
MARGIN = 0.1
REFINEMENTS = 5
TRAIN_VIEWS = ("none", "jpeg80", "jpeg90")


def jpeg_pixels(pixels, quality_value):
    """Same Pillow RGB/quality/default-subsampling operation as m2_evaluate."""
    validate_rgb_u8(pixels)
    if quality_value not in (80, 90):
        raise ValueError("the frozen EOT bank contains JPEG80 and JPEG90 only")
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format="JPEG", quality=quality_value)
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        return np.asarray(decoded.convert("RGB"), dtype=np.uint8).copy()


class _JPEGForwardIdentityBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, saved, pixels, quality_value):
        require_finite(saved, "JPEG BPDA input")
        return image_tensor(jpeg_pixels(pixels, quality_value), saved.device).to(saved.dtype)

    @staticmethod
    def backward(ctx, gradient):
        return gradient, None, None


def jpeg_bpda(saved, pixels, quality_value):
    """Actual JPEG bytes in forward; identity approximation in backward only."""
    return _JPEGForwardIdentityBackward.apply(saved, pixels, quality_value)


def seed_for(seed, source, restart):
    payload = f"M3-LAO-v3|{int(seed)}|{source}|{restart}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def choose_candidate(candidates):
    if not candidates:
        raise ValueError("empty candidate pool")
    success = [c for c in candidates if c["metrics"]["success"]]
    if success:
        return min(success, key=lambda c: c["metrics"]["lpips"]), "minimum_lpips_success"
    valid = [c for c in candidates if c["metrics"]["low_distortion"]]
    if valid:
        return max(valid, key=lambda c: c["metrics"]["worst_view_margin"]), "maximum_worst_margin_low_distortion_failure"
    return min(candidates, key=lambda c: c["metrics"]["lpips"]), "minimum_lpips_failure_no_quality_candidate"


class CandidatePool:
    """Keep all metric observations, but only the best image/state in memory."""
    def __init__(self):
        self.best, self.policy, self.history = None, None, []

    def add(self, candidate):
        self.history.append(dict(candidate["metrics"]))
        self.best, self.policy = choose_candidate(([self.best] if self.best else []) + [candidate])


class Objective:
    def __init__(self, detector, lpips_fn, reference, device, eot=False, lpips_weight=1):
        assert_frozen(detector, "E detector")
        assert_frozen(lpips_fn, "LPIPS")
        validate_rgb_u8(reference)
        if not math.isfinite(lpips_weight) or lpips_weight < 0:
            raise ValueError("LPIPS weight must be finite and nonnegative")
        self.detector, self.lpips_fn = detector, lpips_fn
        self.reference, self.device = reference, device
        self.clean = image_tensor(reference, device)
        self.views = TRAIN_VIEWS if eot else ("none",)
        self.lpips_weight = lpips_weight
        self.compute = {"observations": 0, "E_forwards": 0, "E_view_examples": 0,
                        "LPIPS_forwards": 0, "JPEG_encodes": 0}

    def observe(self, raw, state, step, stage, restart=0, phase="trajectory", fraction=None,
                exact_pixels=None):
        require_finite(raw, "objective raw image")
        pixels = projected_uint8(raw, self.reference, BUDGET) if exact_pixels is None else exact_pixels.copy()
        # exact_pixels is reserved for independently loaded/saved PNGs; it never
        # supplies a backward objective or bypasses the original-image budget.
        if exact_pixels is None:
            saved = projected_png_ste(raw, self.reference, BUDGET)
        else:
            if raw.requires_grad:
                raise ValueError("exact-pixels verification must not carry an optimization graph")
            saved = image_tensor(pixels, raw.device)
        logits_by_view, margins = {}, []
        for view in self.views:
            transformed = saved if view == "none" else jpeg_bpda(saved, pixels, int(view[4:]))
            logits = normalized_logits(self.detector, transformed)
            logits_by_view[view] = logits
            margins.append((logits[:, 1] - logits[:, 0]).squeeze(0))
            self.compute["E_forwards"] += 1
            self.compute["E_view_examples"] += 1
            self.compute["JPEG_encodes"] += view != "none"
        margin_tensor = torch.stack(margins)
        perceptual = self.lpips_fn(self.clean * 2 - 1, saved * 2 - 1).mean()
        require_finite(perceptual, "saved-PNG LPIPS")
        if float(perceptual.item()) < 0:
            raise FloatingPointError("LPIPS is negative")
        loss = F.relu(3 - margin_tensor).mean() + self.lpips_weight * perceptual
        require_finite(loss, "v3 loss")
        scores = {view: float(margin.item()) for view, margin in zip(self.views, margins)}
        preds = {view: int(logits.argmax(1).item()) for view, logits in logits_by_view.items()}
        metrics = {**quality(pixels, self.reference, float(perceptual.item())),
                   "step": int(step), "stage": stage, "restart": int(restart), "phase": phase,
                   "fraction": fraction, "loss": float(loss.item()), "real_margin": scores["none"],
                   "pred": preds["none"], "worst_view_margin": min(scores.values()),
                   "view_margins": scores, "view_predictions": preds, "pixels_sha256": pixels_sha256(pixels)}
        if metrics["max_difference_pixels"] > BUDGET:
            raise RuntimeError("candidate exceeds the original-image integer budget")
        metrics["success"] = metrics["low_distortion"] and metrics["worst_view_margin"] >= MARGIN
        self.compute["observations"] += 1
        self.compute["LPIPS_forwards"] += 1
        return {"metrics": metrics, "pixels": pixels, "state": state.detach().clone()}, loss

    def observe_pixels(self, pixels, step, stage, restart=0, phase="retained"):
        with torch.no_grad():
            raw = image_tensor(pixels, self.device)
            return self.observe(raw, raw, step, stage, restart, phase, exact_pixels=pixels)[0]


def gradient_of(loss, variable, label):
    if not loss.requires_grad:
        raise RuntimeError(f"{label} gradient is missing; loss is disconnected")
    gradient = torch.autograd.grad(loss, variable, only_inputs=True, allow_unused=True)[0]
    require_finite(gradient, f"{label} gradient")
    return gradient


def continuous_bounds(reference, device):
    signed = reference.astype(np.int16)
    return (image_tensor(np.maximum(signed - BUDGET, 0).astype(np.uint8), device),
            image_tensor(np.minimum(signed + BUDGET, 255).astype(np.uint8), device))


def pixel_update(current, gradient, lower, upper, alpha_pixels):
    """Retain fractions of a pixel across updates; floor only inside forward."""
    if not math.isfinite(alpha_pixels) or alpha_pixels <= 0:
        raise ValueError("alpha_pixels must be positive and finite")
    require_finite(gradient, "pixel gradient")
    updated = current.detach() - (alpha_pixels / 255) * gradient.sign()
    updated = torch.maximum(torch.minimum(updated, upper), lower).detach()
    require_finite(updated, "updated pixel buffer")
    return updated


def gradient_record(gradient, before, after, step, stage, restart):
    norms, delta = tensor_norms(gradient, "gradient"), tensor_norms(after - before, "update")
    return {"step": step, "score_step": step - 1, "score_timing": "pre_update",
            "stage": stage, "restart": restart, "grad_l2": norms["l2"], "grad_linf": norms["linf"],
            "update_l2": delta["l2"], "update_linf": delta["linf"]}


@torch.no_grad()
def refine_segment(objective, origin, endpoint, renderer, first, pool, stage, restart):
    lower, upper, logs = 0.0, 1.0, []
    for probe in range(REFINEMENTS):
        fraction = (lower + upper) / 2
        variable = origin + fraction * (endpoint - origin)
        raw = renderer(variable)
        candidate, _ = objective.observe(raw, variable, first["metrics"]["step"], stage, restart,
                                         "refinement", fraction)
        pool.add(candidate)
        logs.append({"probe": probe + 1, "bracket_before": [lower, upper], **candidate["metrics"]})
        if candidate["metrics"]["success"]:
            upper = fraction
        else:
            lower = fraction
    return logs


def stage_result(pool, gradients, probes, stops, objective, started, sync):
    sync()
    return {"candidate": pool.best, "selection": pool.policy, "history": pool.history,
            "gradient_steps": gradients, "refinement_history": probes, "stop_reasons": stops,
            "actual_gradient_steps": len(gradients), "compute": dict(objective.compute),
            "seconds": time.perf_counter() - started}


def optimize_latent(renderer, z0, objective, max_steps=40, lr=.02, sync=lambda: None):
    if max_steps < 0 or not math.isfinite(lr) or lr <= 0:
        raise ValueError("invalid latent optimization parameters")
    require_finite(z0, "initial latent")
    z0 = z0.detach().clone()
    z = z0.clone().requires_grad_(max_steps > 0)
    optimizer = torch.optim.AdamW([z], lr=lr, weight_decay=0) if max_steps else None
    pool, gradients, probes = CandidatePool(), [], []
    sync()
    started = time.perf_counter()
    initial = None
    for step in range(max_steps + 1):
        with torch.set_grad_enabled(step < max_steps):
            item, loss = objective.observe(renderer(z), z, step, "latent")
        pool.add(item)
        if initial is None:
            initial = item["pixels"].copy()
        if item["metrics"]["success"] and max_steps:
            del loss
            probes = refine_segment(objective, z0, item["state"], renderer, item, pool, "latent", 0)
            reason = "first_feasible_success_then_five_refinements"
            break
        if step == max_steps:
            reason = "independent_zero_step" if not max_steps else "maximum_steps_without_success"
            break
        gradient = gradient_of(loss, z, "latent")
        before = z.detach().clone()
        optimizer.zero_grad(set_to_none=True)
        z.grad = gradient
        optimizer.step()
        require_finite(z, "updated latent")
        gradients.append(gradient_record(gradient, before, z, step + 1, "latent", 0))
    # Counterfactual reset must execute the actual renderer, not return a cache.
    with torch.no_grad():
        reset, _ = objective.observe(renderer(z0), z0, 0, "latent", phase="reset")
    if not np.array_equal(initial, reset["pixels"]):
        raise RuntimeError("actual latent reset differs from independent initial render")
    result = stage_result(pool, gradients, probes, [{"restart": 0, "reason": reason}], objective, started, sync)
    result["reset"] = reset
    return result


def optimize_pixels(objective, initial_pixels=None, max_steps=80, restarts=3, alpha_pixels=.5,
                    seed=20261018, source="", sync=lambda: None):
    if max_steps < 0 or restarts not in (1, 3):
        raise ValueError("pixel protocol requires nonnegative steps and one/three restarts")
    if not math.isfinite(alpha_pixels) or alpha_pixels <= 0:
        raise ValueError("alpha_pixels must be positive and finite")
    reference = objective.reference
    initial_pixels = reference if initial_pixels is None else initial_pixels
    validate_rgb_u8(initial_pixels)
    lower, upper = continuous_bounds(reference, objective.device)
    clean, center = objective.clean, image_tensor(initial_pixels, objective.device)
    pool, gradients, probes, stops = CandidatePool(), [], [], []
    sync()
    started = time.perf_counter()
    # Original clean and the actual selected LAO PNG remain eligible throughout.
    pool.add(objective.observe_pixels(reference, 0, "pixel", -1, "original_clean"))
    pool.add(objective.observe_pixels(initial_pixels, 0, "pixel", -1, "stage_initial_png"))
    for restart in range(restarts):
        if restart == 0:
            current = center.detach().clone()
        else:
            generator = np.random.default_rng(seed_for(seed, source, restart))
            noise = generator.uniform(-restart, restart, size=tuple(center.shape)).astype(np.float32)
            current = center.detach() + torch.from_numpy(noise).to(center.device) / 255
            current = torch.maximum(torch.minimum(current, upper), lower).detach()
        for step in range(max_steps + 1):
            current.requires_grad_(step < max_steps)
            with torch.set_grad_enabled(step < max_steps):
                item, loss = objective.observe(current, current, step, "pixel", restart)
            pool.add(item)
            if item["metrics"]["success"]:
                del loss
                # Refine toward the actual observed PNG, not an unobserved float endpoint.
                endpoint = image_tensor(item["pixels"], objective.device)
                probes.extend(refine_segment(objective, clean, endpoint, lambda x: x,
                                              item, pool, "pixel", restart))
                stops.append({"restart": restart, "step": step,
                              "reason": "first_feasible_success_then_five_refinements"})
                break
            if step == max_steps:
                stops.append({"restart": restart, "step": step, "reason": "maximum_steps_without_success"})
                break
            gradient = gradient_of(loss, current, "pixel")
            before = current.detach().clone()
            current = pixel_update(current, gradient, lower, upper, alpha_pixels)
            gradients.append(gradient_record(gradient, before, current, step + 1, "pixel", restart))
    return stage_result(pool, gradients, probes, stops, objective, started, sync)


def normalize_configs(configs):
    if not isinstance(configs, list) or not configs:
        raise ValueError("config must be a nonempty JSON list")
    result, names = [], set()
    allowed = {"name", "kind", "eot", "alpha_pixels", "lpips_weight", "max_steps", "restarts",
               "lao_steps", "lao_lr", "lao_lpips_weight"}
    for entry in configs:
        if not isinstance(entry, dict) or set(entry) - allowed:
            raise ValueError("unknown configuration field")
        cfg = {"eot": False, "alpha_pixels": .5, "lpips_weight": 1., "max_steps": 80,
               "restarts": 3, "lao_steps": 40, "lao_lr": .02, "lao_lpips_weight": 1., **entry}
        name, kind = cfg.get("name"), cfg.get("kind")
        if not isinstance(name, str) or not name or Path(name).name != name or name in (".", "..") or name in names:
            raise ValueError("method names must be unique simple directory names")
        if kind not in ("P", "L", "H", "Z") or not isinstance(cfg["eot"], bool):
            raise ValueError("kind must be P/L/H/Z and eot must be boolean")
        for key in ("alpha_pixels", "lpips_weight", "lao_lr", "lao_lpips_weight"):
            if isinstance(cfg[key], bool) or not isinstance(cfg[key], (float, int)) or not math.isfinite(cfg[key]) or cfg[key] <= 0:
                raise ValueError(f"{key} must be positive and finite")
        for key in ("max_steps", "restarts", "lao_steps"):
            if isinstance(cfg[key], bool) or not isinstance(cfg[key], int) or cfg[key] < 0:
                raise ValueError(f"{key} must be a nonnegative integer")
        if cfg["restarts"] not in (1, 3) or cfg["max_steps"] > 80 or cfg["lao_steps"] > 40:
            raise ValueError("step/restart budget exceeds v3 protocol")
        names.add(name)
        result.append(cfg)
    return result


def public_stage(stage):
    return {**{key: value for key, value in stage.items() if key not in ("candidate", "reset")},
            "selected": dict(stage["candidate"]["metrics"])}


@torch.no_grad()
def verify_saved(objective, path, candidate):
    reread = load_rgb(path)
    if not np.array_equal(reread, candidate["pixels"]):
        raise RuntimeError("saved PNG differs from chosen pixels")
    checked = objective.observe_pixels(reread, candidate["metrics"]["step"], "verification")
    expected, actual = candidate["metrics"], checked["metrics"]
    for view in objective.views:
        if actual["view_predictions"][view] != expected["view_predictions"][view] or not math.isclose(
                actual["view_margins"][view], expected["view_margins"][view], rel_tol=1e-5, abs_tol=2e-5):
            raise RuntimeError(f"saved PNG {view} score differs from chosen candidate")
    if not math.isclose(actual["lpips"], expected["lpips"], rel_tol=1e-5, abs_tol=1e-6):
        raise RuntimeError("saved PNG LPIPS differs from chosen candidate")
    if actual["success"] != expected["success"]:
        raise RuntimeError("saved PNG crosses the candidate success boundary")
    return actual


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--images_root", type=Path, required=True)
    p.add_argument("--out_root", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="mps")
    p.add_argument("--pretrained_diffusion_path", default="Manojb/stable-diffusion-2-1-base")
    p.add_argument("--encoder_weights", default="./checkpoints/Controlvae.pt")
    p.add_argument("--seed", type=int, default=20261018)
    p.add_argument("--res", type=int, choices=(224,), default=224)
    p.add_argument("--diffusion_steps", type=int, choices=(20,), default=20)
    p.add_argument("--start_step", type=int, choices=(18,), default=18)
    p.add_argument("--guidance", type=float, choices=(0.,), default=0.)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    configs = normalize_configs(json.loads(args.config.read_text()))
    args.images_root, args.out_root = args.images_root.resolve(), args.out_root.resolve()
    sources = image_paths(args.images_root)
    if not sources:
        raise ValueError("no source images")
    for cfg in configs:
        if (args.out_root / cfg["name"]).exists():
            raise FileExistsError(f"refusing to overwrite {args.out_root / cfg['name']}")
    import m2_device
    device = m2_device.enable(args.device)
    import m2_stealth as legacy
    legacy.seed_everything(args.seed)
    detector = legacy.other_attacks.model_selection("E").eval().requires_grad_(False)
    lpips_fn = legacy.build_lpips().eval().requires_grad_(False)
    model, cvae = None, None
    if any(c["kind"] in ("L", "H", "Z") for c in configs):
        sd_path = m2_device.local_model(args.pretrained_diffusion_path, ignore_fp16=device != "cuda")
        model = legacy.StableDiffusionPipeline.from_pretrained(
            sd_path, safety_checker=None, requires_safety_checker=False).to(device)
        model.scheduler = legacy.DDIMScheduler.from_config(model.scheduler.config)
        for module in (model.unet, model.vae, model.text_encoder):
            module.eval().requires_grad_(False)
        cvae = legacy.load_control_vae(model, args.encoder_weights)
        for module in cvae:
            module.eval().requires_grad_(False)
    runtime_files = ("m3_lao_v3.py", "m3_lao_v2.py", "m3_lao_pgd.py", "m3_lao_budget.py",
                     "m3_lao_support.py", "m2_stealth.py", "m2_device.py", "other_attacks.py",
                     "ControlVAE.py", "diff_latent_attack.py")
    common = {**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "created_at": datetime.now().astimezone().isoformat(), "device": device,
              "n_images": len(sources), "torch": torch.__version__, "pillow": PILLOW_VERSION,
              "surrogates": "E", "evaluate": "E", "budget_pixels": BUDGET,
              "loss": "mean(relu(3-view_margin)) + LPIPS_weight*LPIPS(original,savedPNG)",
              "success_rule": {"all_training_view_margins_min": MARGIN, "lpips_max": .03, "psnr_min": 30},
              "refinement_probes_per_successful_start": REFINEMENTS,
              "EOT_bank": list(TRAIN_VIEWS), "EOT_distribution": "uniform deterministic finite bank",
              "JPEG_forward": "Pillow JPEG RGB, quality80/90, default encoder/subsampling; actual decoded pixels",
              "JPEG_backward": "identity BPDA surrogate; not the derivative of JPEG",
              "detector_batch_size": 1, "quantization": "CPU numpy floor then original integer +/-8; identity STE",
              "lpips_forward": "canonical original vs actual saved PNG in [-1,1], no channel transformation",
              "encode_source": "clean", "control_source": "clean", "control_scale": .5,
              "lao_residual_anchor": True, "lao_weight_decay": 0, "model_training": False,
              "selection": "minimum LPIPS all-view feasible success; otherwise highest worst-view margin quality failure; otherwise minimum LPIPS failure",
              "source_sha256": {p.name: sha256(p) for p in sources},
              "runtime_source_sha256": {n: sha256(Path(__file__).parent / n) for n in runtime_files},
              "command": " ".join(sys.argv if argv is None else [sys.argv[0], *argv])}
    streams = {}
    args.out_root.mkdir(parents=True, exist_ok=True)
    try:
        for cfg in configs:
            folder = args.out_root / cfg["name"]
            folder.mkdir(exist_ok=False)
            write_json_exclusive(folder / "run_config.json", {**common, **cfg, "variant": cfg["name"],
                "train_views": list(TRAIN_VIEWS if cfg["eot"] else ("none",)),
                "iterations": 0 if cfg["kind"] == "Z" else cfg["lao_steps"] if cfg["kind"] == "L" else cfg["max_steps"],
                "timing_is_shared": False, "save_dir": str(folder)})
            streams[cfg["name"]] = (folder / "records.jsonl").open("x")
        for index, source in enumerate(sources):
            image_id, reference = f"{index:04d}", canonical_source(source, args.res)
            for cfg in configs:
                name, kind = cfg["name"], cfg["kind"]
                folder = args.out_root / name
                legacy.seed_everything(seed_for(args.seed, source.name, -1))
                synchronize(device)
                started = time.perf_counter()
                sync = lambda: synchronize(device)
                stages, renderer, setup_seconds = {}, None, 0.
                preparation = {}
                if kind in ("L", "H", "Z"):
                    renderer = ResidualRenderer(model, cvae, reference, args)
                    synchronize(device)
                    setup_seconds = time.perf_counter() - started
                    preparation = {"configured_diffusion_steps": args.diffusion_steps,
                        "ddim_inversion_unet_forwards": args.diffusion_steps - 1,
                        "reference_diffusion_unet_forwards": len(renderer.timesteps), "reference_decoder_forwards": 2}
                    objective_l = Objective(detector, lpips_fn, reference, device, cfg["eot"], cfg["lao_lpips_weight"])
                    stages["latent"] = optimize_latent(renderer, renderer.z0, objective_l,
                        0 if kind == "Z" else cfg["lao_steps"], cfg["lao_lr"], sync)
                if kind in ("P", "H"):
                    initial = stages["latent"]["candidate"]["pixels"] if kind == "H" else reference
                    objective_p = Objective(detector, lpips_fn, reference, device, cfg["eot"], cfg["lpips_weight"])
                    stages["pixel"] = optimize_pixels(objective_p, initial, cfg["max_steps"], cfg["restarts"],
                        cfg["alpha_pixels"], args.seed, source.name, sync)
                selected_stage = stages["pixel"] if "pixel" in stages else stages["latent"]
                candidate = selected_stage["candidate"]
                paths_out = {"origin": folder / f"{image_id}_originImage.png", "adv": folder / f"{image_id}_adv_image.png"}
                origin_info = save_png_verified(paths_out["origin"], reference)
                saved = save_png_verified(paths_out["adv"], candidate["pixels"], reference, BUDGET)
                verification = Objective(detector, lpips_fn, reference, device, cfg["eot"], 1)
                verified = verify_saved(verification, paths_out["adv"], candidate)
                clean = verification.observe_pixels(reference, 0, "verification", phase="original_clean")["metrics"]
                extra = {}
                if "latent" in stages:
                    reset = stages["latent"]["reset"]
                    paths_out["reset"] = folder / f"{image_id}_reset_image.png"
                    reset_saved = save_png_verified(paths_out["reset"], reset["pixels"], reference, BUDGET)
                    extra.update(reset={**reset["metrics"], **reset_saved}, reset_path=str(paths_out["reset"]))
                if kind == "H":
                    paths_out["lao"] = folder / f"{image_id}_lao_image.png"
                    lao = stages["latent"]["candidate"]
                    lao_saved = save_png_verified(paths_out["lao"], lao["pixels"], reference, BUDGET)
                    extra.update(lao_stage={**lao["metrics"], **lao_saved}, lao_path=str(paths_out["lao"]))
                latent_grad = stages.get("latent", {}).get("gradient_steps", [])
                pixel_grad = stages.get("pixel", {}).get("gradient_steps", [])
                compute = {k: sum(s["compute"][k] for s in stages.values()) + verification.compute[k]
                           for k in verification.compute}
                compute.update(renderer_forwards=renderer.calls if renderer else 0,
                    renderer_unet_forwards=(renderer.calls * len(renderer.timesteps)) if renderer else 0,
                    preparation=preparation, verification=dict(verification.compute))
                compute.update(detector_forwards=compute["E_forwards"],
                    detector_view_examples=compute["E_view_examples"],
                    lpips_forwards=compute["LPIPS_forwards"], jpeg_forwards=compute["JPEG_encodes"],
                    latent_gradient_steps=len(latent_grad), pixel_gradient_steps=len(pixel_grad))
                sync()
                record = {"image_id": image_id, "source": source.name, "kind": kind, "eot": cfg["eot"],
                    "seconds": time.perf_counter() - started, "preparation_seconds": setup_seconds,
                    "timing_is_shared": False, "timing_note": "actual independent configuration wall time; includes preparation, both H stages, refinements, reset, exports and saved-PNG verification; excludes once-per-process model loading",
                    "clean": {"E": {"pred": clean["pred"], "real_margin": clean["real_margin"]}},
                    "final": {"E": {"pred": verified["pred"], "real_margin": verified["real_margin"]}},
                    "pgd": None, "pgd_info": {}, "selected": {**candidate["metrics"], **saved},
                    "selected_step": candidate["metrics"]["step"], "selected_restart": candidate["metrics"]["restart"],
                    "selected_phase": candidate["metrics"]["phase"], "selection": selected_stage["selection"],
                    "low_distortion_success": verified["success"], "saved_verification": verified,
                    "history": [h for s in stages.values() for h in s["history"]],
                    "gradient_steps": latent_grad + pixel_grad,
                    "latent_gradient_steps": latent_grad, "pixel_gradient_steps": pixel_grad,
                    "actual_gradient_steps": len(latent_grad) + len(pixel_grad),
                    "actual_latent_gradient_steps": len(latent_grad), "actual_pixel_gradient_steps": len(pixel_grad),
                    "refinement_history": [h for s in stages.values() for h in s["refinement_history"]],
                    "stages": {k: public_stage(v) for k, v in stages.items()}, "compute": compute,
                    "origin_path": str(paths_out["origin"]), "adv_path": str(paths_out["adv"]),
                    "source_file_sha256": sha256(source), "output_sha256": saved["sha256"],
                    "canonical_source_png_sha256": origin_info["sha256"], "canonical_source_pixels_sha256": pixels_sha256(reference),
                    "budget_pixels": BUDGET, "max_difference_pixels": saved["max_difference_pixels"],
                    "linf": saved["max_difference_pixels"] / 255., "lpips": verified["lpips"],
                    "psnr": verified["psnr"], "psnr_infinite": verified["psnr_infinite"], **extra}
                streams[name].write(json.dumps(record, allow_nan=False) + "\n")
                streams[name].flush()
                print(f"[{name}] {index+1}/{len(sources)} {image_id}: margin={verified['worst_view_margin']:.4f} "
                      f"LPIPS={verified['lpips']:.5f} success={verified['success']} "
                      f"steps=L{len(latent_grad)}+P{len(pixel_grad)}", flush=True)
                del stages, candidate, selected_stage, renderer
                m2_device.release_memory()
    finally:
        for stream in streams.values():
            stream.close()
    print(json.dumps({"status": "verified", "methods": [c["name"] for c in configs], "images": len(sources)}), flush=True)


if __name__ == "__main__":
    main()
