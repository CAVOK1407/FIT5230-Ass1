"""Frozen, integer-budget pixel baselines; no diffusion model is loaded.

PGD10: clean start, 10 updates minimizing margin hinge + saved-image LPIPS,
       final iterate retained. PGD40: 40 CE-ascent updates on each of 3 starts,
       retain the largest saved-image E margin, including the clean input.
"""
import argparse
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from m3_lao_budget import (
    canonical_source, image_paths, max_difference_pixels, pixels_sha256,
    project_uint8, save_png_verified, sha256, validate_budget, validate_rgb_u8,
    write_json_exclusive,
)


def require_finite(value, name):
    if value is None:
        raise RuntimeError(f"{name} is missing; the input gradient path is disconnected")
    if not bool(torch.isfinite(value).all().item()):
        raise FloatingPointError(f"{name} contains NaN or infinity")


def image_tensor(pixels, device):
    validate_rgb_u8(pixels)
    return torch.from_numpy(pixels.astype(np.float32) / 255.0).permute(2, 0, 1)[None].to(device)


def candidate_uint8(image01):
    require_finite(image01, "candidate image")
    if image01.ndim != 4 or image01.shape[0] != 1 or image01.shape[1] != 3:
        raise ValueError("pixel baseline expects one BCHW RGB image")
    return np.uint8(image01.detach().clamp(0, 1).cpu().numpy()[0].transpose(1, 2, 0) * 255.0)


def projected_uint8(image01, reference, budget_pixels):
    return project_uint8(candidate_uint8(image01), reference, budget_pixels)


class _ProjectedPNG(torch.autograd.Function):
    @staticmethod
    def forward(ctx, image01, reference, budget_pixels):
        pixels = projected_uint8(image01, reference, budget_pixels)
        return image_tensor(pixels, image01.device).to(image01.dtype)

    @staticmethod
    def backward(ctx, gradient):
        return gradient, None, None


def projected_png_ste(image01, reference, budget_pixels):
    """Exact integer-projected PNG forward; identity BPDA derivative backward."""
    return _ProjectedPNG.apply(image01, reference, budget_pixels)


def normalized_logits(detector, image01):
    if detector.training:
        raise RuntimeError("baseline detector must remain in eval mode")
    mean = image01.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = image01.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    logits = detector((image01 - mean) / std)
    require_finite(logits, "detector logits")
    if logits.shape != (1, 2):
        raise ValueError("expected E wrapper logits with shape [1, 2]")
    return logits


def prediction(logits):
    require_finite(logits, "prediction logits")
    return {"pred": int(logits.argmax(1).item()),
            "real_margin": float((logits[:, 1] - logits[:, 0]).item())}


@torch.no_grad()
def score_pixels(detector, pixels, device):
    return prediction(normalized_logits(detector, image_tensor(pixels, device)))


def stable_restart_seed(seed, source_key, restart):
    payload = f"M3-strict-PGD|{int(seed)}|{source_key}|{restart}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def bounded_continuous_update(image01, gradient, reference, budget_pixels, direction):
    require_finite(gradient, "pixel gradient")
    clean = reference.astype(np.int16)
    lower = image_tensor(np.maximum(clean - budget_pixels, 0).astype(np.uint8), image01.device)
    upper = image_tensor(np.minimum(clean + budget_pixels, 255).astype(np.uint8), image01.device)
    updated = image01.detach() + direction * (2.0 / 255.0) * gradient.sign()
    updated = torch.maximum(torch.minimum(updated, upper), lower).detach()
    require_finite(updated, "updated image")
    return updated


def pgd10(detector, lpips_fn, reference, device="cpu", budget_pixels=8):
    """Fixed 10-step matched-style baseline; selection never uses held-out models."""
    budget_pixels = validate_budget(budget_pixels)
    validate_rgb_u8(reference)
    if lpips_fn is None or lpips_fn.training:
        raise ValueError("PGD10 requires a frozen LPIPS module in eval mode")
    initial = image_tensor(reference, device)
    current = initial.detach().clone()
    clean_score = score_pixels(detector, reference, device)
    history = []
    for step in range(10):
        current.requires_grad_(True)
        saved = projected_png_ste(current, reference, budget_pixels)
        logits = normalized_logits(detector, saved)
        margin = logits[:, 1] - logits[:, 0]
        hinge = F.relu(3.0 - margin).mean()
        perceptual = lpips_fn(initial * 2 - 1, saved * 2 - 1).mean()
        loss = hinge + perceptual
        require_finite(loss, "PGD10 loss")
        gradient = torch.autograd.grad(loss, current, only_inputs=True)[0]
        require_finite(gradient, "PGD10 gradient")
        history.append({"step": step, "timing": "pre_update", "margin": float(margin.item()),
                        "hinge": float(hinge.item()), "lpips": float(perceptual.item()),
                        "gradient_linf": float(gradient.abs().max().item())})
        current = bounded_continuous_update(current, gradient, reference, budget_pixels, -1)
    output = projected_uint8(current, reference, budget_pixels)
    return output, {"clean": clean_score, "final": score_pixels(detector, output, device),
                    "steps": 10, "actual_gradient_steps": 10, "restarts": 1,
                    "selected_step": 10, "selected_restart": 0,
                    "selection": "final iterate; no best-iterate selection",
                    "detector_forwards": 12, "lpips_forwards": 10, "history": history}


def pgd40(detector, reference, source_key, device="cpu", budget_pixels=8, seed=20261018):
    """Three starts x 40 CE-ascent updates, best saved E margin across the trajectory."""
    budget_pixels = validate_budget(budget_pixels)
    validate_rgb_u8(reference)
    clean_score = score_pixels(detector, reference, device)
    best_pixels = reference.copy()
    best_score = clean_score
    best_restart, best_step = 0, 0
    seeds = [None] + [stable_restart_seed(seed, source_key, restart) for restart in (1, 2)]
    label = torch.tensor([0], device=device)
    history = []
    gradient_steps = 0
    for restart, restart_seed in enumerate(seeds):
        if restart == 0:
            start = reference.copy()
        else:
            rng = np.random.default_rng(restart_seed)
            offset = rng.integers(-budget_pixels, budget_pixels + 1, size=reference.shape, dtype=np.int16)
            start = np.clip(reference.astype(np.int16) + offset, 0, 255).astype(np.uint8)
        current = image_tensor(start, device)
        for step in range(41):
            current.requires_grad_(step < 40)
            saved = projected_png_ste(current, reference, budget_pixels)
            logits = normalized_logits(detector, saved)
            current_score = prediction(logits)
            if current_score["real_margin"] > best_score["real_margin"]:
                best_pixels = projected_uint8(current, reference, budget_pixels)
                best_score = current_score
                best_restart, best_step = restart, step
            item = {"restart": restart, "step": step, "margin": current_score["real_margin"]}
            if step < 40:
                loss = F.cross_entropy(logits, label)
                require_finite(loss, "PGD40 CE objective")
                gradient = torch.autograd.grad(loss, current, only_inputs=True)[0]
                require_finite(gradient, "PGD40 gradient")
                item["gradient_linf"] = float(gradient.abs().max().item())
                current = bounded_continuous_update(current, gradient, reference, budget_pixels, +1)
                gradient_steps += 1
            history.append(item)
    return best_pixels, {"clean": clean_score, "final": best_score,
                         "steps": 40, "actual_gradient_steps": gradient_steps, "restarts": 3,
                         "selected_step": best_step, "selected_restart": best_restart,
                         "restart_seeds": seeds,
                         "selection": "maximum saved-image E margin across clean input and all starts/iterates; first wins ties",
                         "detector_forwards": 124, "lpips_forwards": 0, "history": history}


def synchronize(device):
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images_root", required=True, type=Path)
    parser.add_argument("--out_root", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="mps")
    parser.add_argument("--budget_pixels", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261018)
    parser.add_argument("--res", type=int, default=224)
    args = parser.parse_args(argv)
    validate_budget(args.budget_pixels)
    args.images_root = args.images_root.resolve()
    args.out_root = args.out_root.resolve()
    if args.out_root.exists() and not args.out_root.is_dir():
        raise FileExistsError(f"output root is not a directory: {args.out_root}")
    for name in ("PGD10", "PGD40", "pgd_manifest.json"):
        if (args.out_root / name).exists():
            raise FileExistsError(f"output target already exists: {args.out_root / name}")
    paths = image_paths(args.images_root)
    canonical = [(path, canonical_source(path, args.res)) for path in paths]
    random.seed(args.seed)
    np.random.seed(args.seed % (2 ** 32))
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    import m2_device
    device = m2_device.enable(args.device)
    import other_attacks
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    from lossers.lpips import LPIPS
    detector = other_attacks.model_selection("E").eval().requires_grad_(False)
    lpips_fn = LPIPS(net="vgg", cache_dir="./.cache").to(device).eval().requires_grad_(False)
    args.out_root.mkdir(parents=True, exist_ok=True)
    config = {"created_at": datetime.now().astimezone().isoformat(), "images_root": str(args.images_root),
              "device": device, "budget_pixels": args.budget_pixels, "seed": args.seed, "res": args.res,
              "n_images": len(paths), "surrogates": "E", "evaluate": "E", "torch": torch.__version__,
              "alpha_pixels": 2, "image_normalization": "ImageNet mean/std; no inverse-normalization roundtrip",
              "saved_budget_definition": "max(abs(int16(saved PNG) - int16(canonical RGB/PIL input))) <= budget_pixels",
              "quantization": "CPU NumPy floor, integer projection, identity BPDA derivative",
              "model_training": False, "diffusion_model_loaded": False,
              "source_sha256": {path.name: sha256(path) for path in paths},
              "runtime_source_sha256": {name: sha256(Path(__file__).parent / name)
                                         for name in ("m3_lao_pgd.py", "m3_lao_budget.py", "m2_device.py", "other_attacks.py")},
              "command": " ".join(sys.argv)}
    manifest = {}
    for name in ("PGD10", "PGD40"):
        folder = args.out_root / name
        folder.mkdir()
        variant_config = {**config, "variant": name, "save_dir": str(folder),
                          "loss": "relu(3-E_margin)+LPIPS(saved_image,clean)" if name == "PGD10" else "CE(label=AI) ascent",
                          "steps_per_start": 10 if name == "PGD10" else 40,
                          "starts": 1 if name == "PGD10" else 3,
                          "lpips_weight": 1 if name == "PGD10" else 0,
                          "selection": "final iterate" if name == "PGD10" else "largest saved E margin, first wins ties",
                          "objective_note": "PGD10 LPIPS is evaluated on the saved-image STE; frozen LAO LPIPS used raw decoded values"}
        write_json_exclusive(folder / "run_config.json", variant_config)
        hashes = {}
        with (folder / "records.jsonl").open("x") as records:
            for index, (source, reference) in enumerate(canonical):
                image_id = f"{index:04d}"
                source_key = source.name + ":" + sha256(source)
                synchronize(device)
                started = time.perf_counter()
                if name == "PGD10":
                    output, info = pgd10(detector, lpips_fn, reference, device, args.budget_pixels)
                else:
                    output, info = pgd40(detector, reference, source_key, device, args.budget_pixels, args.seed)
                clean_saved = save_png_verified(folder / f"{image_id}_originImage.png", reference)
                saved = save_png_verified(folder / f"{image_id}_adv_image.png", output, reference, args.budget_pixels)
                # Re-score the exact reloaded bytes, not any in-memory unquantized candidate.
                from m3_lao_budget import load_rgb
                reloaded_score = score_pixels(detector, load_rgb(folder / f"{image_id}_adv_image.png"), device)
                if reloaded_score["pred"] != info["final"]["pred"] or not math.isclose(
                        reloaded_score["real_margin"], info["final"]["real_margin"], rel_tol=1e-5, abs_tol=1e-5):
                    raise RuntimeError("saved-PNG score differs from the selected projected candidate")
                synchronize(device)
                seconds = time.perf_counter() - started
                history = info.pop("history")
                item = {"image_id": image_id, "source": source.name, "seconds": seconds,
                        "clean": {"E": info["clean"]}, "final": {"E": reloaded_score},
                        "pgd": {"E": reloaded_score}, "pgd_info": {name: info}, "history": history,
                        "source_file_sha256": sha256(source), "canonical_source_png_sha256": clean_saved["sha256"],
                        "canonical_source_pixels_sha256": pixels_sha256(reference), "output_sha256": saved["sha256"],
                        "max_difference_pixels": saved["max_difference_pixels"], "budget_pixels": args.budget_pixels,
                        "actual_gradient_steps": info["actual_gradient_steps"],
                        "selected_step": info["selected_step"], "selected_restart": info["selected_restart"],
                        "linf": max_difference_pixels(output, reference) / 255.0,
                        "timing_includes_serialization_and_verification": True,
                        "verification_detector_forwards": 1}
                records.write(json.dumps(item, allow_nan=False) + "\n")
                records.flush()
                hashes[image_id] = saved["sha256"]
                print(f"[{name}] {index + 1}/{len(paths)} {image_id}: margin={reloaded_score['real_margin']:.4f}, "
                      f"maxdiff={saved['max_difference_pixels']}, steps={info['actual_gradient_steps']}, {seconds:.2f}s", flush=True)
                m2_device.release_memory()
        manifest[name] = hashes
    write_json_exclusive(args.out_root / "pgd_manifest.json", {"status": "verified", "variants": manifest, **config})
    print(json.dumps({"status": "verified", "out_root": str(args.out_root), "n_images": len(paths)}))


if __name__ == "__main__":
    main()
