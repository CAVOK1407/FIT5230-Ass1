"""Original PGD10 plus a low-distortion, selection-matched pixel comparator.

Both outputs use one fixed ten-update trajectory. The matched output can use
only the prefix through its first feasible success, plus five scored bisection
probes along the pixel variable. Later PGD10 updates cannot alter that choice.
"""
import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from m3_lao_budget import (canonical_source, image_paths, load_rgb,
                           max_difference_pixels, pixels_sha256,
                           save_png_verified, sha256, write_json_exclusive)
from m3_lao_pgd import (bounded_continuous_update, image_tensor,
                        normalized_logits, prediction, projected_png_ste,
                        projected_uint8, require_finite, score_pixels,
                        synchronize)


MARGIN = 0.1
LPIPS_CAP = 0.03
REFINEMENT_STEPS = 5
BUDGET = 8


def psnr(reference, output):
    mse = float(np.mean((reference.astype(np.float64) - output.astype(np.float64)) ** 2))
    return None if mse == 0 else 10 * math.log10(255 ** 2 / mse)


def candidate(reference, pixels, margin, perceptual, step, phase="trajectory", alpha=1.0):
    if not math.isfinite(margin) or not math.isfinite(perceptual) or perceptual < 0:
        raise FloatingPointError("nonfinite/negative candidate metric")
    distance = max_difference_pixels(pixels, reference)
    if distance > BUDGET:
        raise RuntimeError("candidate violates the integer PNG budget")
    quality = perceptual <= LPIPS_CAP
    return {"pixels": pixels.copy(), "step": step, "phase": phase, "alpha": alpha,
            "margin": margin, "lpips": perceptual, "psnr": psnr(reference, pixels),
            "max_difference_pixels": distance, "quality_qualified": quality,
            "buffered_success": quality and margin >= MARGIN,
            "pixels_sha256": pixels_sha256(pixels)}


def public(item):
    return {k: v for k, v in item.items() if k != "pixels"}


def choose_fallback(pool):
    qualified = [p for p in pool if p["quality_qualified"]]
    # Python's stable min preserves the earliest observed tie.
    return min(qualified, key=lambda p: -p["margin"]) if qualified else min(pool, key=lambda p: p["lpips"])


@torch.no_grad()
def refinement(detector, lpips_fn, initial, successful_variable, reference, first):
    best = first
    lower, upper = 0.0, 1.0
    probes = []
    for index in range(REFINEMENT_STEPS):
        alpha = (lower + upper) / 2
        variable = initial + alpha * (successful_variable - initial)
        pixels = projected_uint8(variable, reference, BUDGET)
        saved = image_tensor(pixels, initial.device)
        logits = normalized_logits(detector, saved)
        lp = float(lpips_fn(initial * 2 - 1, saved * 2 - 1).mean().item())
        item = candidate(reference, pixels, prediction(logits)["real_margin"], lp,
                         first["step"], "refinement", alpha)
        item["probe"] = index + 1
        probes.append(public(item))
        if item["buffered_success"]:
            upper = alpha
            if item["lpips"] < best["lpips"]:
                best = item
        else:
            lower = alpha
    return best, probes


def run_pair(detector, lpips_fn, reference, device="cpu"):
    """Do not change the original PGD10 objective, gradient or final iterate."""
    if detector.training or lpips_fn.training:
        raise ValueError("detector and LPIPS must be in eval mode")
    initial = image_tensor(reference, device)
    current = initial.detach().clone()
    clean = score_pixels(detector, reference, device)
    history, prefix_pool, probes = [], [], []
    selected = None
    first_success_step = None
    matched_variable = None
    for step in range(11):
        current.requires_grad_(step < 10)
        saved = projected_png_ste(current, reference, BUDGET)
        logits = normalized_logits(detector, saved)
        margin = logits[:, 1] - logits[:, 0]
        perceptual = lpips_fn(initial * 2 - 1, saved * 2 - 1).mean()
        require_finite(perceptual, "PGD perceptual loss")
        pixels = projected_uint8(current, reference, BUDGET)
        item = candidate(reference, pixels, float(margin.item()), float(perceptual.item()), step)
        history_item = public(item)
        if selected is None:
            prefix_pool.append(item)
            if item["buffered_success"]:
                selected = item
                first_success_step = step
                matched_variable = current.detach().clone()
        if step < 10:
            hinge = F.relu(3.0 - margin).mean()
            loss = hinge + perceptual
            require_finite(loss, "PGD10 loss")
            gradient = torch.autograd.grad(loss, current, only_inputs=True)[0]
            require_finite(gradient, "PGD10 gradient")
            history_item.update(hinge=float(hinge.item()), gradient_linf=float(gradient.abs().max().item()))
            current = bounded_continuous_update(current, gradient, reference, BUDGET, -1)
        history.append(history_item)
    final = item
    if selected is None:
        selected = choose_fallback(prefix_pool)
    elif first_success_step > 0:
        selected, probes = refinement(detector, lpips_fn, initial, matched_variable, reference, selected)
    common = {"clean": clean, "history": history, "actual_shared_gradient_steps": 10,
              "restarts": 1, "loss": "relu(3-E_margin)+LPIPS(saved,clean)",
              "alpha_pixels": 2, "budget_pixels": BUDGET}
    final_info = {**common, "selected": public(final), "actual_gradient_steps": 10,
                  "selection": "original PGD10 final iterate", "refinement": []}
    matched_steps = 10 if first_success_step is None else first_success_step
    matched_info = {**common, "selected": public(selected), "actual_gradient_steps": matched_steps,
                    "first_success_step": first_success_step,
                    "selection": "first feasible prefix success, then minimum-LPIPS successful probe; otherwise feasible maximum margin",
                    "selection_reads_only_prefix_through": matched_steps, "refinement": probes,
                    "margin_buffer": MARGIN, "lpips_cap": LPIPS_CAP,
                    "stop_reason": "first_feasible_success" if first_success_step is not None else "iteration_limit"}
    return {"PGD10": (final["pixels"], final_info), "PGD10_MATCHED": (selected["pixels"], matched_info)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images_root", type=Path, required=True)
    parser.add_argument("--out_root", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="mps")
    args = parser.parse_args(argv)
    source, root = args.images_root.resolve(), args.out_root.resolve()
    targets = ["PGD10", "PGD10_MATCHED", "pgd_v2_manifest.json"]
    for name in targets:
        if (root / name).exists():
            raise FileExistsError(root / name)
    paths = image_paths(source)
    if not paths:
        raise ValueError("no input images")
    import m2_device
    device = m2_device.enable(args.device)
    import other_attacks
    from lossers.lpips import LPIPS
    torch.manual_seed(20261018)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    detector = other_attacks.model_selection("E").eval().requires_grad_(False)
    lpips_fn = LPIPS(net="vgg", cache_dir="./.cache").to(device).eval().requires_grad_(False)
    config = {"created_at": datetime.now().astimezone().isoformat(), "images_root": str(source),
              "device": device, "n_images": len(paths), "budget_pixels": BUDGET,
              "lpips_cap": LPIPS_CAP, "margin_buffer": MARGIN, "refinement_steps": REFINEMENT_STEPS,
              "surrogates": "E", "evaluate": "E", "model_training": False,
              "source_sha256": {p.name: sha256(p) for p in paths},
              "runtime_source_sha256": {str(Path(__file__).parent / n): sha256(Path(__file__).parent / n)
                                         for n in ["m3_lao_pgd_v2.py", "m3_lao_pgd.py", "m3_lao_budget.py", "m2_device.py", "other_attacks.py"]},
              "timing_note": "Both outputs share one ten-step computation; seconds is shared wall time including five possible refinement forwards and two saved-PNG verifications, not an independent matched-method benchmark"}
    root.mkdir(parents=True, exist_ok=True)
    streams, hashes = {}, {}
    try:
        for method in targets[:2]:
            folder = root / method
            folder.mkdir()
            write_json_exclusive(folder / "run_config.json", {**config, "variant": method})
            streams[method] = (folder / "records.jsonl").open("x")
            hashes[method] = {}
        for index, path in enumerate(paths):
            image_id = f"{index:04d}"
            reference = canonical_source(path, 224)
            synchronize(device)
            started = time.perf_counter()
            results = run_pair(detector, lpips_fn, reference, device)
            completed = []
            for method, (pixels, info) in results.items():
                folder = root / method
                clean_saved = save_png_verified(folder / f"{image_id}_originImage.png", reference)
                saved = save_png_verified(folder / f"{image_id}_adv_image.png", pixels, reference, BUDGET)
                reread = load_rgb(folder / f"{image_id}_adv_image.png")
                score = score_pixels(detector, reread, device)
                if not math.isclose(score["real_margin"], info["selected"]["margin"], rel_tol=1e-5, abs_tol=1e-5):
                    raise RuntimeError("saved PNG score differs from selected candidate")
                completed.append((method, {"image_id": image_id, "source": path.name,
                    "clean": {"E": info["clean"]}, "final": {"E": score}, "pgd": {"E": score},
                    "pgd_info": {method: {k: v for k, v in info.items() if k != "history"}},
                    "history": info["history"], "refinement": info["refinement"],
                    "source_file_sha256": sha256(path), "canonical_source_png_sha256": clean_saved["sha256"],
                    "canonical_source_pixels_sha256": pixels_sha256(reference), "output_sha256": saved["sha256"],
                    "max_difference_pixels": saved["max_difference_pixels"], "budget_pixels": BUDGET,
                    "actual_gradient_steps": info["actual_gradient_steps"], "actual_shared_gradient_steps": 10,
                    "selected_step": info["selected"]["step"], "selected_restart": 0,
                    "selected": info["selected"], "linf": saved["max_difference_pixels"] / 255,
                    "timing_is_shared": True}))
                hashes[method][image_id] = saved["sha256"]
            synchronize(device)
            seconds = time.perf_counter() - started
            for method, record in completed:
                record["seconds"] = seconds
                streams[method].write(json.dumps(record, allow_nan=False) + "\n")
                streams[method].flush()
            print(f"[PGD pair] {index+1}/{len(paths)}: saved and verified", flush=True)
            m2_device.release_memory()
    finally:
        for stream in streams.values():
            stream.close()
    write_json_exclusive(root / "pgd_v2_manifest.json", {"status": "verified", "variants": hashes, **config})


if __name__ == "__main__":
    main()
