"""Latent-only, saved-PNG-aware LAO with a frozen residual renderer.

The old implementation is never modified. Selection uses E and saved-image quality
only. A requested 10-update prefix never sees later candidates. The integer PNG
projection has an identity BPDA/STE derivative, not an exact discrete derivative.
"""
import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from m3_lao_budget import (canonical_source, image_paths, load_rgb, max_difference_pixels,
                           pixels_sha256, projection_stats, save_png_verified, sha256,
                           validate_rgb_u8, write_json_exclusive)
from m3_lao_pgd import (candidate_uint8, image_tensor, normalized_logits, prediction,
                        projected_png_ste, projected_uint8, require_finite, score_pixels,
                        synchronize)
from m3_lao_support import residual_anchor, scale_control_features, tensor_norms

BUDGET = 8
MARGIN_MIN = 0.1
LPIPS_MAX = 0.03
PSNR_MIN = 30.0
REFINEMENTS = 5


def quality(pixels, reference, lpips):
    difference = pixels.astype(np.float64) - reference.astype(np.float64)
    mse = float(np.mean(difference ** 2))
    # JSON has no infinity. null plus an explicit flag denotes an identical image.
    psnr = None if mse == 0 else 10 * math.log10(255 ** 2 / mse)
    maximum = max_difference_pixels(pixels, reference)
    return {"lpips": float(lpips), "psnr": psnr, "psnr_infinite": mse == 0,
            "max_difference_pixels": maximum,
            "low_distortion": maximum <= BUDGET and lpips <= LPIPS_MAX
            and (psnr is None or psnr >= PSNR_MIN)}


def choose_candidate(candidates):
    """Retain every failure; first observation wins exact ties."""
    if not candidates:
        raise ValueError("cannot select from an empty candidate set")
    successful = [c for c in candidates if c["metrics"]["success"]]
    if successful:
        return min(successful, key=lambda c: c["metrics"]["lpips"]), "minimum_lpips_success"
    valid = [c for c in candidates if c["metrics"]["low_distortion"]]
    if valid:
        return max(valid, key=lambda c: c["metrics"]["real_margin"]), "highest_margin_low_distortion_failure"
    return min(candidates, key=lambda c: c["metrics"]["lpips"]), "minimum_lpips_failure_no_quality_candidate"


def assert_frozen(module, label):
    if module.training or any(p.requires_grad for p in module.parameters()):
        raise ValueError(f"{label} must be frozen and in eval mode")


def observe(renderer, latent, detector, lpips_fn, reference, step, phase="adam", fraction=None):
    """One actual render, one E and one LPIPS forward; keep exact output bytes."""
    raw = renderer(latent)
    require_finite(raw, "rendered image")
    saved = projected_png_ste(raw, reference, BUDGET)
    logits = normalized_logits(detector, saved)
    margin = logits[:, 1] - logits[:, 0]
    perceptual = lpips_fn(image_tensor(reference, saved.device) * 2 - 1, saved * 2 - 1).mean()
    require_finite(perceptual, "saved-image LPIPS")
    loss = F.relu(3.0 - margin).mean() + perceptual
    require_finite(loss, "LAO v2 loss")
    # Use the authoritative integer projection, never quantize saved float values again.
    pixels = projected_uint8(raw, reference, BUDGET)
    metrics = {**prediction(logits), **quality(pixels, reference, float(perceptual.item())),
               "loss": float(loss.item()), "step": step, "phase": phase,
               "fraction": fraction, "pixels_sha256": pixels_sha256(pixels)}
    metrics["success"] = metrics["low_distortion"] and metrics["real_margin"] >= MARGIN_MIN
    return {"metrics": metrics, "pixels": pixels, "latent": latent.detach().clone(),
            "raw": raw.detach().clone()}, loss


def refine(renderer, z0, successful, detector, lpips_fn, reference, candidates):
    """Five measured midpoint probes; the bracket is a heuristic, not monotonicity proof.

    All observed successes remain eligible even if subsequent midpoint probes fail.
    The segment endpoint is the first successful latent, never a moving candidate.
    """
    lower, upper = 0.0, 1.0
    endpoint = successful["latent"]
    logs = []
    with torch.no_grad():
        for index in range(REFINEMENTS):
            fraction = (lower + upper) / 2
            latent = z0 + fraction * (endpoint - z0)
            candidate, _ = observe(renderer, latent, detector, lpips_fn, reference,
                                   successful["metrics"]["step"], "refinement", fraction)
            candidates.append(candidate)
            logs.append({"probe": index + 1, "bracket_before": [lower, upper], **candidate["metrics"]})
            if candidate["metrics"]["success"]:
                upper = fraction
            else:
                lower = fraction
    return logs


def optimize_latent(renderer, z0, detector, lpips_fn, reference, max_iterations=40,
                    lr=0.01, prefix=False, synchronize_fn=lambda: None):
    """Model-independent core, also exercised with tiny CPU-only toy models."""
    if max_iterations not in (0, 10, 40) or lr not in (0.01, 0.02):
        raise ValueError("frozen v2 protocol requires iterations 0/10/40 and lr .01/.02")
    if prefix and max_iterations != 40:
        raise ValueError("the optional 10-step prefix requires a 40-step maximum")
    validate_rgb_u8(reference)
    require_finite(z0, "initial latent")
    assert_frozen(detector, "E detector")
    assert_frozen(lpips_fn, "LPIPS")
    z0 = z0.detach().clone()
    latent = z0.clone().requires_grad_(max_iterations > 0)
    optimizer = torch.optim.AdamW([latent], lr=lr, weight_decay=0) if max_iterations else None
    candidates, gradients, refinement_history = [], [], []
    outputs = {}
    synchronize_fn()
    started = time.perf_counter()

    def result(limit, reason):
        # Refinements inherit the step of their endpoint; >10 endpoints cannot enter Q10.
        available = [c for c in candidates if c["metrics"]["step"] <= limit]
        selected, policy = choose_candidate(available)
        synchronize_fn()
        return {"candidate": selected, "selection": policy, "stop_reason": reason,
                "actual_steps": len([g for g in gradients if g["step"] <= limit]),
                "history": [dict(c["metrics"]) for c in available],
                "gradient_steps": [dict(g) for g in gradients if g["step"] <= limit],
                "refinement_history": [dict(r) for r in refinement_history if r["step"] <= limit],
                "optimization_seconds": time.perf_counter() - started,
                "objective_renderer_forwards": len(available),
                "objective_detector_forwards": len(available), "objective_lpips_forwards": len(available)}

    for step in range(max_iterations + 1):
        with torch.set_grad_enabled(step < max_iterations):
            candidate, loss = observe(renderer, latent, detector, lpips_fn, reference, step)
        candidates.append(candidate)
        if candidate["metrics"]["success"] and max_iterations > 0:
            # No backward follows this successful observation. Release its large
            # diffusion/decoder graph before the no-grad refinement renders.
            del loss
            refinement_history = refine(renderer, z0, candidate, detector, lpips_fn, reference, candidates)
            outputs["final"] = result(max_iterations, "first_low_distortion_success_then_five_refinements")
            if prefix and step <= 10:
                outputs["prefix"] = result(10, "first_low_distortion_success_then_five_refinements")
            break
        if prefix and step == 10:
            outputs["prefix"] = result(10, "ten_step_prefix_exhausted_without_success")
        if step == max_iterations:
            outputs["final"] = result(max_iterations, "independent_zero_step" if step == 0
                                      else "maximum_steps_exhausted_without_success")
            break
        optimizer.zero_grad(set_to_none=True)
        if not loss.requires_grad:
            raise RuntimeError("latent gradient is missing; renderer is disconnected")
        gradient = torch.autograd.grad(loss, latent, only_inputs=True, allow_unused=True)[0]
        require_finite(gradient, "latent gradient")
        before = latent.detach().clone()
        latent.grad = gradient
        optimizer.step()
        require_finite(latent, "updated latent")
        grad_norms = tensor_norms(gradient, "latent gradient")
        update_norms = tensor_norms(latent - before, "latent update")
        gradients.append({"step": step + 1, "score_step": step, "score_timing": "pre_update",
                          "grad_l2": grad_norms["l2"], "grad_linf": grad_norms["linf"],
                          "update_l2": update_norms["l2"], "update_linf": update_norms["linf"],
                          "gradient": grad_norms, "update": update_norms,
                          "distance_from_initial": tensor_norms(latent - z0, "latent distance")})
    # This is a genuine rerender. It is not copied from an anchor reference or state-0 cache.
    with torch.no_grad():
        reset, _ = observe(renderer, z0, detector, lpips_fn, reference, 0, "reset")
    if not np.array_equal(reset["pixels"], candidates[0]["pixels"]):
        raise RuntimeError("actual reset renderer differs from initial zero-step rendering")
    for value in outputs.values():
        value["reset"] = reset
        value["shared_actual_steps"] = len(gradients)
        value["shared_objective_forwards"] = len(candidates)
        value["shared_refinement_evaluations"] = len(refinement_history)
        value["reset_renderer_forwards"] = 1
    return outputs


class ResidualRenderer:
    """Same frozen clean/clean batch-2 Control-VAE residual renderer as v1."""
    def __init__(self, model, cvae, reference, args):
        from m2_stealth import FINAL_SCALE, ddim_reverse_sample, diffusion_step, preprocess
        self.model, self.decoder = model, cvae[1]
        self.diffusion_step, self.scale, self.guidance = diffusion_step, FINAL_SCALE, args.guidance
        self.calls = 0
        pil = Image.fromarray(reference)
        prompt = [""] * 2
        with torch.no_grad():
            _, inversion = ddim_reverse_sample(pil, prompt, model, args.diffusion_steps, 0, res=args.res)
            self.z0 = inversion[::-1][args.start_step - 1].detach().clone()
            uncond = model.tokenizer([""], padding="max_length", max_length=77, return_tensors="pt")
            uncond_embeddings = model.text_encoder(uncond.input_ids.to(model.device))[0]
            text = model.tokenizer(prompt, padding="max_length", max_length=model.tokenizer.model_max_length,
                                   truncation=True, return_tensors="pt")
            embeddings = model.text_encoder(text.input_ids.to(model.device))[0]
            self.timesteps = model.scheduler.timesteps[args.start_step:]
            context = torch.cat([torch.cat([uncond_embeddings.detach()] * 2), embeddings]).detach()
            self.context = [context for _ in self.timesteps]
            full_features = [f.detach() for f in cvae[0](preprocess(pil, args.res))[1]]
            self.features = scale_control_features(full_features, 0.5)
            decoded_latent = self.diffuse(self.z0)
            self.full_reference = self.decoder(decoded_latent, full_features)[1:].detach()
            self.gated_reference = self.decoder(decoded_latent, self.features)[1:].detach()
            require_finite(self.full_reference, "full reference")
            require_finite(self.gated_reference, "gated reference")

    def diffuse(self, latent):
        latents = torch.cat([self.z0, latent])
        for index, timestep in enumerate(self.timesteps):
            latents = self.diffusion_step(self.model, latents, self.context[index], timestep, self.guidance)
        return self.model.vae.post_quant_conv(1 / self.scale * latents)

    def __call__(self, latent):
        self.calls += 1
        current = self.decoder(self.diffuse(latent), self.features)[1:]
        anchored = residual_anchor(current, self.full_reference, self.gated_reference)
        return (anchored / 2 + 0.5).clamp(0, 1)


def image_seed(seed, source_name):
    data = f"M3-LAO-v2|{seed}|{source_name}".encode()
    return int.from_bytes(hashlib.sha256(data).digest()[:4], "big")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--images_root", type=Path, required=True)
    p.add_argument("--out_root", type=Path, required=True)
    p.add_argument("--name", default=None)
    p.add_argument("--prefix_name", default=None, help="explicit optional Q10 output for a 40-step run; none disables")
    p.add_argument("--max_iterations", type=int, choices=(0, 10, 40), default=40)
    p.add_argument("--lr", type=float, choices=(0.01, 0.02), default=0.01)
    p.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="mps")
    p.add_argument("--pretrained_diffusion_path", default="Manojb/stable-diffusion-2-1-base")
    p.add_argument("--encoder_weights", default="./checkpoints/Controlvae.pt")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--res", type=int, choices=(224,), default=224)
    p.add_argument("--diffusion_steps", type=int, choices=(20,), default=20)
    p.add_argument("--start_step", type=int, choices=(18,), default=18)
    p.add_argument("--guidance", type=float, choices=(0.0,), default=0.0)
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    args.name = args.name or {0: "ZERO", 10: "Q10", 40: "Q40"}[args.max_iterations]
    if args.prefix_name and args.prefix_name.lower() == "none":
        args.prefix_name = None
    if args.prefix_name and args.max_iterations != 40:
        p.error("--prefix_name requires --max_iterations 40")
    names = [args.name] + ([args.prefix_name] if args.prefix_name else [])
    if len(names) != len(set(names)) or any(Path(n).name != n or n in (".", "..") for n in names):
        p.error("method names must be unique simple directory names")
    args.images_root, args.out_root = args.images_root.resolve(), args.out_root.resolve()
    paths = image_paths(args.images_root)
    if not paths:
        raise FileNotFoundError("no source images")
    for name in names:
        if (args.out_root / name).exists():
            raise FileExistsError(f"refusing to overwrite {args.out_root / name}")
    canonical = [(path, canonical_source(path, args.res)) for path in paths]
    import m2_device
    device = m2_device.enable(args.device)
    import m2_stealth as legacy
    legacy.seed_everything(args.seed)
    sd_path = m2_device.local_model(args.pretrained_diffusion_path, ignore_fp16=device != "cuda")
    model = legacy.StableDiffusionPipeline.from_pretrained(
        sd_path, safety_checker=None, requires_safety_checker=False).to(device)
    model.scheduler = legacy.DDIMScheduler.from_config(model.scheduler.config)
    for module in (model.unet, model.vae, model.text_encoder):
        module.eval().requires_grad_(False)
    cvae = legacy.load_control_vae(model, args.encoder_weights)
    detector = legacy.other_attacks.model_selection("E").eval().requires_grad_(False)
    lpips_fn = legacy.build_lpips().eval().requires_grad_(False)
    runtime_files = ("m3_lao_v2.py", "m3_lao_budget.py", "m3_lao_pgd.py", "m3_lao_support.py",
                     "m2_stealth.py", "m2_device.py", "other_attacks.py", "ControlVAE.py", "diff_latent_attack.py")
    common = {**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "created_at": datetime.now().astimezone().isoformat(), "device": device,
              "n_images": len(paths), "torch": torch.__version__, "surrogates": "E", "evaluate": "E",
              "encode_source": "clean", "control_source": "clean", "decode_path": "final",
              "control_scale": 0.5, "lao_residual_anchor": True, "lao_quantization": "integer_projection_ste",
              "lao_weight_decay": 0, "model_training": False, "pixel_pgd": False,
              "loss": "margin", "kappa": 3.0, "lpips_weight": 1.0, "budget_pixels": BUDGET,
              "success_rule": {"E_margin_min": MARGIN_MIN, "lpips_max": LPIPS_MAX, "psnr_min": PSNR_MIN},
              "refinement_evaluations": REFINEMENTS, "quantization": "CPU NumPy floor then int16 projection; identity BPDA",
              "lpips_forward": "canonical clean and actual projected PNG; both mapped to [-1,1]",
              "selection": "minimum LPIPS success; otherwise highest margin low-distortion failure; otherwise minimum LPIPS failure",
              "source_sha256": {path.name: sha256(path) for path in paths},
              "runtime_source_sha256": {name: sha256(Path(__file__).parent / name) for name in runtime_files},
              "seed_per_image": "first 32 bits SHA256(M3-LAO-v2|seed|source filename)",
              "command": " ".join(sys.argv if argv is None else [sys.argv[0], *argv])}
    args.out_root.mkdir(parents=True, exist_ok=True)
    handles = {}
    try:
        for name in names:
            folder = args.out_root / name
            folder.mkdir(exist_ok=False)
            is_prefix = name == args.prefix_name
            write_json_exclusive(folder / "run_config.json", {**common, "variant": name,
                "iterations": 10 if is_prefix else args.max_iterations, "is_shared_prefix": is_prefix,
                "save_dir": str(folder), "shared_run_variant": args.name,
                "independent_zero_step": args.max_iterations == 0})
            handles[name] = (folder / "records.jsonl").open("x")
        for index, (source, reference) in enumerate(canonical):
            image_id = f"{index:04d}"
            legacy.seed_everything(image_seed(args.seed, source.name))
            synchronize(device)
            started = time.perf_counter()
            renderer = ResidualRenderer(model, cvae, reference, args)
            synchronize(device)
            setup_seconds = time.perf_counter() - started
            clean_score = score_pixels(detector, reference, device)
            results = optimize_latent(renderer, renderer.z0, detector, lpips_fn, reference,
                                      args.max_iterations, args.lr, bool(args.prefix_name),
                                      lambda: synchronize(device))
            shared_results_seconds = time.perf_counter() - started
            outputs = [(args.name, results["final"])]
            if args.prefix_name:
                outputs.append((args.prefix_name, results["prefix"]))
            for name, result in outputs:
                export_started = time.perf_counter()
                folder = args.out_root / name
                candidate, reset = result["candidate"], result["reset"]
                paths_out = {"origin": folder / f"{image_id}_originImage.png",
                             "adv": folder / f"{image_id}_adv_image.png",
                             "reset": folder / f"{image_id}_reset_image.png"}
                clean_info = save_png_verified(paths_out["origin"], reference)
                saved = save_png_verified(paths_out["adv"], candidate["pixels"], reference, BUDGET)
                reset_info = save_png_verified(paths_out["reset"], reset["pixels"], reference, BUDGET)
                with torch.no_grad():
                    reloaded = load_rgb(paths_out["adv"])
                    final_score = score_pixels(detector, reloaded, device)
                    final_lpips = float(lpips_fn(image_tensor(reference, device) * 2 - 1,
                                               image_tensor(reloaded, device) * 2 - 1).mean().item())
                    float_score = prediction(normalized_logits(detector, candidate["raw"]))
                metrics = candidate["metrics"]
                if final_score["pred"] != metrics["pred"] or not math.isclose(
                        final_score["real_margin"], metrics["real_margin"], rel_tol=1e-5, abs_tol=1e-5):
                    raise RuntimeError("saved PNG E score differs from selected candidate")
                if not math.isfinite(final_lpips) or not math.isclose(final_lpips, metrics["lpips"], rel_tol=1e-5, abs_tol=1e-6):
                    raise RuntimeError("saved PNG LPIPS differs from selected candidate")
                synchronize(device)
                export_seconds = time.perf_counter() - export_started
                record = {"image_id": image_id, "source": source.name,
                    "seconds": shared_results_seconds + export_seconds,
                    "method_available_trajectory_seconds": setup_seconds + result["optimization_seconds"],
                    "export_seconds": export_seconds,
                    "shared_execution_seconds": shared_results_seconds,
                    "timing_note": "seconds = actual full shared preparation/optimization/reset + own export; prefix inherits shared cost and cannot be summed with final; method_available_trajectory_seconds ends at that method's candidate selection and excludes later updates/reset/export",
                    "clean": {"E": clean_score}, "final": {"E": final_score}, "pgd": None, "pgd_info": {},
                    "history": result["history"], "gradient_steps": result["gradient_steps"],
                    "refinement_history": result["refinement_history"],
                    "selected_step": metrics["step"], "selected_phase": metrics["phase"],
                    "actual_steps": result["actual_steps"], "actual_gradient_steps": result["actual_steps"],
                    "shared_actual_steps": result["shared_actual_steps"], "stop_reason": result["stop_reason"],
                    "actual_shared_gradient_steps": result["shared_actual_steps"],
                    "seconds_is_shared": bool(args.prefix_name),
                    "selection": result["selection"], "low_distortion_success": metrics["success"],
                    "selected": {"saved_png": {**metrics, **saved}, "raw_float_real_margin": float_score["real_margin"]},
                    "reset": {**reset["metrics"], **reset_info}, "reset_path": str(paths_out["reset"]),
                    "adv_path": str(paths_out["adv"]), "origin_path": str(paths_out["origin"]),
                    "source_file_sha256": sha256(source), "output_sha256": saved["sha256"],
                    "canonical_source_png_sha256": clean_info["sha256"],
                    "canonical_source_pixels_sha256": pixels_sha256(reference),
                    "max_difference_pixels": saved["max_difference_pixels"], "budget_pixels": BUDGET,
                    "linf": saved["max_difference_pixels"] / 255.0,
                    "psnr": metrics["psnr"], "psnr_infinite": metrics["psnr_infinite"], "lpips": final_lpips,
                    "projection": projection_stats(candidate_uint8(candidate["raw"]), reference, candidate["pixels"], BUDGET),
                    "compute": {"method_objective_forwards": result["objective_renderer_forwards"],
                        "shared_objective_forwards": result["shared_objective_forwards"],
                        "shared_refinement_evaluations": result["shared_refinement_evaluations"],
                        "shared_reset_forwards": 1, "renderer_calls_including_reset": renderer.calls,
                        "preparation": {"configured_diffusion_steps": args.diffusion_steps,
                            "ddim_inversion_unet_forwards": args.diffusion_steps - 1,
                            "reference_diffusion_steps": len(renderer.timesteps), "reference_decoder_forwards": 2},
                        "E_forwards_outside_objective_and_reset": 1 + 2 * len(outputs),
                        "LPIPS_forwards_outside_objective_and_reset": len(outputs),
                        "shared_group": args.name, "accounting_note": "one E and LPIPS per objective/reset; two UNet steps per renderer call; diagnostic raw float score never selects"}}
                handles[name].write(json.dumps(record, allow_nan=False) + "\n")
                handles[name].flush()
                print(f"[{name}] {index + 1}/{len(paths)} {image_id}: E={final_score['real_margin']:.4f} "
                      f"LPIPS={final_lpips:.5f} success={metrics['success']} "
                      f"steps={result['actual_steps']} selected={metrics['step']}/{metrics['phase']}", flush=True)
            del renderer, results, outputs, result, candidate, reset
            m2_device.release_memory()
    finally:
        for handle in handles.values():
            handle.close()
    print(json.dumps({"status": "verified", "methods": names, "images": len(paths)}), flush=True)


if __name__ == "__main__":
    main()
