"""
FIT5230 · Dark.ModelBreakers · Milestone 2
Modified StealthDiffusion attack (PGD -> Latent Adversarial Optimisation -> Control-VAE).

Run this file from inside the official StealthDiffusion checkout (pinned commit a156fb0)
with the same Python 3.10 environment the official code uses.  It imports the official
modules (diff_latent_attack, ControlVAE, other_attacks) and re-implements only the LAO
loop, so that every change is an explicit command-line switch:

  --surrogates E          detectors used in the PGD stage and in the LAO loss.
                          A comma list (e.g. E,R,D) turns on the M2 *surrogate ensemble*.
  --loss ce|margin        ce     = official objective  -CE(logits/10, label=AI) * 10
                          margin = M2 objective        relu(kappa - (logit_real - logit_ai))
  --decode_path plain|final
                          plain = official: the LAO loss is computed on the ordinary
                                  VAE decode, the saved image comes from Control-VAE.
                          final = M2: the LAO loss is computed on the *same* Control-VAE
                                  decoding path that produces the saved image.
  --lpips_weight 0        M2: weight of LPIPS(decoded image, input) inside the LAO loss.
                          (The official loop computes LPIPS but never uses it.)
  --di_prob 0             M2: input diversity - with this probability the decoded image is
                          randomly resized and padded before it is shown to the surrogates.

Milestone 3 switches (all default to the official behaviour; names follow the paper draft,
Section "Pipeline configurations", Table of configurations A1-A5 / X1-X6):

  --encode_source pgd1|clean
                          which image is DDIM-inverted into the latent.
                          pgd1  = official: the output of the first PGD call (x1).
                          clean = the unmodified input (configurations A4a, A4b, X4).
                          With clean the first PGD call is skipped.
  --control_source pgd|encoded|clean
                          which image is fed to the Control-VAE encoder (the "control image").
                          pgd     = official: a second PGD call on the encoded image (x2).
                          encoded = the encoded image itself, no second PGD call (X3).
                          clean   = the unmodified input (X2, A4a, X4).
  --iterations 0          no latent optimisation (A2 with the defaults, X4 with clean/clean).
  --pgd_impl torchattacks|own
                          own = a re-implementation of the same PGD loop (needed for the
                          switches below). With the official parameters it must give the same
                          pixels as torchattacks; m3_run.sh pgdcheck verifies that.
  --cap none|<float>|auto --pgd_alpha --pgd_steps
                          capped PGD (paper, H2 "constructive check"): every PGD call stops at
                          the first step at which the surrogate score reaches the cap. auto =
                          median surrogate score of the real development images (--real_dev_dir).
                          The paper's capped variant is  --cap auto --pgd_alpha eps/10
                          --pgd_steps 40 --loss margin --kappa cap.
  --cap_bisect            with --cap: a step that crosses the cap is shortened by bisection so the
                          score lands at the cap (one sign step of eps/10 can overshoot a sensitive
                          detector several times over).
  --kappa <float>|cap     margin target of --loss margin; cap = the value of --cap.
  --pgd_eot / --lao_eot none|jpeg|resize|both
                          expectation over transformation: the attack loss is averaged over
                          the identity view and the listed channel views. The forward pass
                          applies the real channel (PIL JPEG at --eot_jpeg_q, or Lanczos
                          down-scaling by --eot_resize and bicubic up-scaling, exactly as
                          m2_evaluate.py does); the backward pass treats the channel as the
                          identity (straight-through estimator).
  --control_scale 1      opt-in scaling of fixed Control-VAE skip features, in [0, 1].
  --lao_quantization none|ste
                         ste evaluates the LAO detector loss on saved 8-bit values.
  --lao_weight_decay .01 explicit AdamW decay; use 0 for a latent-only attribution study.
  --lao_diagnostics     record latent updates and final-path scores; save a reset-latent
                         counterfactual. Adds inference work, so timings are diagnostic.

With all defaults the script reproduces the official pipeline (variant "B00" = paper A5).
Everything else (DDIM inversion, 2 diffusion steps, AdamW lr 1e-2, Control-VAE weights,
the two PGD stages, 224x224 resolution, seed 8888 for the VAE sampling) is kept as in
the official code so that variants are comparable.

Outputs per image (in --save_dir):
  NNNN_originImage.png   input resized to 224x224 (the reference for PSNR/SSIM/LPIPS)
  NNNN_pgd1.png          output of the first PGD call, x1 (= configuration A1 on its own)
  NNNN_control.png       the control image fed to the Control-VAE encoder, quantised to 8 bit
                         (= configuration X1 on its own when --control_source pgd)
  NNNN_plain_vae.png     ordinary-VAE decode of the final latent (diagnostic only)
  NNNN_adv_image.png     final Control-VAE output = the attack result
  records.jsonl          one line per image: predictions of all detectors at the stages
                         clean / pgd (x1) / control (x2) / plain_vae / final, PSNR, SSIM,
                         time, the LAO loss history and the PGD step counts
  run_config.json        the exact configuration of this run (incl. the resolved cap)
"""
import io
import argparse
import glob
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from natsort import natsorted, ns
import torchattacks

# The official other_attacks.py hides all GPUs but "1" at import time (the authors used a
# multi-GPU server).  Import it first, then restore the single visible Colab GPU.
import other_attacks  # noqa: E402
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import m2_device  # noqa: E402

from diffusers import StableDiffusionPipeline, DDIMScheduler  # noqa: E402
from lossers.lpips import LPIPS  # noqa: E402
from diff_latent_attack import (  # noqa: E402
    ddim_reverse_sample, diffusion_step, preprocess, calculate_psnr, calculate_ssim,
)
from ControlVAE import NewEncoder, NewDecoder  # noqa: E402
from m3_lao_support import (  # noqa: E402
    changed_pixel_fraction, make_latent_optimizer, quantize_like_save,
    require_finite, saved_image_ste, scale_control_features, tensor_norms,
    validate_control_scale,
)

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
normalize = transforms.Normalize(mean=MEAN, std=STD)
denormalize = transforms.Normalize(mean=[-m / s for m, s in zip(MEAN, STD)], std=[1 / s for s in STD])

DETECTOR_NAMES = {"E": "efficientnet-b0", "R": "resnet50", "D": "deit-base", "S": "swin-base"}

# Control-VAE architecture, copied from the official diff_latent_attack.py
CVAE_PARAMS = {
    "act_fn": "silu",
    "block_out_channels": [128, 256, 512, 512],
    "down_block_types": ["DownEncoderBlock2D"] * 4,
    "in_channels": 3,
    "latent_channels": 4,
    "layers_per_block": 2,
    "norm_num_groups": 32,
    "out_channels": 3,
    "up_block_types": ["UpDecoderBlock2D"] * 4,
}
CVAE_ENCODER_PARAMS = {
    "in_channels": CVAE_PARAMS["in_channels"],
    "out_channels": CVAE_PARAMS["latent_channels"],
    "down_block_types": CVAE_PARAMS["down_block_types"],
    "block_out_channels": CVAE_PARAMS["block_out_channels"],
    "layers_per_block": CVAE_PARAMS["layers_per_block"],
    "act_fn": CVAE_PARAMS["act_fn"],
    "norm_num_groups": CVAE_PARAMS["norm_num_groups"],
    "double_z": True,
}
CVAE_DECODER_PARAMS = {
    "in_channels": CVAE_PARAMS["latent_channels"],
    "out_channels": CVAE_PARAMS["out_channels"],
    "up_block_types": CVAE_PARAMS["up_block_types"],
    "block_out_channels": CVAE_PARAMS["block_out_channels"],
    "layers_per_block": CVAE_PARAMS["layers_per_block"],
    "act_fn": CVAE_PARAMS["act_fn"],
    "norm_num_groups": CVAE_PARAMS["norm_num_groups"],
}
PLAIN_SCALE = 0.18215   # scale used by the official LAO loop (ordinary VAE)
FINAL_SCALE = 0.1825    # scale used by the official final Control-VAE decode (kept as is)


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class Ensemble(nn.Module):
    """Average of the official [0, logit_real] detector outputs -> used by the PGD stage.

    The eval() call below is load-bearing, not tidiness. A freshly built nn.Module starts in
    TRAINING mode. torchattacks snapshots model.training, forces eval for the attack, then
    restores the snapshot when it returns - so a wrapper left in training mode puts every
    detector inside it back into training mode. EfficientNet-B0 and ResNet-50 then run their
    BatchNorm layers on batch statistics (batch size 1 here) and keep overwriting their
    running statistics, which silently corrupts every later prediction; DeiT and Swin use
    LayerNorm with zero dropout and are unaffected. Symptom: the clean-image predictions of
    E and R differ between a single-surrogate run and an ensemble run, while D and S do not.
    """

    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.eval()
        self.requires_grad_(False)

    def forward(self, x):
        return torch.stack([m(x) for m in self.models], dim=0).mean(dim=0)


def real_margin(logits):
    """Official detector wrapper returns [0, raw]; argmax==1 means 'real'. Margin = raw."""
    return logits[:, 1] - logits[:, 0]


def input_diversity(x, prob, low_ratio=0.9):
    """Random resize + zero pad back to the original size (differentiable), DI-style."""
    if prob <= 0 or torch.rand(1).item() > prob:
        return x
    h, w = x.shape[-2:]
    new_h = int(torch.randint(int(h * low_ratio), h + 1, (1,)).item())
    new_w = int(torch.randint(int(w * low_ratio), w + 1, (1,)).item())
    resized = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)
    pad_h, pad_w = h - new_h, w - new_w
    top = int(torch.randint(0, pad_h + 1, (1,)).item())
    left = int(torch.randint(0, pad_w + 1, (1,)).item())
    return F.pad(resized, (left, pad_w - left, top, pad_h - top), value=0.0)


def attack_objective(logits_list, label, loss_type, kappa, attack_loss_weight):
    """LAO objective averaged over surrogate detectors (lower = better for the attacker)."""
    losses = []
    for logits in logits_list:
        if loss_type == "ce":
            # official: pred = classifier(x)/10 ; loss = -CE(pred, label) * attack_loss_weight
            losses.append(-F.cross_entropy(logits / 10, label) * attack_loss_weight)
        elif loss_type == "margin":
            losses.append(F.relu(kappa - real_margin(logits)).mean())
        else:
            raise ValueError(loss_type)
    return torch.stack(losses).mean()


def build_lpips():
    return LPIPS(net="vgg", cache_dir="./.cache").cuda()


def load_control_vae(model, encoder_weights):
    newencoder = NewEncoder(**CVAE_ENCODER_PARAMS)
    state = torch.load(encoder_weights, map_location="cpu")
    newencoder.load_state_dict(state["encoder"], strict=False)
    newencoder.requires_grad_(False)
    newencoder = newencoder.cuda().eval()
    decoder = NewDecoder(**CVAE_DECODER_PARAMS)
    decoder.load_state_dict(model.vae.decoder.state_dict(), strict=False)
    decoder.requires_grad_(False)
    decoder = decoder.cuda().eval()
    return newencoder, decoder


def to_uint8(img01):
    return np.uint8(img01.detach().clamp(0, 1).cpu().numpy()[0].transpose(1, 2, 0) * 255.0)


def force_eval(models):
    """Put every detector back in eval mode. See the note in Ensemble for why this is needed."""
    for m in models:
        m.eval()


@torch.no_grad()
def predict(detectors, img01):
    x = normalize(img01)
    out = {}
    for name, det in detectors.items():
        # Loud failure beats silent corruption: a detector in training mode reports
        # BatchNorm batch statistics and rewrites its running statistics as a side effect.
        assert not det.training, (
            f"detector {name} is in training mode; its BatchNorm statistics are being "
            f"corrupted and every prediction after this point is meaningless")
        logits = det(x)
        out[name] = {"pred": int(torch.argmax(logits, 1).item()),
                     "real_margin": float(real_margin(logits).item())}
    return out


def run_pgd(pgd_model, eps, x_norm, label, restore_eval):
    """Official PGD stage, with every detector forced back to eval mode afterwards.

    torchattacks restores model.training on return, so the mode the detectors are left in
    depends on the mode of whatever wrapper was handed to it. We do not rely on that.
    """
    attack = torchattacks.PGD(pgd_model, eps=eps, alpha=4 / 255, steps=10, random_start=False)
    attack.set_normalization_used(mean=MEAN, std=STD)
    out = attack(x_norm, label)
    force_eval(restore_eval)
    return out


# ---------------------------------------------------------------------------------------
# M3: channel views for expectation over transformation (straight-through estimator)
# ---------------------------------------------------------------------------------------
def _pil_channel(x01, fn):
    """Apply a PIL operation to every image of a [B,3,H,W] tensor in [0,1]. No gradient."""
    outs = []
    for i in range(x01.shape[0]):
        pil = fn(Image.fromarray(to_uint8(x01[i:i + 1])))
        outs.append(torch.from_numpy(np.float32(pil)[:, :, :3] / 255.0).permute(2, 0, 1))
    return torch.stack(outs).to(x01.device)


def jpeg_channel(pil, quality):
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def resize_channel(pil, scale):
    # identical to m2_evaluate.apply_condition: Lanczos down, bicubic back up
    w, h = pil.size
    small = (max(8, int(round(w * scale))), max(8, int(round(h * scale))))
    return pil.resize(small, resample=Image.LANCZOS).resize((w, h), resample=Image.BICUBIC)


def eot_views(x01, mode, jpeg_q=75, resize=0.5):
    """[identity] + channel views of x01. Each channel view is  x + (channel(x) - x).detach(),
    so its value is the real channel output and its gradient is the identity."""
    views = [x01]
    if mode in ("jpeg", "both"):
        views.append(x01 + (_pil_channel(x01, lambda p: jpeg_channel(p, jpeg_q)) - x01).detach())
    if mode in ("resize", "both"):
        views.append(x01 + (_pil_channel(x01, lambda p: resize_channel(p, resize)) - x01).detach())
    return views


# ---------------------------------------------------------------------------------------
# M3: own PGD loop = torchattacks.PGD(random_start=False) + set_normalization_used, plus
#     an optional score cap and optional EOT. Written to be pixel-identical to torchattacks
#     with the official parameters (eps 4/255, alpha 4/255, 10 steps, no cap, no EOT).
# ---------------------------------------------------------------------------------------
_MEAN_T = None
_STD_T = None


def _mean_std(device):
    global _MEAN_T, _STD_T
    if _MEAN_T is None or _MEAN_T.device != device:
        _MEAN_T = torch.tensor(MEAN, device=device).reshape(1, 3, 1, 1)
        _STD_T = torch.tensor(STD, device=device).reshape(1, 3, 1, 1)
    return _MEAN_T, _STD_T


def pgd_own(pgd_model, x_norm, label, eps, alpha, steps, cap=None, eot="none", eot_jpeg_q=75, eot_resize=0.5,
            cap_bisect=False, bisect_iters=10):
    """Returns (adv_norm, info). adv_norm is normalised like the torchattacks output."""
    mean, std = _mean_std(x_norm.device)
    x01 = (x_norm * std + mean).detach()               # torchattacks.inverse_normalize
    adv = x01.clone().detach()
    ce = nn.CrossEntropyLoss()
    info = {"steps": 0, "stopped_by_cap": False, "margin_start": None, "margin_end": None, "cap": cap}
    for step in range(steps):
        adv.requires_grad_(True)
        views = eot_views(adv, eot, eot_jpeg_q, eot_resize)
        outs = [pgd_model((v - mean) / std) for v in views]   # torchattacks.normalize
        m_id = float(real_margin(outs[0]).item())
        if step == 0:
            info["margin_start"] = m_id
        if cap is not None and m_id >= cap:
            info["stopped_by_cap"] = True
            adv = adv.detach()
            break
        cost = torch.stack([ce(o, label) for o in outs]).mean()
        grad = torch.autograd.grad(cost, adv, retain_graph=False, create_graph=False)[0]
        prev = adv.detach()
        adv = adv.detach() + alpha * grad.sign()
        delta = torch.clamp(adv - x01, min=-eps, max=eps)
        adv = torch.clamp(x01 + delta, min=0, max=1).detach()
        info["steps"] = step + 1
        if cap is not None and cap_bisect:
            # If this step crossed the cap, shrink it by bisection so the score lands just above the
            # cap instead of far beyond it (a single sign step can overshoot a sensitive detector).
            with torch.no_grad():
                m_new = float(real_margin(pgd_model((adv - mean) / std)).item())
            if m_new >= cap:
                lo, hi = 0.0, 1.0          # fraction of the step: lo stays below the cap, hi is above
                full = adv - prev
                for _ in range(bisect_iters):
                    mid = (lo + hi) / 2
                    cand = (prev + mid * full).detach()
                    with torch.no_grad():
                        m_mid = float(real_margin(pgd_model((cand - mean) / std)).item())
                    if m_mid >= cap:
                        hi = mid
                    else:
                        lo = mid
                adv = (prev + hi * full).detach()
                info["stopped_by_cap"] = True
                info["bisected"] = True
                break
    with torch.no_grad():
        info["margin_end"] = float(real_margin(pgd_model((adv - mean) / std)).item())
    return ((adv - mean) / std).detach(), info


def pgd_stage(pgd_model, x_norm, label, args, restore_eval, cap):
    """One PGD call, official or own implementation. Returns (adv_norm, info)."""
    if args.pgd_impl == "torchattacks":
        out = run_pgd(pgd_model, args.pgd_eps, x_norm, label, restore_eval)
        return out, {"impl": "torchattacks", "steps": 10}
    out, info = pgd_own(pgd_model, x_norm, label, args.pgd_eps, args.pgd_alpha, args.pgd_steps,
                        cap=cap, eot=args.pgd_eot, eot_jpeg_q=args.eot_jpeg_q, eot_resize=args.eot_resize,
                        cap_bisect=args.cap_bisect, bisect_iters=args.bisect_iters)
    force_eval(restore_eval)
    info["impl"] = "own"
    return out, info


@torch.no_grad()
def median_real_margin(pgd_model, real_dir, res):
    """Cap for --cap auto: median surrogate score of the real development images."""
    files = natsorted([f for f in os.listdir(real_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))], alg=ns.PATH)
    if not files:
        raise FileNotFoundError(f"no images in {real_dir}")
    margins = []
    for f in files:
        pil = Image.open(os.path.join(real_dir, f)).convert("RGB").resize((res, res), resample=Image.LANCZOS)
        x01 = torch.from_numpy(np.float32(pil)[:, :, :3] / 255.0).permute(2, 0, 1)[None].cuda()
        margins.append(float(real_margin(pgd_model(normalize(x01))).item()))
    return float(np.median(margins)), margins


@torch.enable_grad()
def attack_one(model, pil_image, args, surrogate_models, detectors, cvae, lpips_fn, out_prefix, pgd_model, cap, kappa):
    """Attack one PIL image. Returns a record dict. Mirrors diff_latent_attack.diffattack."""
    res = args.res
    label = torch.tensor([0]).cuda()  # official label convention: 0 = AI-generated
    # Every detector that is forwarded anywhere in this function, deduplicated by identity.
    all_models = list({id(m): m for m in list(surrogate_models) + list(detectors.values())}.values())
    force_eval(all_models)
    pgd_infos = {}

    # ---------- stage 0: PGD in pixel space (official; skipped with --encode_source clean) ----------
    origin_pil = pil_image.resize((res, res), resample=Image.LANCZOS)
    origin01 = torch.from_numpy(np.float32(origin_pil)[:, :, :3] / 255.0).permute(2, 0, 1)[None].cuda()
    test_image = normalize(origin01)
    if args.encode_source == "pgd1":
        adv_norm, pgd_infos["pgd1"] = pgd_stage(pgd_model, test_image, label, args, all_models, cap)
        pgd_pil = Image.fromarray(to_uint8(denormalize(adv_norm)))   # x1 (8-bit, as the official code)
        pgd01 = denormalize(adv_norm).clamp(0, 1)
        enc_pil = pgd_pil
    else:
        adv_norm, pgd_pil, pgd01 = None, None, None
        enc_pil = origin_pil

    if args.pgd_only:
        # pixel-space baseline: the first PGD call is the whole attack (paper, "Baselines")
        origin_u8, pgd_u8 = to_uint8(origin01), to_uint8(pgd01)
        Image.fromarray(origin_u8).save(out_prefix + "_originImage.png")
        pgd_pil.save(out_prefix + "_pgd1.png")
        pgd_pil.save(out_prefix + "_adv_image.png")
        force_eval(all_models)
        return {"clean": predict(detectors, origin01), "pgd": predict(detectors, pgd01), "control": None,
                "plain_vae": None, "final": predict(detectors, pgd01),
                "psnr": float(calculate_psnr(pgd_u8, origin_u8)), "ssim": float(calculate_ssim(pgd_u8, origin_u8)),
                "linf": float(np.abs(pgd_u8.astype(np.float32) - origin_u8.astype(np.float32)).max() / 255.0),
                "pgd_info": pgd_infos, "history": []}

    # ---------- stage 1: DDIM inversion (official) ----------
    prompt = [""] * 2
    _, inversion_latents = ddim_reverse_sample(enc_pil, prompt, model, args.diffusion_steps, 0, res=res)
    inversion_latents = inversion_latents[::-1]
    latent = inversion_latents[args.start_step - 1].detach().clone()

    uncond_input = model.tokenizer([""], padding="max_length", max_length=77, return_tensors="pt")
    uncond_embeddings = model.text_encoder(uncond_input.input_ids.to(model.device))[0]
    text_input = model.tokenizer(prompt, padding="max_length", max_length=model.tokenizer.model_max_length,
                                 truncation=True, return_tensors="pt")
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    timesteps = model.scheduler.timesteps[1 + args.start_step - 1:]
    context = [torch.cat([torch.cat([uncond_embeddings.detach()] * 2), text_embeddings]) for _ in timesteps]

    original_latent = latent.clone()
    latent.requires_grad_(True)
    optimizer = make_latent_optimizer(latent, args.lr, args.lao_weight_decay)
    lao_research = (args.lao_diagnostics or args.lao_quantization != "none"
                    or args.control_scale != 1.0 or args.lao_weight_decay != 0.01)
    if lao_research:
        require_finite(original_latent, "initial latent")
        if lpips_fn is not None:
            lpips_fn.eval().requires_grad_(False)

    init_image = preprocess(enc_pil, res)             # encoded image in [-1, 1] (official: x1)
    init_image_compare = preprocess(origin_pil, res)  # clean input in [-1, 1]

    # ---------- Control-VAE conditioning (official final path, computed once) ----------
    newencoder, decoder = cvae
    if args.control_source == "pgd":
        # second PGD call of the official final path -> control image x2
        adv2, pgd_infos["pgd2"] = pgd_stage(pgd_model, normalize(init_image * 0.5 + 0.5), label, args, all_models, cap)
        init_image1 = (2.0 * denormalize(adv2) - 1).detach()
    elif args.control_source == "encoded":
        init_image1 = init_image.detach()
    else:  # clean
        init_image1 = init_image_compare.detach()
    control01 = (init_image1 / 2 + 0.5).clamp(0, 1)
    with torch.no_grad():
        down_features = [f.detach() for f in newencoder(init_image1)[1]]
        down_features = scale_control_features(down_features, args.control_scale)

    def decode_plain(latents):
        return model.vae.decode(1 / PLAIN_SCALE * latents)["sample"][1:]

    def decode_final(latents):
        return decoder(model.vae.post_quant_conv(1 / FINAL_SCALE * latents), down_features)[1:]

    decode_for_loss = decode_final if args.decode_path == "final" else decode_plain

    @torch.no_grad()
    def final_path_scores(image01):
        """Identity-view margins, not the DI/EOT average used by the objective."""
        values = {}
        for name, image in (("float_margins", image01),
                            ("quantized_margins", quantize_like_save(image01))):
            margins = [real_margin(det(normalize(image))) for det in surrogate_models]
            for margin in margins:
                require_finite(margin, "diagnostic final-path margin")
            values[name] = [float(margin.item()) for margin in margins]
        return values

    # ---------- stage 2: latent adversarial optimisation ----------
    history = []
    for it in range(args.iterations):
        latents = torch.cat([original_latent, latent])
        for ind, t in enumerate(timesteps):
            latents = diffusion_step(model, latents, context[ind], t, args.guidance)
        dec = decode_for_loss(latents)                       # [-1, 1]
        out01 = (dec / 2 + 0.5).clamp(0, 1)
        step_diagnostics = None
        if args.lao_diagnostics:
            latent_before_step = latent.detach().clone()
            step_diagnostics = {"score_timing": "pre_update"}
            if it in (0, args.iterations - 1):
                with torch.no_grad():
                    final_before_step = out01 if args.decode_path == "final" else \
                        (decode_final(latents) / 2 + 0.5).clamp(0, 1)
                    step_diagnostics["final_path"] = final_path_scores(final_before_step)
        if args.lao_quantization == "ste":
            out01 = saved_image_ste(out01)
        x_in = normalize(input_diversity(out01, args.di_prob))
        views = eot_views(x_in, "none") if args.lao_eot == "none" else \
            [normalize(v) for v in eot_views(denormalize(x_in).clamp(0, 1), args.lao_eot, args.eot_jpeg_q, args.eot_resize)]
        logits_views = [[m(v) for m in surrogate_models] for v in views]
        logits_list = logits_views[0]                        # identity view, logged in history
        loss_attack = torch.stack([attack_objective(lv, label, args.loss, kappa, args.attack_loss_weight)
                                   for lv in logits_views]).mean()
        loss = loss_attack
        loss_lpips = torch.zeros(())
        if args.lpips_weight > 0:
            loss_lpips = lpips_fn(init_image_compare, dec).mean()
            loss = loss + args.lpips_weight * loss_lpips
        if lao_research:
            require_finite(loss, "loss")
        optimizer.zero_grad()
        loss.backward()
        if lao_research:
            require_finite(latent.grad, "latent gradient")
        if args.lao_diagnostics:
            step_diagnostics["gradient"] = tensor_norms(latent.grad, "latent gradient")
        optimizer.step()
        if lao_research:
            require_finite(latent, "updated latent")
        history_item = {"iter": it, "loss": float(loss.item()), "attack": float(loss_attack.item()),
                        "lpips": float(loss_lpips.item()),
                        "margins": [float(real_margin(l).item()) for l in logits_list]}
        if args.lao_diagnostics:
            step_diagnostics["update"] = tensor_norms(latent - latent_before_step, "latent update")
            step_diagnostics["distance_from_initial"] = tensor_norms(
                latent - original_latent, "latent distance from initial")
            history_item["lao_diagnostics"] = step_diagnostics
        history.append(history_item)

    # ---------- final decode (official) ----------
    with torch.no_grad():
        latents = torch.cat([original_latent, latent.detach()])
        for ind, t in enumerate(timesteps):
            latents = diffusion_step(model, latents, context[ind], t, args.guidance)
        plain01 = (decode_plain(latents) / 2 + 0.5).clamp(0, 1)
        final01 = (decode_final(latents) / 2 + 0.5).clamp(0, 1)
        if lao_research:
            require_finite(final01, "final decoded image")
        if args.lao_diagnostics:
            # The first batch member followed the same diffusion trajectory from
            # original_latent. Reuse it instead of performing another UNet pass.
            reset_latents = torch.cat([latents[:1], latents[:1]])
            reset01 = (decode_final(reset_latents) / 2 + 0.5).clamp(0, 1)
            reset_scores = final_path_scores(reset01)
            final_scores = final_path_scores(final01)

    origin_u8, plain_u8, final_u8 = to_uint8(origin01), to_uint8(plain01), to_uint8(final01)
    Image.fromarray(origin_u8).save(out_prefix + "_originImage.png")
    Image.fromarray(plain_u8).save(out_prefix + "_plain_vae.png")
    Image.fromarray(final_u8).save(out_prefix + "_adv_image.png")
    if args.lao_diagnostics:
        reset_u8 = to_uint8(reset01)
        Image.fromarray(reset_u8).save(out_prefix + "_reset_image.png")
    if pgd_pil is not None:
        pgd_pil.save(out_prefix + "_pgd1.png")                       # x1
    Image.fromarray(to_uint8(control01)).save(out_prefix + "_control.png")   # x2 (quantised)

    force_eval(all_models)   # nothing above may leave a detector in training mode
    record = {
        "clean": predict(detectors, origin01),
        "pgd": predict(detectors, pgd01) if pgd01 is not None else None,
        "control": predict(detectors, control01),
        "plain_vae": predict(detectors, plain01),
        "final": predict(detectors, final01),
        "psnr": float(calculate_psnr(final_u8, origin_u8)),
        "ssim": float(calculate_ssim(final_u8, origin_u8)),
        "linf": float(np.abs(final_u8.astype(np.float32) - origin_u8.astype(np.float32)).max() / 255.0),
        "pgd_info": pgd_infos,
        "history": history,
    }
    if args.lao_diagnostics:
        record["lao_diagnostics"] = {
            "surrogate_order": [name.strip() for name in args.surrogates.split(",") if name.strip()],
            "score_definition": "identity-view real-minus-AI margin on the final decoder path",
            "initial_latent_reset": reset_scores,
            "final_after_all_updates": final_scores,
            "distance_from_initial": tensor_norms(latent - original_latent, "final latent distance"),
            "changed_saved_pixel_fraction": changed_pixel_fraction(reset_u8, final_u8),
            "reset_image": os.path.basename(out_prefix + "_reset_image.png"),
            "timing_includes_diagnostics": True,
        }
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images_root", required=True, help="directory of AI-generated input images")
    parser.add_argument("--save_dir", required=True, help="output directory (created; must not exist unless --resume)")
    parser.add_argument("--pretrained_diffusion_path", default="stabilityai/stable-diffusion-2-1-base")
    parser.add_argument("--encoder_weights", default="./checkpoints/Controlvae.pt")
    parser.add_argument("--surrogates", default="E", help="comma list of E,R,D,S used in the attack loss")
    parser.add_argument("--evaluate", default="E,R,D,S", help="comma list of detectors reported per image")
    parser.add_argument("--loss", default="ce", choices=["ce", "margin"])
    parser.add_argument("--kappa", default="3.0", help="margin target for --loss margin; a number, or 'cap' = value of --cap")
    parser.add_argument("--decode_path", default="plain", choices=["plain", "final"])
    parser.add_argument("--control_scale", type=float, default=1.0,
                        help="scale fixed Control-VAE features in [0,1]; 1 preserves the original path")
    parser.add_argument("--lao_quantization", default="none", choices=["none", "ste"],
                        help="ste uses exact saved 8-bit image values in the LAO detector forward")
    parser.add_argument("--lao_weight_decay", type=float, default=0.01,
                        help="AdamW latent weight decay; use 0 for the new latent-only study")
    parser.add_argument("--lao_diagnostics", action="store_true",
                        help="record latent gradients/updates and save the initial-latent counterfactual")
    # ---- M3 switches (defaults = official behaviour) ----
    parser.add_argument("--encode_source", default="pgd1", choices=["pgd1", "clean"],
                        help="image that is DDIM-inverted: pgd1 = first PGD output (official), clean = input")
    parser.add_argument("--control_source", default="pgd", choices=["pgd", "encoded", "clean"],
                        help="control image for the Control-VAE encoder: pgd = second PGD call on the encoded "
                             "image (official), encoded = the encoded image itself, clean = the input")
    parser.add_argument("--pgd_impl", default="torchattacks", choices=["torchattacks", "own"])
    parser.add_argument("--pgd_alpha", type=float, default=4 / 255, help="step size of the own PGD loop")
    parser.add_argument("--pgd_steps", type=int, default=10, help="max steps of the own PGD loop")
    parser.add_argument("--cap", default="none", help="none | <float> | auto (median real-dev score of the surrogate)")
    parser.add_argument("--real_dev_dir", default="data/real/dev", help="real development images for --cap auto")
    parser.add_argument("--cap_bisect", action="store_true",
                        help="with --cap: when a PGD step crosses the cap, bisect the step so the score lands at the cap")
    parser.add_argument("--bisect_iters", type=int, default=10)
    parser.add_argument("--pgd_eot", default="none", choices=["none", "jpeg", "resize", "both"])
    parser.add_argument("--lao_eot", default="none", choices=["none", "jpeg", "resize", "both"])
    parser.add_argument("--eot_jpeg_q", type=int, default=75)
    parser.add_argument("--eot_resize", type=float, default=0.5)
    parser.add_argument("--pgd_only", action="store_true",
                        help="pixel-space baseline: save the first PGD output as the attack result and skip the rest")
    parser.add_argument("--lpips_weight", type=float, default=0.0)
    parser.add_argument("--di_prob", type=float, default=0.0)
    parser.add_argument("--iterations", type=int, default=10, help="LAO iterations (official code default 10)")
    parser.add_argument("--diffusion_steps", type=int, default=20)
    parser.add_argument("--start_step", type=int, default=18)
    parser.add_argument("--guidance", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--attack_loss_weight", type=float, default=10.0)
    parser.add_argument("--pgd_eps", type=float, default=4 / 255)
    parser.add_argument("--res", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_images", type=int, default=0, help="0 = all images in images_root")
    parser.add_argument("--resume", action="store_true", help="skip images that already have an output")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"],
                        help="auto = cuda if available, else mps (Apple Silicon), else cpu")
    args = parser.parse_args(argv)

    validate_control_scale(args.control_scale)
    if not np.isfinite(args.lao_weight_decay) or args.lao_weight_decay < 0:
        raise ValueError("--lao_weight_decay must be finite and non-negative")

    m2_device.enable(args.device)
    seed_everything(args.seed)
    surrogate_names = [s.strip() for s in args.surrogates.split(",") if s.strip()]
    evaluate_names = [s.strip() for s in args.evaluate.split(",") if s.strip()]
    for n in surrogate_names + evaluate_names:
        if n not in DETECTOR_NAMES:
            raise ValueError(f"unknown detector {n}; choose from E,R,D,S")
    if args.pgd_impl == "torchattacks" and (args.cap != "none" or args.pgd_eot != "none"
                                            or args.pgd_steps != 10 or abs(args.pgd_alpha - 4 / 255) > 1e-12):
        raise ValueError("--cap, --pgd_eot, --pgd_steps and --pgd_alpha need --pgd_impl own")
    if args.kappa == "cap" and args.cap == "none":
        raise ValueError("--kappa cap needs --cap")
    if args.pgd_only and args.encode_source != "pgd1":
        raise ValueError("--pgd_only needs --encode_source pgd1")

    if os.path.isdir(args.save_dir) and os.listdir(args.save_dir) and not args.resume:
        raise FileExistsError(f"{args.save_dir} exists and is not empty; use a new save_dir or --resume")
    os.makedirs(args.save_dir, exist_ok=True)

    image_paths = natsorted([p for p in glob.glob(os.path.join(args.images_root, "*"))
                             if p.lower().endswith((".png", ".jpg", ".jpeg"))], alg=ns.PATH)
    if not image_paths:
        raise FileNotFoundError(f"no PNG/JPG images in {args.images_root}")
    if args.max_images > 0:
        image_paths = image_paths[: args.max_images]

    config = dict(vars(args))
    config.update({"n_images": len(image_paths), "torch": torch.__version__,
                   "command": " ".join(sys.argv)})
    with open(os.path.join(args.save_dir, "run_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"[m2] loading Stable Diffusion from {args.pretrained_diffusion_path}", flush=True)
    sd_path = m2_device.local_model(args.pretrained_diffusion_path, ignore_fp16=(m2_device.DEVICE != "cuda"))
    try:
        model = StableDiffusionPipeline.from_pretrained(
            sd_path, safety_checker=None, requires_safety_checker=False).to(m2_device.DEVICE)
    except TypeError:
        model = StableDiffusionPipeline.from_pretrained(sd_path, safety_checker=None).to(m2_device.DEVICE)
    model.scheduler = DDIMScheduler.from_config(model.scheduler.config)
    model.unet.requires_grad_(False)
    model.vae.requires_grad_(False)
    model.text_encoder.requires_grad_(False)

    print(f"[m2] loading detectors {sorted(set(surrogate_names + evaluate_names))}", flush=True)
    detectors = {}
    for name in sorted(set(surrogate_names + evaluate_names)):
        det = other_attacks.model_selection(name).eval()
        det.requires_grad_(False)
        detectors[name] = det
    surrogate_models = [detectors[n] for n in surrogate_names]
    eval_detectors = {n: detectors[n] for n in evaluate_names}
    pgd_model = surrogate_models[0] if len(surrogate_models) == 1 else Ensemble(surrogate_models)

    # ---- M3: resolve the cap and kappa once, record them in run_config.json ----
    cap = None
    if args.cap == "auto":
        cap, dev_margins = median_real_margin(pgd_model, args.real_dev_dir, args.res)
        force_eval(list(detectors.values()))
        config["cap_resolved"] = cap
        config["cap_real_dev_margins"] = dev_margins
        print(f"[m3] cap = median real-dev surrogate score = {cap:.3f} over {len(dev_margins)} images "
              f"(range {min(dev_margins):.2f} .. {max(dev_margins):.2f})", flush=True)
    elif args.cap != "none":
        cap = float(args.cap)
        config["cap_resolved"] = cap
    kappa = cap if args.kappa == "cap" else float(args.kappa)
    config["kappa_resolved"] = kappa
    with open(os.path.join(args.save_dir, "run_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    cvae = load_control_vae(model, args.encoder_weights)
    lpips_fn = build_lpips() if args.lpips_weight > 0 else None

    records_path = os.path.join(args.save_dir, "records.jsonl")
    done = set()
    if args.resume and os.path.exists(records_path):
        with open(records_path) as f:
            done = {json.loads(line)["image_id"] for line in f if line.strip()}

    for ind, path in enumerate(image_paths):
        image_id = str(ind).rjust(4, "0")
        out_prefix = os.path.join(args.save_dir, image_id)
        if image_id in done and os.path.exists(out_prefix + "_adv_image.png"):
            continue
        pil_image = Image.open(path).convert("RGB")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        record = attack_one(model, pil_image, args, surrogate_models, eval_detectors, cvae, lpips_fn, out_prefix,
                            pgd_model, cap, kappa)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        record.update({"image_id": image_id, "source": os.path.basename(path),
                       "seconds": round(time.time() - t0, 2)})
        with open(records_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        finals = " ".join(f"{k}:{'real' if v['pred'] == 1 else 'AI'}" for k, v in record["final"].items())
        steps = " ".join(f"{k}:{v.get('steps')}" for k, v in record["pgd_info"].items()) if record["pgd_info"] else "-"
        print(f"[m2] {image_id} {record['seconds']:.1f}s psnr={record['psnr']:.2f} "
              f"ssim={record['ssim']:.3f} pgd_steps[{steps}] final-> {finals}", flush=True)
        m2_device.release_memory()   # keeps per-image time flat; see the note in m2_device

    print(f"[m2] done. outputs in {args.save_dir}", flush=True)


if __name__ == "__main__":
    main()
