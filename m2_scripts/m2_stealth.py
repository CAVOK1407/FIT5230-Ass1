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

With all defaults the script reproduces the official pipeline (variant "B00").
Everything else (DDIM inversion, 2 diffusion steps, AdamW lr 1e-2, Control-VAE weights,
the two PGD stages, 224x224 resolution, seed 8888 for the VAE sampling) is kept as in
the official code so that variants are comparable.

Outputs per image (in --save_dir):
  NNNN_originImage.png   input resized to 224x224 (the reference for PSNR/SSIM/LPIPS)
  NNNN_plain_vae.png     ordinary-VAE decode of the final latent (diagnostic only)
  NNNN_adv_image.png     final Control-VAE output = the attack result
  records.jsonl          one line per image: predictions of all detectors, PSNR, SSIM, time
  run_config.json        the exact configuration of this run
"""
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


@torch.enable_grad()
def attack_one(model, pil_image, args, surrogate_models, detectors, cvae, lpips_fn, out_prefix):
    """Attack one PIL image. Returns a record dict. Mirrors diff_latent_attack.diffattack."""
    res = args.res
    label = torch.tensor([0]).cuda()  # official label convention: 0 = AI-generated
    pgd_model = surrogate_models[0] if len(surrogate_models) == 1 else Ensemble(surrogate_models)
    # Every detector that is forwarded anywhere in this function, deduplicated by identity.
    all_models = list({id(m): m for m in list(surrogate_models) + list(detectors.values())}.values())
    force_eval(all_models)

    # ---------- stage 0: PGD in pixel space (official) ----------
    origin_pil = pil_image.resize((res, res), resample=Image.LANCZOS)
    origin01 = torch.from_numpy(np.float32(origin_pil)[:, :, :3] / 255.0).permute(2, 0, 1)[None].cuda()
    test_image = normalize(origin01)
    adv_norm = run_pgd(pgd_model, args.pgd_eps, test_image, label, all_models)
    pgd_pil = Image.fromarray(to_uint8(denormalize(adv_norm)))

    # ---------- stage 1: DDIM inversion (official) ----------
    prompt = [""] * 2
    _, inversion_latents = ddim_reverse_sample(pgd_pil, prompt, model, args.diffusion_steps, 0, res=res)
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
    optimizer = torch.optim.AdamW([latent], lr=args.lr)

    init_image = preprocess(pgd_pil, res)          # PGD image in [-1, 1]
    init_image_compare = preprocess(origin_pil, res)  # clean input in [-1, 1]

    # ---------- Control-VAE conditioning (official final path, computed once) ----------
    newencoder, decoder = cvae
    # second PGD stage of the official final path
    adv2 = run_pgd(pgd_model, args.pgd_eps, normalize(init_image * 0.5 + 0.5), label, all_models)
    init_image1 = (2.0 * denormalize(adv2) - 1).detach()
    with torch.no_grad():
        down_features = [f.detach() for f in newencoder(init_image1)[1]]

    def decode_plain(latents):
        return model.vae.decode(1 / PLAIN_SCALE * latents)["sample"][1:]

    def decode_final(latents):
        return decoder(model.vae.post_quant_conv(1 / FINAL_SCALE * latents), down_features)[1:]

    decode_for_loss = decode_final if args.decode_path == "final" else decode_plain

    # ---------- stage 2: latent adversarial optimisation ----------
    history = []
    for it in range(args.iterations):
        latents = torch.cat([original_latent, latent])
        for ind, t in enumerate(timesteps):
            latents = diffusion_step(model, latents, context[ind], t, args.guidance)
        dec = decode_for_loss(latents)                       # [-1, 1]
        out01 = (dec / 2 + 0.5).clamp(0, 1)
        x_in = normalize(input_diversity(out01, args.di_prob))
        logits_list = [m(x_in) for m in surrogate_models]
        loss_attack = attack_objective(logits_list, label, args.loss, args.kappa, args.attack_loss_weight)
        loss = loss_attack
        loss_lpips = torch.zeros(())
        if args.lpips_weight > 0:
            loss_lpips = lpips_fn(init_image_compare, dec).mean()
            loss = loss + args.lpips_weight * loss_lpips
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        history.append({"iter": it, "loss": float(loss.item()), "attack": float(loss_attack.item()),
                        "lpips": float(loss_lpips.item()),
                        "margins": [float(real_margin(l).item()) for l in logits_list]})

    # ---------- final decode (official) ----------
    with torch.no_grad():
        latents = torch.cat([original_latent, latent.detach()])
        for ind, t in enumerate(timesteps):
            latents = diffusion_step(model, latents, context[ind], t, args.guidance)
        plain01 = (decode_plain(latents) / 2 + 0.5).clamp(0, 1)
        final01 = (decode_final(latents) / 2 + 0.5).clamp(0, 1)

    origin_u8, plain_u8, final_u8 = to_uint8(origin01), to_uint8(plain01), to_uint8(final01)
    Image.fromarray(origin_u8).save(out_prefix + "_originImage.png")
    Image.fromarray(plain_u8).save(out_prefix + "_plain_vae.png")
    Image.fromarray(final_u8).save(out_prefix + "_adv_image.png")

    force_eval(all_models)   # nothing above may leave a detector in training mode
    record = {
        "clean": predict(detectors, origin01),
        "pgd": predict(detectors, denormalize(adv_norm).clamp(0, 1)),
        "plain_vae": predict(detectors, plain01),
        "final": predict(detectors, final01),
        "psnr": float(calculate_psnr(final_u8, origin_u8)),
        "ssim": float(calculate_ssim(final_u8, origin_u8)),
        "linf": float(np.abs(final_u8.astype(np.float32) - origin_u8.astype(np.float32)).max() / 255.0),
        "history": history,
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
    parser.add_argument("--kappa", type=float, default=3.0, help="margin target for --loss margin")
    parser.add_argument("--decode_path", default="plain", choices=["plain", "final"])
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

    m2_device.enable(args.device)
    seed_everything(args.seed)
    surrogate_names = [s.strip() for s in args.surrogates.split(",") if s.strip()]
    evaluate_names = [s.strip() for s in args.evaluate.split(",") if s.strip()]
    for n in surrogate_names + evaluate_names:
        if n not in DETECTOR_NAMES:
            raise ValueError(f"unknown detector {n}; choose from E,R,D,S")

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
        record = attack_one(model, pil_image, args, surrogate_models, eval_detectors, cvae, lpips_fn, out_prefix)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        record.update({"image_id": image_id, "source": os.path.basename(path),
                       "seconds": round(time.time() - t0, 2)})
        with open(records_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        finals = " ".join(f"{k}:{'real' if v['pred'] == 1 else 'AI'}" for k, v in record["final"].items())
        print(f"[m2] {image_id} {record['seconds']:.1f}s psnr={record['psnr']:.2f} "
              f"ssim={record['ssim']:.3f} final-> {finals}", flush=True)
        m2_device.release_memory()   # keeps per-image time flat; see the note in m2_device

    print(f"[m2] done. outputs in {args.save_dir}", flush=True)


if __name__ == "__main__":
    main()
