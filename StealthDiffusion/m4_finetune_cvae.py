#!/usr/bin/env python3
"""M4 unconstrained-enhancement demo: fine-tune the Control-VAE control branch
so that the *reconstruction itself* evades the detectors (no per-image PGD, no
latent optimisation at test time).

Why.  The released train_vae.py trains the control encoder on real photographs
with LPIPS + L1 + a frequency loss only; the noise-prototype term the paper
describes is computed but never enters the loss (loss_a = 0).  M3 showed the
per-image attack that reaches the saved file is a pixel PGD step costing
~10 s/image.  With unlimited data and GPU time, the attacker would instead
put the detector objective into the *training* of the control branch once,
and then launder every generated image at the cost of one encode-decode
pass (~2 s/image on a laptop, nothing on a GPU).

What this script does (small-scale feasibility version):
  * loads the released Controlvae.pt control encoder (NewEncoder) and the
    SD VAE decoder with skip connections (NewDecoder), exactly as
    m2_stealth.load_control_vae does;
  * trains ONLY the NewEncoder (encoder + zero convs); SD VAE and decoder are
    frozen; detectors are frozen and kept in eval mode;
  * proxy for the final path: z = 0.18215 * vae.encode(x).mean,
    rec = NewDecoder(post_quant_conv(z / 0.1825), NewEncoder(x)[1]);
  * loss = l1 * L1(x, rec) + lpips * LPIPS(x, rec)
           + adv * mean_over_surrogates( margin loss on rec, label = real );
  * logs per-epoch dev metrics (PSNR / LPIPS / flagged fraction on PNG-
    quantised reconstructions, per detector) and saves a checkpoint in the
    same {"encoder": state_dict} format, so m2_stealth.py can be run with
    --encoder_weights <new file> and the existing evaluation is reused.

Nothing under results/m3_* is touched.  Run from the StealthDiffusion folder
inside the fit5230m2 environment.
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import other_attacks  # noqa: E402  (official detector loader; hides GPUs at import)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import m2_device  # noqa: E402
from m2_stealth import (  # noqa: E402
    CVAE_DECODER_PARAMS, CVAE_ENCODER_PARAMS, FINAL_SCALE, PLAIN_SCALE,
    build_lpips, normalize, real_margin, seed_everything,
)
from ControlVAE import NewDecoder, NewEncoder  # noqa: E402
from diff_latent_attack import calculate_psnr  # noqa: E402


def list_images(d):
    return sorted(p for p in glob.glob(os.path.join(d, "*")) if p.lower().endswith((".png", ".jpg", ".jpeg")))


def load01(path, res):
    pil = Image.open(path).convert("RGB").resize((res, res), resample=Image.LANCZOS)
    return torch.from_numpy(np.float32(pil)[:, :, :3] / 255.0).permute(2, 0, 1)


def quantize01(x01):
    """Same 8-bit rounding the PNG writer applies (m2_stealth.to_uint8)."""
    return torch.round(x01.clamp(0, 1) * 255.0) / 255.0


def to_uint8(x01):
    return np.uint8(x01.detach().clamp(0, 1).cpu().numpy().transpose(1, 2, 0) * 255.0)


def margin_loss(logits, kappa):
    # push margin (logit_real - logit_AI) up to kappa, then stop: no overshoot
    return F.relu(kappa - real_margin(logits)).mean()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dirs", nargs="+", required=True, help="folders of generated 224px PNGs (training)")
    ap.add_argument("--dev_dir", required=True, help="generated dev images (monitoring only)")
    ap.add_argument("--real_dev_dir", default="", help="optional real dev images, to report the real-image margin range")
    ap.add_argument("--pretrained_diffusion_path", default="Manojb/stable-diffusion-2-1-base")
    ap.add_argument("--encoder_weights", default="./checkpoints/Controlvae.pt")
    ap.add_argument("--surrogates", default="E,R,D")
    ap.add_argument("--monitor", default="E,R,D,S")
    ap.add_argument("--loss", default="margin", choices=["margin", "ce"])
    ap.add_argument("--kappa", type=float, default=3.0)
    ap.add_argument("--adv_weight", type=float, default=0.05)
    ap.add_argument("--lpips_weight", type=float, default=1.0)
    ap.add_argument("--l1_weight", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--res", type=int, default=224)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="./checkpoints/Controlvae_m4.pt")
    ap.add_argument("--log_dir", default="./results/m4_cvae/train")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    ap.add_argument("--max_train", type=int, default=0)
    ap.add_argument("--min_psnr", type=float, default=32.0, help="dev quality gate for checkpoint selection")
    ap.add_argument("--max_lpips", type=float, default=0.05, help="dev quality gate for checkpoint selection")
    ap.add_argument("--train_zero_convs_only", action="store_true",
                    help="freeze the encoder trunk; train only the four zero convs that feed the skip connections")
    args = ap.parse_args(argv)

    m2_device.enable(args.device)
    seed_everything(args.seed)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(os.path.join(args.log_dir, "train_config.json"), "w") as f:
        json.dump({**vars(args), "torch": torch.__version__, "command": " ".join(sys.argv)}, f, indent=2)

    # ---- SD VAE (frozen) ----
    from diffusers.models import AutoencoderKL
    sd_path = m2_device.local_model(args.pretrained_diffusion_path, ignore_fp16=(m2_device.DEVICE != "cuda"))
    vae = AutoencoderKL.from_pretrained(sd_path, subfolder="vae").cuda().eval()
    vae.requires_grad_(False)

    # ---- Control-VAE: encoder (trainable) + skip decoder (frozen), as in m2_stealth.load_control_vae ----
    enc = NewEncoder(**CVAE_ENCODER_PARAMS)
    state = torch.load(args.encoder_weights, map_location="cpu")
    missing, unexpected = enc.load_state_dict(state["encoder"], strict=False)
    print(f"[m4] control encoder loaded from {args.encoder_weights}: {len(missing)} missing, {len(unexpected)} unexpected keys", flush=True)
    enc = enc.cuda().train()
    dec = NewDecoder(**CVAE_DECODER_PARAMS)
    dec.load_state_dict(vae.decoder.state_dict(), strict=False)
    dec = dec.cuda().eval()
    dec.requires_grad_(False)

    # ---- detectors (frozen, eval) ----
    names = sorted(set([s for s in args.surrogates.split(",") if s] + [s for s in args.monitor.split(",") if s]))
    dets = {}
    for n in names:
        d = other_attacks.model_selection(n).eval()
        d.requires_grad_(False)
        dets[n] = d
    surr = [dets[n] for n in args.surrogates.split(",") if n]
    lpips_fn = build_lpips().eval()
    lpips_fn.requires_grad_(False)

    # ---- data ----
    train_paths = []
    for d in args.train_dirs:
        train_paths += list_images(d)
    if args.max_train > 0:
        train_paths = train_paths[: args.max_train]
    dev_paths = list_images(args.dev_dir)
    print(f"[m4] train images {len(train_paths)}  dev images {len(dev_paths)}", flush=True)
    train01 = torch.stack([load01(p, args.res) for p in train_paths])
    dev01 = torch.stack([load01(p, args.res) for p in dev_paths])

    # real-image margin range on the dev photographs (for the report; not used in training)
    if args.real_dev_dir:
        real01 = torch.stack([load01(p, args.res) for p in list_images(args.real_dev_dir)]).cuda()
        with torch.no_grad():
            real_rng = {n: [float(v) for v in real_margin(dets[n](normalize(real01))).cpu()] for n in names}
        with open(os.path.join(args.log_dir, "real_dev_margins.json"), "w") as f:
            json.dump(real_rng, f, indent=2)
        print("[m4] real dev margin median:", {n: round(float(np.median(v)), 2) for n, v in real_rng.items()}, flush=True)

    @torch.no_grad()
    def encode_latents(x01):
        x = 2.0 * x01 - 1.0
        return PLAIN_SCALE * vae.encode(x).latent_dist.mean

    # precompute latents once (VAE is frozen); keep on device
    train_z = torch.cat([encode_latents(train01[i:i + 4].cuda()) for i in range(0, len(train01), 4)])
    dev_z = torch.cat([encode_latents(dev01[i:i + 4].cuda()) for i in range(0, len(dev01), 4)])

    def reconstruct(x01, z):
        x = 2.0 * x01 - 1.0
        feats = enc(x)[1]
        rec = dec(vae.post_quant_conv(z / FINAL_SCALE), feats)
        return rec  # [-1, 1]

    @torch.no_grad()
    def evaluate(tag):
        enc.eval()
        out = {"tag": tag, "n": len(dev01)}
        flagged = {n: 0 for n in names}
        margins = {n: [] for n in names}
        psnr, lp = [], []
        for i in range(len(dev01)):
            x01 = dev01[i:i + 1].cuda()
            rec = reconstruct(x01, dev_z[i:i + 1])
            rec01 = quantize01((rec + 1) / 2)
            for n in names:
                lg = dets[n](normalize(rec01))
                flagged[n] += int(torch.argmax(lg, 1).item() == 0)
                margins[n].append(float(real_margin(lg).item()))
            psnr.append(float(calculate_psnr(to_uint8(rec01[0]), to_uint8(x01[0]))))
            lp.append(float(lpips_fn(2 * x01 - 1, 2 * rec01 - 1).mean().item()))
        out.update({f"flagged_{n}": flagged[n] / len(dev01) for n in names})
        out.update({f"margin_med_{n}": float(np.median(margins[n])) for n in names})
        out.update({"psnr": float(np.mean(psnr)), "lpips": float(np.mean(lp))})
        enc.train()
        return out

    log_path = os.path.join(args.log_dir, "train_log.jsonl")
    logf = open(log_path, "a")

    def log(rec):
        logf.write(json.dumps(rec) + "\n")
        logf.flush()
        print("[m4]", json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}), flush=True)

    log(evaluate("epoch0_released_weights"))

    if args.train_zero_convs_only:
        enc.requires_grad_(False)
        for zc in enc.zero_convs:
            zc.requires_grad_(True)
        params = [p for zc in enc.zero_convs for p in zc.parameters()]
    else:
        params = list(enc.parameters())
    print(f"[m4] trainable parameters: {sum(p.numel() for p in params) / 1e6:.2f} M", flush=True)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    best = None  # (score, epoch); score = mean flagged fraction over monitored detectors, lower = better
    label_real = None
    n = len(train01)
    g = torch.Generator().manual_seed(args.seed)
    step = 0
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        perm = torch.randperm(n, generator=g)
        ep_loss = []
        for b in range(0, n, args.batch_size):
            idx = perm[b:b + args.batch_size]
            x01 = train01[idx].cuda()
            z = train_z[idx]
            rec = reconstruct(x01, z)
            rec01 = ((rec + 1) / 2).clamp(0, 1)
            l1 = F.l1_loss(rec, 2 * x01 - 1)
            lp = lpips_fn(2 * x01 - 1, rec).mean()
            adv = 0.0
            for d in surr:
                lg = d(normalize(rec01))
                if args.loss == "margin":
                    adv = adv + margin_loss(lg, args.kappa)
                else:
                    if label_real is None:
                        label_real = torch.ones(lg.shape[0], dtype=torch.long, device=lg.device)
                    adv = adv + F.cross_entropy(lg, label_real[: lg.shape[0]])
            adv = adv / len(surr)
            loss = args.l1_weight * l1 + args.lpips_weight * lp + args.adv_weight * adv
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
            ep_loss.append((float(loss), float(l1), float(lp), float(adv)))
        m = np.mean(ep_loss, axis=0)
        rec = evaluate(f"epoch{ep}")
        rec.update({"epoch": ep, "step": step, "loss": float(m[0]), "l1": float(m[1]), "lpips_train": float(m[2]),
                    "adv": float(m[3]), "elapsed_s": time.time() - t0})
        passed = rec["psnr"] >= args.min_psnr and rec["lpips"] <= args.max_lpips
        score = float(np.mean([rec[f"flagged_{n}"] for n in names]))
        rec.update({"quality_gate": passed, "flagged_mean": score})
        last_path = args.out.replace(".pt", "_last.pt")
        torch.save({"encoder": enc.state_dict(),
                    "m4_meta": {"epoch": ep, "config": vars(args), "source": args.encoder_weights}}, last_path)
        if passed and (best is None or score < best[0]):
            best = (score, ep)
            torch.save({"encoder": enc.state_dict(),
                        "m4_meta": {"epoch": ep, "config": vars(args), "source": args.encoder_weights,
                                    "selected_by": f"lowest mean flagged on dev with psnr>={args.min_psnr}, lpips<={args.max_lpips}"}},
                       args.out)
            rec["saved_as_best"] = True
        log(rec)
    logf.close()
    if best is None:
        print(f"[m4] WARNING: no epoch passed the quality gate; {args.out} was NOT written. Last weights: {last_path}", flush=True)
    else:
        print(f"[m4] best epoch {best[1]} (mean flagged {best[0]:.3f}) saved to {args.out}", flush=True)


if __name__ == "__main__":
    main()
