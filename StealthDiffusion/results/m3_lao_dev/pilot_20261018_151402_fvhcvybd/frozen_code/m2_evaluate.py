"""
FIT5230 · Dark.ModelBreakers · Milestone 2 — evaluation of saved attack outputs.

Run from inside the official StealthDiffusion checkout with its Python 3.10 environment.

  python m2_evaluate.py --runs B00=results/B00 M2_full=results/M2_full \
        --real_dir data/real/test --out results/eval \
        --detectors E,R,D,S --conditions none,jpeg75,resize05

Everything is measured on the PNG files written by m2_stealth.py (re-loaded from disk),
never on in-memory tensors.  For every (run, detector, condition) it reports:

  n                 images in the run
  n_clean_detected  images the detector correctly flags as AI *before* the attack
                    (under the same post-processing condition)
  cond_asr          conditional attack success rate = among n_clean_detected, the fraction
                    whose attacked version is classified as real
  uncond_escape     attacked images classified as real / n  (no conditioning)
  real_fpr          real test photos classified as AI / n_real   (detector false positives)
  auc_real_vs_adv   ROC-AUC of the detector score separating real photos from attacked
                    images (1.0 = still perfectly separable by moving the threshold;
                    0.5 = attacked images are indistinguishable from real photos)
  auc_real_vs_clean the same for un-attacked generated images
  psnr/ssim/lpips/linf  distortion of the attacked image w.r.t. the 224x224 input
  plain_ok_final_fail  images where the ordinary-VAE decode evaded the detector but the
                    final Control-VAE output did not (motivation for --decode_path final)

Post-processing conditions (applied identically to clean and attacked images):
  none, jpeg<Q> (PIL JPEG quality Q), resize<S> (S read as the decimals of the scale:
  resize05 = 0.5x, resize075 = 0.75x; Lanczos down, bicubic back to 224),
  combined with '+', e.g. jpeg75+resize05.
"""
import argparse
import csv
import io
import json
import os

import numpy as np
import torch
from PIL import Image
from natsort import natsorted, ns
from torchvision import transforms

import other_attacks  # noqa: E402
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import m2_device  # noqa: E402
from diff_latent_attack import calculate_psnr, calculate_ssim  # noqa: E402

normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
RES = 224


def apply_condition(pil, cond):
    for token in cond.split("+"):
        token = token.strip()
        if token == "none":
            continue
        if token.startswith("jpeg"):
            q = int(token[4:])
            buf = io.BytesIO()
            pil.save(buf, format="JPEG", quality=q)
            buf.seek(0)
            pil = Image.open(buf).convert("RGB")
        elif token.startswith("resize"):
            digits = token[6:]                      # resize05 = 0.5x, resize075 = 0.75x, resize5 = 0.5x
            scale = int(digits) / (10.0 ** len(digits))
            small = max(8, int(round(RES * scale)))
            pil = pil.resize((small, small), resample=Image.LANCZOS).resize((RES, RES), resample=Image.BICUBIC)
        else:
            raise ValueError(f"unknown condition token {token}")
    return pil


def pil_to_tensor(pil):
    return torch.from_numpy(np.float32(pil.convert("RGB")) / 255.0).permute(2, 0, 1)


@torch.no_grad()
def predict_batch(det, pils, batch_size=16):
    preds, margins = [], []
    for i in range(0, len(pils), batch_size):
        x = torch.stack([pil_to_tensor(p) for p in pils[i:i + batch_size]]).cuda()
        logits = det(normalize(x))
        preds += torch.argmax(logits, 1).tolist()
        margins += (logits[:, 1] - logits[:, 0]).tolist()
    return preds, margins


def auc(pos_scores, neg_scores):
    """ROC-AUC with real photos as positives (higher margin = more 'real')."""
    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return float("nan")
    from sklearn.metrics import roc_auc_score
    y = [1] * len(pos_scores) + [0] * len(neg_scores)
    return float(roc_auc_score(y, list(pos_scores) + list(neg_scores)))


def load_run(run_dir, adv_suffix="_adv_image.png"):
    """adv_suffix selects which saved file counts as the attack output. M3: a run directory
    also holds NNNN_pgd1.png (x1, configuration A1) and NNNN_control.png (x2, configuration X1),
    so  --runs A1=results/m3/A5:_pgd1.png  evaluates the first PGD output on its own."""
    ids = natsorted([f[:-len("_originImage.png")] for f in os.listdir(run_dir) if f.endswith("_originImage.png")], alg=ns.PATH)
    items = []
    for i in ids:
        adv = os.path.join(run_dir, i + adv_suffix)
        if not os.path.exists(adv):
            continue
        plain = os.path.join(run_dir, i + "_plain_vae.png")
        items.append({"image_id": i,
                      "clean": Image.open(os.path.join(run_dir, i + "_originImage.png")).convert("RGB"),
                      "adv": Image.open(adv).convert("RGB"),
                      "plain": Image.open(plain).convert("RGB") if os.path.exists(plain) else None})
    seconds = {}
    rec = os.path.join(run_dir, "records.jsonl")
    if os.path.exists(rec):
        with open(rec) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    seconds[r["image_id"]] = r.get("seconds")
    return items, seconds


def mean(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and np.isnan(x))]
    return float(np.mean(xs)) if xs else float("nan")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", required=True,
                   help="name=dir pairs; name=dir:SUFFIX evaluates dir/NNNN<SUFFIX> instead of NNNN_adv_image.png")
    p.add_argument("--real_dir", required=True, help="directory of real 224x224 PNGs (test split)")
    p.add_argument("--out", required=True)
    p.add_argument("--detectors", default="E,R,D,S")
    p.add_argument("--conditions", default="none,jpeg75,resize05")
    p.add_argument("--no_lpips", action="store_true")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"],
                   help="auto = cuda if available, else mps (Apple Silicon), else cpu")
    args = p.parse_args(argv)

    m2_device.enable(args.device)
    os.makedirs(args.out, exist_ok=True)
    runs = {}
    for spec in args.runs:
        name, d = spec.split("=", 1)
        suffix = "_adv_image.png"
        if ":" in d:
            d, suffix = d.rsplit(":", 1)
        items, seconds = load_run(d, suffix)
        if not items:
            raise FileNotFoundError(f"no completed images in {d}")
        runs[name] = {"dir": d, "items": items, "seconds": seconds}
        print(f"[eval] run {name}: {len(items)} images from {d} ({suffix})", flush=True)
    real_files = natsorted([f for f in os.listdir(args.real_dir) if f.endswith(".png")], alg=ns.PATH)
    real_pils = [Image.open(os.path.join(args.real_dir, f)).convert("RGB") for f in real_files]
    print(f"[eval] {len(real_pils)} real test images", flush=True)
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]

    # ---- distortion (condition-free), once per run ----
    lpips_fn = None
    if not args.no_lpips:
        from lossers.lpips import LPIPS
        lpips_fn = LPIPS(net="vgg", cache_dir="./.cache").cuda().eval()
    quality = {}
    for name, run in runs.items():
        for it in run["items"]:
            a, c = np.array(it["adv"]), np.array(it["clean"])
            q = {"psnr": float(calculate_psnr(a, c)), "ssim": float(calculate_ssim(a, c)),
                 "linf": float(np.abs(a.astype(np.float32) - c.astype(np.float32)).max() / 255.0), "lpips": None}
            if lpips_fn is not None:
                with torch.no_grad():
                    ta = (pil_to_tensor(it["adv"])[None].cuda() * 2 - 1)
                    tc = (pil_to_tensor(it["clean"])[None].cuda() * 2 - 1)
                    q["lpips"] = float(lpips_fn(tc, ta).mean().item())
            quality[(name, it["image_id"])] = q
    del lpips_fn
    torch.cuda.empty_cache()

    per_image, real_rows, summary = [], [], []
    for det_name in [d.strip() for d in args.detectors.split(",")]:
        det = other_attacks.model_selection(det_name).eval()
        det.requires_grad_(False)
        assert not det.training, f"detector {det_name} must be in eval mode (BatchNorm statistics)"
        for cond in conditions:
            r_pred, r_margin = predict_batch(det, [apply_condition(p, cond) for p in real_pils])
            for f, pr, mg in zip(real_files, r_pred, r_margin):
                real_rows.append({"detector": det_name, "condition": cond, "image": f, "pred": pr, "real_margin": mg})
            real_fpr = float(np.mean([pr == 0 for pr in r_pred])) if r_pred else float("nan")
            for name, run in runs.items():
                items = run["items"]
                c_pred, c_margin = predict_batch(det, [apply_condition(it["clean"], cond) for it in items])
                a_pred, a_margin = predict_batch(det, [apply_condition(it["adv"], cond) for it in items])
                has_plain = all(it["plain"] is not None for it in items)
                if has_plain:
                    p_pred, p_margin = predict_batch(det, [apply_condition(it["plain"], cond) for it in items])
                else:
                    p_pred, p_margin = [None] * len(items), [None] * len(items)
                n = len(items)
                detected = [i for i in range(n) if c_pred[i] == 0]
                evaded = [i for i in detected if a_pred[i] == 1]
                plain_ok_final_fail = sum(1 for i in detected if has_plain and p_pred[i] == 1 and a_pred[i] == 0)
                for i, it in enumerate(items):
                    q = quality[(name, it["image_id"])] if cond == "none" else {"psnr": None, "ssim": None, "linf": None, "lpips": None}
                    per_image.append({"run": name, "detector": det_name, "condition": cond, "image_id": it["image_id"],
                                      "clean_pred": c_pred[i], "clean_margin": c_margin[i],
                                      "adv_pred": a_pred[i], "adv_margin": a_margin[i],
                                      "plain_pred": p_pred[i], "plain_margin": p_margin[i],
                                      "psnr": q["psnr"], "ssim": q["ssim"], "linf": q["linf"], "lpips": q["lpips"],
                                      "seconds": run["seconds"].get(it["image_id"])})
                row = {"run": name, "detector": det_name, "condition": cond, "n": n,
                       "n_clean_detected": len(detected),
                       "cond_asr": (len(evaded) / len(detected)) if detected else float("nan"),
                       "uncond_escape": float(np.mean([pr == 1 for pr in a_pred])),
                       "mean_adv_margin": float(np.mean(a_margin)),
                       "mean_clean_margin": float(np.mean(c_margin)),
                       "real_fpr": real_fpr,
                       "auc_real_vs_adv": auc(r_margin, a_margin),
                       "auc_real_vs_clean": auc(r_margin, c_margin),
                       "plain_ok_final_fail": plain_ok_final_fail if has_plain else None,
                       "psnr": mean([quality[(name, it["image_id"])]["psnr"] for it in items]),
                       "ssim": mean([quality[(name, it["image_id"])]["ssim"] for it in items]),
                       "lpips": mean([quality[(name, it["image_id"])]["lpips"] for it in items]),
                       "linf": mean([quality[(name, it["image_id"])]["linf"] for it in items]),
                       "seconds": mean([run["seconds"].get(it["image_id"]) for it in items])}
                summary.append(row)
                print(f"[eval] {name:<12} {det_name} {cond:<16} detected {len(detected):>3}/{n:<3} "
                      f"cond_ASR {row['cond_asr']:.2f}  escape {row['uncond_escape']:.2f}  "
                      f"real_FPR {real_fpr:.2f}  AUC {row['auc_real_vs_adv']:.2f}", flush=True)
        del det
        torch.cuda.empty_cache()

    def write_csv(path, rows):
        if not rows:
            return
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    write_csv(os.path.join(args.out, "per_image.csv"), per_image)
    write_csv(os.path.join(args.out, "real_predictions.csv"), real_rows)
    write_csv(os.path.join(args.out, "summary.csv"), summary)
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump({"runs": {k: v["dir"] for k, v in runs.items()}, "real_dir": args.real_dir,
                   "conditions": conditions, "summary": summary}, f, indent=2)
    print(f"[eval] wrote {args.out}/summary.csv, per_image.csv, real_predictions.csv", flush=True)


if __name__ == "__main__":
    main()
