"""
FIT5230 · Dark.ModelBreakers · Milestone 2 — data preparation.

Run from inside the official StealthDiffusion checkout with its Python 3.10 environment.

  python m2_data.py real     --out data --n_dev 10 --n_test 100
      Downloads Imagenette (a public 10-class subset of ImageNet, real photographs),
      takes validation images round-robin over the 10 classes, centre-crops and resizes
      them to 224x224 PNG:  data/real/dev/*.png  and  data/real/test/*.png

  python m2_data.py generate --out data --n_dev 10 --n_test 100 --model stable-diffusion-v1-5/stable-diffusion-v1-5
      Generates AI images with Stable Diffusion for the same 10 classes
      ("a photo of a <class>"), 512x512, then saves 224x224 PNG copies:
      data/gen/dev/*.png, data/gen/test/*.png  (512px originals in data/gen512/)

  python m2_data.py check    --out data
      Runs the four official detectors (E,R,D,S) on the dev sets and prints the clean
      accuracy and the label polarity (official convention: class 0 = AI, class 1 = real).

Every image is listed in a manifest CSV with its origin (file path or prompt+seed+model).
"""
import argparse
import csv
import glob
import os
import shutil
import sys
import tarfile
import urllib.request

import numpy as np
from PIL import Image
from natsort import natsorted, ns

IMAGENETTE_URLS = [
    "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz",
    "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz",
]
IMAGENETTE_CLASSES = [
    ("n01440764", "tench"), ("n02102040", "English springer spaniel"), ("n02979186", "cassette player"),
    ("n03000684", "chain saw"), ("n03028079", "church"), ("n03394916", "French horn"),
    ("n03417042", "garbage truck"), ("n03425413", "gas pump"), ("n03445777", "golf ball"),
    ("n03888605", "parachute"),
]
RES = 224


def center_square_resize(pil, res=RES):
    pil = pil.convert("RGB")
    w, h = pil.size
    s = min(w, h)
    left, top = (w - s) // 2, (h - s) // 2
    return pil.crop((left, top, left + s, top + s)).resize((res, res), resample=Image.LANCZOS)


def write_manifest(path, rows, fields):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _imagenette_root(out):
    for d in glob.glob(os.path.join(out, "imagenette2*")):
        if os.path.isdir(os.path.join(d, "val")):
            return d
    return None


def download_imagenette(out):
    """Download + extract Imagenette robustly: tries several URLs, atomic download, re-download on a bad archive."""
    root = _imagenette_root(out)
    if root:
        return root
    os.makedirs(out, exist_ok=True)
    last_err = None
    for url in IMAGENETTE_URLS:
        tgz = os.path.join(out, os.path.basename(url))
        for attempt in range(2):
            try:
                if not os.path.exists(tgz) or os.path.getsize(tgz) < 50 * 1024 * 1024:
                    print(f"[data] downloading {url} (attempt {attempt + 1})", flush=True)
                    part = tgz + ".part"
                    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                    with urllib.request.urlopen(req, timeout=180) as r, open(part, "wb") as f:
                        shutil.copyfileobj(r, f)
                    os.replace(part, tgz)
                print("[data] extracting", flush=True)
                with tarfile.open(tgz) as tf:
                    tf.extractall(out)
                root = _imagenette_root(out)
                if root:
                    return root
            except Exception as e:  # noqa: BLE001 - report and try the next URL/attempt
                last_err = e
                print(f"[data] {url} failed: {e}", flush=True)
                if os.path.exists(tgz):
                    os.remove(tgz)
    raise RuntimeError(
        f"Imagenette download failed ({last_err}). Pass --real_src <folder of real images> to skip it "
        "(e.g. REAL_SRC=/path bash run_local_mac.sh data).")


def cmd_real(args):
    names = dict(IMAGENETTE_CLASSES)
    # Escape hatch: use an existing folder of real photos instead of downloading Imagenette.
    if args.real_src:
        src = args.real_src
        if not os.path.isdir(src):
            raise FileNotFoundError(f"--real_src folder not found: {src}")
        files = natsorted([os.path.join(dp, f) for dp, _, fs in os.walk(src)
                           for f in fs if f.lower().endswith((".jpeg", ".jpg", ".png"))], alg=ns.PATH)
        if len(files) < args.n_dev + args.n_test:
            raise ValueError(f"--real_src has only {len(files)} images (< {args.n_dev + args.n_test}); "
                             f"add more or lower N_DEV/N_TEST")
        rng = np.random.default_rng(args.seed)
        rng.shuffle(files)
        order = [(None, f) for f in files[: args.n_dev + args.n_test]]
        print(f"[data] using {len(order)} real images from {src}", flush=True)
    else:
        root = download_imagenette(args.out)
        val_root = os.path.join(root, "val")
        # Use whatever class folders actually exist (robust to a missing class / varied layout).
        class_dirs = sorted(d for d in os.listdir(val_root) if os.path.isdir(os.path.join(val_root, d)))
        if not class_dirs:
            raise RuntimeError(f"no class folders under {val_root}")
        per_class = {}
        for ci, wnid in enumerate(class_dirs):
            files = natsorted([f for f in os.listdir(os.path.join(val_root, wnid)) if f.lower().endswith((".jpeg", ".jpg", ".png"))], alg=ns.PATH)
            rng = np.random.default_rng(args.seed + ci)
            rng.shuffle(files)
            per_class[wnid] = [os.path.join(val_root, wnid, f) for f in files]
        # round-robin over classes so every split is class balanced
        order = []
        i = 0
        need_total = args.n_dev + args.n_test
        while len(order) < need_total and any(i < len(per_class[w]) for w in class_dirs):
            for wnid in class_dirs:
                if i < len(per_class[wnid]):
                    order.append((wnid, per_class[wnid][i]))
            i += 1
        print(f"[data] {len(class_dirs)} real classes found; collected {len(order)} candidate images", flush=True)

    # Save images; skip any that fail to open (e.g. a truncated download) and pull the next one.
    def save_split(out_dir, items):
        os.makedirs(out_dir, exist_ok=True)
        rows = []
        for wnid, path in items:
            if len(rows) >= out_dir_target[out_dir]:
                break
            try:
                img = center_square_resize(Image.open(path))
            except Exception as e:  # noqa: BLE001
                print(f"[data] skip unreadable {path}: {e}", flush=True)
                continue
            image_id = str(len(rows)).rjust(4, "0")
            img.save(os.path.join(out_dir, image_id + ".png"))
            try:
                source = os.path.relpath(path, args.out)
            except ValueError:
                source = path
            rows.append({"image_id": image_id, "class": names.get(wnid, "unknown"),
                         "wnid": wnid or "", "source": source})
        write_manifest(os.path.join(out_dir, "manifest.csv"), rows, ["image_id", "class", "wnid", "source"])
        print(f"[data] {out_dir}: {len(rows)} images", flush=True)
        return len(rows)

    dev_dir = os.path.join(args.out, "real", "dev")
    test_dir = os.path.join(args.out, "real", "test")
    out_dir_target = {dev_dir: args.n_dev, test_dir: args.n_test}
    n_dev_saved = save_split(dev_dir, order[: args.n_dev])
    n_test_saved = save_split(test_dir, order[args.n_dev:])
    if n_dev_saved < args.n_dev or n_test_saved < args.n_test:
        print(f"[data] WARNING: wanted dev {args.n_dev}/test {args.n_test}, got {n_dev_saved}/{n_test_saved}. "
              "If the archive looks truncated, delete data/imagenette2-320* and re-run, or use --real_src.", flush=True)


def cmd_generate(args):
    import torch
    import m2_device
    from diffusers import StableDiffusionPipeline

    dev = m2_device.enable(args.device)
    # fp16 only helps on CUDA; MPS/CPU are more reliable in fp32
    kwargs = {"torch_dtype": torch.float16 if dev == "cuda" else torch.float32, "safety_checker": None}
    model_path = m2_device.local_model(args.model, ignore_fp16=(dev != "cuda"))
    try:
        pipe = StableDiffusionPipeline.from_pretrained(model_path, requires_safety_checker=False, **kwargs)
    except TypeError:
        pipe = StableDiffusionPipeline.from_pretrained(model_path, **kwargs)
    pipe = pipe.to(dev)
    pipe.set_progress_bar_config(disable=True)
    total = args.n_dev + args.n_test
    names = [n for _, n in IMAGENETTE_CLASSES]
    prompts = [f"a photo of a {names[k % len(names)]}" for k in range(total)]
    seeds = [args.seed * 1000 + k for k in range(total)]
    out512 = os.path.join(args.out, "gen512")
    os.makedirs(out512, exist_ok=True)
    gen_device = dev if dev in ("cuda", "mps") else "cpu"
    rows = []
    for start in range(0, total, args.batch_size):
        idx = list(range(start, min(start + args.batch_size, total)))
        gens = [torch.Generator(device=gen_device).manual_seed(seeds[k]) for k in idx]
        with torch.no_grad():
            images = pipe([prompts[k] for k in idx], num_inference_steps=args.steps, guidance_scale=args.guidance,
                          height=512, width=512, generator=gens).images
        for k, im in zip(idx, images):
            image_id = str(k).rjust(4, "0")
            im.save(os.path.join(out512, image_id + ".png"))
            rows.append({"image_id": image_id, "prompt": prompts[k], "seed": seeds[k], "model": args.model,
                         "steps": args.steps, "guidance": args.guidance})
        print(f"[data] generated {min(start + args.batch_size, total)}/{total}", flush=True)
    for split, lo, hi in [("dev", 0, args.n_dev), ("test", args.n_dev, total)]:
        out_dir = os.path.join(args.out, "gen", split)
        os.makedirs(out_dir, exist_ok=True)
        split_rows = []
        for k in range(lo, hi):
            src = str(k).rjust(4, "0")
            image_id = str(k - lo).rjust(4, "0")
            Image.open(os.path.join(out512, src + ".png")).resize((RES, RES), resample=Image.LANCZOS).save(
                os.path.join(out_dir, image_id + ".png"))
            r = dict(rows[k]); r["image_id"] = image_id; r["source512"] = f"gen512/{src}.png"
            split_rows.append(r)
        write_manifest(os.path.join(out_dir, "manifest.csv"), split_rows,
                       ["image_id", "prompt", "seed", "model", "steps", "guidance", "source512"])
        print(f"[data] gen/{split}: {len(split_rows)} images -> {out_dir}", flush=True)


def cmd_check(args):
    import torch
    from torchvision import transforms
    import other_attacks
    import m2_device
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    m2_device.enable(args.device)
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def load_dir(d):
        files = natsorted([f for f in os.listdir(d) if f.endswith(".png")], alg=ns.PATH)
        x = np.stack([np.float32(Image.open(os.path.join(d, f)).convert("RGB")) / 255.0 for f in files])
        return torch.from_numpy(x).permute(0, 3, 1, 2)

    real = load_dir(os.path.join(args.out, "real", "dev"))
    gen = load_dir(os.path.join(args.out, "gen", "dev"))
    print(f"[check] dev real {tuple(real.shape)} | dev generated {tuple(gen.shape)}")
    print(f"{'detector':<10}{'acc(real->1)':>14}{'acc(gen->0)':>13}{'margin real':>13}{'margin gen':>12}")
    rows = []
    for name in args.detectors.split(","):
        det = other_attacks.model_selection(name).eval()
        det.requires_grad_(False)
        out = {}
        for tag, x in [("real", real), ("gen", gen)]:
            preds, margins = [], []
            with torch.no_grad():
                for i in range(0, len(x), 16):
                    logits = det(normalize(x[i:i + 16].cuda()))
                    preds += torch.argmax(logits, 1).tolist()
                    margins += (logits[:, 1] - logits[:, 0]).tolist()
            out[tag] = (np.array(preds), np.array(margins))
        acc_real = float((out["real"][0] == 1).mean())
        acc_gen = float((out["gen"][0] == 0).mean())
        print(f"{name:<10}{acc_real:>14.2f}{acc_gen:>13.2f}{out['real'][1].mean():>13.2f}{out['gen'][1].mean():>12.2f}")
        if acc_real < 0.5 and acc_gen < 0.5:
            print(f"   WARNING: {name} looks label-inverted on this data; check the checkpoint/polarity")
        rows.append({"detector": name, "acc_real": acc_real, "acc_gen": acc_gen,
                     "margin_real": float(out["real"][1].mean()), "margin_gen": float(out["gen"][1].mean())})
        del det
        torch.cuda.empty_cache()
    write_manifest(os.path.join(args.out, "dev_check.csv"), rows,
                   ["detector", "acc_real", "acc_gen", "margin_real", "margin_gen"])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["real", "generate", "check"])
    p.add_argument("--out", default="data")
    p.add_argument("--n_dev", type=int, default=10)
    p.add_argument("--n_test", type=int, default=100)
    p.add_argument("--seed", type=int, default=5230)
    p.add_argument("--real_src", default="",
                   help="folder of real photos to use instead of downloading Imagenette (searched recursively)")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"],
                   help="auto = cuda if available, else mps (Apple Silicon), else cpu")
    p.add_argument("--model", default="stable-diffusion-v1-5/stable-diffusion-v1-5",
                   help="Hugging Face id of the generator (alternative: Manojb/stable-diffusion-2-1-base)")
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--guidance", type=float, default=7.5)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--detectors", default="E,R,D,S")
    args = p.parse_args(argv)
    {"real": cmd_real, "generate": cmd_generate, "check": cmd_check}[args.mode](args)


if __name__ == "__main__":
    main()
