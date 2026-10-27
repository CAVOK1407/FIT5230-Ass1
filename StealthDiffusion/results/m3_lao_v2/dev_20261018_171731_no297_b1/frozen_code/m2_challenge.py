"""
FIT5230 · Dark.ModelBreakers · Milestone 2 — challenge package for other teams.

  python m2_challenge.py build --real_dir data/real/test --gen_dir data/gen/test \
        --attack_dir results/M2_full --n_per_group 30 --out challenge --seed 5230

Builds
  challenge/public/images/<random id>.png    shuffled mix of real photos, raw generated
                                             images and our attacked images (labels hidden)
  challenge/public/README.md                 task + submission format
  challenge/public/submission_template.csv
  challenge/challenge_public.zip             <- share only this
  challenge/private/challenge_key.csv        <- never share

  python m2_challenge.py score --key challenge/private/challenge_key.csv --submission their.csv
Scores a returned CSV (image_id, pred_label with 1 = generated, 0 = real): overall accuracy,
detection rate on raw generated images, detection rate on attacked images, false-positive
rate on real photos.
"""
import argparse
import csv
import hashlib
import json
import os
import random
import shutil
import zipfile

from natsort import natsorted, ns

README = """# Dark.ModelBreakers — Milestone 2 challenge: can you detect our stealthed images?

`images/` contains {n_total} PNG images (224x224) in random order:
- real photographs,
- AI-generated images (Stable Diffusion, unmodified),
- AI-generated images processed by our modified StealthDiffusion attack.

Task: label every image as **1 = generated (AI)** or **0 = real**.
Fill `submission_template.csv` (columns `image_id,pred_label`) and send it back to us on Ed;
we run the scorer against the private key and post: overall accuracy, detection rate on raw
generated images, detection rate on attacked images, and false-positive rate on real photos.

If you want a white-box / adaptive comparison instead, share a detector wrapper with us:
`load_detector() -> torch.nn.Module` whose forward takes float RGB `[B,3,H,W]` in `[0,1]`
and returns fake logits `[B]` (positive = generated).
"""


def cmd_build(args):
    random.seed(args.seed)
    groups = {
        "real": [os.path.join(args.real_dir, f) for f in natsorted(os.listdir(args.real_dir), alg=ns.PATH) if f.endswith(".png")],
        "generated": [os.path.join(args.gen_dir, f) for f in natsorted(os.listdir(args.gen_dir), alg=ns.PATH) if f.endswith(".png")],
        "attacked": [os.path.join(args.attack_dir, f) for f in natsorted(os.listdir(args.attack_dir), alg=ns.PATH) if f.endswith("_adv_image.png")],
    }
    public_img = os.path.join(args.out, "public", "images")
    private = os.path.join(args.out, "private")
    if os.path.exists(os.path.join(args.out, "public")):
        raise FileExistsError(f"{args.out}/public exists; choose a new --out")
    os.makedirs(public_img)
    os.makedirs(private, exist_ok=True)
    key_rows = []
    for group, files in groups.items():
        if len(files) < args.n_per_group:
            raise ValueError(f"group {group} has only {len(files)} images (< {args.n_per_group})")
        chosen = random.sample(files, args.n_per_group)
        for src in chosen:
            image_id = "img_" + hashlib.sha256((str(args.seed) + src).encode()).hexdigest()[:10]
            shutil.copyfile(src, os.path.join(public_img, image_id + ".png"))
            key_rows.append({"image_id": image_id, "group": group, "label": 0 if group == "real" else 1, "source": src})
    random.shuffle(key_rows)
    with open(os.path.join(private, "challenge_key.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image_id", "group", "label", "source"])
        w.writeheader()
        w.writerows(key_rows)
    with open(os.path.join(args.out, "public", "submission_template.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "pred_label"])
        for r in natsorted(key_rows, key=lambda r: r["image_id"], alg=ns.PATH):
            w.writerow([r["image_id"], ""])
    with open(os.path.join(args.out, "public", "README.md"), "w") as f:
        f.write(README.format(n_total=len(key_rows)))
    zip_path = os.path.join(args.out, "challenge_public.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(os.path.join(args.out, "public")):
            for fn in files:
                full = os.path.join(root, fn)
                z.write(full, os.path.relpath(full, os.path.join(args.out, "public")))
    print(f"[challenge] {len(key_rows)} images -> {zip_path} (share this); key in {private}/challenge_key.csv (private)")


def cmd_score(args):
    with open(args.key) as f:
        key = {r["image_id"]: r for r in csv.DictReader(f)}
    with open(args.submission) as f:
        sub = {r["image_id"]: r for r in csv.DictReader(f)}
    missing = [i for i in key if i not in sub or sub[i].get("pred_label", "") == ""]
    stats = {g: {"n": 0, "correct": 0} for g in ("real", "generated", "attacked")}
    for image_id, r in key.items():
        if image_id in missing:
            continue
        pred = int(float(sub[image_id]["pred_label"]))
        g = stats[r["group"]]
        g["n"] += 1
        g["correct"] += int(pred == int(r["label"]))
    n = sum(g["n"] for g in stats.values())
    result = {
        "team": args.team,
        "n_scored": n, "n_missing": len(missing),
        "overall_accuracy": sum(g["correct"] for g in stats.values()) / n if n else None,
        "detection_rate_generated": stats["generated"]["correct"] / stats["generated"]["n"] if stats["generated"]["n"] else None,
        "detection_rate_attacked": stats["attacked"]["correct"] / stats["attacked"]["n"] if stats["attacked"]["n"] else None,
        "false_positive_rate_real": 1 - stats["real"]["correct"] / stats["real"]["n"] if stats["real"]["n"] else None,
    }
    print(json.dumps(result, indent=2))
    if args.save:
        with open(args.save, "w") as f:
            json.dump(result, f, indent=2)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="mode", required=True)
    b = sub.add_parser("build")
    b.add_argument("--real_dir", required=True)
    b.add_argument("--gen_dir", required=True)
    b.add_argument("--attack_dir", required=True)
    b.add_argument("--n_per_group", type=int, default=30)
    b.add_argument("--out", default="challenge")
    b.add_argument("--seed", type=int, default=5230)
    s = sub.add_parser("score")
    s.add_argument("--key", required=True)
    s.add_argument("--submission", required=True)
    s.add_argument("--team", default="unknown")
    s.add_argument("--save", default="")
    args = p.parse_args(argv)
    {"build": cmd_build, "score": cmd_score}[args.mode](args)


if __name__ == "__main__":
    main()
