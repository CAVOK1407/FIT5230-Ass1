"""
FIT5230 · Dark.ModelBreakers · Milestone 3 — T1 report (Hypothesis 1: where does the attack reach the saved image?)

    python m3_report.py --runs B00=results/B00 A5=results/m3_sd21/A5 A2=results/m3_sd21/A2 \
        X2=results/m3_sd21/X2 X3=results/m3_sd21/X3 X4=results/m3_sd21/X4 \
        --pairs A5:A2 A5:X3 A5:X2 X2:X4 A5:B00 [--out results/m3_sd21/t1_report.md]

Reads only what m2_stealth.py wrote (records.jsonl and the PNG files). Prints

  0. clean-image consistency across runs  (must be identical: the clean image is never touched)
  1. stage table per run                   (caught images that read 'real' after x1 / x2 / plain decode / final)
  2. pair contrasts                        (decisions that differ on the saved image, score differences,
                                            grey-level differences between the two saved files)
  3. H1(b): success of X2 minus X4 per detector

Configurations (paper draft, Table "Pipeline configurations"):
  A5 = released pipeline (= M2's B00)        A2 = A5 without latent optimisation (--iterations 0)
  X2 = clean control image                   X3 = control image = x1 (second PGD call removed)
  X4 = clean encoded + clean control + no LAO (reconstruction floor)
Contrasts: A5 vs A2 -> what LAO adds to the saved image (H1a, discordance);
           X2 vs X4 -> what the latent route carries when the control image is clean (H1b);
           A5 vs X3 -> what the second PGD call adds;  A5 vs B00 -> noise floor (same configuration twice).
numpy + Pillow only.
"""
import argparse
import json
import math
import os

import numpy as np
from PIL import Image

STAGES = ["pgd", "control", "plain_vae", "final"]
STAGE_LABEL = {"pgd": "x1 (PGD 1)", "control": "x2 (control)", "plain_vae": "plain decode", "final": "saved image"}


def read_records(run_dir):
    recs = {}
    with open(os.path.join(run_dir, "records.jsonl")) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                recs[r["image_id"]] = r
    return recs


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def load_png(path):
    return np.asarray(Image.open(path).convert("RGB")).astype(np.float64)


def psnr(a, b):
    mse = float(np.mean((a - b) ** 2))
    return float("inf") if mse == 0 else 20.0 * math.log10(255.0 / math.sqrt(mse))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", required=True, help="name=dir")
    p.add_argument("--pairs", nargs="*", default=[], help="A:B contrasts between run names")
    p.add_argument("--out", default="", help="also write the report to this markdown file")
    args = p.parse_args(argv)

    runs = {}
    for spec in args.runs:
        name, d = spec.split("=", 1)
        runs[name] = {"dir": d, "rec": read_records(d)}
    names = list(runs)
    ids = sorted(set.intersection(*[set(r["rec"]) for r in runs.values()]))
    dets = sorted(runs[names[0]]["rec"][ids[0]]["final"].keys())
    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    run_list = ", ".join(f"{n} ({runs[n]['dir']})" for n in names)
    out(f"# T1 report — {len(ids)} images present in all runs: {run_list}")
    for n in names:
        cfg = json.load(open(os.path.join(runs[n]["dir"], "run_config.json")))
        keys = ["surrogates", "decode_path", "loss", "iterations", "encode_source", "control_source", "pgd_impl",
                "cap_resolved", "pgd_eot", "lao_eot", "pretrained_diffusion_path", "device", "seed"]
        out(f"- {n}: " + ", ".join(f"{k}={cfg[k]}" for k in keys if k in cfg))

    # ---- 0. clean consistency ----
    out("\n## 0. Clean image, all runs (must agree: the clean image is never modified)")
    out("| detector | caught (pred=AI) per run | max |Δ score| across runs | decisions that differ |")
    out("|---|---|---|---|")
    for d in dets:
        caught = {n: sum(1 for i in ids if runs[n]["rec"][i]["clean"][d]["pred"] == 0) for n in names}
        maxd, ndiff = 0.0, 0
        for i in ids:
            s = [runs[n]["rec"][i]["clean"][d]["real_margin"] for n in names]
            pr = [runs[n]["rec"][i]["clean"][d]["pred"] for n in names]
            maxd = max(maxd, max(s) - min(s))
            ndiff += int(len(set(pr)) > 1)
        out(f"| {d} | {', '.join(f'{n}:{caught[n]}' for n in names)} | {maxd:.4f} | {ndiff} |")

    # ---- 1. stage table ----
    out("\n## 1. Stage table — of the caught images, how many read 'real' at each stage (mean score over all images)")
    for n in names:
        rec = runs[n]["rec"]
        out(f"\n### {n}")
        out("| detector | caught | " + " | ".join(STAGE_LABEL[s] for s in STAGES) + " |")
        out("|---|---|" + "---|" * len(STAGES))
        for d in dets:
            caught = [i for i in ids if rec[i]["clean"][d]["pred"] == 0]
            cells = []
            for s in STAGES:
                vals = [rec[i].get(s) for i in ids]
                if any(v is None for v in vals):
                    cells.append("—")
                    continue
                ok = sum(1 for i in caught if rec[i][s][d]["pred"] == 1)
                m = np.mean([rec[i][s][d]["real_margin"] for i in ids])
                cells.append(f"{ok}/{len(caught)} ({m:+.1f})")
            out(f"| {d} | {len(caught)} | " + " | ".join(cells) + " |")
        secs = [rec[i].get("seconds") for i in ids if rec[i].get("seconds") is not None]
        steps = {}
        for i in ids:
            for k, v in (rec[i].get("pgd_info") or {}).items():
                steps.setdefault(k, []).append(v.get("steps"))
        extra = f"mean {np.mean(secs):.1f} s/image" if secs else ""
        if steps:
            extra += "; PGD steps " + ", ".join(f"{k}: mean {np.mean([x for x in v if x is not None]):.1f}" for k, v in steps.items())
        if extra:
            out(f"({extra})")

    # ---- 2. pairs ----
    if args.pairs:
        out("\n## 2. Contrasts between runs (saved image)")
    for pair in args.pairs:
        a, b = pair.split(":")
        ra, rb = runs[a]["rec"], runs[b]["rec"]
        out(f"\n### {a} vs {b}")
        rms, mx, same = [], [], []
        pa_in, pb_in = [], []
        for i in ids:
            A = load_png(os.path.join(runs[a]["dir"], f"{i}_adv_image.png"))
            B = load_png(os.path.join(runs[b]["dir"], f"{i}_adv_image.png"))
            X = load_png(os.path.join(runs[a]["dir"], f"{i}_originImage.png"))
            diff = np.abs(A - B)
            rms.append(math.sqrt(float(np.mean(diff ** 2))))
            mx.append(float(diff.max()))
            same.append(float(np.mean(diff == 0)))
            pa_in.append(psnr(A, X))
            pb_in.append(psnr(B, X))
        out(f"Saved files: RMS difference mean {np.mean(rms):.3f} grey levels (max over images {np.max(rms):.3f}); "
            f"largest per-pixel difference mean {np.mean(mx):.1f}; identical pixels {np.mean(same):.3f}; "
            f"PSNR to input {a} {np.mean(pa_in):.2f} dB, {b} {np.mean(pb_in):.2f} dB")
        out(f"| detector | caught | real after {a} | real after {b} | decisions differ (all {len(ids)}) | Wilson 95% of discordance | mean |Δ score| | max |Δ score| |")
        out("|---|---|---|---|---|---|---|---|")
        tot_diff = 0
        for d in dets:
            caught = [i for i in ids if ra[i]["clean"][d]["pred"] == 0]
            ok_a = sum(1 for i in caught if ra[i]["final"][d]["pred"] == 1)
            ok_b = sum(1 for i in caught if rb[i]["final"][d]["pred"] == 1)
            diffs = [i for i in ids if ra[i]["final"][d]["pred"] != rb[i]["final"][d]["pred"]]
            ds = [abs(ra[i]["final"][d]["real_margin"] - rb[i]["final"][d]["real_margin"]) for i in ids]
            lo, hi = wilson(len(diffs), len(ids))
            tot_diff += len(diffs)
            out(f"| {d} | {len(caught)} | {ok_a} | {ok_b} | {len(diffs)} | [{lo:.2f}, {hi:.2f}] | {np.mean(ds):.2f} | {np.max(ds):.2f} |")
        out(f"Decisions that differ, summed over detectors: {tot_diff} of {len(ids) * len(dets)}")

    # ---- 3. H1(b) ----
    if "X2" in runs and "X4" in runs:
        out("\n## 3. H1(b): success rate of X2 minus X4 (conditional on images caught before the attack)")
        out("| detector | caught | X4 real | X2 real | difference |")
        out("|---|---|---|---|---|")
        for d in dets:
            caught = [i for i in ids if runs["X4"]["rec"][i]["clean"][d]["pred"] == 0]
            x4 = sum(1 for i in caught if runs["X4"]["rec"][i]["final"][d]["pred"] == 1)
            x2 = sum(1 for i in caught if runs["X2"]["rec"][i]["final"][d]["pred"] == 1)
            n = len(caught)
            out(f"| {d} | {n} | {x4} ({x4 / n:.2f}) | {x2} ({x2 / n:.2f}) | {(x2 - x4) / n:+.2f} |" if n else f"| {d} | 0 | – | – | – |")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"\n[m3_report] wrote {args.out}")


if __name__ == "__main__":
    main()
