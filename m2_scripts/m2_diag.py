#!/usr/bin/env python3
"""Decompose each attack variant stage by stage, from the saved records.jsonl.

No torch, no GPU, no model loading - it only reads what the attack already wrote.
Answers three questions:

  1. WHERE does the attack gain and lose?  The pipeline has four checkpoints
     (clean input -> PGD -> ordinary-VAE decode -> Control-VAE decode) and every
     detector's verdict was recorded at each one.  If a variant is fooling a
     detector after PGD but not at the end, the decode is undoing the attack.

  2. IS THE OPTIMISER WORKING?  history[] holds the LAO loss and the surrogate
     margins at every iteration.  Margins must rise (towards "real").  If they
     fall while that detector is in the loss, the gradient has the wrong sign.

  3. Conditional ASR with explicit denominators.

Usage:
    python m2_diag.py --runs B00=results/B00 B10=results/B10 ENS=results/ENS FULL=results/FULL
"""
import argparse
import json
import os

STAGES = ["clean", "pgd", "plain_vae", "final"]
STAGE_LABEL = {"clean": "clean", "pgd": "after PGD", "plain_vae": "plain VAE", "final": "Control-VAE"}


def load_run(run_dir):
    recs = []
    path = os.path.join(run_dir, "records.jsonl")
    if not os.path.exists(path):
        return None, None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    cfg = {}
    cfg_path = os.path.join(run_dir, "run_config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
    return recs, cfg


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


def stage_table(recs, dets):
    """For every detector and stage: how many images read 'real', and the mean margin."""
    print(f"  {'detector':<9}" + "".join(f"{STAGE_LABEL[s]:>22}" for s in STAGES))
    print(f"  {'':<9}" + "".join(f"{'fooled':>10}{'margin':>12}" for _ in STAGES))
    n = len(recs)
    for d in dets:
        row = f"  {d:<9}"
        for s in STAGES:
            fooled = sum(1 for r in recs if r.get(s, {}).get(d, {}).get("pred") == 1)
            m = mean([r.get(s, {}).get(d, {}).get("real_margin") for r in recs])
            row += f"{f'{fooled}/{n}':>10}{m:>+12.2f}"
        print(row)


def lao_health(recs, surrogates, loss_type="ce", kappa=3.0):
    """Is the LAO optimiser doing its job?

    The right test depends on the objective, and they are not the same test:

    "ce" is -CE(logits, AI), which is UNBOUNDED BELOW. There is always more to gain, so
    the loss should keep falling and the margins should keep rising, without limit. A
    falling margin here means the gradient is pointing the wrong way.

    "margin" is relu(kappa - margin), which is BOUNDED BELOW BY ZERO. Once an image is
    past kappa its term is exactly 0 and contributes no gradient; the optimiser then
    spends the remaining capacity on the other terms (LPIPS), which pulls the margin back
    down towards kappa. So a falling margin that stays above kappa is the objective
    working as designed - it is the whole point of using a margin instead of CE - and the
    correct test is whether the final margin cleared kappa, not whether it rose.
    """
    hist = [r.get("history", []) for r in recs if r.get("history")]
    if not hist:
        print("  (no history recorded)")
        return
    n_iter = len(hist[0])
    first_loss = mean([h[0]["attack"] for h in hist])
    last_loss = mean([h[-1]["attack"] for h in hist])
    if loss_type == "margin":
        note = ("at/near 0 = every image already past kappa, nothing left to minimise"
                if abs(last_loss) < 1.0 else "should fall towards 0")
        print(f"  attack loss over {n_iter} iterations: {first_loss:+.3f} -> {last_loss:+.3f}   ({note})")
    else:
        arrow = "down (good)" if last_loss < first_loss else "UP (BAD - optimiser is not minimising)"
        print(f"  attack loss over {n_iter} iterations: {first_loss:+.3f} -> {last_loss:+.3f}   {arrow}")
    if not surrogates:
        return
    if loss_type == "margin":
        print(f"  surrogate margin (mean over {len(hist)} images; must END above kappa={kappa:g}):")
    else:
        print(f"  surrogate margin in the loss (mean over {len(hist)} images; must RISE towards 'real'):")
    for i, name in enumerate(surrogates):
        try:
            f0 = mean([h[0]["margins"][i] for h in hist])
            f1 = mean([h[-1]["margins"][i] for h in hist])
        except (IndexError, KeyError):
            continue
        if loss_type == "margin":
            verdict = (f"clears kappa by {f1 - kappa:+.1f}" if f1 > kappa
                       else "BELOW kappa  <-- objective not satisfied")
        else:
            verdict = "rose" if f1 > f0 else "FELL  <-- wrong direction"
        print(f"    {name}: {f0:+8.2f} -> {f1:+8.2f}   {verdict}")


def conditional_asr(recs, dets):
    """ASR counted only over images the detector caught on the clean input."""
    print(f"  {'detector':<9}{'denominator':>13}{'ASR (final)':>14}{'ASR (after PGD only)':>23}")
    for d in dets:
        caught = [r for r in recs if r.get("clean", {}).get(d, {}).get("pred") == 0]
        if not caught:
            print(f"  {d:<9}{'0':>13}{'n/a':>14}{'n/a':>23}")
            continue
        final_ok = sum(1 for r in caught if r.get("final", {}).get(d, {}).get("pred") == 1)
        pgd_ok = sum(1 for r in caught if r.get("pgd", {}).get(d, {}).get("pred") == 1)
        print(f"  {d:<9}{len(caught):>13}"
              f"{f'{100.0 * final_ok / len(caught):.1f}% ({final_ok}/{len(caught)})':>14}"
              f"{f'{100.0 * pgd_ok / len(caught):.1f}% ({pgd_ok}/{len(caught)})':>23}")


def quality(recs):
    print(f"  PSNR {mean([r['psnr'] for r in recs]):.2f} dB | "
          f"SSIM {mean([r['ssim'] for r in recs]):.3f} | "
          f"Linf {mean([r.get('linf', 0) for r in recs]):.4f} | "
          f"{mean([r.get('seconds', 0) for r in recs]):.1f} s/image")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="NAME=path pairs")
    args = ap.parse_args()

    for spec in args.runs:
        name, _, run_dir = spec.partition("=")
        recs, cfg = load_run(run_dir)
        if not recs:
            print(f"\n### {name}: no records.jsonl in {run_dir} - skipped")
            continue
        surrogates = [s.strip() for s in cfg.get("surrogates", "").split(",") if s.strip()]
        dets = [s.strip() for s in cfg.get("evaluate", "E,R,D,S").split(",") if s.strip()]
        print(f"\n{'=' * 78}")
        print(f"### {name}   surrogates={cfg.get('surrogates')}  loss={cfg.get('loss')}  "
              f"decode_path={cfg.get('decode_path')}  lpips={cfg.get('lpips_weight')}  "
              f"di={cfg.get('di_prob')}  n={len(recs)}")
        print(f"{'=' * 78}")
        print("\n[1] stage by stage  (fooled = detector says 'real'; margin > 0 means 'real')")
        stage_table(recs, dets)
        print("\n[2] is the optimiser working?")
        lao_health(recs, surrogates, cfg.get("loss", "ce"), float(cfg.get("kappa", 3.0)))
        print("\n[3] conditional ASR  (denominator = images the detector caught when clean)")
        conditional_asr(recs, dets)
        print("\n[4] quality")
        quality(recs)
    print()


if __name__ == "__main__":
    main()
