#!/usr/bin/env python3
"""M4 role-reversal defence demo 1: two-sided score test.

Idea (from our own M2/M3 finding): the attack pushes the detector's
real-vs-AI margin far *beyond* the range of real photographs (E: attacked
median +152 vs real max +47).  A defender who keeps the released detector
and only adds an upper bound on the margin -- "too real to be real" -- can
recover most escaped images without retraining anything.

Rule:  predict AI  if  margin < 0  (original rule)
                   or  margin > U_q,   U_q = q-quantile of real-image margins.

Calibration without leakage: the 100 real test images are split into two
folds by image index (even / odd).  U_q is fitted on one fold, the false
positive rate is measured on the other fold, and the recall on attacked
images uses that same U_q.  The two folds are swapped and the results are
averaged.  Attacked images are never used to set U_q.

Inputs are the archived M3 measurements (100 SD1.5 images, 4 detectors):
    per_image.csv         attacked-image margins (clean / adv) per condition
    real_predictions.csv  real-image margins per condition
All numbers are recomputed here from those files.
"""
import argparse
import numpy as np
import pandas as pd

DETS = ["E", "R", "D", "S"]


def fit_upper(real_margins, q):
    return float(np.quantile(real_margins, q))


def run(data_dir, runs, conditions, qs, out_csv):
    pi = pd.read_csv(f"{data_dir}/per_image.csv")
    rp = pd.read_csv(f"{data_dir}/real_predictions.csv")
    rp["idx"] = rp["image"].str.replace(".png", "", regex=False).astype(int)
    rp["fold"] = rp["idx"] % 2

    rows = []
    for cond in conditions:
        for det in DETS:
            r = rp[(rp.detector == det) & (rp.condition == cond)]
            for q in qs:
                fprs, lower_fprs = [], []
                per_run = {run: [] for run in runs}
                per_run_n = {run: 0 for run in runs}
                for fold in (0, 1):
                    cal = r[r.fold == fold].real_margin.values
                    hold = r[r.fold != fold].real_margin.values
                    U = fit_upper(cal, q)
                    # FPR of the full two-sided rule on held-out reals
                    fprs.append(np.mean((hold < 0) | (hold > U)))
                    lower_fprs.append(np.mean(hold < 0))
                    for run in runs:
                        a = pi[(pi.run == run) & (pi.detector == det) & (pi.condition == cond)]
                        # conditional set: caught before the attack (same denominator as M3)
                        a = a[a.clean_pred == 0]
                        escaped = a[a.adv_pred == 1]
                        n_esc = len(escaped)
                        caught_back = int((escaped.adv_margin > U).sum())
                        per_run[run].append((n_esc, caught_back, len(a)))
                for run in runs:
                    (n1, c1, d1), (n2, c2, d2) = per_run[run]
                    n_esc = n1  # identical across folds (does not depend on U)
                    recall = (c1 + c2) / (2 * n_esc) if n_esc else np.nan
                    asr_before = n_esc / d1 if d1 else np.nan
                    asr_after = (n_esc - (c1 + c2) / 2) / d1 if d1 else np.nan
                    rows.append(dict(condition=cond, detector=det, q=q, run=run,
                                     n_clean_detected=d1, n_escaped=n_esc,
                                     cond_asr_before=round(asr_before, 3),
                                     caught_back_frac=round(recall, 3),
                                     cond_asr_after=round(asr_after, 3),
                                     real_fpr_one_sided=round(float(np.mean(lower_fprs)), 3),
                                     real_fpr_two_sided=round(float(np.mean(fprs)), 3)))
    out = pd.DataFrame(rows)
    out.to_csv(out_csv, index=False)
    return out


def print_table(out, cond, q):
    sub = out[(out.condition == cond) & (out.q == q)]
    print(f"\n=== condition={cond}  q={q}  (cond. ASR before -> after two-sided test; "
          f"real FPR one-sided -> two-sided) ===")
    runs = list(dict.fromkeys(sub.run))
    hdr = "run".ljust(10) + "".join(d.ljust(22) for d in DETS)
    print(hdr)
    for run in runs:
        line = run.ljust(10)
        for det in DETS:
            x = sub[(sub.run == run) & (sub.detector == det)].iloc[0]
            line += f"{x.cond_asr_before:.2f}->{x.cond_asr_after:.2f} ({x.caught_back_frac:.2f})".ljust(22)
        print(line)
    line = "real FPR".ljust(10)
    for det in DETS:
        x = sub[sub.detector == det].iloc[0]
        line += f"{x.real_fpr_one_sided:.2f}->{x.real_fpr_two_sided:.2f}".ljust(22)
    print(line)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=".")
    ap.add_argument("--runs", default="PGD8,A5,ENS,FULL,EOT_ENS,CAP_E,CAP_ENS")
    ap.add_argument("--conditions", default="none,jpeg95,jpeg75")
    ap.add_argument("--q", default="0.95,1.0")
    ap.add_argument("--out", default="twosided_results.csv")
    a = ap.parse_args()
    runs = a.runs.split(",")
    conds = a.conditions.split(",")
    qs = [float(x) for x in a.q.split(",")]
    out = run(a.data_dir, runs, conds, qs, a.out)
    for cond in conds:
        for q in qs:
            print_table(out, cond, q)
