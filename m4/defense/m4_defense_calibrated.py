#!/usr/bin/env python3
"""M4 role-reversal defence demo 3: pre-process + re-calibrated two-sided
thresholds (the combination of demos 1 and 2).

Demo 2 showed that "JPEG then detect" with the *unchanged* threshold is
not free: the released detectors were trained on lossless images and start
flagging real photographs after compression (real FPR 0.11-0.63 at q=75).
The fix is cheap: the defender re-calibrates the decision interval on
*compressed real photographs*.  For each detector and pre-processing
condition c the rule becomes

    predict AI  if  margin_c < L_c  or  margin_c > U_c

with L_c the alpha-quantile and U_c the (1-beta)-quantile of real-image
margins measured under the same pre-processing.  alpha controls the
false-positive budget; beta=0 (U_c = max) keeps the two-sided test of
demo 1.

Calibration is cross-fitted on the 100 real test images (even/odd folds):
thresholds come from one fold, FPR is measured on the other, attacked
images never enter calibration.  Numbers are averaged over the two folds.

Outputs, per condition / detector / run:
    det_rate    fraction of attacked images flagged (unconditional, n=100)
    clean_det   fraction of un-attacked generated images flagged
    real_fpr    held-out real false positive rate
"system" = OR over the four detectors (flag if any flags);
"vote2"  = flag if at least two of the four detectors flag.
"""
import argparse
import numpy as np
import pandas as pd

DETS = ["E", "R", "D", "S"]
CONDS = ["none", "jpeg95", "jpeg75", "jpeg50", "resize075", "resize05"]


def run(data_dir, runs, alpha, beta, out_csv):
    pi = pd.read_csv(f"{data_dir}/per_image.csv")
    rp = pd.read_csv(f"{data_dir}/real_predictions.csv")
    rp["idx"] = rp["image"].str.replace(".png", "", regex=False).astype(int)
    rp["fold"] = rp["idx"] % 2
    rows = []
    for cond in CONDS:
        acc = {}  # (run, det) -> list of (det_rate, clean_det, fpr)
        sys_acc = {run: [] for run in runs}
        vote_acc = {}
        for fold in (0, 1):
            thr = {}
            hold_flags = {}
            for det in DETS:
                r = rp[(rp.detector == det) & (rp.condition == cond)]
                cal = r[r.fold == fold].real_margin.values
                hold = r[r.fold != fold].sort_values("idx")
                L = float(np.quantile(cal, alpha))
                U = float(np.quantile(cal, 1 - beta))
                thr[det] = (L, U)
                hold_flags[det] = ((hold.real_margin < L) | (hold.real_margin > U)).values
            hold_cnt = np.column_stack([hold_flags[d] for d in DETS]).sum(axis=1)
            fpr_any = (hold_cnt >= 1).mean()
            fpr_vote = (hold_cnt >= 2).mean()
            for run in runs:
                a = pi[(pi.run == run) & (pi.condition == cond)]
                adv = a.pivot(index="image_id", columns="detector", values="adv_margin")
                cln = a.pivot(index="image_id", columns="detector", values="clean_margin")
                adv_flag, cln_flag = {}, {}
                for det in DETS:
                    L, U = thr[det]
                    adv_flag[det] = ((adv[det] < L) | (adv[det] > U)).values
                    cln_flag[det] = ((cln[det] < L) | (cln[det] > U)).values
                    acc.setdefault((run, det), []).append(
                        (adv_flag[det].mean(), cln_flag[det].mean(), hold_flags[det].mean()))
                adv_cnt = np.column_stack([adv_flag[d] for d in DETS]).sum(axis=1)
                cln_cnt = np.column_stack([cln_flag[d] for d in DETS]).sum(axis=1)
                sys_acc[run].append(((adv_cnt >= 1).mean(), (cln_cnt >= 1).mean(), fpr_any))
                vote_acc.setdefault(run, []).append(((adv_cnt >= 2).mean(), (cln_cnt >= 2).mean(), fpr_vote))
        for run in runs:
            for det in DETS:
                m = np.mean(acc[(run, det)], axis=0)
                rows.append(dict(condition=cond, run=run, detector=det, alpha=alpha, beta=beta,
                                 det_rate=round(m[0], 3), clean_det=round(m[1], 3), real_fpr=round(m[2], 3)))
            m = np.mean(sys_acc[run], axis=0)
            rows.append(dict(condition=cond, run=run, detector="system", alpha=alpha, beta=beta,
                             det_rate=round(m[0], 3), clean_det=round(m[1], 3), real_fpr=round(m[2], 3)))
            m = np.mean(vote_acc[run], axis=0)
            rows.append(dict(condition=cond, run=run, detector="vote2", alpha=alpha, beta=beta,
                             det_rate=round(m[0], 3), clean_det=round(m[1], 3), real_fpr=round(m[2], 3)))
    out = pd.DataFrame(rows)
    out.to_csv(out_csv, index=False)
    return out


def print_tables(out, runs):
    for det in DETS + ["system", "vote2"]:
        print(f"\n=== detector={det}: detection rate of attacked images, calibrated two-sided rule ===")
        print("condition".ljust(11) + "".join(r.ljust(9) for r in runs) + "clean".ljust(9) + "realFPR")
        for cond in CONDS:
            s = out[(out.condition == cond) & (out.detector == det)]
            line = cond.ljust(11)
            for run in runs:
                line += f"{s[s.run == run].det_rate.iloc[0]:.2f}".ljust(9)
            line += f"{s.clean_det.iloc[0]:.2f}".ljust(9) + f"{s.real_fpr.iloc[0]:.2f}"
            print(line)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=".")
    ap.add_argument("--runs", default="PGD8,A5,ENS,FULL,EOT_ENS,CAP_E,CAP_ENS")
    ap.add_argument("--alpha", type=float, default=0.05, help="lower-tail FPR budget per detector")
    ap.add_argument("--beta", type=float, default=0.0, help="upper-tail budget; 0 = use max of real")
    ap.add_argument("--out", default="calibrated_results.csv")
    a = ap.parse_args()
    runs = a.runs.split(",")
    out = run(a.data_dir, runs, a.alpha, a.beta, a.out)
    print_tables(out, runs)
