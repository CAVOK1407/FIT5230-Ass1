#!/usr/bin/env python3
"""M4 role-reversal defence demo 2: pre-process before detection.

Idea: the attack perturbation that reaches the saved image is a pixel-space
PGD perturbation (M3 finding), and M3 measured that it does not survive
JPEG re-compression.  A defender who re-encodes every incoming image
(JPEG q=95/75/50) or down-scales it (0.75x / 0.5x) before running the
unchanged detector removes most of the attack.  The price is a higher
false positive rate on real photographs, which this script also reports.

Metrics (unconditional, i.e. over all 100 attacked images, no denominator
trick):
    det_rate   fraction of attacked images flagged as AI after pre-processing
    real_fpr   fraction of real photographs flagged as AI after pre-processing
    clean_det  fraction of un-attacked generated images flagged (sanity: does
               the pre-processing destroy the detector's own signal?)
"system" rows combine the four detectors with an OR rule (flag if any
detector flags).

Inputs: archived M3 measurements (per_image.csv, real_predictions.csv).
"""
import argparse
import numpy as np
import pandas as pd

DETS = ["E", "R", "D", "S"]
CONDS = ["none", "jpeg95", "jpeg75", "jpeg50", "resize075", "resize05"]


def run(data_dir, runs, out_csv):
    pi = pd.read_csv(f"{data_dir}/per_image.csv")
    rp = pd.read_csv(f"{data_dir}/real_predictions.csv")
    rows = []
    for cond in CONDS:
        r = rp[rp.condition == cond].pivot(index="image", columns="detector", values="pred")
        real_any = (r[DETS] == 0).any(axis=1).mean()
        for run in runs:
            a = pi[(pi.run == run) & (pi.condition == cond)]
            adv = a.pivot(index="image_id", columns="detector", values="adv_pred")
            cln = a.pivot(index="image_id", columns="detector", values="clean_pred")
            for det in DETS:
                rows.append(dict(condition=cond, run=run, detector=det,
                                 det_rate=round((adv[det] == 0).mean(), 3),
                                 clean_det=round((cln[det] == 0).mean(), 3),
                                 real_fpr=round((r[det] == 0).mean(), 3)))
            rows.append(dict(condition=cond, run=run, detector="system",
                             det_rate=round((adv[DETS] == 0).any(axis=1).mean(), 3),
                             clean_det=round((cln[DETS] == 0).any(axis=1).mean(), 3),
                             real_fpr=round(real_any, 3)))
    out = pd.DataFrame(rows)
    out.to_csv(out_csv, index=False)
    return out


def print_tables(out, runs):
    for det in DETS + ["system"]:
        print(f"\n=== detector={det}: detection rate of attacked images (unconditional) ===")
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
    ap.add_argument("--out", default="preprocess_results.csv")
    a = ap.parse_args()
    runs = a.runs.split(",")
    out = run(a.data_dir, runs, a.out)
    print_tables(out, runs)
