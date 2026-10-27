#!/usr/bin/env python3
"""Generate the LaTeX tables of the M4 defence section directly from the
result CSVs, so that no number is typed by hand."""
import sys
import numpy as np
import pandas as pd

data = sys.argv[1]      # dir with per_image.csv / real_predictions.csv
res = sys.argv[2]       # dir with *_results.csv
out = sys.argv[3]       # output .tex
DETS = ["E", "R", "D", "S"]
RUNS = ["PGD8", "A5", "ENS", "FULL", "EOT_ENS", "CAP_E", "CAP_ENS"]
RUN_LABEL = {"PGD8": "PGD-8", "A5": "Official", "ENS": "ENS", "FULL": "FULL",
             "EOT_ENS": "EOT-ENS", "CAP_E": "CAP-E", "CAP_ENS": "CAP-ENS"}
COND_LABEL = {"none": "none (PNG)", "jpeg95": "JPEG 95", "jpeg75": "JPEG 75",
              "jpeg50": "JPEG 50", "resize075": "resize 0.75", "resize05": "resize 0.5"}
CONDS = list(COND_LABEL)

pi = pd.read_csv(f"{data}/per_image.csv")
rp = pd.read_csv(f"{data}/real_predictions.csv")
L = []

# ---- Table I: margin ranges -------------------------------------------------
L.append(r"""\begin{table}[!t]
\caption{Real-vs-AI margin $m$ (logit\textsubscript{real} $-$ logit\textsubscript{AI}) on lossless PNG. Real: 100 Imagenette photographs. Attacked: SD\,1.5 images caught before the attack (denominator of M3). ``$>$max'' is the fraction of attacked images whose margin exceeds the largest real-image margin.}
\label{tab:margins}
\centering\footnotesize\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{l rrr rr rr rr}
\toprule
 & \multicolumn{3}{c}{Real} & \multicolumn{2}{c}{Official} & \multicolumn{2}{c}{ENS} & \multicolumn{2}{c}{CAP-ENS} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}
Det. & min & med & max & med & $>$max & med & $>$max & med & $>$max \\
\midrule""")
for det in DETS:
    r = rp[(rp.detector == det) & (rp.condition == "none")].real_margin
    row = [det, f"{r.min():.1f}", f"{r.median():.1f}", f"{r.max():.1f}"]
    for run in ["A5", "ENS", "CAP_ENS"]:
        a = pi[(pi.run == run) & (pi.detector == det) & (pi.condition == "none") & (pi.clean_pred == 0)]
        row += [f"{a.adv_margin.median():.1f}", f"{(a.adv_margin > r.max()).mean():.2f}"]
    L.append(" & ".join(row) + r" \\")
L.append(r"""\bottomrule
\end{tabular}
\end{table}
""")

# ---- Table II: two-sided test, q = 1.0, none --------------------------------
ts = pd.read_csv(f"{res}/twosided_results.csv")
ts = ts[(ts.condition == "none") & (ts.q == 1.0)]
L.append(r"""\begin{table}[!t]
\caption{Defence 1, two-sided test on lossless PNG. Upper bound $U$ = largest margin of the calibration real images (two-fold cross-fitting on 100 real images). Cells: conditional ASR before $\rightarrow$ after adding the rule $m > U \Rightarrow$ AI. Last row: real-image false positive rate before $\rightarrow$ after.}
\label{tab:twosided}
\centering\footnotesize
\begin{tabular}{l cccc}
\toprule
Attack & E & R & D & S \\
\midrule""")
for run in RUNS:
    row = [RUN_LABEL[run]]
    for det in DETS:
        x = ts[(ts.run == run) & (ts.detector == det)].iloc[0]
        row.append(f"{x.cond_asr_before:.2f}$\\rightarrow${x.cond_asr_after:.2f}")
    L.append(" & ".join(row) + r" \\")
L.append(r"\midrule")
row = ["Real FPR"]
for det in DETS:
    x = ts[ts.detector == det].iloc[0]
    row.append(f"{x.real_fpr_one_sided:.2f}$\\rightarrow${x.real_fpr_two_sided:.2f}")
L.append(" & ".join(row) + r" \\")
L.append(r"""\bottomrule
\end{tabular}
\end{table}
""")

# ---- Table III: preprocessing, unchanged threshold, ENS ---------------------
pp = pd.read_csv(f"{res}/preprocess_results.csv")
L.append(r"""\begin{table}[!t]
\caption{Defence 2, pre-process then detect with the \emph{unchanged} threshold. Attack = ENS (strongest M2/M3 variant). Left: fraction of the 100 attacked images flagged as AI. Right: real-image false positive rate under the same pre-processing. ``Sys'' = flag if any of the four detectors flags.}
\label{tab:preprocess}
\centering\footnotesize\setlength{\tabcolsep}{2.6pt}
\begin{tabular}{l ccccc ccccc}
\toprule
 & \multicolumn{5}{c}{Attacked flagged (ENS)} & \multicolumn{5}{c}{Real FPR} \\
\cmidrule(lr){2-6}\cmidrule(lr){7-11}
Pre-proc. & E & R & D & S & Sys & E & R & D & S & Sys \\
\midrule""")
for cond in CONDS:
    s = pp[(pp.condition == cond) & (pp.run == "ENS")]
    row = [COND_LABEL[cond]]
    for det in DETS + ["system"]:
        row.append(f"{s[s.detector == det].det_rate.iloc[0]:.2f}")
    for det in DETS + ["system"]:
        row.append(f"{s[s.detector == det].real_fpr.iloc[0]:.2f}")
    L.append(" & ".join(row) + r" \\")
L.append(r"""\bottomrule
\end{tabular}
\end{table}
""")

# ---- Table IV: calibrated two-sided + 2-of-4 vote ---------------------------
cb = pd.read_csv(f"{res}/calibrated_a05_b0.csv")
L.append(r"""\begin{table*}[!t]
\caption{Defence 3, calibrated two-sided thresholds ($L$ = 5\% quantile, $U$ = max of calibration real images, two-fold cross-fitting) and a 2-of-4 vote. Cells: fraction of the 100 attacked images flagged as AI. ``Clean'': un-attacked generated images flagged. ``Real FPR'': held-out real photographs flagged. Rows: pre-processing applied before detection (calibration is repeated under the same pre-processing).}
\label{tab:calibrated}
\centering
\begin{tabular}{l ccccccc cc}
\toprule
Pre-proc. & PGD-8 & Official & ENS & FULL & EOT-ENS & CAP-E & CAP-ENS & Clean & Real FPR \\
\midrule""")
for cond in CONDS:
    s = cb[(cb.condition == cond) & (cb.detector == "vote2")]
    row = [COND_LABEL[cond]]
    for run in RUNS:
        row.append(f"{s[s.run == run].det_rate.iloc[0]:.2f}")
    row.append(f"{s.clean_det.iloc[0]:.2f}")
    row.append(f"{s.real_fpr.iloc[0]:.2f}")
    L.append(" & ".join(row) + r" \\")
L.append(r"""\bottomrule
\end{tabular}
\end{table*}
""")

# ---- Table V: per-detector view of Defence 3 on PNG -------------------------
s = cb[(cb.condition == "none")]
L.append(r"""\begin{table}[!t]
\caption{Defence 3 on lossless PNG, per detector (same thresholds as Table~\ref{tab:calibrated}). Fraction of attacked images flagged; last row real FPR.}
\label{tab:calibrated_det}
\centering
\begin{tabular}{l cccc c}
\toprule
Attack & E & R & D & S & 2-of-4 \\
\midrule""")
for run in RUNS:
    row = [RUN_LABEL[run]]
    for det in DETS + ["vote2"]:
        row.append(f"{s[(s.run == run) & (s.detector == det)].det_rate.iloc[0]:.2f}")
    L.append(" & ".join(row) + r" \\")
L.append(r"\midrule")
row = ["Real FPR"]
for det in DETS + ["vote2"]:
    row.append(f"{s[s.detector == det].real_fpr.iloc[0]:.2f}")
L.append(" & ".join(row) + r" \\")
L.append(r"""\bottomrule
\end{tabular}
\end{table}
""")

open(out, "w").write("\n".join(L))
print("wrote", out)


# =============================================================================
# Enhancement tables (fine-tuned Control-VAE), written to a second file
# =============================================================================
import json
import os
cv = os.environ.get("CVAE_DIR")
if cv:
    L2 = []
    sm = pd.read_csv(f"{cv}/summary.csv")
    RUNS2 = ["X4", "X4_m4", "A5", "A5_m4", "ENS", "ENS_m4"]
    LAB2 = {"X4": "Recon. only, released", "X4_m4": "Recon. only, fine-tuned",
            "A5": "Official, released", "A5_m4": "Official, fine-tuned",
            "ENS": "ENS, released", "ENS_m4": "ENS, fine-tuned"}

    # training log
    log = [json.loads(l) for l in open(f"{cv}/train_log.jsonl") if l.strip()]
    best_tag = [r["tag"] for r in log if r.get("saved_as_best")][-1]
    L2.append(r"""\begin{table}[!t]
\caption{Fine-tuning log on the 10 development images (reconstruction only, PNG-quantised). ``Flagged'': fraction judged AI by each detector. Gate: PSNR $\geq 32$ and LPIPS $\leq 0.05$. Epoch 2 was selected (lowest mean flagged among epochs passing the gate).}
\label{tab:ft_log}
\centering\footnotesize
\begin{tabular}{l cccc cc c}
\toprule
Epoch & E & R & D & S & PSNR & LPIPS & Gate \\
\midrule""")
    for rec in log:
        tag = rec["tag"]
        if tag == "epoch0_released_weights":
            ep = "0 (released)"
        else:
            ep = tag.replace("epoch", "")
            if int(ep) not in (1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30):
                continue
        gate = "--" if "quality_gate" not in rec else ("yes" if rec["quality_gate"] else "no")
        if tag == best_tag:
            ep += r" $\star$"
        L2.append(f"{ep} & {rec['flagged_E']:.1f} & {rec['flagged_R']:.1f} & {rec['flagged_D']:.1f} & {rec['flagged_S']:.1f} & "
                  f"{rec['psnr']:.1f} & {rec['lpips']:.3f} & {gate} \\\\")
    L2.append(r"""\bottomrule
\end{tabular}
\end{table}
""")

    # main result: none and jpeg75
    L2.append(r"""\begin{table*}[!t]
\caption{Fine-tuned control branch vs.\ released weights on the 100 SD\,1.5 test images (never used in training). Conditional ASR per detector (denominator in parentheses in the header), lossless PNG and after JPEG 75. Distortion is measured against the clean input. ``s/img'': attack time per image on the M5 Max (MPS). The fine-tuned weights come from 2 epochs on 150 disjoint images (80 s).}
\label{tab:ft_main}
\centering\footnotesize
\begin{tabular}{l cccc cccc ccc c}
\toprule
 & \multicolumn{4}{c}{Lossless PNG} & \multicolumn{4}{c}{JPEG 75} & & & & \\
\cmidrule(lr){2-5}\cmidrule(lr){6-9}
Configuration & E (94) & R (84) & D (99) & S (98) & E (82) & R (88) & D (99) & S (97) & PSNR & LPIPS & $L_\infty$ & s/img \\
\midrule""")
    for run in RUNS2:
        row = [LAB2[run]]
        for cond in ["none", "jpeg75"]:
            for det in DETS:
                x = sm[(sm.run == run) & (sm.condition == cond) & (sm.detector == det)].iloc[0]
                row.append(f"{x.cond_asr:.2f}")
        x = sm[(sm.run == run) & (sm.condition == "none") & (sm.detector == "E")].iloc[0]
        row += [f"{x.psnr:.1f}", f"{x.lpips:.3f}", f"{x.linf:.3f}", f"{x.seconds:.1f}"]
        L2.append(" & ".join(row) + r" \\")
        if run.endswith("_m4") and run != "ENS_m4":
            L2.append(r"\addlinespace")
    L2.append(r"""\bottomrule
\end{tabular}
\end{table*}
""")

    # AUC + defence cross-check
    cb2 = pd.read_csv(f"{res}/calibrated_cvae_a05.csv")
    cb2 = cb2[cb2.condition == "none"]
    L2.append(r"""\begin{table}[!t]
\caption{The fine-tuned attacker against the Section~\ref{sec:defence} defender (lossless PNG). AUC: real vs.\ attacked images under the released detectors ($0.5$ = inseparable; $0$ = separable because of overshoot). Right: fraction flagged by Defence~3 (calibrated two-sided thresholds, 2-of-4 vote), same thresholds as Table~\ref{tab:calibrated}.}
\label{tab:ft_defence}
\centering\footnotesize\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{l cccc ccccc}
\toprule
 & \multicolumn{4}{c}{AUC real vs.\ attacked} & \multicolumn{5}{c}{Flagged by Defence 3} \\
\cmidrule(lr){2-5}\cmidrule(lr){6-10}
Configuration & E & R & D & S & E & R & D & S & 2-of-4 \\
\midrule""")
    for run in RUNS2:
        row = [LAB2[run]]
        for det in DETS:
            x = sm[(sm.run == run) & (sm.condition == "none") & (sm.detector == det)].iloc[0]
            row.append(f"{x.auc_real_vs_adv:.2f}")
        for det in DETS + ["vote2"]:
            row.append(f"{cb2[(cb2.run == run) & (cb2.detector == det)].det_rate.iloc[0]:.2f}")
        L2.append(" & ".join(row) + r" \\")
    L2.append(r"\midrule")
    row = ["Real FPR", "", "", "", ""]
    for det in DETS + ["vote2"]:
        row.append(f"{cb2[cb2.detector == det].real_fpr.iloc[0]:.2f}")
    L2.append(" & ".join(row) + r" \\")
    L2.append(r"""\bottomrule
\end{tabular}
\end{table}
""")
    out2 = os.path.join(os.path.dirname(out), "tables_enhance.tex")
    open(out2, "w").write("\n".join(L2))
    print("wrote", out2)
